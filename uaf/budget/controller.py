"""BudgetController — мониторинг бюджета UAF-сессии в отдельном потоке."""

import logging
import os
import shutil
import signal
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from uaf.budget.convergence import check_convergence_with_llm_signal
from uaf.budget.status_file import (
    AlertEntry,
    BudgetStatusV21,
    DataQuality,
    SoftwareHealth,
    write_budget_status,
)

logger = logging.getLogger(__name__)

# 12 алерт-кодов из дизайн-документа
AlertCode = Literal[
    "SW-HANG",
    "SW-DISK-FULL",
    "MQ-NAN-CASCADE",
    "DQ-DATA-MODIFIED",
    "MQ-DEGRADATION",
    "MQ-CONSECUTIVE-FAILS",
    "BQ-BUDGET-80PCT",
    "BQ-TIME-80PCT",
    "SW-MLFLOW-DOWN",
    "DQ-SCHEMA-DRIFT",
    "MQ-NEW-BEST",
    "MQ-CONVERGENCE",
    "BQ-ITER-COMPLETE",
    "MQ-LEAKAGE-SUSPECT",
]

_ALERT_LEVELS: dict[str, str] = {
    "SW-HANG": "CRITICAL",
    "SW-DISK-FULL": "CRITICAL",
    "MQ-NAN-CASCADE": "CRITICAL",
    "DQ-DATA-MODIFIED": "CRITICAL",
    "MQ-LEAKAGE-SUSPECT": "CRITICAL",
    "MQ-DEGRADATION": "WARNING",
    "MQ-CONSECUTIVE-FAILS": "WARNING",
    "BQ-BUDGET-80PCT": "WARNING",
    "BQ-TIME-80PCT": "WARNING",
    "SW-MLFLOW-DOWN": "WARNING",
    "DQ-SCHEMA-DRIFT": "WARNING",
    "MQ-NEW-BEST": "INFO",
    "MQ-CONVERGENCE": "INFO",
    "BQ-ITER-COMPLETE": "INFO",
}

_MIN_DISK_FREE_GB = 1.0
_HANG_TIMEOUT_SECONDS = 7200.0  # 2 часа — Optuna и ансамбли молчат долго
_HANG_RECHECK_SECONDS = 60.0
_NAN_CASCADE_THRESHOLD = 3  # подряд NaN => cascade
_CONSECUTIVE_FAIL_THRESHOLD = 3


@dataclass
class BudgetConfig:
    """Конфигурация бюджета сессии.

    Атрибуты:
        mode: режим бюджета (fixed/dynamic).
        max_iterations: максимальное число итераций (fixed mode).
        max_cost_usd: максимальная стоимость (fixed mode, не используется в v0.1).
        max_time_hours: максимальное время (fixed/dynamic).
        patience: число итераций без улучшения для конвергенции (dynamic).
        min_delta: минимальное улучшение метрики (dynamic).
        min_iterations: минимум итераций до проверки конвергенции (dynamic).
        safety_cap_iterations: жёсткий лимит итераций (dynamic).
        safety_cap_time_hours: жёсткий лимит времени (dynamic).
        soft_warning_fraction: доля бюджета для мягкого предупреждения.
        poll_interval_seconds: интервал опроса MLflow.
        grace_period_seconds: время ожидания после hard_stop перед SIGTERM.
        metric_direction: направление оптимизации (maximize/minimize).
    """

    mode: Literal["fixed", "dynamic"] = "fixed"
    max_iterations: int | None = 20
    max_cost_usd: float | None = None
    max_time_hours: float | None = 8.0
    patience: int = 3
    min_delta: float = 0.001
    min_iterations: int = 3
    safety_cap_iterations: int = 50
    safety_cap_time_hours: float = 24.0
    soft_warning_fraction: float = 0.8
    poll_interval_seconds: int = 30
    grace_period_seconds: int = 60
    metric_direction: Literal["maximize", "minimize"] = "maximize"
    metric_name: str = "roi"
    leakage_sanity_threshold: float | None = None
    leakage_soft_warning: float | None = None


@dataclass
class _MonitoringState:
    """Внутреннее состояние мониторинга.

    Атрибуты:
        best_metric: лучшее значение метрики за сессию.
        consecutive_fails: число подряд идущих failed runs.
        consecutive_nan: число подряд идущих NaN метрик.
        llm_signal_consecutive: число подряд идущих высоких LLM сигналов.
        triggered_alerts: множество уже сработавших кодов (для дедупликации WARNING).
        data_schema_hash: хэш схемы входных данных при старте.
    """

    best_metric: float | None = None
    consecutive_fails: int = 0
    consecutive_nan: int = 0
    llm_signal_consecutive: int = 0
    triggered_alerts: set[str] = field(default_factory=set)
    data_schema_hash: str = ""
    investigate_leakage: bool = False
    leakage_investigated: bool = False


class BudgetController:
    """Мониторинг бюджета в отдельном polling-потоке.

    Каждые `poll_interval_seconds` читает MLflow runs, считает итерации,
    проверяет метрики и пишет budget_status.json атомарно.

    При исчерпании бюджета:
        1. Устанавливает hard_stop=True в budget_status.json
        2. Ждёт grace_period_seconds
        3. Отправляет SIGTERM процессу claude
        4. Ждёт 30 секунд, при необходимости SIGKILL

    Атрибуты:
        budget_status_path: путь к budget_status.json.
        experiment_id: MLflow experiment ID сессии.
        tracking_uri: MLflow tracking URI.
        config: конфигурация бюджета.
        session_id: идентификатор сессии.
        data_files: список входных файлов для DQ-DATA-MODIFIED мониторинга.
    """

    def __init__(
        self,
        budget_status_path: Path,
        experiment_id: str,
        tracking_uri: str,
        config: BudgetConfig,
        session_id: str,
        data_files: list[Path] | None = None,
    ) -> None:
        self.budget_status_path = budget_status_path
        self.experiment_id = experiment_id
        self.tracking_uri = tracking_uri
        self.config = config
        self.session_id = session_id
        self.data_files = data_files or []

        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._state = _MonitoringState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._claude_pid: int | None = None
        self._session_start_time = time.time()
        self._last_stdout_time = time.time()

        # Вычисляем хэши входных данных для DQ-DATA-MODIFIED
        if self.data_files:
            self._state.data_schema_hash = self._compute_data_hash()

    def set_claude_pid(self, pid: int) -> None:
        """Устанавливает PID Claude Code subprocess после его старта."""
        self._claude_pid = pid
        logger.info("BudgetController: claude_pid установлен = %d", pid)

    def start(self, claude_pid: int | None = None) -> None:
        """Запускает polling-поток мониторинга.

        Args:
            claude_pid: PID процесса Claude Code для отправки сигналов.
        """
        self._claude_pid = claude_pid
        self._session_start_time = time.time()

        # Инициализируем budget_status.json
        initial_status = self._build_status(alerts=[])
        write_budget_status(initial_status, self.budget_status_path)

        self._thread = threading.Thread(
            target=self._polling_loop,
            name=f"BudgetController-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "BudgetController запущен (interval=%ds, pid=%s)",
            self.config.poll_interval_seconds,
            claude_pid,
        )

    def stop(self) -> None:
        """Останавливает polling-поток."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("BudgetController остановлен")

    def update_stdout_time(self) -> None:
        """Обновляет время последнего вывода (для детекции зависания)."""
        self._last_stdout_time = time.time()

    def _polling_loop(self) -> None:
        """Основной цикл опроса MLflow и обновления budget_status.json."""
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                logger.error("Ошибка в BudgetController polling: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self.config.poll_interval_seconds)

    def _poll_once(self) -> None:
        """Один цикл опроса: читает MLflow, считает метрики, пишет статус."""
        time_elapsed = time.time() - self._session_start_time
        alerts: list[AlertEntry] = []

        # Проверка SW health
        sw_health = self._check_software_health()
        if not sw_health.mlflow_reachable:
            alerts.append(self._make_alert("SW-MLFLOW-DOWN", "MLflow сервер недоступен"))

        if sw_health.disk_free_gb < _MIN_DISK_FREE_GB:
            alerts.append(
                self._make_alert(
                    "SW-DISK-FULL",
                    f"Диск почти заполнен: {sw_health.disk_free_gb:.1f} ГБ свободно",
                )
            )

        # Чтение runs из MLflow
        runs = self._fetch_runs()
        iterations_used = self._count_iterations(runs)
        metrics_history = self._extract_metrics_history(runs)
        llm_signal = self._extract_llm_signal(runs)

        # Обновление LLM signal consecutive counter
        if llm_signal >= 0.9:
            self._state.llm_signal_consecutive += 1
        else:
            self._state.llm_signal_consecutive = 0

        # Подсчёт failed runs
        consecutive_fails = self._count_consecutive_fails(runs)
        _nan_count = self._count_nan_metrics(runs)

        # Алерты по метрикам
        if metrics_history:
            latest = metrics_history[-1]
            import math

            if math.isnan(latest):
                self._state.consecutive_nan += 1
            else:
                self._state.consecutive_nan = 0

            if self._state.consecutive_nan >= _NAN_CASCADE_THRESHOLD:
                alerts.append(
                    self._make_alert(
                        "MQ-NAN-CASCADE",
                        f"NaN метрика {self._state.consecutive_nan} раз подряд",
                    )
                )

            if self._state.best_metric is None:
                leakage_alerts = self._check_leakage_thresholds(latest, is_first=True)
                alerts.extend(leakage_alerts)
                # Устанавливаем best_metric только если нет CRITICAL leakage алерта
                if not any(a.code == "MQ-LEAKAGE-SUSPECT" and a.level == "CRITICAL" for a in leakage_alerts):
                    self._state.best_metric = latest
                    if not leakage_alerts:
                        alerts.append(self._make_alert("MQ-NEW-BEST", f"Первый результат: {latest:.6f}"))
            else:
                is_better = (
                    latest > self._state.best_metric
                    if self.config.metric_direction == "maximize"
                    else latest < self._state.best_metric
                )
                if is_better and not math.isnan(latest):
                    leakage_alerts = self._check_leakage_thresholds(latest, is_first=False)
                    alerts.extend(leakage_alerts)
                    # Обновляем best_metric только если нет CRITICAL leakage алерта
                    if not any(a.code == "MQ-LEAKAGE-SUSPECT" and a.level == "CRITICAL" for a in leakage_alerts):
                        self._state.best_metric = latest
                        if not leakage_alerts:
                            alerts.append(
                                self._make_alert("MQ-NEW-BEST", f"Новый лучший результат: {latest:.6f}")
                            )
                elif (
                    len(metrics_history) >= 3
                    and "MQ-DEGRADATION" not in self._state.triggered_alerts
                ):
                    # Деградация: последние 3 результата хуже лучшего
                    recent_3 = metrics_history[-3:]
                    if self.config.metric_direction == "maximize":
                        degraded = all(
                            v < self._state.best_metric * 0.95
                            for v in recent_3
                            if not math.isnan(v)
                        )
                    else:
                        degraded = all(
                            v > self._state.best_metric * 1.05
                            for v in recent_3
                            if not math.isnan(v)
                        )
                    if degraded:
                        alerts.append(
                            self._make_alert(
                                "MQ-DEGRADATION",
                                f"Деградация метрики: лучшее={self._state.best_metric:.4f}",
                            )
                        )
                        self._state.triggered_alerts.add("MQ-DEGRADATION")

        if consecutive_fails >= _CONSECUTIVE_FAIL_THRESHOLD:
            alerts.append(
                self._make_alert(
                    "MQ-CONSECUTIVE-FAILS",
                    f"{consecutive_fails} провалов подряд",
                )
            )

        # Проверка данных
        dq = self._check_data_quality()
        if dq.files_modified:
            alerts.append(
                self._make_alert("DQ-DATA-MODIFIED", "Входные данные изменены во время сессии")
            )

        # Проверка зависания
        stdout_age = time.time() - self._last_stdout_time
        if stdout_age > _HANG_TIMEOUT_SECONDS and sw_health.process_alive:
            time.sleep(_HANG_RECHECK_SECONDS)
            # Повторная проверка после паузы
            new_stdout_age = time.time() - self._last_stdout_time
            if new_stdout_age > _HANG_TIMEOUT_SECONDS:
                alerts.append(
                    self._make_alert(
                        "SW-HANG",
                        f"Нет вывода {new_stdout_age:.0f} сек — возможное зависание",
                    )
                )

        # Подсчёт бюджетной доли
        budget_fraction, time_fraction = self._compute_budget_fractions(
            iterations_used, time_elapsed
        )

        # 80% предупреждения
        budget_warn_not_seen = "BQ-BUDGET-80PCT" not in self._state.triggered_alerts
        if budget_fraction >= self.config.soft_warning_fraction and budget_warn_not_seen:
            alerts.append(
                self._make_alert(
                    "BQ-BUDGET-80PCT",
                    f"Использовано {budget_fraction * 100:.0f}% итерационного бюджета",
                )
            )
            self._state.triggered_alerts.add("BQ-BUDGET-80PCT")

        time_warn_not_seen = "BQ-TIME-80PCT" not in self._state.triggered_alerts
        if time_fraction >= self.config.soft_warning_fraction and time_warn_not_seen:
            alerts.append(
                self._make_alert(
                    "BQ-TIME-80PCT",
                    f"Использовано {time_fraction * 100:.0f}% временного бюджета",
                )
            )
            self._state.triggered_alerts.add("BQ-TIME-80PCT")

        # Определение hard stop
        hard_stop, hard_stop_reason = self._check_hard_stop(
            iterations_used=iterations_used,
            time_elapsed=time_elapsed,
            metrics_history=metrics_history,
            llm_signal=llm_signal,
            alerts=alerts,
        )

        # Конвергенция
        if hard_stop and hard_stop_reason in ("metric_convergence", "llm_signal"):
            alerts.append(
                self._make_alert(
                    "MQ-CONVERGENCE",
                    f"Конвергенция: {hard_stop_reason}",
                )
            )

        # Подсказки для Claude Code
        hints = self._generate_hints(
            iterations_used, time_elapsed, budget_fraction, metrics_history
        )

        status = self._build_status(
            alerts=alerts,
            iterations_used=iterations_used,
            time_elapsed=time_elapsed,
            metrics_history=metrics_history,
            budget_fraction=max(budget_fraction, time_fraction),
            hard_stop=hard_stop,
            hard_stop_reason=hard_stop_reason,
            llm_signal=llm_signal,
            sw_health=sw_health,
            dq=dq,
            hints=hints,
        )
        write_budget_status(status, self.budget_status_path)

        # Логируем CRITICAL алерты
        for alert in alerts:
            if alert.level == "CRITICAL":
                logger.critical("[%s] %s", alert.code, alert.message)
            elif alert.level == "WARNING":
                logger.warning("[%s] %s", alert.code, alert.message)
            else:
                logger.info("[%s] %s", alert.code, alert.message)

        if hard_stop:
            logger.warning("Hard stop инициирован: %s", hard_stop_reason)
            self._execute_hard_stop(hard_stop_reason or "budget_exhausted")

    def _fetch_runs(self) -> list[Run]:
        """Читает все experiment runs из MLflow.

        Returns:
            Список MLflow Run объектов.
        """
        try:
            return self._client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.session_id = '{self.session_id}'",
                max_results=500,
            )
        except Exception as exc:
            logger.warning("Ошибка чтения MLflow runs: %s", exc)
            return []

    def _count_iterations(self, runs: list) -> int:  # type: ignore[type-arg]
        """Считает количество итераций (runs с type=experiment).

        Args:
            runs: список MLflow runs.

        Returns:
            Количество завершённых итераций.
        """
        return sum(
            1
            for r in runs
            if r.data.tags.get("type") == "experiment"
            and r.data.tags.get("status") in ("success", "failed")
        )

    def _extract_metrics_history(self, runs: list) -> list[float]:  # type: ignore[type-arg]
        """Извлекает историю метрик из успешных runs, сортированных по времени.

        Args:
            runs: список MLflow runs.

        Returns:
            Список значений метрик по порядку запуска.
        """
        successful = [
            r
            for r in runs
            if r.data.tags.get("type") in ("experiment", "chain_verify")
            and r.data.tags.get("status") == "success"
            and r.data.metrics
        ]
        # Сортируем по времени старта
        successful.sort(key=lambda r: r.info.start_time)
        import math

        target_metric = self.config.metric_name
        maximize = self.config.metric_direction == "maximize"
        history = []
        for run in successful:
            metrics = run.data.metrics
            if not metrics:
                continue
            val = self._pick_metric(metrics, target_metric, maximize)
            if val is not None and not math.isnan(val):
                history.append(val)
        return history

    def _pick_metric(
        self,
        metrics: dict[str, float],
        target_metric: str,
        maximize: bool,
    ) -> float | None:
        """Выбирает значение целевой метрики из словаря MLflow-метрик.

        Стратегия (в порядке приоритета):
        1. Точное совпадение ключа с target_metric.
        2. Ключи вида "{target}_test_best" или "best_{target}_test" — тест-метрика явная.
        3. Ключи содержащие target И "test", но НЕ содержащие "val" — test ROI.
        4. Ключи вида "{target}_best*" без "val".
        5. Любые ключи с target, кроме явных val-метрик.
        6. Fallback: первая не-NaN метрика.

        Val-метрики (содержащие "val") намеренно исключаются: они всегда выше test
        и не отражают реальное качество модели.
        """
        import math

        clean = {k: v for k, v in metrics.items() if not math.isnan(v)}
        if not clean:
            return None

        # 1. Точное совпадение
        if target_metric in clean:
            return clean[target_metric]

        # 2. Явные test-метрики: {target}_test_best, best_{target}_test, {target}_test
        priority_patterns = [
            f"{target_metric}_test_best",
            f"best_{target_metric}_test",
            f"{target_metric}_test",
        ]
        for pat in priority_patterns:
            if pat in clean:
                return clean[pat]

        # 3. Ключи содержащие target И "test", без "val"
        test_keys = [
            k for k in clean
            if target_metric in k and "test" in k and "val" not in k
        ]
        if test_keys:
            vals = [clean[k] for k in test_keys]
            return max(vals) if maximize else min(vals)

        # 4. Ключи {target}_best* без val
        best_keys = [
            k for k in clean
            if k.startswith(f"{target_metric}_best") and "val" not in k
        ]
        if best_keys:
            vals = [clean[k] for k in best_keys]
            return max(vals) if maximize else min(vals)

        # 5. Любые ключи с target, кроме явных val-метрик
        non_val_keys = [
            k for k in clean
            if target_metric in k and "val" not in k
        ]
        if non_val_keys:
            vals = [clean[k] for k in non_val_keys]
            return max(vals) if maximize else min(vals)

        # 6. Fallback (включая val, если больше ничего нет)
        return next(iter(clean.values()))

    def _extract_llm_signal(self, runs: list) -> float:  # type: ignore[type-arg]
        """Читает последний convergence_signal от Claude Code.

        Args:
            runs: список MLflow runs.

        Returns:
            Последний convergence_signal (0.0-1.0).
        """
        experiment_runs = [
            r
            for r in runs
            if r.data.tags.get("type") == "experiment" and r.data.tags.get("status") == "success"
        ]
        if not experiment_runs:
            return 0.0
        latest = max(experiment_runs, key=lambda r: r.info.start_time)
        try:
            return float(latest.data.tags.get("convergence_signal", "0.0"))
        except (ValueError, TypeError):
            return 0.0

    def _count_consecutive_fails(self, runs: list) -> int:  # type: ignore[type-arg]
        """Считает число провалов подряд в конце истории runs.

        Args:
            runs: список MLflow runs.

        Returns:
            Число последовательных failed runs.
        """
        experiment_runs = [r for r in runs if r.data.tags.get("type") == "experiment"]
        experiment_runs.sort(key=lambda r: r.info.start_time)
        count = 0
        for run in reversed(experiment_runs):
            if run.data.tags.get("status") == "failed":
                count += 1
            else:
                break
        return count

    def _count_nan_metrics(self, runs: list) -> int:  # type: ignore[type-arg]
        """Считает число runs с NaN метриками.

        Args:
            runs: список MLflow runs.

        Returns:
            Количество runs с NaN метриками.
        """
        import math

        count = 0
        for run in runs:
            has_metrics = run.data.tags.get("type") == "experiment" and run.data.metrics
            if has_metrics and any(math.isnan(v) for v in run.data.metrics.values()):
                count += 1
        return count

    def _check_software_health(self) -> SoftwareHealth:
        """Проверяет состояние программного окружения.

        Returns:
            SoftwareHealth с текущими показателями.
        """
        mlflow_ok = self._ping_mlflow()
        disk_free = shutil.disk_usage("/").free / (1024**3)
        process_alive = self._is_claude_alive()
        stdout_age = time.time() - self._last_stdout_time
        return SoftwareHealth(
            mlflow_reachable=mlflow_ok,
            disk_free_gb=round(disk_free, 2),
            process_alive=process_alive,
            last_stdout_age_seconds=round(stdout_age, 1),
        )

    def _ping_mlflow(self) -> bool:
        """Проверяет доступность MLflow сервера.

        Returns:
            True если сервер отвечает.
        """
        try:
            health_url = self.tracking_uri.rstrip("/") + "/health"
            urllib.request.urlopen(health_url, timeout=3)
            return True
        except Exception:
            return False

    def _is_claude_alive(self) -> bool:
        """Проверяет жив ли процесс Claude Code.

        Returns:
            True если процесс жив или PID не задан.
        """
        if self._claude_pid is None:
            return True
        try:
            os.kill(self._claude_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # процесс существует, нет прав

    def _check_data_quality(self) -> DataQuality:
        """Проверяет неизменность входных данных.

        Returns:
            DataQuality с текущим состоянием.
        """
        if not self.data_files:
            return DataQuality(schema_hash=self._state.data_schema_hash, files_modified=False)

        current_hash = self._compute_data_hash()
        schema_was_set = bool(self._state.data_schema_hash)
        modified = current_hash != self._state.data_schema_hash and schema_was_set
        return DataQuality(schema_hash=current_hash, files_modified=modified)

    def _compute_data_hash(self) -> str:
        """Вычисляет хэш mtime+size всех входных файлов.

        Returns:
            Строковый хэш состояния файлов.
        """
        import hashlib

        hasher = hashlib.sha256()
        for f in sorted(self.data_files):
            if f.exists():
                stat = f.stat()
                hasher.update(f"{f}:{stat.st_mtime}:{stat.st_size}".encode())
        return hasher.hexdigest()

    def _compute_budget_fractions(
        self, iterations_used: int, time_elapsed: float
    ) -> tuple[float, float]:
        """Вычисляет доли использованного итерационного и временного бюджета.

        Args:
            iterations_used: использованные итерации.
            time_elapsed: прошедшее время в секундах.

        Returns:
            Tuple (budget_fraction, time_fraction).
        """
        if self.config.mode == "fixed" and self.config.max_iterations:
            budget_fraction = iterations_used / self.config.max_iterations
        elif self.config.mode == "dynamic":
            budget_fraction = iterations_used / self.config.safety_cap_iterations
        else:
            budget_fraction = 0.0

        time_limit_sec = (self.config.max_time_hours or self.config.safety_cap_time_hours) * 3600
        time_fraction = time_elapsed / time_limit_sec if time_limit_sec > 0 else 0.0

        return min(budget_fraction, 1.0), min(time_fraction, 1.0)

    def _check_hard_stop(
        self,
        iterations_used: int,
        time_elapsed: float,
        metrics_history: list[float],
        llm_signal: float,
        alerts: list[AlertEntry],
    ) -> tuple[bool, str]:
        """Определяет нужен ли hard stop и по какой причине.

        Args:
            iterations_used: использованные итерации.
            time_elapsed: прошедшее время.
            metrics_history: история метрик.
            llm_signal: последний LLM convergence signal.
            alerts: текущие активные алерты.

        Returns:
            Tuple (hard_stop, reason).
        """
        # CRITICAL алерты всегда вызывают hard stop
        for alert in alerts:
            if alert.level == "CRITICAL":
                return True, f"critical_alert:{alert.code}"

        time_limit_sec = (self.config.max_time_hours or self.config.safety_cap_time_hours) * 3600
        if time_elapsed >= time_limit_sec:
            return True, "time_limit_exceeded"

        if self.config.mode == "fixed":
            if self.config.max_iterations and iterations_used >= self.config.max_iterations:
                return True, "iterations_limit_reached"
        elif self.config.mode == "dynamic":
            if iterations_used >= self.config.safety_cap_iterations:
                return True, "safety_cap_iterations"

            converged, reason = check_convergence_with_llm_signal(
                metrics_history=metrics_history,
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                min_iterations=self.config.min_iterations,
                llm_convergence_signal=llm_signal,
                llm_consecutive_count=self._state.llm_signal_consecutive,
            )
            if converged:
                return True, reason

        return False, ""

    def _generate_hints(
        self,
        iterations_used: int,
        time_elapsed: float,
        budget_fraction: float,
        metrics_history: list[float],
    ) -> list[str]:
        """Генерирует подсказки для Claude Code на основе текущего состояния.

        Args:
            iterations_used: использованные итерации.
            time_elapsed: прошедшее время.
            budget_fraction: доля использованного бюджета.
            metrics_history: история метрик.

        Returns:
            Список строк-подсказок.
        """
        hints = []
        if self._state.investigate_leakage and not self._state.leakage_investigated:
            hints.append(
                "investigate_leakage: true — метрика превысила soft_warning порог. "
                "Запусти Leakage Investigation Protocol (STEP L.1–L.5 в program.md)."
            )
        if budget_fraction >= 0.8:
            iter_cap = self.config.max_iterations or self.config.safety_cap_iterations
            remaining = int((1 - budget_fraction) * iter_cap)
            hints.append(
                f"Использовано {budget_fraction * 100:.0f}% бюджета. "
                f"Осталось ~{remaining} итераций."
            )
        if len(metrics_history) >= 5 and self._state.best_metric is not None:
            latest = metrics_history[-1]
            delta_pct = (
                abs(latest - self._state.best_metric)
                / (abs(self._state.best_metric) + 1e-10)
                * 100
            )
            if delta_pct < 0.1:
                hints.append(
                    "Метрика стабилизировалась (изменение < 0.1%). "
                    "Рассмотри завершение или смену подхода."
                )
        return hints

    def _build_status(
        self,
        alerts: list[AlertEntry],
        iterations_used: int = 0,
        time_elapsed: float = 0.0,
        metrics_history: list[float] | None = None,
        budget_fraction: float = 0.0,
        hard_stop: bool = False,
        hard_stop_reason: str | None = None,
        llm_signal: float = 0.0,
        sw_health: SoftwareHealth | None = None,
        dq: DataQuality | None = None,
        hints: list[str] | None = None,
    ) -> BudgetStatusV21:
        """Собирает полную структуру BudgetStatusV21.

        Args:
            alerts: список алертов.
            iterations_used: использованные итерации.
            time_elapsed: прошедшее время.
            metrics_history: история метрик.
            budget_fraction: доля бюджета.
            hard_stop: флаг hard stop.
            hard_stop_reason: причина.
            llm_signal: LLM convergence signal.
            sw_health: состояние ПО.
            dq: качество данных.
            hints: подсказки.

        Returns:
            Готовая структура BudgetStatusV21.
        """
        time_limit_sec = (
            (self.config.max_time_hours or self.config.safety_cap_time_hours) * 3600
            if (self.config.max_time_hours or self.config.safety_cap_time_hours)
            else None
        )
        iterations_limit = (
            self.config.max_iterations
            if self.config.mode == "fixed"
            else self.config.safety_cap_iterations
        )
        warning_triggered = (
            "BQ-BUDGET-80PCT" in self._state.triggered_alerts
            or "BQ-TIME-80PCT" in self._state.triggered_alerts
        )
        phase = "monitoring"
        if hard_stop:
            phase = "grace_period"

        return BudgetStatusV21(
            session_id=self.session_id,
            iterations_used=iterations_used,
            iterations_limit=iterations_limit,
            time_elapsed=round(time_elapsed, 1),
            time_limit=time_limit_sec,
            hard_stop=hard_stop,
            hard_stop_reason=hard_stop_reason,
            phase=phase,
            alerts=alerts,
            hints=hints or [],
            software_health=sw_health or SoftwareHealth(),
            data_quality=dq or DataQuality(),
            metrics_history=metrics_history or [],
            budget_fraction_used=round(budget_fraction, 4),
            warning_triggered=warning_triggered,
            convergence_signal=llm_signal,
            investigate_leakage=self._state.investigate_leakage,
            leakage_investigated=self._state.leakage_investigated,
            timestamp=time.time(),
        )

    def _execute_hard_stop(self, reason: str) -> None:
        """Выполняет процедуру принудительной остановки.

        Порядок:
        1. Устанавливает phase=grace_period в budget_status.json
        2. Ждёт grace_period_seconds
        3. Отправляет SIGTERM
        4. Ждёт 30 секунд, при необходимости SIGKILL

        Args:
            reason: причина остановки.
        """
        logger.warning(
            "Инициирован hard stop: %s. Grace period %d сек",
            reason,
            self.config.grace_period_seconds,
        )

        # При leakage — не ждём, сразу убиваем
        effective_grace = 0 if "leakage" in reason.lower() else self.config.grace_period_seconds
        grace_end = time.time() + effective_grace
        while time.time() < grace_end:
            if self._stop_event.is_set():
                return
            if self._claude_pid is not None and not self._is_claude_alive():
                logger.info("Claude Code завершился самостоятельно во время grace period")
                return
            time.sleep(5)

        if self._claude_pid is None:
            logger.info("claude_pid не задан, пропускаем SIGTERM")
            return

        if not self._is_claude_alive():
            logger.info("Claude Code уже завершился")
            return

        # SIGTERM
        logger.warning("Отправка SIGTERM процессу claude (pid=%d)", self._claude_pid)
        try:
            os.kill(self._claude_pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.info("Процесс claude уже завершился до SIGTERM")
            return

        # Ждём 30 секунд после SIGTERM
        sigterm_time = time.time()
        while time.time() - sigterm_time < 30:
            if not self._is_claude_alive():
                logger.info("Claude Code завершился после SIGTERM")
                return
            time.sleep(1)

        # SIGKILL
        if self._is_claude_alive():
            import contextlib

            logger.error("Claude Code не ответил на SIGTERM, отправка SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                os.kill(self._claude_pid, signal.SIGKILL)

        self._stop_event.set()

    def _check_leakage_thresholds(
        self, metric_value: float, is_first: bool
    ) -> list[AlertEntry]:
        """Проверяет soft_warning и sanity thresholds для leakage detection.

        Двухуровневая система:
        - soft_warning: добавляет WARNING алерт, устанавливает investigate_leakage флаг.
        - sanity_threshold: проверяет MLflow на clean verdict, если нет — CRITICAL.

        Args:
            metric_value: текущее значение метрики.
            is_first: True если это первое наблюдение метрики.

        Returns:
            Список AlertEntry (пустой если нет проблем).
        """
        result: list[AlertEntry] = []
        abs_val = abs(metric_value)
        soft_thr = self.config.leakage_soft_warning
        hard_thr = self.config.leakage_sanity_threshold
        ctx = "Первый результат" if is_first else "Метрика"

        if hard_thr is not None and abs_val > hard_thr:
            # Проверяем есть ли уже верифицированный clean run
            if self._has_clean_leakage_verdict(hard_thr):
                self._state.leakage_investigated = True
                logger.info(
                    "[MQ-LEAKAGE-SUSPECT] %s %.4f > sanity threshold %.4f, "
                    "но найден clean leakage verdict — продолжаем.",
                    ctx,
                    metric_value,
                    hard_thr,
                )
            else:
                result.append(
                    AlertEntry(
                        code="MQ-LEAKAGE-SUSPECT",
                        level="CRITICAL",
                        message=(
                            f"{ctx} {metric_value:.4f} превышает sanity threshold {hard_thr}. "
                            f"Вероятный leakage — best_metric НЕ обновлена. "
                            f"Запусти Leakage Investigation Protocol и залогируй leakage_verdict=clean."
                        ),
                    )
                )
                logger.warning(
                    "[MQ-LEAKAGE-SUSPECT] %s %.4f > sanity threshold %.4f. "
                    "Нет clean verdict в MLflow. CRITICAL.",
                    ctx,
                    metric_value,
                    hard_thr,
                )
        elif soft_thr is not None and abs_val > soft_thr:
            self._state.investigate_leakage = True
            result.append(
                AlertEntry(
                    code="MQ-LEAKAGE-SUSPECT",
                    level="WARNING",
                    message=(
                        f"{ctx} {metric_value:.4f} превышает soft_warning {soft_thr}. "
                        f"Требуется Leakage Investigation Protocol (см. program.md)."
                    ),
                )
            )
            logger.warning(
                "[MQ-LEAKAGE-SUSPECT] %s %.4f > soft_warning %.4f. investigate_leakage=True.",
                ctx,
                metric_value,
                soft_thr,
            )

        return result

    def _has_clean_leakage_verdict(self, threshold: float) -> bool:
        """Проверяет наличие в MLflow run с leakage_verdict=clean и метрикой ниже threshold.

        Args:
            threshold: sanity threshold — clean metric должна быть ниже этого значения.

        Returns:
            True если найден верифицированный clean run.
        """
        try:
            runs = self._client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=(
                    f"tags.session_id = '{self.session_id}' "
                    f"AND tags.leakage_verdict = 'clean'"
                ),
                max_results=10,
            )
            for run in runs:
                clean_metric = run.data.metrics.get("metric_clean")
                if clean_metric is not None and abs(clean_metric) <= threshold:
                    return True
            return False
        except Exception as exc:
            logger.warning("Ошибка проверки leakage verdict в MLflow: %s", exc)
            return False

    @staticmethod
    def _make_alert(code: str, message: str) -> AlertEntry:
        """Создаёт AlertEntry с автоматическим определением уровня.

        Args:
            code: код алерта из реестра.
            message: текстовое описание.

        Returns:
            Готовая запись алерта.
        """
        level = _ALERT_LEVELS.get(code, "INFO")
        return AlertEntry(code=code, level=level, message=message)
