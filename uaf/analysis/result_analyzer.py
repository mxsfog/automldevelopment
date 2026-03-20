"""ResultAnalyzer — 8-шаговый post-session анализ ML-экспериментов."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# 7 категорий ошибок по keyword matching
FailureCategory = Literal[
    "import_error",
    "oom_error",
    "data_error",
    "timeout_error",
    "assertion_error",
    "runtime_error",
    "other",
]

# Ключевые слова для классификации ошибок (порядок важен: более специфичные первые)
_FAILURE_KEYWORDS: list[tuple[FailureCategory, list[str]]] = [
    ("import_error", ["ModuleNotFoundError", "ImportError", "No module named"]),
    (
        "oom_error",
        [
            "OutOfMemoryError",
            "CUDA out of memory",
            "MemoryError",
            "Killed",
            "OOM",
            "out of memory",
        ],
    ),
    (
        "data_error",
        [
            "FileNotFoundError",
            "KeyError",
            "ValueError: could not convert",
            "DataError",
            "ParserError",
            "EmptyDataError",
        ],
    ),
    ("timeout_error", ["TimeoutError", "signal.SIGTERM", "Timeout", "timed out"]),
    ("assertion_error", ["AssertionError"]),
    (
        "runtime_error",
        [
            "RuntimeError",
            "ZeroDivisionError",
            "AttributeError",
            "TypeError",
            "IndexError",
            "NameError",
        ],
    ),
]

# Минимум runs для корреляционного анализа (шаг 5)
_MIN_RUNS_FOR_CORRELATION = 5

# Порог systemic failure: одна категория >= этой доли failed runs
_SYSTEMIC_FAILURE_THRESHOLD = 0.5


@dataclass
class RunSummary:
    """Краткая сводка одного MLflow run.

    Атрибуты:
        run_id: ID run в MLflow.
        run_name: имя run.
        status: completed/failed/partial.
        metrics: словарь метрик.
        params: словарь параметров.
        start_time: время старта (unix ms).
        failure_category: категория ошибки для failed runs.
        failure_reason: краткое описание ошибки.
    """

    run_id: str
    run_name: str
    status: str
    metrics: dict[str, float]
    params: dict[str, str]
    start_time: int
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None


@dataclass
class MetricProfile:
    """Метрический профиль набора runs.

    Атрибуты:
        metric_name: название метрики.
        mean: среднее значение.
        std: стандартное отклонение.
        best: лучшее значение.
        worst: худшее значение.
        count: количество runs с этой метрикой.
    """

    metric_name: str
    mean: float
    std: float
    best: float
    worst: float
    count: int


@dataclass
class ParamCorrelation:
    """Корреляция параметра с метрикой.

    Атрибуты:
        param_name: название параметра.
        metric_name: название метрики.
        spearman_r: коэффициент Spearman.
        p_value: p-значение (если доступно).
        n_samples: количество пар.
    """

    param_name: str
    metric_name: str
    spearman_r: float
    p_value: float | None
    n_samples: int


@dataclass
class FailureAnalysis:
    """Анализ failed runs.

    Атрибуты:
        total_failed: общее число failed runs.
        by_category: словарь {категория: количество}.
        systemic_category: категория с >= 50% failed runs (если есть).
        examples: примеры run_id для каждой категории.
    """

    total_failed: int
    by_category: dict[str, int]
    systemic_category: FailureCategory | None
    examples: dict[str, list[str]]


@dataclass
class Hypothesis:
    """Одна гипотеза для следующей итерации.

    Атрибуты:
        code: код гипотезы (H-01..H-09).
        description: описание.
        priority: приоритет (1 = высший).
        evidence: на чём основана гипотеза.
    """

    code: str
    description: str
    priority: int
    evidence: str


@dataclass
class SessionAnalysis:
    """Полный анализ сессии — результат 8-шагового алгоритма.

    Атрибуты:
        session_id: ID сессии.
        total_runs: общее число runs.
        completed_runs: runs со статусом success.
        failed_runs: runs со статусом failed.
        partial_runs: runs со статусом partial/running.
        ranked_runs: runs отсортированные по целевой метрике.
        metric_profile: профиль целевой метрики.
        failure_analysis: анализ ошибок.
        param_correlations: корреляции параметров (если >= 5 runs).
        hypotheses: список гипотез (максимум 5).
        target_metric: название целевой метрики.
        metric_direction: maximize/minimize.
        has_predictions: наличие predictions.csv в артефактах.
        segment_analysis_note: заметка о сегментации (если predictions доступны).
    """

    session_id: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    partial_runs: int
    ranked_runs: list[RunSummary]
    metric_profile: MetricProfile | None
    failure_analysis: FailureAnalysis
    param_correlations: list[ParamCorrelation]
    hypotheses: list[Hypothesis]
    target_metric: str
    metric_direction: Literal["maximize", "minimize"]
    has_predictions: bool
    segment_analysis_note: str | None


class ResultAnalyzer:
    """Выполняет 8-шаговый post-session анализ экспериментов.

    Читает MLflow runs, анализирует результаты, генерирует гипотезы детерминированно
    (без LLM). Сохраняет результат в session_analysis.json.

    Атрибуты:
        session_id: ID сессии.
        experiment_id: MLflow experiment ID.
        tracking_uri: MLflow tracking URI.
        session_dir: директория сессии.
        target_metric: название целевой метрики.
        metric_direction: maximize/minimize.
    """

    def __init__(
        self,
        session_id: str,
        experiment_id: str,
        tracking_uri: str,
        session_dir: Path,
        target_metric: str,
        metric_direction: Literal["maximize", "minimize"] = "maximize",
    ) -> None:
        self.session_id = session_id
        self.experiment_id = experiment_id
        self.tracking_uri = tracking_uri
        self.session_dir = session_dir
        self.target_metric = target_metric
        self.metric_direction = metric_direction

    def analyze(self) -> SessionAnalysis:
        """Запускает полный 8-шаговый анализ.

        Шаги:
        1. Разделение runs по статусу
        2. Ранжирование по целевой метрике
        3. Метрический профиль
        4. Анализ failed runs
        5. Корреляции param→metric (Spearman)
        6. Проверка predictions.csv
        7. Генерация гипотез H-01..H-09
        8. Сборка SessionAnalysis и сохранение

        Returns:
            SessionAnalysis с полными результатами.
        """
        logger.info("Начало анализа сессии: %s", self.session_id)

        runs = self._fetch_runs()
        logger.info("Загружено %d runs из MLflow", len(runs))

        # Шаг 1: разделение по статусу
        completed, failed, partial = self._split_by_status(runs)
        logger.info(
            "Runs: completed=%d, failed=%d, partial=%d",
            len(completed),
            len(failed),
            len(partial),
        )

        # Шаг 2: ранжирование
        ranked = self._rank_runs(completed)

        # Шаг 3: метрический профиль
        metric_profile = self._compute_metric_profile(completed)

        # Шаг 4: анализ ошибок
        failure_analysis = self._analyze_failures(failed)

        # Шаг 5: корреляции (только при >= 5 runs)
        param_correlations: list[ParamCorrelation] = []
        all_successful = completed + partial
        if len(all_successful) >= _MIN_RUNS_FOR_CORRELATION:
            param_correlations = self._compute_param_correlations(all_successful)
            logger.info("Вычислено %d корреляций param→metric", len(param_correlations))
        else:
            logger.info(
                "Пропуск корреляций: %d runs < %d",
                len(all_successful),
                _MIN_RUNS_FOR_CORRELATION,
            )

        # Шаг 6: проверка predictions
        has_predictions = self._check_predictions()
        segment_note = None
        if has_predictions:
            segment_note = (
                "predictions.csv найден в артефактах. "
                "Сегментный анализ доступен для ReportGenerator."
            )

        # Шаг 7: генерация гипотез
        hypotheses = self._generate_hypotheses(
            ranked=ranked,
            failure_analysis=failure_analysis,
            param_correlations=param_correlations,
            metric_profile=metric_profile,
        )

        # Шаг 8: сборка
        analysis = SessionAnalysis(
            session_id=self.session_id,
            total_runs=len(runs),
            completed_runs=len(completed),
            failed_runs=len(failed),
            partial_runs=len(partial),
            ranked_runs=ranked,
            metric_profile=metric_profile,
            failure_analysis=failure_analysis,
            param_correlations=param_correlations,
            hypotheses=hypotheses,
            target_metric=self.target_metric,
            metric_direction=self.metric_direction,
            has_predictions=has_predictions,
            segment_analysis_note=segment_note,
        )

        self._save(analysis)
        logger.info("Анализ сессии завершён: %s", self.session_dir / "session_analysis.json")
        return analysis

    def _fetch_runs(self) -> list[RunSummary]:
        """Читает все experiment runs из MLflow.

        Returns:
            Список RunSummary.
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            raw_runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.session_id = '{self.session_id}' "
                f"and tags.type = 'experiment'",
                max_results=500,
            )
        except Exception as exc:
            logger.error("Ошибка чтения MLflow runs: %s", exc)
            return []

        result = []
        for r in raw_runs:
            status_tag = r.data.tags.get("status", "unknown")
            if status_tag in ("success",):
                status = "completed"
            elif status_tag == "failed":
                status = "failed"
            else:
                status = "partial"

            result.append(
                RunSummary(
                    run_id=r.info.run_id,
                    run_name=r.info.run_name or "",
                    status=status,
                    metrics=dict(r.data.metrics),
                    params=dict(r.data.params),
                    start_time=r.info.start_time or 0,
                    failure_reason=r.data.tags.get("failure_reason"),
                )
            )
        return result

    def _split_by_status(
        self, runs: list[RunSummary]
    ) -> tuple[list[RunSummary], list[RunSummary], list[RunSummary]]:
        """Делит runs на completed/failed/partial.

        Args:
            runs: все runs.

        Returns:
            Тройка (completed, failed, partial).
        """
        completed = [r for r in runs if r.status == "completed"]
        failed = [r for r in runs if r.status == "failed"]
        partial = [r for r in runs if r.status == "partial"]
        return completed, failed, partial

    def _rank_runs(self, completed: list[RunSummary]) -> list[RunSummary]:
        """Ранжирует completed runs по целевой метрике.

        Args:
            completed: runs со статусом completed.

        Returns:
            Отсортированный список (лучший первый).
        """
        with_metric = [
            r
            for r in completed
            if self.target_metric in r.metrics and not math.isnan(r.metrics[self.target_metric])
        ]
        reverse = self.metric_direction == "maximize"
        return sorted(with_metric, key=lambda r: r.metrics[self.target_metric], reverse=reverse)

    def _compute_metric_profile(self, completed: list[RunSummary]) -> MetricProfile | None:
        """Вычисляет mean/std/best/worst для целевой метрики.

        Args:
            completed: completed runs.

        Returns:
            MetricProfile или None если нет данных.
        """
        values = [
            r.metrics[self.target_metric]
            for r in completed
            if self.target_metric in r.metrics and not math.isnan(r.metrics[self.target_metric])
        ]
        if not values:
            return None

        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        return MetricProfile(
            metric_name=self.target_metric,
            mean=round(mean, 6),
            std=round(std, 6),
            best=max(values) if self.metric_direction == "maximize" else min(values),
            worst=min(values) if self.metric_direction == "maximize" else max(values),
            count=n,
        )

    def _analyze_failures(self, failed: list[RunSummary]) -> FailureAnalysis:
        """Классифицирует failed runs по 7 категориям keyword matching.

        Args:
            failed: failed runs.

        Returns:
            FailureAnalysis с категориями и systemic failure.
        """
        by_category: dict[str, int] = {cat: 0 for cat, _ in _FAILURE_KEYWORDS}
        by_category["other"] = 0
        examples: dict[str, list[str]] = {cat: [] for cat, _ in _FAILURE_KEYWORDS}
        examples["other"] = []

        # Обогащаем runs категориями
        for run in failed:
            reason = run.failure_reason or ""
            category = self._classify_failure(reason)
            run.failure_category = category
            by_category[category] += 1
            if len(examples[category]) < 3:
                examples[category].append(run.run_id)

        systemic_category: FailureCategory | None = None
        total_failed = len(failed)
        if total_failed > 0:
            for cat, count in by_category.items():
                if count / total_failed >= _SYSTEMIC_FAILURE_THRESHOLD:
                    systemic_category = cat  # type: ignore[assignment]
                    logger.warning(
                        "Systemic failure обнаружен: категория=%s (%d/%d)",
                        cat,
                        count,
                        total_failed,
                    )
                    break

        return FailureAnalysis(
            total_failed=total_failed,
            by_category=by_category,
            systemic_category=systemic_category,
            examples=examples,
        )

    def _classify_failure(self, reason: str) -> FailureCategory:
        """Классифицирует ошибку по keyword matching.

        Args:
            reason: строка с причиной ошибки.

        Returns:
            Категория ошибки.
        """
        for category, keywords in _FAILURE_KEYWORDS:
            if any(kw.lower() in reason.lower() for kw in keywords):
                return category
        return "other"

    def _compute_param_correlations(self, runs: list[RunSummary]) -> list[ParamCorrelation]:
        """Вычисляет Spearman корреляции числовых параметров с целевой метрикой.

        Args:
            runs: runs с метриками и параметрами.

        Returns:
            Список корреляций, отсортированных по abs(r).
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            logger.warning("scipy недоступен, корреляции пропущены")
            return []

        # Собираем числовые параметры
        param_names: set[str] = set()
        for run in runs:
            for k, v in run.params.items():
                try:
                    float(v)
                    param_names.add(k)
                except (ValueError, TypeError):
                    pass

        correlations: list[ParamCorrelation] = []
        for param_name in sorted(param_names):
            pairs = []
            for run in runs:
                if param_name in run.params and self.target_metric in run.metrics:
                    try:
                        p_val = float(run.params[param_name])
                        m_val = run.metrics[self.target_metric]
                        if not math.isnan(m_val) and not math.isnan(p_val):
                            pairs.append((p_val, m_val))
                    except (ValueError, TypeError):
                        pass

            if len(pairs) < _MIN_RUNS_FOR_CORRELATION:
                continue

            param_vals = [p[0] for p in pairs]
            metric_vals = [p[1] for p in pairs]
            try:
                result = spearmanr(param_vals, metric_vals)
                correlations.append(
                    ParamCorrelation(
                        param_name=param_name,
                        metric_name=self.target_metric,
                        spearman_r=round(float(result.statistic), 4),
                        p_value=round(float(result.pvalue), 4),
                        n_samples=len(pairs),
                    )
                )
            except Exception as exc:
                logger.debug("Ошибка корреляции для %s: %s", param_name, exc)

        # Сортируем по abs(r) убыванием
        correlations.sort(key=lambda c: abs(c.spearman_r), reverse=True)
        return correlations

    def _check_predictions(self) -> bool:
        """Проверяет наличие predictions.csv в артефактах MLflow.

        Returns:
            True если predictions.csv найден.
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.session_id = '{self.session_id}'",
                max_results=500,
            )
            for run in runs:
                artifacts = client.list_artifacts(run.info.run_id)
                for art in artifacts:
                    if "predictions" in art.path.lower() and art.path.endswith(".csv"):
                        return True
        except Exception as exc:
            logger.debug("Ошибка проверки predictions: %s", exc)
        return False

    def _generate_hypotheses(
        self,
        ranked: list[RunSummary],
        failure_analysis: FailureAnalysis,
        param_correlations: list[ParamCorrelation],
        metric_profile: MetricProfile | None,
    ) -> list[Hypothesis]:
        """Генерирует детерминированные гипотезы H-01..H-09 без LLM.

        Правила (приоритет определяется важностью сигнала):
        H-01: systemic import errors -> исправить зависимости
        H-02: systemic OOM -> уменьшить batch size / упростить модель
        H-03: systemic data errors -> проверить пути и форматы данных
        H-04: высокий std (cv > 0.1) -> добавить cross-validation / усреднение
        H-05: сильная корреляция параметра (|r| > 0.7) -> оптимизировать его
        H-06: метрика плато (лучший - худший < 1%) -> сменить архитектуру
        H-07: много failed при мало completed -> debugging session
        H-08: много partial runs -> проверить таймауты и budget_status
        H-09: нет корреляций, нестабильность -> SHAP feature importance

        Args:
            ranked: ранжированные runs.
            failure_analysis: анализ ошибок.
            param_correlations: корреляции параметров.
            metric_profile: профиль метрики.

        Returns:
            До 5 гипотез по приоритету.
        """
        candidates: list[Hypothesis] = []

        # H-01: systemic import errors
        if failure_analysis.systemic_category == "import_error":
            import_count = failure_analysis.by_category.get("import_error", 0)
            candidates.append(
                Hypothesis(
                    code="H-01",
                    description=(
                        "Критические ошибки импорта: исправить зависимости"
                        " перед следующей сессией"
                    ),
                    priority=1,
                    evidence=(
                        f"import_error у {import_count}"
                        f"/{failure_analysis.total_failed} failed runs"
                    ),
                )
            )

        # H-02: systemic OOM
        if failure_analysis.systemic_category == "oom_error":
            oom_count = failure_analysis.by_category.get("oom_error", 0)
            candidates.append(
                Hypothesis(
                    code="H-02",
                    description=(
                        "OOM ошибки доминируют: уменьшить batch_size"
                        " или переключиться на более лёгкую модель"
                    ),
                    priority=1,
                    evidence=(
                        f"oom_error у {oom_count}"
                        f"/{failure_analysis.total_failed} failed runs"
                    ),
                )
            )

        # H-03: systemic data errors
        if failure_analysis.systemic_category == "data_error":
            data_count = failure_analysis.by_category.get("data_error", 0)
            candidates.append(
                Hypothesis(
                    code="H-03",
                    description="Ошибки данных доминируют: проверить пути, форматы, кодировки",
                    priority=1,
                    evidence=(
                        f"data_error у {data_count}"
                        f"/{failure_analysis.total_failed} failed runs"
                    ),
                )
            )

        # H-04: высокий std
        if metric_profile is not None and metric_profile.mean != 0:
            cv = metric_profile.std / abs(metric_profile.mean)
            if cv > 0.1:
                candidates.append(
                    Hypothesis(
                        code="H-04",
                        description=f"Нестабильность метрики (CV={cv:.2f}): добавить k-fold CV"
                        f" или ensemble для снижения дисперсии",
                        priority=2,
                        evidence=f"std={metric_profile.std:.4f},"
                        f" mean={metric_profile.mean:.4f}",
                    )
                )

        # H-05: сильная корреляция параметра
        if param_correlations:
            top = param_correlations[0]
            if abs(top.spearman_r) > 0.7:
                direction = "увеличить" if top.spearman_r > 0 else "уменьшить"
                candidates.append(
                    Hypothesis(
                        code="H-05",
                        description=f"Сильная корреляция {top.param_name}→{top.metric_name}"
                        f" (r={top.spearman_r:.2f}): {direction} {top.param_name}",
                        priority=2,
                        evidence=f"Spearman r={top.spearman_r:.3f},"
                        f" p={top.p_value}, n={top.n_samples}",
                    )
                )

        # H-06: плато метрики
        if metric_profile is not None and metric_profile.count >= 3:
            span = abs(metric_profile.best - metric_profile.worst)
            if metric_profile.best != 0 and span / abs(metric_profile.best) < 0.01:
                candidates.append(
                    Hypothesis(
                        code="H-06",
                        description=f"Метрика в плато (диапазон {span:.6f}): попробовать"
                        f" другую архитектуру или feature engineering",
                        priority=3,
                        evidence=f"best={metric_profile.best:.6f},"
                        f" worst={metric_profile.worst:.6f}",
                    )
                )

        # H-07: много failed при мало completed
        completed_count = len(ranked)
        if failure_analysis.total_failed > completed_count and completed_count < 3:
            candidates.append(
                Hypothesis(
                    code="H-07",
                    description="Большинство runs failed: нужна debugging session"
                    " перед полноценным исследованием",
                    priority=1,
                    evidence=f"failed={failure_analysis.total_failed},"
                    f" completed={completed_count}",
                )
            )

        # H-08: много partial runs — не актуально без partial счётчика, добавляем если нет другого

        # H-09: нет корреляций, нестабильность — SHAP
        if not param_correlations and metric_profile is not None and metric_profile.count >= 3:
            candidates.append(
                Hypothesis(
                    code="H-09",
                    description="Корреляции параметров не найдены: запустить SHAP analysis"
                    " для понимания важности признаков",
                    priority=4,
                    evidence=f"0 корреляций при {metric_profile.count} runs",
                )
            )

        # Если гипотез нет — добавляем базовую
        if not candidates and metric_profile is not None:
            candidates.append(
                Hypothesis(
                    code="H-05",
                    description=f"Продолжить оптимизацию гиперпараметров"
                    f" (текущий лучший {self.target_metric}={metric_profile.best:.6f})",
                    priority=5,
                    evidence=f"best={metric_profile.best:.6f},"
                    f" mean={metric_profile.mean:.6f}",
                )
            )

        # Возвращаем максимум 5 по приоритету
        candidates.sort(key=lambda h: h.priority)
        return candidates[:5]

    def _save(self, analysis: SessionAnalysis) -> None:
        """Сохраняет SessionAnalysis в session_analysis.json.

        Args:
            analysis: результат анализа.
        """
        output_path = self.session_dir / "session_analysis.json"
        data = _analysis_to_dict(analysis)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("session_analysis.json сохранён: %s", output_path)


def _analysis_to_dict(analysis: SessionAnalysis) -> dict[str, Any]:
    """Конвертирует SessionAnalysis в словарь для JSON.

    Args:
        analysis: результат анализа.

    Returns:
        Словарь готовый к сериализации.
    """
    return {
        "session_id": analysis.session_id,
        "total_runs": analysis.total_runs,
        "completed_runs": analysis.completed_runs,
        "failed_runs": analysis.failed_runs,
        "partial_runs": analysis.partial_runs,
        "target_metric": analysis.target_metric,
        "metric_direction": analysis.metric_direction,
        "has_predictions": analysis.has_predictions,
        "segment_analysis_note": analysis.segment_analysis_note,
        "ranked_runs": [
            {
                "run_id": r.run_id,
                "run_name": r.run_name,
                "status": r.status,
                "metrics": r.metrics,
                "params": r.params,
            }
            for r in analysis.ranked_runs
        ],
        "metric_profile": (
            {
                "metric_name": analysis.metric_profile.metric_name,
                "mean": analysis.metric_profile.mean,
                "std": analysis.metric_profile.std,
                "best": analysis.metric_profile.best,
                "worst": analysis.metric_profile.worst,
                "count": analysis.metric_profile.count,
            }
            if analysis.metric_profile
            else None
        ),
        "failure_analysis": {
            "total_failed": analysis.failure_analysis.total_failed,
            "by_category": analysis.failure_analysis.by_category,
            "systemic_category": analysis.failure_analysis.systemic_category,
            "examples": analysis.failure_analysis.examples,
        },
        "param_correlations": [
            {
                "param_name": c.param_name,
                "metric_name": c.metric_name,
                "spearman_r": c.spearman_r,
                "p_value": c.p_value,
                "n_samples": c.n_samples,
            }
            for c in analysis.param_correlations
        ],
        "hypotheses": [
            {
                "code": h.code,
                "description": h.description,
                "priority": h.priority,
                "evidence": h.evidence,
            }
            for h in analysis.hypotheses
        ],
    }


def load_session_analysis(session_dir: Path) -> SessionAnalysis | None:
    """Загружает SessionAnalysis из session_analysis.json.

    Args:
        session_dir: директория сессии.

    Returns:
        SessionAnalysis или None если файл не найден.
    """
    path = session_dir / "session_analysis.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _dict_to_analysis(data)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Ошибка загрузки session_analysis.json: %s", exc)
        return None


def _dict_to_analysis(data: dict[str, Any]) -> SessionAnalysis:
    """Десериализует SessionAnalysis из словаря.

    Args:
        data: словарь из JSON.

    Returns:
        SessionAnalysis.
    """
    ranked_runs = [
        RunSummary(
            run_id=r["run_id"],
            run_name=r["run_name"],
            status=r["status"],
            metrics=r["metrics"],
            params=r["params"],
            start_time=0,
        )
        for r in data.get("ranked_runs", [])
    ]

    mp_data = data.get("metric_profile")
    metric_profile = (
        MetricProfile(
            metric_name=mp_data["metric_name"],
            mean=mp_data["mean"],
            std=mp_data["std"],
            best=mp_data["best"],
            worst=mp_data["worst"],
            count=mp_data["count"],
        )
        if mp_data
        else None
    )

    fa_data = data.get("failure_analysis", {})
    failure_analysis = FailureAnalysis(
        total_failed=fa_data.get("total_failed", 0),
        by_category=fa_data.get("by_category", {}),
        systemic_category=fa_data.get("systemic_category"),
        examples=fa_data.get("examples", {}),
    )

    param_correlations = [
        ParamCorrelation(
            param_name=c["param_name"],
            metric_name=c["metric_name"],
            spearman_r=c["spearman_r"],
            p_value=c["p_value"],
            n_samples=c["n_samples"],
        )
        for c in data.get("param_correlations", [])
    ]

    hypotheses = [
        Hypothesis(
            code=h["code"],
            description=h["description"],
            priority=h["priority"],
            evidence=h["evidence"],
        )
        for h in data.get("hypotheses", [])
    ]

    return SessionAnalysis(
        session_id=data["session_id"],
        total_runs=data["total_runs"],
        completed_runs=data["completed_runs"],
        failed_runs=data["failed_runs"],
        partial_runs=data["partial_runs"],
        ranked_runs=ranked_runs,
        metric_profile=metric_profile,
        failure_analysis=failure_analysis,
        param_correlations=param_correlations,
        hypotheses=hypotheses,
        target_metric=data["target_metric"],
        metric_direction=data["metric_direction"],
        has_predictions=data.get("has_predictions", False),
        segment_analysis_note=data.get("segment_analysis_note"),
    )
