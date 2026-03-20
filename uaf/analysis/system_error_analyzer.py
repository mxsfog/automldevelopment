"""SystemErrorAnalyzer — самодиагностика UAF (SE-01..SE-09 категории)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Пороговые значения для SE-категорий
_CRASH_RATE_THRESHOLD = 0.3  # SE-01: > 30% failed — высокий crash rate
_RUFF_VIOLATIONS_THRESHOLD = 0.05  # SE-02: files with violations threshold
_MLFLOW_COMPLIANCE_THRESHOLD = 1.0  # SE-04: 100% runs должны иметь session_id тег


@dataclass
class SystemErrorEntry:
    """Одна запись системной ошибки.

    Атрибуты:
        code: SE-01..SE-09.
        category: человекочитаемое название.
        severity: critical/warning/ok.
        value: измеренное значение.
        threshold: порог для severity.
        description: описание проблемы.
    """

    code: str
    category: str
    severity: str
    value: float | int | bool | str
    threshold: float | int | bool | str | None
    description: str


@dataclass
class SystemErrorReport:
    """Полный отчёт самодиагностики UAF.

    Атрибуты:
        session_id: ID сессии.
        errors: список системных ошибок по SE категориям.
        overall_health: overall_ok/has_warnings/has_critical.
        critical_count: количество CRITICAL ошибок.
        warning_count: количество WARNING ошибок.
    """

    session_id: str
    errors: list[SystemErrorEntry]
    overall_health: str
    critical_count: int
    warning_count: int


class SystemErrorAnalyzer:
    """Анализирует системные ошибки UAF по SE-01..SE-09 категориям.

    SE-01: crash_rate — доля failed runs
    SE-02: ruff_violations — качество кода
    SE-03: budget_overrun — превышение бюджета
    SE-04: mlflow_compliance — все runs залогированы корректно
    SE-05: dvc_commit_failures — DVC commit ошибки
    SE-06: sigterm_stop — сессия прервана принудительно
    SE-07: report_failure — ReportGenerator не смог создать PDF
    SE-08: oversight_timeout — HumanOversightGate таймаут
    SE-09: antigoal_violations — нарушения antigoals

    Атрибуты:
        session_id: ID сессии.
        session_dir: директория сессии.
        experiment_id: MLflow experiment ID.
        tracking_uri: MLflow tracking URI.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        experiment_id: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.experiment_id = experiment_id
        self.tracking_uri = tracking_uri

    def analyze(self) -> SystemErrorReport:
        """Запускает полный анализ SE-01..SE-09.

        Returns:
            SystemErrorReport с результатами всех категорий.
        """
        logger.info("SystemErrorAnalyzer: начало анализа сессии %s", self.session_id)

        errors: list[SystemErrorEntry] = [
            self._check_crash_rate(),
            self._check_ruff_violations(),
            self._check_budget_overrun(),
            self._check_mlflow_compliance(),
            self._check_dvc_commit_failures(),
            self._check_sigterm_stop(),
            self._check_report_failure(),
            self._check_oversight_timeout(),
            self._check_antigoal_violations(),
        ]

        critical_count = sum(1 for e in errors if e.severity == "critical")
        warning_count = sum(1 for e in errors if e.severity == "warning")

        if critical_count > 0:
            overall = "has_critical"
        elif warning_count > 0:
            overall = "has_warnings"
        else:
            overall = "overall_ok"

        report = SystemErrorReport(
            session_id=self.session_id,
            errors=errors,
            overall_health=overall,
            critical_count=critical_count,
            warning_count=warning_count,
        )

        self._save(report)
        logger.info(
            "SystemErrorAnalyzer завершён: health=%s, critical=%d, warnings=%d",
            overall,
            critical_count,
            warning_count,
        )
        return report

    def _check_crash_rate(self) -> SystemErrorEntry:
        """SE-01: crash_rate — доля failed runs из MLflow.

        Returns:
            SystemErrorEntry для SE-01.
        """
        code = "SE-01"
        category = "crash_rate"
        runs = self._fetch_experiment_runs()
        if not runs:
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="ok",
                value=0.0,
                threshold=_CRASH_RATE_THRESHOLD,
                description="Нет runs для анализа crash rate",
            )

        experiment_runs = [r for r in runs if r.get("type") == "experiment"]
        total = len(experiment_runs)
        if total == 0:
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="ok",
                value=0.0,
                threshold=_CRASH_RATE_THRESHOLD,
                description="Нет experiment runs",
            )

        failed = sum(1 for r in experiment_runs if r.get("status") == "failed")
        rate = failed / total

        severity = "warning" if rate > _CRASH_RATE_THRESHOLD else "ok"
        if rate > 0.5:
            severity = "critical"

        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=round(rate, 4),
            threshold=_CRASH_RATE_THRESHOLD,
            description=f"Crash rate: {failed}/{total} runs failed ({rate * 100:.1f}%)",
        )

    def _check_ruff_violations(self) -> SystemErrorEntry:
        """SE-02: ruff_violations — читает ruff_report.json.

        Returns:
            SystemErrorEntry для SE-02.
        """
        code = "SE-02"
        category = "ruff_violations"
        ruff_report_path = self.session_dir / "ruff_report.json"

        if not ruff_report_path.exists():
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="warning",
                value="missing",
                threshold=None,
                description="ruff_report.json не найден — RuffEnforcer не запускался",
            )

        try:
            report = json.loads(ruff_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="warning",
                value="parse_error",
                threshold=None,
                description=f"Ошибка парсинга ruff_report.json: {exc}",
            )

        total_files = report.get("total_files", 0)
        clean_rate = report.get("clean_rate", 1.0)
        total_violations = report.get("total_violations", 0)

        if total_files == 0:
            severity = "ok"
        elif clean_rate < (1 - _RUFF_VIOLATIONS_THRESHOLD):
            severity = "warning"
            if clean_rate < 0.8:
                severity = "critical"
        else:
            severity = "ok"

        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=round(clean_rate, 4),
            threshold=1 - _RUFF_VIOLATIONS_THRESHOLD,
            description=f"Ruff clean rate: {clean_rate * 100:.1f}%"
            f" ({total_violations} нарушений в {total_files} файлах)",
        )

    def _check_budget_overrun(self) -> SystemErrorEntry:
        """SE-03: budget_overrun — читает budget_status.json.

        Returns:
            SystemErrorEntry для SE-03.
        """
        code = "SE-03"
        category = "budget_overrun"
        budget_file = self.session_dir.parent / "budget_status.json"

        if not budget_file.exists():
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="ok",
                value=False,
                threshold=None,
                description="budget_status.json не найден (сессия завершена нормально)",
            )

        try:
            status = json.loads(budget_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="warning",
                value="parse_error",
                threshold=None,
                description="Ошибка парсинга budget_status.json",
            )

        budget_fraction = status.get("budget_fraction_used", 0.0)
        hard_stop_reason = status.get("hard_stop_reason", "")
        hard_stop = status.get("hard_stop", False)

        is_overrun = hard_stop and "limit" in str(hard_stop_reason)
        severity = "warning" if is_overrun else "ok"

        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=round(budget_fraction, 4),
            threshold=1.0,
            description=f"Использовано {budget_fraction * 100:.1f}% бюджета"
            + (f", hard_stop_reason={hard_stop_reason}" if hard_stop else ""),
        )

    def _check_mlflow_compliance(self) -> SystemErrorEntry:
        """SE-04: mlflow_compliance — все experiment runs имеют session_id тег.

        Returns:
            SystemErrorEntry для SE-04.
        """
        code = "SE-04"
        category = "mlflow_compliance"
        runs = self._fetch_experiment_runs()

        experiment_runs = [r for r in runs if r.get("type") == "experiment"]
        if not experiment_runs:
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="ok",
                value=1.0,
                threshold=_MLFLOW_COMPLIANCE_THRESHOLD,
                description="Нет experiment runs для проверки compliance",
            )

        compliant = sum(1 for r in experiment_runs if r.get("session_id") == self.session_id)
        compliance_rate = compliant / len(experiment_runs)

        severity = "critical" if compliance_rate < _MLFLOW_COMPLIANCE_THRESHOLD else "ok"

        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=round(compliance_rate, 4),
            threshold=_MLFLOW_COMPLIANCE_THRESHOLD,
            description=f"MLflow compliance: {compliant}/{len(experiment_runs)} runs"
            f" с корректным session_id ({compliance_rate * 100:.1f}%)",
        )

    def _check_dvc_commit_failures(self) -> SystemErrorEntry:
        """SE-05: dvc_commit_failures — читает session.log на наличие DVC ошибок.

        Returns:
            SystemErrorEntry для SE-05.
        """
        code = "SE-05"
        category = "dvc_commit_failures"
        session_log = self.session_dir / "session.log"

        if not session_log.exists():
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="ok",
                value=0,
                threshold=None,
                description="session.log не найден",
            )

        try:
            log_content = session_log.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return SystemErrorEntry(
                code=code,
                category=category,
                severity="warning",
                value="read_error",
                threshold=None,
                description=f"Ошибка чтения session.log: {exc}",
            )

        dvc_error_lines = [
            line
            for line in log_content.splitlines()
            if "dvc" in line.lower()
            and any(kw in line.lower() for kw in ["error", "failed", "exception"])
        ]
        count = len(dvc_error_lines)
        severity = "warning" if count > 0 else "ok"
        if count > 5:
            severity = "critical"

        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=count,
            threshold=0,
            description=f"DVC commit failures в session.log: {count}",
        )

    def _check_sigterm_stop(self) -> SystemErrorEntry:
        """SE-06: sigterm_stop — сессия прервана принудительно через SIGTERM.

        Returns:
            SystemErrorEntry для SE-06.
        """
        code = "SE-06"
        category = "sigterm_stop"
        budget_file = self.session_dir.parent / "budget_status.json"

        was_sigterm = False
        reason = None
        if budget_file.exists():
            try:
                status = json.loads(budget_file.read_text(encoding="utf-8"))
                reason = status.get("hard_stop_reason", "")
                was_sigterm = status.get("hard_stop", False) and bool(reason)
            except json.JSONDecodeError:
                pass

        # Проверяем session.log на SIGTERM
        session_log = self.session_dir / "session.log"
        if session_log.exists():
            try:
                content = session_log.read_text(encoding="utf-8", errors="ignore")
                if "sigterm" in content.lower() or "killed" in content.lower():
                    was_sigterm = True
            except OSError:
                pass

        severity = "warning" if was_sigterm else "ok"
        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=was_sigterm,
            threshold=False,
            description=f"Сессия прервана SIGTERM: {reason or 'нет'}"
            if was_sigterm
            else "Сессия завершилась штатно",
        )

    def _check_report_failure(self) -> SystemErrorEntry:
        """SE-07: report_failure — PDF не был создан.

        Returns:
            SystemErrorEntry для SE-07.
        """
        code = "SE-07"
        category = "report_failure"
        report_dir = self.session_dir / "report"

        # Проверяем наличие PDF или .tex как fallback
        pdf_exists = any(report_dir.glob("*.pdf")) if report_dir.exists() else False
        tex_exists = any(report_dir.glob("*.tex")) if report_dir.exists() else False

        if pdf_exists:
            severity = "ok"
            value = "pdf_generated"
            desc = "PDF отчёт создан успешно"
        elif tex_exists:
            severity = "warning"
            value = "tex_only"
            desc = "PDF не создан, есть .tex fallback (LaTeX не установлен?)"
        else:
            severity = "critical"
            value = "no_report"
            desc = "Отчёт не создан (ни PDF, ни .tex)"

        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=value,
            threshold=None,
            description=desc,
        )

    def _check_oversight_timeout(self) -> SystemErrorEntry:
        """SE-08: oversight_timeout — HumanOversightGate завершился по таймауту.

        Returns:
            SystemErrorEntry для SE-08.
        """
        code = "SE-08"
        category = "oversight_timeout"

        # Проверяем session_state.json
        state_file = self.session_dir / "session_state.json"
        timed_out = False

        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                timed_out = state.get("approval_result", {}).get("notes", "").startswith(
                    "Auto-rejected: timeout"
                )
            except (json.JSONDecodeError, AttributeError):
                pass

        severity = "warning" if timed_out else "ok"
        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=timed_out,
            threshold=False,
            description="HumanOversightGate завершился по таймауту"
            if timed_out
            else "HumanOversightGate прошёл штатно",
        )

    def _check_antigoal_violations(self) -> SystemErrorEntry:
        """SE-09: antigoal_violations — нарушения antigoals (читает session_state.json).

        Returns:
            SystemErrorEntry для SE-09.
        """
        code = "SE-09"
        category = "antigoal_violations"

        violations: list[str] = []

        # Проверка antigoal 3: все failed runs должны быть в отчёте
        analysis_file = self.session_dir / "session_analysis.json"
        if analysis_file.exists():
            try:
                analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
                failed_runs = analysis.get("failed_runs", 0)
                # Если есть failed runs но failure_analysis пустой — нарушение
                fa = analysis.get("failure_analysis", {})
                if failed_runs > 0 and fa.get("total_failed", 0) == 0:
                    violations.append("antigoal-3: failed runs не включены в анализ")
            except json.JSONDecodeError:
                pass

        # Проверка antigoal 4: данные не модифицированы
        budget_file = self.session_dir.parent / "budget_status.json"
        if budget_file.exists():
            try:
                status = json.loads(budget_file.read_text(encoding="utf-8"))
                dq = status.get("data_quality", {})
                if dq.get("files_modified", False):
                    violations.append("antigoal-4: входные данные изменены во время сессии")
            except json.JSONDecodeError:
                pass

        count = len(violations)
        severity = "critical" if count > 0 else "ok"
        return SystemErrorEntry(
            code=code,
            category=category,
            severity=severity,
            value=count,
            threshold=0,
            description=(
                f"Нарушения antigoals ({count}): {'; '.join(violations)}"
                if violations
                else "Нарушений antigoals не обнаружено"
            ),
        )

    def _fetch_experiment_runs(self) -> list[dict[str, Any]]:
        """Читает runs из MLflow и возвращает упрощённые словари.

        Returns:
            Список словарей с tags runs.
        """
        if not self.experiment_id or not self.tracking_uri:
            return []
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            raw = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.session_id = '{self.session_id}'",
                max_results=500,
            )
            return [
                {
                    "run_id": r.info.run_id,
                    "type": r.data.tags.get("type", ""),
                    "status": r.data.tags.get("status", ""),
                    "session_id": r.data.tags.get("session_id", ""),
                }
                for r in raw
            ]
        except Exception as exc:
            logger.warning("Ошибка чтения MLflow для SystemErrorAnalyzer: %s", exc)
            return []

    def _save(self, report: SystemErrorReport) -> None:
        """Сохраняет SystemErrorReport в system_error_report.json.

        Args:
            report: результат анализа.
        """
        output_path = self.session_dir / "system_error_report.json"
        data = {
            "session_id": report.session_id,
            "overall_health": report.overall_health,
            "critical_count": report.critical_count,
            "warning_count": report.warning_count,
            "errors": [
                {
                    "code": e.code,
                    "category": e.category,
                    "severity": e.severity,
                    "value": e.value,
                    "threshold": e.threshold,
                    "description": e.description,
                }
                for e in report.errors
            ],
        }
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("system_error_report.json сохранён: %s", output_path)
