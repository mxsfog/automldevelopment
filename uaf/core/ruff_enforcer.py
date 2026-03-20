"""RuffEnforcer — post-processing ruff на всех .py файлах SESSION_DIR."""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Целевое значение clean_rate для M-UAF метрик
_TARGET_CLEAN_RATE = 0.95


@dataclass
class RuffViolation:
    """Одно нарушение ruff.

    Атрибуты:
        rule: код правила (например E501).
        message: описание нарушения.
        line: номер строки.
        col: номер колонки.
        fixable: может ли быть исправлено автоматически.
    """

    rule: str
    message: str
    line: int
    col: int
    fixable: bool = False


@dataclass
class RuffFileResult:
    """Результат проверки одного файла.

    Атрибуты:
        file: путь к файлу.
        formatted: был ли успешно отформатирован (ruff format).
        violations_before_fix: нарушений до --fix.
        violations_after_fix: нарушений после --fix (unfixable).
        unfixable_violations: список неисправляемых нарушений.
        format_error: ошибка при форматировании (синтаксическая ошибка и т.п.).
    """

    file: Path
    formatted: bool
    violations_before_fix: int
    violations_after_fix: int
    unfixable_violations: list[RuffViolation] = field(default_factory=list)
    format_error: str | None = None


@dataclass
class RuffReport:
    """Агрегированный отчёт по всем файлам сессии.

    Атрибуты:
        total_files: всего проверено .py файлов.
        clean_files: файлов без нарушений после fix.
        files_with_unfixable: файлов с нарушениями которые не удалось исправить.
        total_violations: суммарное количество нарушений (после fix).
        clean_rate: доля чистых файлов (0.0-1.0).
        files: список результатов по файлам.
        ruff_version: версия ruff (если удалось определить).
        target_met: достигнут ли целевой clean_rate >= 0.95.
    """

    total_files: int
    clean_files: int
    files_with_unfixable: int
    total_violations: int
    clean_rate: float
    files: list[RuffFileResult]
    ruff_version: str
    target_met: bool


class RuffEnforcer:
    """Применяет ruff format + ruff check --fix ко всем .py файлам SESSION_DIR.

    Два прохода:
    1. ruff format — форматирование (изменяет файл на месте)
    2. ruff check --fix — исправление автоматических нарушений

    Результат записывается в ruff_report.json и логируется в MLflow.

    Атрибуты:
        session_dir: директория сессии с .py файлами.
        ruff_config_path: опциональный путь к pyproject.toml с ruff конфигом.
            Если не задан — используется ruff конфиг из ближайшего pyproject.toml.
    """

    def __init__(
        self,
        session_dir: Path,
        ruff_config_path: Path | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.ruff_config_path = ruff_config_path
        self._ruff_version = self._detect_ruff_version()

    def enforce(self) -> RuffReport:
        """Запускает ruff на всех .py файлах в session_dir.

        Возвращает:
            RuffReport с результатами по всем файлам.
        """
        py_files = sorted(self.session_dir.rglob("*.py"))
        # Исключаем venv и __pycache__
        py_files = [
            f
            for f in py_files
            if ".venv" not in f.parts
            and "venv" not in f.parts
            and "__pycache__" not in f.parts
        ]

        logger.info(
            "RuffEnforcer: найдено %d .py файлов в %s", len(py_files), self.session_dir
        )

        file_results: list[RuffFileResult] = []
        for py_file in py_files:
            result = self._process_file(py_file)
            file_results.append(result)

        report = self._aggregate(file_results)
        self._save(report)

        logger.info(
            "RuffEnforcer завершён: clean_rate=%.1f%%, violations=%d, target_met=%s",
            report.clean_rate * 100,
            report.total_violations,
            report.target_met,
        )
        return report

    def log_to_mlflow(
        self,
        report: RuffReport,
        tracking_uri: str,
        experiment_id: str,
        session_id: str,
    ) -> None:
        """Логирует метрики RuffReport в MLflow Session Summary Run.

        Args:
            report: результаты ruff.
            tracking_uri: MLflow tracking URI.
            experiment_id: ID эксперимента.
            session_id: ID сессии.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(f"uaf/{session_id}")

            with mlflow.start_run(
                run_name="session_summary/ruff",
                experiment_id=experiment_id,
            ):
                mlflow.log_metrics(
                    {
                        "ruff_clean_rate": report.clean_rate,
                        "ruff_total_violations": float(report.total_violations),
                        "ruff_files_with_issues": float(report.files_with_unfixable),
                        "ruff_total_files": float(report.total_files),
                    }
                )
                mlflow.set_tag("type", "ruff_report")
                mlflow.set_tag("session_id", session_id)
                mlflow.set_tag("ruff_target_met", str(report.target_met))

                # Логируем ruff_report.json как артефакт
                ruff_report_path = self.session_dir / "ruff_report.json"
                if ruff_report_path.exists():
                    mlflow.log_artifact(str(ruff_report_path))

            logger.info(
                "RuffEnforcer: метрики залогированы в MLflow (clean_rate=%.2f)",
                report.clean_rate,
            )
        except Exception as exc:
            logger.warning("Не удалось залогировать ruff метрики в MLflow: %s", exc)

    def _process_file(self, py_file: Path) -> RuffFileResult:
        """Применяет ruff format и ruff check --fix к одному файлу.

        Args:
            py_file: путь к .py файлу.

        Returns:
            RuffFileResult с результатами.
        """
        base_cmd = ["ruff"]
        if self.ruff_config_path:
            base_cmd += ["--config", str(self.ruff_config_path)]

        # Шаг 1: ruff format
        format_result = subprocess.run(
            [*base_cmd, "format", str(py_file)],
            capture_output=True,
            text=True,
        )
        formatted = format_result.returncode == 0
        format_error = format_result.stderr.strip() if not formatted else None

        # Шаг 2: ruff check (без --fix) для подсчёта нарушений до
        check_before = subprocess.run(
            [*base_cmd, "check", str(py_file), "--output-format", "json"],
            capture_output=True,
            text=True,
        )
        violations_before = self._parse_json_violations(check_before.stdout)
        count_before = len(violations_before)

        # Шаг 3: ruff check --fix
        subprocess.run(
            [*base_cmd, "check", str(py_file), "--fix", "--exit-zero"],
            capture_output=True,
            text=True,
        )

        # Шаг 4: ruff check после --fix (unfixable)
        check_after = subprocess.run(
            [*base_cmd, "check", str(py_file), "--output-format", "json"],
            capture_output=True,
            text=True,
        )
        violations_after = self._parse_json_violations(check_after.stdout)
        count_after = len(violations_after)

        return RuffFileResult(
            file=py_file,
            formatted=formatted,
            violations_before_fix=count_before,
            violations_after_fix=count_after,
            unfixable_violations=violations_after,
            format_error=format_error,
        )

    def _parse_json_violations(self, output: str) -> list[RuffViolation]:
        """Парсит JSON вывод ruff check --output-format json.

        Args:
            output: JSON строка от ruff.

        Returns:
            Список RuffViolation.
        """
        if not output.strip():
            return []
        try:
            items = json.loads(output)
        except json.JSONDecodeError:
            return []

        violations = []
        for item in items:
            # ruff json format: {"filename": ..., "code": ..., "message": ...,
            #                    "location": {"row": ..., "column": ...}, "fix": ...}
            violations.append(
                RuffViolation(
                    rule=item.get("code", ""),
                    message=item.get("message", ""),
                    line=item.get("location", {}).get("row", 0),
                    col=item.get("location", {}).get("column", 0),
                    fixable=item.get("fix") is not None,
                )
            )
        return violations

    def _aggregate(self, file_results: list[RuffFileResult]) -> RuffReport:
        """Агрегирует результаты по файлам в общий отчёт.

        Args:
            file_results: список результатов по файлам.

        Returns:
            RuffReport.
        """
        total_files = len(file_results)
        clean_files = sum(1 for r in file_results if r.violations_after_fix == 0)
        files_with_unfixable = sum(1 for r in file_results if r.violations_after_fix > 0)
        total_violations = sum(r.violations_after_fix for r in file_results)
        clean_rate = clean_files / total_files if total_files > 0 else 1.0

        return RuffReport(
            total_files=total_files,
            clean_files=clean_files,
            files_with_unfixable=files_with_unfixable,
            total_violations=total_violations,
            clean_rate=round(clean_rate, 4),
            files=file_results,
            ruff_version=self._ruff_version,
            target_met=clean_rate >= _TARGET_CLEAN_RATE,
        )

    def _save(self, report: RuffReport) -> None:
        """Сохраняет RuffReport в ruff_report.json.

        Args:
            report: отчёт для сохранения.
        """
        output_path = self.session_dir / "ruff_report.json"

        files_data = []
        for fr in report.files:
            files_data.append(
                {
                    "file": str(fr.file),
                    "formatted": fr.formatted,
                    "violations_before_fix": fr.violations_before_fix,
                    "violations_after_fix": fr.violations_after_fix,
                    "format_error": fr.format_error,
                    "unfixable": [
                        {
                            "rule": v.rule,
                            "message": v.message,
                            "line": v.line,
                            "col": v.col,
                        }
                        for v in fr.unfixable_violations
                    ],
                }
            )

        data = {
            "total_files": report.total_files,
            "clean_files": report.clean_files,
            "files_with_unfixable": report.files_with_unfixable,
            "total_violations": report.total_violations,
            "clean_rate": report.clean_rate,
            "ruff_version": report.ruff_version,
            "target_met": report.target_met,
            "target_clean_rate": _TARGET_CLEAN_RATE,
            "files": files_data,
        }
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("ruff_report.json сохранён: %s", output_path)

    @staticmethod
    def _detect_ruff_version() -> str:
        """Определяет версию установленного ruff.

        Returns:
            Строка с версией или 'unknown'.
        """
        try:
            result = subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return "unknown"
