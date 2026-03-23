"""SmokeTestRunner — 12 тестов ST-01..ST-12 перед каждым запуском эксперимента."""

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from uaf import MLFLOW_DEFAULT_URI

logger = logging.getLogger(__name__)

# Статус одного теста
TestStatus = Literal["passed", "failed", "skipped", "warning"]

# Обязательные UAF-SECTION маркеры (11 секций)
_REQUIRED_SECTIONS = [
    "# UAF-SECTION: IMPORTS",
    "# UAF-SECTION: CONFIG",
    "# UAF-SECTION: MLFLOW-INIT",
    "# UAF-SECTION: DATA-LOADING",
    "# UAF-SECTION: PREPROCESSING",
    "# UAF-SECTION: MODEL-DEFINITION",
    "# UAF-SECTION: TRAINING",
    "# UAF-SECTION: EVALUATION",
    "# UAF-SECTION: MLFLOW-LOGGING",
    "# UAF-SECTION: ARTIFACT-SAVING",
    "# UAF-SECTION: BUDGET-CHECK",
]

# Секции, которые должны быть заполнены (не содержать NotImplementedError)
_SECTIONS_MUST_BE_FILLED = {
    "DATA-LOADING",
    "MODEL-DEFINITION",
    "TRAINING",
}

# Обязательные ключи experiment_config.yaml
_REQUIRED_CONFIG_KEYS = [
    "session",
    "iteration",
    "task",
    "random_seed",
]

# Паттерн для поиска абсолютных путей (исключаем /tmp и пути к данным)
_HARDCODED_PATH_PATTERN = re.compile(
    r'(?<!["\'])((?:/home/|/root/|/Users/|/mnt/(?!d/)|C:\\|D:\\)[^\s\'"]+)',
    re.MULTILINE,
)

# DRY-RUN таймаут
_DRY_RUN_TIMEOUT_SECONDS = 90


@dataclass
class TestResult:
    """Результат одного smoke test.

    Атрибуты:
        id: код теста (ST-01..ST-12).
        status: passed / failed / skipped / warning.
        message: описание результата.
        blocking: блокирует ли запуск при failed.
    """

    id: str
    status: TestStatus
    message: str
    blocking: bool = True


@dataclass
class SmokeTestReport:
    """Полный отчёт по smoke tests.

    Атрибуты:
        session_id: ID сессии.
        iteration: номер итерации.
        timestamp: ISO timestamp.
        passed: True если нет блокирующих failures.
        tests: список результатов.
        blocking_failures: список ID провалившихся блокирующих тестов.
    """

    session_id: str
    iteration: int
    timestamp: str
    passed: bool
    tests: list[TestResult]
    blocking_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для записи в smoke_test_report.json.

        Returns:
            JSON-совместимый словарь.
        """
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "blocking_failures": self.blocking_failures,
            "tests": [
                {
                    "id": t.id,
                    "status": t.status,
                    "message": t.message,
                    "blocking": t.blocking,
                }
                for t in self.tests
            ],
        }


def _get_section_content(code: str, section_name: str) -> str:
    """Извлекает содержимое секции из кода.

    Секция начинается с '# UAF-SECTION: {name}' и заканчивается
    следующим '# UAF-SECTION:' или концом файла.

    Args:
        code: содержимое experiment.py.
        section_name: имя секции (например 'DATA-LOADING').

    Returns:
        Содержимое секции.
    """
    marker = f"# UAF-SECTION: {section_name}"
    start = code.find(marker)
    if start == -1:
        return ""
    # Ищем следующую секцию
    next_marker = code.find("# UAF-SECTION:", start + len(marker))
    if next_marker == -1:
        return code[start:]
    return code[start:next_marker]


class SmokeTestRunner:
    """Запускает 12 smoke tests перед каждым экспериментом.

    Args:
        session_dir: директория сессии.
        session_id: ID сессии.
        mlflow_tracking_uri: URI MLflow для ST-09.
        task_type: тип задачи для определения применимости ST-11.
    """

    def __init__(
        self,
        session_dir: Path,
        session_id: str,
        mlflow_tracking_uri: str = MLFLOW_DEFAULT_URI,
        task_type: str = "tabular_classification",
    ) -> None:
        self.session_dir = session_dir
        self.session_id = session_id
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.task_type = task_type

    def run(
        self,
        experiment_py_path: Path,
        config_yaml_path: Path,
        iteration: int,
        baseline_run_id: str | None = None,
        is_shadow_step: bool = False,
    ) -> SmokeTestReport:
        """Запускает все применимые smoke tests.

        Args:
            experiment_py_path: путь к experiment.py.
            config_yaml_path: путь к experiment_config.yaml.
            iteration: номер итерации.
            baseline_run_id: ID baseline run (для ST-12, только shadow шаги).
            is_shadow_step: True если это shadow feature шаг Phase 2.

        Returns:
            SmokeTestReport с результатами всех тестов.
        """
        from datetime import datetime, timezone

        timestamp = datetime.now(tz=timezone.utc).isoformat()
        code = experiment_py_path.read_text(encoding="utf-8") if experiment_py_path.exists() else ""

        results: list[TestResult] = []

        results.append(self._st01_scaffold_sections(code))
        results.append(self._st02_python_syntax(experiment_py_path))
        results.append(self._st03_ruff_lint(experiment_py_path))
        results.append(self._st04_mlflow_start_run(code))
        results.append(self._st05_check_budget(code))
        results.append(self._st06_seed_fixed(code))
        results.append(self._st07_no_not_implemented(code))
        results.append(self._st08_config_valid(config_yaml_path))
        results.append(self._st09_mlflow_reachable())
        results.append(self._st10_no_hardcoded_paths(code))
        results.append(self._st11_dry_run(experiment_py_path))
        results.append(
            self._st12_baseline_run_valid(baseline_run_id, is_shadow_step)
        )

        blocking_failures = [r.id for r in results if r.status == "failed" and r.blocking]
        passed = len(blocking_failures) == 0

        report = SmokeTestReport(
            session_id=self.session_id,
            iteration=iteration,
            timestamp=timestamp,
            passed=passed,
            tests=results,
            blocking_failures=blocking_failures,
        )

        # Пишем отчёт в SESSION_DIR
        report_path = self.session_dir / "smoke_test_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "SmokeTestRunner iter=%d: passed=%s, failures=%s",
            iteration,
            passed,
            blocking_failures,
        )
        return report

    # === ST-01: scaffold sections ===

    def _st01_scaffold_sections(self, code: str) -> TestResult:
        """ST-01: все 11 UAF-SECTION: маркеров присутствуют."""
        missing = [s for s in _REQUIRED_SECTIONS if s not in code]
        if missing:
            return TestResult(
                "ST-01",
                "failed",
                f"Отсутствуют секции: {missing}",
                blocking=True,
            )
        return TestResult("ST-01", "passed", f"Все {len(_REQUIRED_SECTIONS)} секций найдены")

    # === ST-02: python syntax ===

    def _st02_python_syntax(self, path: Path) -> TestResult:
        """ST-02: синтаксис Python корректен (compile())."""
        if not path.exists():
            return TestResult("ST-02", "failed", f"Файл не найден: {path}", blocking=True)
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            return TestResult(
                "ST-02",
                "failed",
                f"SyntaxError: {exc.msg} (line {exc.lineno})",
                blocking=True,
            )
        return TestResult("ST-02", "passed", "Синтаксис корректен")

    # === ST-03: ruff lint ===

    def _st03_ruff_lint(self, path: Path) -> TestResult:
        """ST-03: ruff check без ошибок E,F,W."""
        if not path.exists():
            return TestResult("ST-03", "failed", f"Файл не найден: {path}", blocking=True)

        ruff = _find_ruff()
        if ruff is None:
            return TestResult("ST-03", "skipped", "ruff не найден в PATH, пропуск ST-03", blocking=False)

        try:
            result = subprocess.run(
                [ruff, "check", str(path), "--select", "E,F,W", "--output-format", "concise"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return TestResult("ST-03", "failed", "ruff check timeout (30s)", blocking=True)
        except FileNotFoundError:
            return TestResult("ST-03", "skipped", "ruff не запустился", blocking=False)

        if result.returncode != 0:
            # Берём первые 3 строки ошибок
            errors = result.stdout.strip().split("\n")[:3]
            return TestResult(
                "ST-03",
                "failed",
                f"ruff: {'; '.join(errors)}",
                blocking=True,
            )
        return TestResult("ST-03", "passed", "ruff lint чист (E,F,W)")

    # === ST-04: mlflow.start_run ===

    def _st04_mlflow_start_run(self, code: str) -> TestResult:
        """ST-04: mlflow.start_run присутствует в коде."""
        if "mlflow.start_run" not in code:
            return TestResult(
                "ST-04",
                "failed",
                "mlflow.start_run не найден в experiment.py",
                blocking=True,
            )
        return TestResult("ST-04", "passed", "mlflow.start_run найден")

    # === ST-05: check_budget ===

    def _st05_check_budget(self, code: str) -> TestResult:
        """ST-05: check_budget / budget_status вызов присутствует."""
        patterns = ["check_budget", "budget_status", "hard_stop", "UAF_BUDGET_STATUS_FILE"]
        found = any(p in code for p in patterns)
        if not found:
            return TestResult(
                "ST-05",
                "failed",
                "Проверка бюджета (check_budget / budget_status) не найдена",
                blocking=True,
            )
        return TestResult("ST-05", "passed", "Проверка бюджета найдена")

    # === ST-06: seed фиксирован ===

    def _st06_seed_fixed(self, code: str) -> TestResult:
        """ST-06: seed зафиксирован (random.seed / np.random.seed / torch.manual_seed)."""
        seed_patterns = [
            "random.seed",
            "np.random.seed",
            "numpy.random.seed",
            "torch.manual_seed",
            "random_seed",
            "seed(",
        ]
        found = any(p in code for p in seed_patterns)
        if not found:
            return TestResult(
                "ST-06",
                "failed",
                "Seed не зафиксирован. Нет вызовов random.seed/np.random.seed/torch.manual_seed",
                blocking=True,
            )
        return TestResult("ST-06", "passed", "Seed зафиксирован")

    # === ST-07: нет NotImplementedError в заполненных секциях ===

    def _st07_no_not_implemented(self, code: str) -> TestResult:
        """ST-07: заполненные секции не содержат raise NotImplementedError."""
        offenders: list[str] = []
        for section_name in _SECTIONS_MUST_BE_FILLED:
            section_content = _get_section_content(code, section_name)
            if "raise NotImplementedError" in section_content:
                offenders.append(section_name)

        if offenders:
            return TestResult(
                "ST-07",
                "failed",
                f"NotImplementedError в секциях: {offenders}",
                blocking=True,
            )
        return TestResult("ST-07", "passed", "NotImplementedError не найден в обязательных секциях")

    # === ST-08: experiment_config.yaml валиден ===

    def _st08_config_valid(self, config_path: Path) -> TestResult:
        """ST-08: experiment_config.yaml существует и содержит обязательные ключи."""
        if not config_path.exists():
            return TestResult(
                "ST-08",
                "failed",
                f"experiment_config.yaml не найден: {config_path}",
                blocking=True,
            )
        try:
            with config_path.open() as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            return TestResult(
                "ST-08",
                "failed",
                f"Ошибка парсинга YAML: {exc}",
                blocking=True,
            )

        if not isinstance(config, dict):
            return TestResult(
                "ST-08",
                "failed",
                "experiment_config.yaml не является словарём",
                blocking=True,
            )

        missing_keys = [k for k in _REQUIRED_CONFIG_KEYS if k not in config]
        if missing_keys:
            return TestResult(
                "ST-08",
                "failed",
                f"Отсутствуют обязательные ключи: {missing_keys}",
                blocking=True,
            )

        # Проверяем вложенные ключи
        task = config.get("task", {})
        if not isinstance(task, dict) or "type" not in task or "metric" not in task:
            return TestResult(
                "ST-08",
                "failed",
                "task секция должна содержать 'type' и 'metric'",
                blocking=True,
            )
        return TestResult("ST-08", "passed", "experiment_config.yaml валиден")

    # === ST-09: MLflow доступен ===

    def _st09_mlflow_reachable(self) -> TestResult:
        """ST-09: MLflow сервер доступен (HTTP ping)."""
        import urllib.request

        health_url = self.mlflow_tracking_uri.rstrip("/") + "/health"
        # Для file:// URI — проверяем директорию
        if self.mlflow_tracking_uri.startswith("file://") or not self.mlflow_tracking_uri.startswith("http"):
            # sqlite:// — проверяем директорию родителя или наличие БД
            if self.mlflow_tracking_uri.startswith("sqlite://"):
                # sqlite:///path/to/mlflow.db или sqlite:////abs/path
                raw = self.mlflow_tracking_uri.split("sqlite:///", 1)[-1]
                db_path = Path("/" + raw) if not raw.startswith("/") else Path(raw)
                if db_path.exists() or db_path.parent.exists():
                    return TestResult("ST-09", "passed", f"MLflow SQLite URI: {db_path}")
                return TestResult(
                    "ST-09",
                    "failed",
                    f"MLflow SQLite: родительская директория не найдена: {db_path.parent}",
                    blocking=True,
                )
            uri_path = self.mlflow_tracking_uri.replace("file://", "")
            if Path(uri_path).exists():
                return TestResult("ST-09", "passed", f"MLflow URI доступен: {uri_path}")
            return TestResult(
                "ST-09",
                "failed",
                f"MLflow URI недоступен: {self.mlflow_tracking_uri}",
                blocking=True,
            )
        try:
            urllib.request.urlopen(health_url, timeout=5)
            return TestResult("ST-09", "passed", f"MLflow сервер доступен: {health_url}")
        except Exception as exc:
            return TestResult(
                "ST-09",
                "failed",
                f"MLflow сервер недоступен ({health_url}): {exc}",
                blocking=True,
            )

    # === ST-10: нет хардкода путей (warning) ===

    def _st10_no_hardcoded_paths(self, code: str) -> TestResult:
        """ST-10: нет хардкода абсолютных путей (предупреждение, не блокирует)."""
        matches = _HARDCODED_PATH_PATTERN.findall(code)
        if matches:
            # Фильтруем заведомо допустимые паттерны в строках документации
            real_matches = [m for m in matches if not m.startswith("/tmp")]
            if real_matches:
                return TestResult(
                    "ST-10",
                    "warning",
                    f"Возможные хардкод-пути: {real_matches[:3]}",
                    blocking=False,
                )
        return TestResult("ST-10", "passed", "Хардкод путей не обнаружен")

    # === ST-11: dry-run для tabular ===

    def _st11_dry_run(self, experiment_py_path: Path) -> TestResult:
        """ST-11: dry-run для tabular задач (< 90 сек, 100 строк данных)."""
        is_tabular = "tabular" in self.task_type.lower()
        if not is_tabular:
            return TestResult("ST-11", "skipped", f"Пропуск dry-run для task_type={self.task_type}", blocking=False)

        if not experiment_py_path.exists():
            return TestResult("ST-11", "failed", "experiment.py не найден", blocking=True)

        # Проверяем наличие --dry-run флага в коде
        code = experiment_py_path.read_text(encoding="utf-8")
        if "--dry-run" not in code and "dry_run" not in code and "dry-run" not in code:
            return TestResult(
                "ST-11",
                "failed",
                "Флаг --dry-run не реализован в experiment.py. "
                "Добавь argparse с --dry-run для табличных задач.",
                blocking=True,
            )

        python_exe = sys.executable
        start = time.time()
        try:
            result = subprocess.run(
                [python_exe, str(experiment_py_path), "--dry-run"],
                capture_output=True,
                text=True,
                timeout=_DRY_RUN_TIMEOUT_SECONDS,
                cwd=str(experiment_py_path.parent),
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                "ST-11",
                "failed",
                f"Dry-run превысил таймаут {_DRY_RUN_TIMEOUT_SECONDS}s",
                blocking=True,
            )
        except FileNotFoundError as exc:
            return TestResult("ST-11", "failed", f"Не удалось запустить Python: {exc}", blocking=True)

        elapsed = time.time() - start
        if result.returncode != 0:
            stderr_lines = (result.stderr or "").strip().split("\n")[-5:]
            return TestResult(
                "ST-11",
                "failed",
                f"Dry-run завершился с ошибкой (exit {result.returncode}): "
                f"{'; '.join(stderr_lines)}",
                blocking=True,
            )
        return TestResult(
            "ST-11",
            "passed",
            f"Dry-run успешен за {elapsed:.1f}s",
        )

    # === ST-12: baseline_run_id валиден (shadow шаги Phase 2) ===

    def _st12_baseline_run_valid(
        self,
        baseline_run_id: str | None,
        is_shadow_step: bool,
    ) -> TestResult:
        """ST-12: baseline_run_id существует в MLflow (только для shadow шагов).

        Args:
            baseline_run_id: ID baseline run из program.md.
            is_shadow_step: True если шаг с method=shadow_feature_trick.

        Returns:
            TestResult.
        """
        if not is_shadow_step:
            return TestResult("ST-12", "skipped", "N/A (не shadow feature шаг)", blocking=False)

        if not baseline_run_id:
            return TestResult(
                "ST-12",
                "failed",
                "baseline_run_id не задан для shadow feature шага. "
                "Заполни baseline_run_id в program.md после Phase 1.",
                blocking=True,
            )

        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(baseline_run_id)
            if run is None:
                raise ValueError("run is None")
            return TestResult(
                "ST-12",
                "passed",
                f"baseline_run_id={baseline_run_id} найден в MLflow",
            )
        except Exception as exc:
            return TestResult(
                "ST-12",
                "failed",
                f"baseline_run_id={baseline_run_id} не найден в MLflow: {exc}",
                blocking=True,
            )


def _find_ruff() -> str | None:
    """Ищет ruff в PATH.

    Returns:
        Путь к ruff или None.
    """
    import shutil

    return shutil.which("ruff")
