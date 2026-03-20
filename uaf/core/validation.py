"""ValidationChecker — проверки схемы валидации UAF (pre-session и post-run)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Тип серьёзности проверки
CheckSeverity = Literal["PASS", "WARN", "ERROR"]

# Допустимые схемы валидации
ValidationScheme = Literal[
    "auto",
    "holdout",
    "kfold",
    "stratified_kfold",
    "group_kfold",
    "time_series_split",
    "leave_one_out",
]

# Матрица совместимости task.type + scheme
# "OK" = ок, "WARN" = предупреждение, "ERROR" = ошибка
_COMPATIBILITY: dict[str, dict[str, str]] = {
    "tabular_classification": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "WARN",
    },
    "tabular_regression": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "WARN",
    },
    "nlp_classification": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "WARN",
    },
    "nlp_ner": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "WARN",
    },
    "nlp_generation": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "WARN",
    },
    "cv_classification": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "ERROR",
    },
    "cv_detection": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "ERROR",
    },
    "cv_segmentation": {
        "holdout": "OK",
        "kfold": "OK",
        "stratified_kfold": "OK",
        "group_kfold": "OK",
        "time_series_split": "ERROR",
    },
    "time_series": {
        "holdout": "ERROR",
        "kfold": "ERROR",
        "stratified_kfold": "ERROR",
        "group_kfold": "ERROR",
        "time_series_split": "OK",
    },
    "recsys": {
        "holdout": "OK",
        "kfold": "WARN",
        "stratified_kfold": "WARN",
        "group_kfold": "OK",
        "time_series_split": "OK",
    },
}


@dataclass
class CheckResult:
    """Результат одной проверки.

    Атрибуты:
        code: код проверки (VS-* или VR-*).
        status: PASS / WARN / ERROR.
        message: описание результата.
        hint: подсказка для Claude Code (опционально).
    """

    code: str
    status: CheckSeverity
    message: str
    hint: str = ""


@dataclass
class ValidationReport:
    """Отчёт по всем проверкам.

    Атрибуты:
        checks: список результатов.
        scheme: итоговая схема (после автовыбора).
        resolved_by: "auto" или "user-specified".
        has_errors: есть ли блокирующие ошибки.
        has_warnings: есть ли предупреждения.
    """

    checks: list[CheckResult] = field(default_factory=list)
    scheme: str = ""
    resolved_by: str = "user-specified"
    has_errors: bool = False
    has_warnings: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для записи в MLflow / JSON.

        Returns:
            Словарь с результатами.
        """
        return {
            "scheme": self.scheme,
            "resolved_by": self.resolved_by,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "checks": [
                {
                    "code": c.code,
                    "status": c.status,
                    "message": c.message,
                    "hint": c.hint,
                }
                for c in self.checks
            ],
        }

    def summary_lines(self) -> list[str]:
        """Текстовые строки для вывода в терминал.

        Returns:
            Список строк с цветовыми метками.
        """
        lines = [f"Схема: {self.scheme} (resolved by: {self.resolved_by})", ""]
        for c in self.checks:
            lines.append(f"[{c.status}] {c.code}: {c.message}")
            if c.hint:
                lines.append(f"       Hint: {c.hint}")
        return lines


@dataclass
class ValidationConfig:
    """Параметры схемы валидации из task.yaml.

    Атрибуты:
        scheme: выбранная схема (auto = автовыбор).
        n_splits: число фолдов.
        shuffle: перемешивание (False обязательно для time_series).
        seed: зерно воспроизводимости.
        group_col: колонка группировки для GroupKFold.
        stratify_col: колонка стратификации (None = target).
        test_holdout: выделять ли test set.
        test_ratio: доля test set.
        train_ratio: доля train (для holdout).
        val_ratio: доля val (для holdout).
        gap: разрыв между train и val (для time_series_split).
        forecast_horizon: горизонт прогноза (для time_series).
    """

    scheme: ValidationScheme = "auto"
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42
    group_col: str | None = None
    stratify_col: str | None = None
    test_holdout: bool = True
    test_ratio: float = 0.1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    gap: int = 0
    forecast_horizon: int = 0


def _auto_select_scheme(
    task_type: str,
    n_rows: int,
    has_document_id: bool = False,
) -> tuple[ValidationScheme, int]:
    """Автовыбор схемы валидации по типу задачи и размеру данных.

    Args:
        task_type: тип ML-задачи из task.yaml.
        n_rows: количество строк в обучающей выборке.
        has_document_id: есть ли document_id / source_id в данных.

    Returns:
        Tuple (scheme, n_splits).
    """
    if "time_series" in task_type:
        return "time_series_split", 5

    if task_type.startswith("nlp") and has_document_id:
        return "group_kfold", 5

    if n_rows < 1000:
        k = 10
        scheme: ValidationScheme = (
            "stratified_kfold" if "classification" in task_type else "kfold"
        )
        return scheme, k

    if n_rows < 50000:
        k = 5
        scheme = "stratified_kfold" if "classification" in task_type else "kfold"
        return scheme, k

    return "holdout", 1


class ValidationChecker:
    """Проверки схемы валидации: 18 pre-session (VS-*) и 7 post-run (VR-*).

    Позиция в архитектуре: встроен в ResearchSessionController.
    Запускается после DataLoader/LeakageAudit/AdversarialValidation.

    Args:
        task_type: тип ML-задачи.
        config: параметры схемы валидации из task.yaml.
        data_schema_path: путь к data_schema.json (для чтения колонок и статистик).
    """

    def __init__(
        self,
        task_type: str,
        config: ValidationConfig,
        data_schema_path: Path | None = None,
    ) -> None:
        self.task_type = task_type
        self.config = config
        self.data_schema_path = data_schema_path
        self._data_schema: dict[str, Any] = {}
        if data_schema_path and data_schema_path.exists():
            import json

            self._data_schema = json.loads(data_schema_path.read_text())

    def run_pre_session(
        self,
        df: pd.DataFrame,
        target_col: str,
        adversarial_auc: float | None = None,
    ) -> ValidationReport:
        """Запускает все 18 pre-session проверок (VS-*).

        Args:
            df: обучающий датасет.
            target_col: имя целевой колонки.
            adversarial_auc: AUC из AdversarialValidation (опционально).

        Returns:
            ValidationReport с результатами всех проверок.
        """
        n_rows = len(df)
        report = ValidationReport()

        # Автовыбор схемы если нужно
        if self.config.scheme == "auto":
            has_doc_id = self._has_document_id(df)
            scheme, n_splits = _auto_select_scheme(self.task_type, n_rows, has_doc_id)
            self.config.scheme = scheme
            if self.config.scheme in ("kfold", "stratified_kfold", "group_kfold", "time_series_split"):
                self.config.n_splits = n_splits
            report.resolved_by = "auto"
            logger.info(
                "Auto-selected scheme: %s (k=%d, N=%d, task=%s)",
                scheme,
                n_splits,
                n_rows,
                self.task_type,
            )
        else:
            report.resolved_by = "user-specified"

        report.scheme = self.config.scheme

        # Запуск всех VS-* проверок
        checks = [
            self._check_vs_t001(df),
            self._check_vs_t002(df),
            self._check_vs_t003(df),
            self._check_vs_t004(df),
            self._check_vs_t005(df),
            self._check_vs_k001(),
            self._check_vs_k002(),
            self._check_vs_k003(df, target_col),
            self._check_vs_s001(),
            self._check_vs_s002(),
            self._check_vs_g001(df),
            self._check_vs_g002(df),
            self._check_vs_l001(df),
            self._check_vs_l002(df),
            self._check_vs_l003(df, target_col),
            self._check_vs_a001(adversarial_auc),
            self._check_vs_a002(adversarial_auc),
            self._check_vs_c001(),
            self._check_vs_c002(),
        ]

        report.checks = [c for c in checks if c is not None]
        report.has_errors = any(c.status == "ERROR" for c in report.checks)
        report.has_warnings = any(c.status == "WARN" for c in report.checks)

        return report

    def run_post_run(
        self,
        mlflow_run: Any,
        metric_name: str,
    ) -> list[CheckResult]:
        """Запускает 7 post-run проверок (VR-*) для одного MLflow run.

        Args:
            mlflow_run: объект MLflow Run.
            metric_name: имя целевой метрики.

        Returns:
            Список результатов проверок.
        """
        results = [
            self._check_vr001(mlflow_run),
            self._check_vr002(mlflow_run),
            self._check_vr003(mlflow_run),
            self._check_vr004(mlflow_run, metric_name),
            self._check_vr005(mlflow_run, metric_name),
            self._check_vr006(mlflow_run, metric_name),
            self._check_vr007(mlflow_run),
        ]
        return [r for r in results if r is not None]

    def run_post_run_fe(self, mlflow_run: Any) -> CheckResult | None:
        """Проверка VR-FE-001: leakage в shadow feature experiments.

        Args:
            mlflow_run: объект MLflow Run с тегами shadow experiment.

        Returns:
            CheckResult или None если не shadow эксперимент.
        """
        tags = mlflow_run.data.tags if hasattr(mlflow_run, "data") else {}
        if tags.get("method") != "shadow_feature_trick":
            return None
        return self._check_vr_fe001(mlflow_run)

    # === VS-T-* : Holdout checks ===

    def _check_vs_t001(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-T-001: train + val + test ratio == 1.0."""
        if self.config.scheme != "holdout":
            return CheckResult("VS-T-001", "PASS", "N/A (не holdout схема)")
        total = self.config.train_ratio + self.config.val_ratio + (
            self.config.test_ratio if self.config.test_holdout else 0.0
        )
        if abs(total - 1.0) > 1e-6:
            return CheckResult(
                "VS-T-001",
                "ERROR",
                f"train+val+test={total:.4f} != 1.0",
                "Исправь train_ratio / val_ratio / test_ratio в task.yaml",
            )
        return CheckResult("VS-T-001", "PASS", f"Сумма долей = {total:.4f}")

    def _check_vs_t002(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-T-002: train_ratio >= 0.5."""
        if self.config.scheme != "holdout":
            return CheckResult("VS-T-002", "PASS", "N/A (не holdout схема)")
        if self.config.train_ratio < 0.5:
            return CheckResult(
                "VS-T-002",
                "ERROR",
                f"train_ratio={self.config.train_ratio} < 0.5",
                "Рекомендуется train_ratio >= 0.7",
            )
        return CheckResult("VS-T-002", "PASS", f"train_ratio={self.config.train_ratio}")

    def _check_vs_t003(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-T-003: val_ratio >= 0.05."""
        if self.config.scheme != "holdout":
            return CheckResult("VS-T-003", "PASS", "N/A (не holdout схема)")
        if self.config.val_ratio < 0.05:
            return CheckResult(
                "VS-T-003",
                "ERROR",
                f"val_ratio={self.config.val_ratio} < 0.05",
                "val_ratio должен быть >= 0.05 для надёжной оценки",
            )
        return CheckResult("VS-T-003", "PASS", f"val_ratio={self.config.val_ratio}")

    def _check_vs_t004(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-T-004: val set >= 30 строк."""
        n_rows = len(df)
        if self.config.scheme == "holdout":
            val_abs = int(n_rows * self.config.val_ratio)
        elif self.config.scheme in ("kfold", "stratified_kfold", "group_kfold"):
            val_abs = n_rows // self.config.n_splits
        else:
            return CheckResult("VS-T-004", "PASS", "N/A для данной схемы")

        if val_abs < 30:
            return CheckResult(
                "VS-T-004",
                "ERROR",
                f"val set ~{val_abs} строк < 30",
                f"Уменьши число фолдов или увеличь датасет (N={n_rows})",
            )
        return CheckResult("VS-T-004", "PASS", f"val_abs ~{val_abs} строк")

    def _check_vs_t005(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-T-005: train set >= 100 строк."""
        n_rows = len(df)
        if self.config.scheme == "holdout":
            train_abs = int(n_rows * self.config.train_ratio)
        elif self.config.scheme in ("kfold", "stratified_kfold", "group_kfold"):
            k = self.config.n_splits
            train_abs = int(n_rows * (k - 1) / k)
        else:
            train_abs = n_rows
        if train_abs < 100:
            return CheckResult(
                "VS-T-005",
                "ERROR",
                f"train set ~{train_abs} строк < 100",
                "Датасет слишком мал. Используй k-fold с большим k или расширь данные",
            )
        return CheckResult("VS-T-005", "PASS", f"train_abs ~{train_abs} строк")

    # === VS-K-* : KFold checks ===

    def _check_vs_k001(self) -> CheckResult | None:
        """VS-K-001: n_splits >= 2."""
        if self.config.scheme not in ("kfold", "stratified_kfold", "group_kfold", "time_series_split"):
            return CheckResult("VS-K-001", "PASS", "N/A (не k-fold схема)")
        if self.config.n_splits < 2:
            return CheckResult(
                "VS-K-001",
                "ERROR",
                f"n_splits={self.config.n_splits} < 2",
                "Минимально допустимое значение n_splits=2",
            )
        return CheckResult("VS-K-001", "PASS", f"n_splits={self.config.n_splits}")

    def _check_vs_k002(self) -> CheckResult | None:
        """VS-K-002: n_splits <= 20 (предупреждение)."""
        if self.config.scheme not in ("kfold", "stratified_kfold", "group_kfold", "time_series_split"):
            return CheckResult("VS-K-002", "PASS", "N/A (не k-fold схема)")
        if self.config.n_splits > 20:
            return CheckResult(
                "VS-K-002",
                "WARN",
                f"n_splits={self.config.n_splits} > 20 (высокая вычислительная стоимость)",
                "Рассмотри уменьшение n_splits до 5-10",
            )
        return CheckResult("VS-K-002", "PASS", f"n_splits={self.config.n_splits}")

    def _check_vs_k003(self, df: pd.DataFrame, target_col: str) -> CheckResult | None:
        """VS-K-003: каждый фолд содержит >= 1 sample каждого класса (stratified)."""
        if self.config.scheme != "stratified_kfold":
            return CheckResult("VS-K-003", "PASS", "N/A (не stratified_kfold)")
        if target_col not in df.columns:
            return CheckResult("VS-K-003", "PASS", f"Колонка {target_col} не найдена для проверки")
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=self.config.shuffle, random_state=self.config.seed)
        y = df[target_col]
        try:
            for _ in skf.split(df, y):
                pass
        except ValueError as exc:
            return CheckResult(
                "VS-K-003",
                "ERROR",
                f"StratifiedKFold не может разбить данные: {exc}",
                "Проверь баланс классов и n_splits. Используй kfold вместо stratified_kfold",
            )

        min_class_count = y.value_counts().min()
        min_fold_size = len(df) // self.config.n_splits
        if min_class_count < self.config.n_splits:
            return CheckResult(
                "VS-K-003",
                "ERROR",
                f"Миноритарный класс ({min_class_count} строк) < n_splits={self.config.n_splits}",
                "Уменьши n_splits или используй смешанную стратегию oversampling",
            )
        return CheckResult(
            "VS-K-003",
            "PASS",
            f"Все классы представлены. Мин. фолд ~{min_fold_size} строк",
        )

    # === VS-S-* : Shuffle / time-series checks ===

    def _check_vs_s001(self) -> CheckResult | None:
        """VS-S-001: shuffle=False для time_series_split."""
        if self.config.scheme != "time_series_split":
            return CheckResult("VS-S-001", "PASS", "N/A (не time_series_split)")
        if self.config.shuffle:
            return CheckResult(
                "VS-S-001",
                "ERROR",
                "shuffle=True запрещён для time_series_split",
                "Установи shuffle: false в task.yaml",
            )
        return CheckResult("VS-S-001", "PASS", "shuffle=False для time_series_split")

    def _check_vs_s002(self) -> CheckResult | None:
        """VS-S-002: gap >= forecast_horizon."""
        if self.config.scheme != "time_series_split":
            return CheckResult("VS-S-002", "PASS", "N/A (не time_series_split)")
        if self.config.forecast_horizon > 0 and self.config.gap < self.config.forecast_horizon:
            return CheckResult(
                "VS-S-002",
                "ERROR",
                f"gap={self.config.gap} < forecast_horizon={self.config.forecast_horizon}",
                f"Установи gap >= {self.config.forecast_horizon} в task.yaml",
            )
        return CheckResult(
            "VS-S-002",
            "PASS",
            f"gap={self.config.gap} >= forecast_horizon={self.config.forecast_horizon}",
        )

    # === VS-G-* : Group checks ===

    def _check_vs_g001(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-G-001: group_col присутствует в данных."""
        if self.config.scheme not in ("group_kfold", "leave_one_out"):
            return CheckResult("VS-G-001", "PASS", "N/A (не group схема)")
        if not self.config.group_col:
            return CheckResult(
                "VS-G-001",
                "ERROR",
                "group_col не задан для group_kfold",
                "Задай group_col в секции validation task.yaml",
            )
        if self.config.group_col not in df.columns:
            return CheckResult(
                "VS-G-001",
                "ERROR",
                f"group_col='{self.config.group_col}' не найдена в данных",
                f"Доступные колонки: {list(df.columns)[:10]}",
            )
        return CheckResult("VS-G-001", "PASS", f"group_col='{self.config.group_col}' найдена")

    def _check_vs_g002(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-G-002: количество уникальных групп >= n_splits."""
        if self.config.scheme not in ("group_kfold", "leave_one_out"):
            return CheckResult("VS-G-002", "PASS", "N/A (не group схема)")
        if not self.config.group_col or self.config.group_col not in df.columns:
            return CheckResult("VS-G-002", "PASS", "N/A (group_col не задан или не найден)")
        n_unique = df[self.config.group_col].nunique()
        if n_unique < self.config.n_splits:
            return CheckResult(
                "VS-G-002",
                "ERROR",
                f"Уникальных групп {n_unique} < n_splits={self.config.n_splits}",
                f"Уменьши n_splits до <= {n_unique}",
            )
        return CheckResult("VS-G-002", "PASS", f"Уникальных групп: {n_unique}")

    # === VS-L-* : Leakage checks ===

    def _check_vs_l001(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-L-001: нет overlap между train и val строками."""
        if self.config.scheme not in ("holdout",):
            return CheckResult("VS-L-001", "PASS", "N/A (применимо только к holdout)")
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_start = train_end
        val_end = train_end + int(n * self.config.val_ratio)
        train_idx = set(range(0, train_end))
        val_idx = set(range(val_start, val_end))
        overlap = train_idx & val_idx
        if overlap:
            return CheckResult(
                "VS-L-001",
                "ERROR",
                f"Пересечение train/val индексов: {len(overlap)} строк",
                "Проверь логику разбиения данных",
            )
        return CheckResult("VS-L-001", "PASS", "Нет overlap train/val")

    def _check_vs_l002(self, df: pd.DataFrame) -> CheckResult | None:
        """VS-L-002: нет overlap между train и test строками."""
        if self.config.scheme != "holdout" or not self.config.test_holdout:
            return CheckResult("VS-L-002", "PASS", "N/A")
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        test_start = int(n * (self.config.train_ratio + self.config.val_ratio))
        train_idx = set(range(0, train_end))
        test_idx = set(range(test_start, n))
        overlap = train_idx & test_idx
        if overlap:
            return CheckResult(
                "VS-L-002",
                "ERROR",
                f"Пересечение train/test индексов: {len(overlap)} строк",
                "Проверь логику разбиения данных",
            )
        return CheckResult("VS-L-002", "PASS", "Нет overlap train/test")

    def _check_vs_l003(self, df: pd.DataFrame, target_col: str) -> CheckResult | None:
        """VS-L-003: target не присутствует в feature columns явно."""
        feature_cols = [c for c in df.columns if c != target_col]
        # Проверяем наличие производных имён
        suspicious = [c for c in feature_cols if target_col.lower() in c.lower() and c != target_col]
        if suspicious:
            return CheckResult(
                "VS-L-003",
                "WARN",
                f"Подозрительные колонки с именем target: {suspicious}",
                "Убедись что это не утечка целевой переменной в признаки",
            )
        return CheckResult("VS-L-003", "PASS", "Target-leakage не обнаружена в именах колонок")

    # === VS-A-* : Adversarial validation checks ===

    def _check_vs_a001(self, auc: float | None) -> CheckResult | None:
        """VS-A-001: AdversarialValidation AUC < 0.85 (блокировка с override)."""
        if auc is None:
            return CheckResult("VS-A-001", "PASS", "AdversarialValidation не запускался")
        if auc >= 0.85:
            return CheckResult(
                "VS-A-001",
                "ERROR",
                f"AdversarialValidation AUC={auc:.3f} >= 0.85 (критический порог)",
                "Train и val/test распределения СИЛЬНО отличаются. "
                "Проверь логику сплита, наличие временного смещения данных. "
                "Для продолжения нужен явный y в HumanOversightGate.",
            )
        return CheckResult("VS-A-001", "PASS", f"AdversarialValidation AUC={auc:.3f} < 0.85")

    def _check_vs_a002(self, auc: float | None) -> CheckResult | None:
        """VS-A-002: AdversarialValidation AUC 0.6..0.85 — предупреждение."""
        if auc is None:
            return CheckResult("VS-A-002", "PASS", "AdversarialValidation не запускался")
        if 0.6 <= auc < 0.85:
            return CheckResult(
                "VS-A-002",
                "WARN",
                f"AdversarialValidation AUC={auc:.3f} (умеренное отличие train/val)",
                "Распределения несколько отличаются. "
                "Рассмотри проверку на covariate shift. "
                "Hint добавлен в program.md.",
            )
        return CheckResult("VS-A-002", "PASS", f"AdversarialValidation AUC={auc:.3f} < 0.6")

    # === VS-C-* : Compatibility checks ===

    def _check_vs_c001(self) -> CheckResult | None:
        """VS-C-001: seed задан."""
        if self.config.seed is None:
            return CheckResult(
                "VS-C-001",
                "WARN",
                "seed не задан — воспроизводимость нарушена",
                "Добавь seed: 42 в секцию validation task.yaml",
            )
        return CheckResult("VS-C-001", "PASS", f"seed={self.config.seed}")

    def _check_vs_c002(self) -> CheckResult | None:
        """VS-C-002: схема совместима с task.type."""
        scheme = self.config.scheme
        if scheme == "auto":
            return CheckResult("VS-C-002", "PASS", "Схема auto — проверка после разрешения")

        compat = _COMPATIBILITY.get(self.task_type, {})
        result = compat.get(scheme, "OK")

        if result == "ERROR":
            return CheckResult(
                "VS-C-002",
                "ERROR",
                f"Схема '{scheme}' несовместима с task.type='{self.task_type}'",
                f"Допустимые схемы для {self.task_type}: "
                f"{[s for s, v in compat.items() if v == 'OK']}",
            )
        if result == "WARN":
            return CheckResult(
                "VS-C-002",
                "WARN",
                f"Схема '{scheme}' нетипична для task.type='{self.task_type}'",
                "Проверь что это сознательный выбор",
            )
        return CheckResult("VS-C-002", "PASS", f"Схема '{scheme}' совместима с '{self.task_type}'")

    # === VR-* : Post-run checks ===

    def _check_vr001(self, run: Any) -> CheckResult:
        """VR-001: validation_scheme залогирован как MLflow param."""
        params = run.data.params if hasattr(run, "data") else {}
        if "validation_scheme" not in params:
            return CheckResult(
                "VR-001",
                "WARN",
                f"run={run.info.run_id}: 'validation_scheme' не залогирован",
                "Добавь mlflow.log_param('validation_scheme', scheme) в experiment.py",
            )
        return CheckResult("VR-001", "PASS", f"validation_scheme={params['validation_scheme']}")

    def _check_vr002(self, run: Any) -> CheckResult:
        """VR-002: fold_idx залогирован для k-fold runs."""
        tags = run.data.tags if hasattr(run, "data") else {}
        params = run.data.params if hasattr(run, "data") else {}
        is_kfold = params.get("validation_scheme", "") in (
            "kfold",
            "stratified_kfold",
            "group_kfold",
        )
        if is_kfold and "fold_idx" not in tags:
            return CheckResult(
                "VR-002",
                "WARN",
                f"run={run.info.run_id}: 'fold_idx' не залогирован для k-fold",
                "Добавь mlflow.set_tag('fold_idx', fold_idx) в experiment.py",
            )
        return CheckResult("VR-002", "PASS", "fold_idx проверен")

    def _check_vr003(self, run: Any) -> CheckResult:
        """VR-003: seed залогирован."""
        params = run.data.params if hasattr(run, "data") else {}
        if "seed" not in params and "random_seed" not in params:
            return CheckResult(
                "VR-003",
                "WARN",
                f"run={run.info.run_id}: seed не залогирован",
                "Добавь mlflow.log_param('seed', config['random_seed'])",
            )
        return CheckResult("VR-003", "PASS", "seed залогирован")

    def _check_vr004(self, run: Any, metric_name: str) -> CheckResult:
        """VR-004: метрика залогирована на каждом фолде."""
        params = run.data.params if hasattr(run, "data") else {}
        metrics = run.data.metrics if hasattr(run, "data") else {}
        is_kfold = params.get("validation_scheme", "") in (
            "kfold",
            "stratified_kfold",
            "group_kfold",
        )
        if not is_kfold:
            return CheckResult("VR-004", "PASS", "N/A (не k-fold)")
        fold_metrics = [k for k in metrics if k.startswith(f"{metric_name}_fold_")]
        if not fold_metrics:
            return CheckResult(
                "VR-004",
                "WARN",
                f"run={run.info.run_id}: fold-метрики {metric_name}_fold_N не найдены",
                f"Залогируй mlflow.log_metric('{metric_name}_fold_0', val) для каждого фолда",
            )
        return CheckResult("VR-004", "PASS", f"fold-метрики найдены: {fold_metrics}")

    def _check_vr005(self, run: Any, metric_name: str) -> CheckResult:
        """VR-005: для k-fold залогирован mean ± std."""
        params = run.data.params if hasattr(run, "data") else {}
        metrics = run.data.metrics if hasattr(run, "data") else {}
        is_kfold = params.get("validation_scheme", "") in (
            "kfold",
            "stratified_kfold",
            "group_kfold",
        )
        if not is_kfold:
            return CheckResult("VR-005", "PASS", "N/A (не k-fold)")
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        missing = [k for k in (mean_key, std_key) if k not in metrics]
        if missing:
            return CheckResult(
                "VR-005",
                "WARN",
                f"run={run.info.run_id}: отсутствуют метрики: {missing}",
                f"Залогируй {mean_key} и {std_key} после k-fold цикла",
            )
        return CheckResult(
            "VR-005",
            "PASS",
            f"{mean_key}={metrics[mean_key]:.4f} ± {metrics[std_key]:.4f}",
        )

    def _check_vr006(self, run: Any, metric_name: str) -> CheckResult:
        """VR-006: test metric залогирована отдельно (не смешана с val)."""
        metrics = run.data.metrics if hasattr(run, "data") else {}
        val_key = f"{metric_name}_val"
        test_key = f"{metric_name}_test"
        if test_key in metrics and val_key not in metrics and metric_name in metrics:
            return CheckResult(
                "VR-006",
                "WARN",
                f"run={run.info.run_id}: найден {test_key} без {val_key} — возможное смешение",
                f"Используй '{metric_name}_val' для val и '{metric_name}_test' только для финального run",
            )
        return CheckResult("VR-006", "PASS", "val/test метрики разделены корректно")

    def _check_vr007(self, run: Any) -> CheckResult:
        """VR-007: n_samples_train и n_samples_val залогированы."""
        params = run.data.params if hasattr(run, "data") else {}
        missing = [k for k in ("n_samples_train", "n_samples_val") if k not in params]
        # n_train / n_val тоже принимаем как альтернативные имена
        if missing:
            alt_missing = [k for k in ("n_train", "n_val") if k not in params]
            if alt_missing:
                return CheckResult(
                    "VR-007",
                    "WARN",
                    f"run={run.info.run_id}: отсутствуют params: {missing}",
                    "Добавь mlflow.log_param('n_samples_train', len(X_train))",
                )
        return CheckResult("VR-007", "PASS", "n_samples_train/val залогированы")

    def _check_vr_fe001(self, run: Any) -> CheckResult:
        """VR-FE-001: target encoding leakage в shadow experiments.

        Ищем тег 'target_enc_fit_on_val' который Claude Code должен выставить
        при обнаружении нарушения.
        """
        tags = run.data.tags if hasattr(run, "data") else {}
        if tags.get("target_enc_fit_on_val") == "true":
            return CheckResult(
                "VR-FE-001",
                "WARN",
                f"run={run.info.run_id}: target encoding fit на val (leakage)",
                "Target encoder должен быть fit только на train. "
                "Этот run помечен как suspect_leakage.",
            )
        return CheckResult("VR-FE-001", "PASS", "target encoding leakage не обнаружен")

    # === Вспомогательные методы ===

    def _has_document_id(self, df: pd.DataFrame) -> bool:
        """Проверяет наличие document_id / source_id колонок.

        Args:
            df: датасет.

        Returns:
            True если документная колонка найдена.
        """
        doc_cols = {"document_id", "doc_id", "source_id", "article_id", "text_id"}
        return bool(doc_cols & set(col.lower() for col in df.columns))
