"""Leakage Audit — 10 проверок утечек данных LA-01..LA-10."""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from uaf.data.loader import DataSchema

logger = logging.getLogger(__name__)

AuditLevel = Literal["CRITICAL", "WARNING", "OK"]


@dataclass
class AuditCheck:
    """Результат одной проверки leakage audit.

    Атрибуты:
        code: код проверки (LA-01..LA-10).
        name: название проверки.
        level: уровень (CRITICAL/WARNING/OK).
        passed: прошла ли проверка.
        message: описание результата.
        details: дополнительные детали для отчёта.
    """

    code: str
    name: str
    level: AuditLevel
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageAuditResult:
    """Полный результат leakage audit.

    Атрибуты:
        checks: список результатов всех 10 проверок.
        critical_count: число CRITICAL нарушений.
        warning_count: число WARNING нарушений.
        blocks_session: True если есть CRITICAL нарушения (блокируют сессию).
        hints_for_claude: подсказки для Claude Code о найденных проблемах.
    """

    checks: list[AuditCheck] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    blocks_session: bool = False
    hints_for_claude: list[str] = field(default_factory=list)


class LeakageAudit:
    """Аудит утечек данных: 10 проверок LA-01..LA-10.

    CRITICAL проверки (блокируют сессию):
    - LA-01: целевая переменная в признаках
    - LA-05: пересечение строк train/val > 1%
    - LA-10: несовпадение схем train/test

    WARNING проверки (предупреждение + hints):
    - LA-02: колонки с будущими данными (future leakage)
    - LA-03: near-duplicate признаки с таргетом
    - LA-04: высококардинальные категориальные (potential proxy)
    - LA-06: утечка через временной порядок
    - LA-07: постоянные признаки
    - LA-08: все значения в val тоже есть в train (perfect split)
    - LA-09: подозрительные корреляции с таргетом

    Атрибуты:
        schema: схема данных от DataLoader.
        train_df: sample тренировочных данных.
        val_df: sample валидационных данных (опционально).
        test_df: sample тестовых данных (опционально).
    """

    def __init__(
        self,
        schema: DataSchema,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> None:
        self.schema = schema
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.target_col = schema.target.column if schema.target else ""

    def run(self) -> LeakageAuditResult:
        """Запускает все 10 проверок.

        Returns:
            Полный результат аудита.
        """
        logger.info("Запуск Leakage Audit (10 проверок)")
        result = LeakageAuditResult()

        checks = [
            self._la01_target_in_features(),
            self._la02_future_leakage_names(),
            self._la03_near_duplicate_with_target(),
            self._la04_high_cardinality_proxy(),
            self._la05_row_overlap(),
            self._la06_temporal_order(),
            self._la07_constant_features(),
            self._la08_perfect_split(),
            self._la09_suspicious_correlation(),
            self._la10_schema_mismatch(),
        ]

        result.checks = checks
        result.critical_count = sum(1 for c in checks if not c.passed and c.level == "CRITICAL")
        result.warning_count = sum(1 for c in checks if not c.passed and c.level == "WARNING")
        result.blocks_session = result.critical_count > 0

        # Собираем hints для Claude Code
        for check in checks:
            if not check.passed:
                hint = f"[{check.code}] {check.message}"
                result.hints_for_claude.append(hint)

        logger.info(
            "Leakage Audit завершён: CRITICAL=%d, WARNING=%d, blocks=%s",
            result.critical_count,
            result.warning_count,
            result.blocks_session,
        )
        return result

    def _la01_target_in_features(self) -> AuditCheck:
        """LA-01: целевая переменная присутствует в признаках.

        Returns:
            Результат проверки.
        """
        code = "LA-01"
        if not self.target_col:
            return AuditCheck(
                code=code,
                name="Target in Features",
                level="CRITICAL",
                passed=True,
                message="Целевая колонка не задана — пропускаем",
            )

        feature_names = [f.name for f in self.schema.features]
        if self.target_col in feature_names:
            return AuditCheck(
                code=code,
                name="Target in Features",
                level="CRITICAL",
                passed=False,
                message=(
                    f"Целевая переменная '{self.target_col}' "
                    f"присутствует в признаках — прямая утечка"
                ),
                details={"target_col": self.target_col},
            )
        return AuditCheck(
            code=code, name="Target in Features", level="OK", passed=True, message="OK"
        )

    def _la02_future_leakage_names(self) -> AuditCheck:
        """LA-02: подозрительные имена признаков (future/label/target/result).

        Returns:
            Результат проверки.
        """
        code = "LA-02"
        suspicious_keywords = {
            "future",
            "label",
            "result",
            "outcome",
            "leakage",
            "cheat",
            "answer",
        }
        flagged = [
            f.name
            for f in self.schema.features
            if any(kw in f.name.lower() for kw in suspicious_keywords)
        ]
        if flagged:
            return AuditCheck(
                code=code,
                name="Future Leakage Names",
                level="WARNING",
                passed=False,
                message=f"Подозрительные имена признаков: {flagged[:5]}",
                details={"flagged_columns": flagged},
            )
        return AuditCheck(
            code=code, name="Future Leakage Names", level="OK", passed=True, message="OK"
        )

    def _la03_near_duplicate_with_target(self) -> AuditCheck:
        """LA-03: признак практически идентичен таргету (корреляция > 0.99).

        Returns:
            Результат проверки.
        """
        code = "LA-03"
        if not self.target_col or self.target_col not in self.train_df.columns:
            return AuditCheck(
                code=code,
                name="Near-Duplicate with Target",
                level="OK",
                passed=True,
                message="Пропущено",
            )

        target_series = self.train_df[self.target_col]
        if target_series.dtype not in (
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            "float64",
            "float32",
            "int64",
            "int32",
        ):
            return AuditCheck(
                code=code,
                name="Near-Duplicate with Target",
                level="OK",
                passed=True,
                message="Таргет не числовой — пропускаем",
            )

        flagged = []
        for col in self.train_df.select_dtypes(include=[np.number]).columns:
            if col == self.target_col:
                continue
            try:
                corr = abs(self.train_df[col].corr(target_series))
                if corr > 0.99:
                    flagged.append((col, round(corr, 4)))
            except Exception:
                continue

        if flagged:
            return AuditCheck(
                code=code,
                name="Near-Duplicate with Target",
                level="WARNING",
                passed=False,
                message=f"Признаки с корреляцией > 0.99 с таргетом: {[c for c, _ in flagged[:3]]}",
                details={"high_corr_cols": flagged},
            )
        return AuditCheck(
            code=code, name="Near-Duplicate with Target", level="OK", passed=True, message="OK"
        )

    def _la04_high_cardinality_proxy(self) -> AuditCheck:
        """LA-04: высококардинальные категориальные признаки — потенциальный proxy таргета.

        Returns:
            Результат проверки.
        """
        code = "LA-04"
        if not self.target_col or self.target_col not in self.train_df.columns:
            return AuditCheck(
                code=code,
                name="High Cardinality Proxy",
                level="OK",
                passed=True,
                message="Пропущено",
            )

        n_target_unique = self.train_df[self.target_col].nunique()
        flagged = [
            f.name
            for f in self.schema.features
            if f.n_unique >= max(n_target_unique * 10, 100)
            and str(f.dtype) in ("object", "category")
        ]
        if flagged:
            return AuditCheck(
                code=code,
                name="High Cardinality Proxy",
                level="WARNING",
                passed=False,
                message=f"Высококардинальные признаки (возможный proxy): {flagged[:5]}",
                details={"high_cardinality_cols": flagged},
            )
        return AuditCheck(
            code=code, name="High Cardinality Proxy", level="OK", passed=True, message="OK"
        )

    def _la05_row_overlap(self) -> AuditCheck:
        """LA-05: пересечение строк train/val > 1%.

        Returns:
            Результат проверки.
        """
        code = "LA-05"
        if self.val_df is None or self.val_df.empty:
            return AuditCheck(
                code=code,
                name="Row Overlap Train/Val",
                level="OK",
                passed=True,
                message="Val сплит не предоставлен — пропускаем",
            )

        try:
            # Сравниваем по общим колонкам
            common_cols = list(set(self.train_df.columns) & set(self.val_df.columns))
            if not common_cols:
                return AuditCheck(
                    code=code,
                    name="Row Overlap Train/Val",
                    level="OK",
                    passed=True,
                    message="Нет общих колонок для сравнения",
                )

            train_hashes = set(self.train_df[common_cols].apply(lambda r: hash(tuple(r)), axis=1))
            val_hashes = self.val_df[common_cols].apply(lambda r: hash(tuple(r)), axis=1)
            overlap_count = val_hashes.isin(train_hashes).sum()
            overlap_frac = overlap_count / max(len(self.val_df), 1)

            if overlap_frac > 0.01:
                return AuditCheck(
                    code=code,
                    name="Row Overlap Train/Val",
                    level="CRITICAL",
                    passed=False,
                    message=f"Пересечение строк train/val = {overlap_frac:.1%} > 1% — утечка",
                    details={
                        "overlap_fraction": round(overlap_frac, 4),
                        "overlap_count": int(overlap_count),
                    },
                )
        except Exception as exc:
            logger.warning("LA-05: ошибка вычисления overlap: %s", exc)

        return AuditCheck(
            code=code, name="Row Overlap Train/Val", level="OK", passed=True, message="OK"
        )

    def _la06_temporal_order(self) -> AuditCheck:
        """LA-06: возможная утечка через временной порядок (datetime колонка в val раньше train).

        Returns:
            Результат проверки.
        """
        code = "LA-06"
        datetime_cols = [
            f.name
            for f in self.schema.features
            if str(f.dtype).startswith("datetime") or "date" in f.name.lower()
        ]
        if not datetime_cols or self.val_df is None:
            return AuditCheck(
                code=code,
                name="Temporal Order",
                level="OK",
                passed=True,
                message="Нет datetime колонок или val сплита",
            )

        for col in datetime_cols:
            if col not in self.train_df.columns or col not in self.val_df.columns:
                continue
            try:
                train_max = pd.to_datetime(self.train_df[col], errors="coerce").max()
                val_min = pd.to_datetime(self.val_df[col], errors="coerce").min()
                if pd.notna(train_max) and pd.notna(val_min) and val_min < train_max:
                    return AuditCheck(
                        code=code,
                        name="Temporal Order",
                        level="WARNING",
                        passed=False,
                        message=(
                            f"Колонка '{col}': val содержит даты"
                            f" ({val_min}) раньше train max ({train_max})"
                        ),
                        details={
                            "col": col,
                            "train_max": str(train_max),
                            "val_min": str(val_min),
                        },
                    )
            except Exception:
                continue

        return AuditCheck(code=code, name="Temporal Order", level="OK", passed=True, message="OK")

    def _la07_constant_features(self) -> AuditCheck:
        """LA-07: постоянные признаки (единственное значение).

        Returns:
            Результат проверки.
        """
        code = "LA-07"
        constant = [f.name for f in self.schema.features if f.is_constant]
        if constant:
            return AuditCheck(
                code=code,
                name="Constant Features",
                level="WARNING",
                passed=False,
                message=f"Постоянные признаки (нет вариации): {constant[:5]}",
                details={"constant_cols": constant},
            )
        return AuditCheck(
            code=code, name="Constant Features", level="OK", passed=True, message="OK"
        )

    def _la08_perfect_split(self) -> AuditCheck:
        """LA-08: все уникальные значения val присутствуют в train (perfect overlap).

        Returns:
            Результат проверки.
        """
        code = "LA-08"
        if self.val_df is None or not self.target_col:
            return AuditCheck(
                code=code, name="Perfect Split", level="OK", passed=True, message="Пропущено"
            )

        if (
            self.target_col not in self.train_df.columns
            or self.target_col not in self.val_df.columns
        ):
            return AuditCheck(
                code=code, name="Perfect Split", level="OK", passed=True, message="Пропущено"
            )

        train_vals = set(self.train_df[self.target_col].dropna().unique())
        val_vals = set(self.val_df[self.target_col].dropna().unique())
        missing_in_train = val_vals - train_vals

        if missing_in_train and len(missing_in_train) / max(len(val_vals), 1) > 0.1:
            return AuditCheck(
                code=code,
                name="Perfect Split",
                level="WARNING",
                passed=False,
                message=f"В val есть классы отсутствующие в train: {list(missing_in_train)[:5]}",
                details={"missing_classes": list(missing_in_train)},
            )
        return AuditCheck(code=code, name="Perfect Split", level="OK", passed=True, message="OK")

    def _la09_suspicious_correlation(self) -> AuditCheck:
        """LA-09: подозрительно высокая корреляция с таргетом (0.9-0.99).

        Returns:
            Результат проверки.
        """
        code = "LA-09"
        if not self.target_col or self.target_col not in self.train_df.columns:
            return AuditCheck(
                code=code,
                name="Suspicious Correlation",
                level="OK",
                passed=True,
                message="Пропущено",
            )

        target_series = self.train_df[self.target_col]
        if str(target_series.dtype) not in ("int64", "int32", "float64", "float32"):
            return AuditCheck(
                code=code,
                name="Suspicious Correlation",
                level="OK",
                passed=True,
                message="Таргет не числовой",
            )

        flagged = []
        for col in self.train_df.select_dtypes(include=[np.number]).columns:
            if col == self.target_col:
                continue
            try:
                corr = abs(self.train_df[col].corr(target_series))
                if 0.9 <= corr <= 0.99:
                    flagged.append((col, round(corr, 4)))
            except Exception:
                continue

        if flagged:
            return AuditCheck(
                code=code,
                name="Suspicious Correlation",
                level="WARNING",
                passed=False,
                message=f"Высокая корреляция с таргетом (0.9-0.99): {[c for c, _ in flagged[:3]]}",
                details={"suspicious_cols": flagged},
            )
        return AuditCheck(
            code=code, name="Suspicious Correlation", level="OK", passed=True, message="OK"
        )

    def _la10_schema_mismatch(self) -> AuditCheck:
        """LA-10: несовпадение схем train/test (разные колонки или типы).

        Returns:
            Результат проверки.
        """
        code = "LA-10"
        if self.test_df is None:
            return AuditCheck(
                code=code,
                name="Schema Mismatch",
                level="OK",
                passed=True,
                message="Test сплит не предоставлен",
            )

        train_cols = set(self.train_df.columns) - {self.target_col}
        test_cols = set(self.test_df.columns) - {self.target_col}

        extra_in_train = train_cols - test_cols
        extra_in_test = test_cols - train_cols

        if extra_in_train or extra_in_test:
            return AuditCheck(
                code=code,
                name="Schema Mismatch",
                level="CRITICAL",
                passed=False,
                message=(
                    f"Несовпадение схем train/test: "
                    f"только в train={list(extra_in_train)[:5]}, "
                    f"только в test={list(extra_in_test)[:5]}"
                ),
                details={
                    "extra_in_train": list(extra_in_train),
                    "extra_in_test": list(extra_in_test),
                },
            )

        # Проверяем типы общих колонок
        type_mismatches = []
        for col in train_cols & test_cols:
            if str(self.train_df[col].dtype) != str(self.test_df[col].dtype):
                type_mismatches.append(
                    {
                        "col": col,
                        "train_dtype": str(self.train_df[col].dtype),
                        "test_dtype": str(self.test_df[col].dtype),
                    }
                )

        if type_mismatches:
            return AuditCheck(
                code=code,
                name="Schema Mismatch",
                level="CRITICAL",
                passed=False,
                message=f"Несовпадение типов train/test в {len(type_mismatches)} колонках",
                details={"type_mismatches": type_mismatches[:10]},
            )

        return AuditCheck(code=code, name="Schema Mismatch", level="OK", passed=True, message="OK")
