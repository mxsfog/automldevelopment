"""Adversarial Validation — LightGBM для обнаружения distribution shift."""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AdversarialStatus = Literal["passed", "warning", "critical"]

_AUC_PASSED_THRESHOLD = 0.6
_AUC_WARNING_THRESHOLD = 0.85
_N_TREES = 100


@dataclass
class AdversarialValidationResult:
    """Результат adversarial validation.

    Атрибуты:
        auc: ROC-AUC классификатора train-vs-val.
        status: итоговый статус (passed/warning/critical).
        important_features: топ признаков отличающих train от val.
        n_train: размер train sample.
        n_val: размер val sample.
        message: текстовое описание результата.
        blocks_session: critical блокирует сессию (требует подтверждения).
        hints_for_claude: подсказки для Claude Code.
    """

    auc: float = 0.0
    status: AdversarialStatus = "passed"
    important_features: list[str] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    message: str = ""
    blocks_session: bool = False
    hints_for_claude: list[str] = field(default_factory=list)


class AdversarialValidator:
    """Запускает adversarial validation: LightGBM для разделения train/val.

    Поддерживается только для tabular и NLP задач.
    CV (image_dir) не поддерживается.

    ROC-AUC пороги:
    - passed: < 0.6 (train и val неразличимы)
    - warning: 0.6-0.85 (умеренный distribution shift)
    - critical: >= 0.85 (сильный shift, блокирует сессию)

    Атрибуты:
        train_df: тренировочный DataFrame.
        val_df: валидационный DataFrame.
        target_col: имя целевой переменной (исключается из признаков).
        id_col: колонка-идентификатор (исключается из признаков).
        n_trees: количество деревьев LightGBM (дефолт 100).
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        id_col: str | None = None,
        n_trees: int = _N_TREES,
    ) -> None:
        self.train_df = train_df
        self.val_df = val_df
        self.target_col = target_col
        self.id_col = id_col
        self.n_trees = n_trees

    def run(self) -> AdversarialValidationResult:
        """Запускает adversarial validation.

        Returns:
            AdversarialValidationResult с AUC и статусом.
        """
        logger.info("Запуск Adversarial Validation (LightGBM, n_trees=%d)", self.n_trees)

        exclude_cols = {self.target_col}
        if self.id_col:
            exclude_cols.add(self.id_col)

        feature_cols = [
            c for c in self.train_df.columns if c not in exclude_cols and c in self.val_df.columns
        ]

        if not feature_cols:
            return AdversarialValidationResult(
                message="Нет общих признаков для adversarial validation",
                status="passed",
            )

        train_sample = self.train_df[feature_cols].copy()
        val_sample = self.val_df[feature_cols].copy()

        # Ограничиваем sample размер для скорости
        max_rows = 50_000
        if len(train_sample) > max_rows:
            train_sample = train_sample.sample(max_rows, random_state=42)
        if len(val_sample) > max_rows:
            val_sample = val_sample.sample(max_rows, random_state=42)

        # Собираем adversarial датасет: 0=train, 1=val
        features_df = pd.concat([train_sample, val_sample], ignore_index=True)
        labels = np.array([0] * len(train_sample) + [1] * len(val_sample))

        # Конвертируем object колонки в числовые через label encoding
        features_df = _encode_categoricals(features_df)

        auc = self._train_and_evaluate(features_df, labels)
        feature_importances = self._get_feature_importances(features_df, labels, feature_cols)

        status = self._classify_auc(auc)
        message = self._build_message(auc, status)
        blocks = status == "critical"

        hints = []
        if status != "passed":
            hints.append(
                f"Distribution shift обнаружен (AUC={auc:.3f}). "
                f"Ключевые отличающие признаки: {feature_importances[:3]}. "
                f"Рассмотри стратифицированную валидацию или коррекцию распределения."
            )

        logger.info("Adversarial Validation: AUC=%.3f, status=%s", auc, status)
        return AdversarialValidationResult(
            auc=round(auc, 4),
            status=status,
            important_features=feature_importances[:10],
            n_train=len(train_sample),
            n_val=len(val_sample),
            message=message,
            blocks_session=blocks,
            hints_for_claude=hints,
        )

    def _train_and_evaluate(self, features_df: pd.DataFrame, labels: np.ndarray) -> float:
        """Обучает LightGBM и возвращает ROC-AUC на кросс-валидации.

        Args:
            features_df: матрица признаков.
            labels: бинарные метки (0=train, 1=val).

        Returns:
            ROC-AUC на 5-fold стратифицированной CV.
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for fold_train_idx, fold_val_idx in skf.split(features_df, labels):
            feat_tr = features_df.iloc[fold_train_idx]
            feat_vl = features_df.iloc[fold_val_idx]
            lbl_tr, lbl_vl = labels[fold_train_idx], labels[fold_val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=self.n_trees,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(feat_tr, lbl_tr)
            preds = model.predict_proba(feat_vl)[:, 1]
            fold_auc = roc_auc_score(lbl_vl, preds)
            aucs.append(fold_auc)

        return float(np.mean(aucs))

    def _get_feature_importances(
        self, features_df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]
    ) -> list[str]:
        """Обучает финальную модель и возвращает топ признаков.

        Args:
            features_df: матрица признаков.
            labels: бинарные метки.
            feature_cols: имена признаков.

        Returns:
            Список имён признаков по убыванию важности.
        """
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=self.n_trees,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(features_df, labels)
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            # Возвращаем оригинальные имена (features_df мог переименовать)
            df_cols = list(features_df.columns)
            return [df_cols[i] if i < len(df_cols) else f"feat_{i}" for i in sorted_idx]
        except Exception as exc:
            logger.warning("Ошибка вычисления feature importances: %s", exc)
            return []

    @staticmethod
    def _classify_auc(auc: float) -> AdversarialStatus:
        """Классифицирует AUC по порогам.

        Args:
            auc: значение ROC-AUC.

        Returns:
            Статус adversarial validation.
        """
        if auc < _AUC_PASSED_THRESHOLD:
            return "passed"
        elif auc < _AUC_WARNING_THRESHOLD:
            return "warning"
        return "critical"

    @staticmethod
    def _build_message(auc: float, status: AdversarialStatus) -> str:
        """Формирует текстовое описание результата.

        Args:
            auc: значение ROC-AUC.
            status: статус.

        Returns:
            Описание для отчёта.
        """
        if status == "passed":
            return f"Distribution shift не обнаружен (AUC={auc:.3f} < {_AUC_PASSED_THRESHOLD})"
        elif status == "warning":
            return (
                f"Умеренный distribution shift (AUC={auc:.3f}, "
                f"порог warning={_AUC_PASSED_THRESHOLD}-{_AUC_WARNING_THRESHOLD})"
            )
        return (
            f"Критический distribution shift (AUC={auc:.3f} >= {_AUC_WARNING_THRESHOLD}) — "
            f"требует проверки перед запуском сессии"
        )


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode все object/category колонки.

    Args:
        df: DataFrame с возможными категориальными колонками.

    Returns:
        DataFrame с числовыми колонками.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    # Заполняем NaN нулями для LightGBM
    df = df.fillna(0)
    return df
