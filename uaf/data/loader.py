"""DataLoader — загрузка метаданных и sample из входных датасетов UAF."""

import hashlib
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SAMPLE_LIMIT_BYTES = 100 * 1024 * 1024  # 100 МБ
_LOAD_TIMEOUT_SECONDS = 60
_SAMPLE_ROWS = 10_000  # строк для статистик и аудита

SupportedFormat = Literal["csv", "parquet", "sql_dump", "jsonl", "txt", "image_dir"]


def _detect_format(path: Path) -> SupportedFormat:
    """Определяет формат файла по расширению или структуре директории.

    Args:
        path: путь к файлу или директории.

    Returns:
        Тип формата.

    Raises:
        ValueError: формат не поддерживается.
    """
    if path.is_dir():
        return "image_dir"
    suffix = path.suffix.lower()
    fmt_map = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".sql": "sql_dump",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".txt": "txt",
    }
    if suffix in fmt_map:
        return fmt_map[suffix]  # type: ignore[return-value]
    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


@dataclass
class SplitInfo:
    """Метаданные одного сплита датасета.

    Атрибуты:
        name: имя сплита (train/val/test).
        path: путь к файлу.
        fmt: формат файла.
        n_rows: количество строк (если определено).
        n_cols: количество колонок (если определено).
        size_bytes: размер файла в байтах.
        columns: список колонок.
        dtypes: словарь {колонка: тип} для tabular данных.
        sample_loaded: был ли загружен sample для анализа.
    """

    name: str
    path: str
    fmt: str
    n_rows: int | None = None
    n_cols: int | None = None
    size_bytes: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    sample_loaded: bool = False


@dataclass
class FeatureInfo:
    """Информация об одном признаке.

    Атрибуты:
        name: имя признака.
        dtype: тип данных.
        null_fraction: доля пропущенных значений (0.0-1.0).
        n_unique: количество уникальных значений.
        is_constant: признак постоянный.
        is_id_like: похоже на идентификатор (высокая кардинальность).
        selected_baseline: вошёл ли в baseline модель (заполняется позже).
    """

    name: str
    dtype: str
    null_fraction: float = 0.0
    n_unique: int = 0
    is_constant: bool = False
    is_id_like: bool = False
    selected_baseline: bool = True


@dataclass
class TargetInfo:
    """Информация о целевой переменной.

    Атрибуты:
        column: имя колонки.
        dtype: тип данных.
        n_unique: количество уникальных значений.
        class_balance: словарь {класс: доля} для классификации.
        null_fraction: доля пропусков.
        task_type_hint: подсказка типа задачи.
    """

    column: str
    dtype: str = ""
    n_unique: int = 0
    class_balance: dict[str, float] = field(default_factory=dict)
    null_fraction: float = 0.0
    task_type_hint: str = "unknown"


@dataclass
class QualityReport:
    """Отчёт о качестве данных.

    Атрибуты:
        total_rows: всего строк (по train).
        total_cols: всего колонок.
        null_cols: колонки с > 0 пропусков.
        constant_cols: постоянные колонки.
        high_null_cols: колонки с > 50% пропусков.
        duplicate_rows_fraction: доля дублей.
        schema_hash: хэш схемы данных.
    """

    total_rows: int = 0
    total_cols: int = 0
    null_cols: list[str] = field(default_factory=list)
    constant_cols: list[str] = field(default_factory=list)
    high_null_cols: list[str] = field(default_factory=list)
    duplicate_rows_fraction: float = 0.0
    schema_hash: str = ""


@dataclass
class TaskHints:
    """Подсказки для Claude Code о типе задачи и данных.

    Атрибуты:
        task_type: тип задачи (binary_classification, multiclass, regression, etc).
        has_datetime: есть ли временные признаки.
        has_text: есть ли текстовые признаки.
        has_high_cardinality: есть ли высококардинальные категориальные.
        class_imbalance: есть ли дисбаланс классов.
        recommended_baseline: рекомендуемая baseline модель.
        data_warnings: предупреждения для учёта при моделировании.
    """

    task_type: str = "unknown"
    has_datetime: bool = False
    has_text: bool = False
    has_high_cardinality: bool = False
    class_imbalance: bool = False
    recommended_baseline: str = "logistic_regression"
    data_warnings: list[str] = field(default_factory=list)


@dataclass
class DataSchema:
    """Полная схема данных — результат работы DataLoader.

    Атрибуты:
        splits: информация по сплитам.
        target: информация о целевой переменной.
        features: список признаков.
        quality: отчёт о качестве.
        leakage_audit: результаты аудита утечек (заполняется LeakageAudit).
        adversarial_validation: результат адверсарной валидации.
        task_hints: подсказки для моделирования.
        generated_at: время генерации схемы.
    """

    splits: list[SplitInfo] = field(default_factory=list)
    target: TargetInfo | None = None
    features: list[FeatureInfo] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)
    leakage_audit: dict[str, Any] = field(default_factory=dict)
    adversarial_validation: dict[str, Any] = field(default_factory=dict)
    task_hints: TaskHints = field(default_factory=TaskHints)
    generated_at: str = ""


class _TimeoutError(Exception):
    """Внутреннее исключение таймаута загрузки."""


class DataLoader:
    """Загружает метаданные и sample из датасета с ограничениями.

    Ограничения:
    - Максимум 100 МБ RAM (sample только)
    - Timeout 60 секунд на загрузку
    - Только чтение (antigoal 4)

    Поддерживаемые форматы:
    - Tabular: CSV, Parquet, SQL Dump
    - NLP: JSONL, TXT
    - CV: image_dir

    Атрибуты:
        train_path: путь к тренировочным данным.
        target_column: имя целевой переменной.
        val_path: путь к валидационным данным (опционально).
        test_path: путь к тестовым данным (опционально).
        id_column: имя колонки-идентификатора (опционально).
        fmt: формат данных (автоопределение если None).
    """

    def __init__(
        self,
        train_path: Path,
        target_column: str,
        val_path: Path | None = None,
        test_path: Path | None = None,
        id_column: str | None = None,
        fmt: SupportedFormat | None = None,
    ) -> None:
        self.train_path = train_path
        self.target_column = target_column
        self.val_path = val_path
        self.test_path = test_path
        self.id_column = id_column
        self.fmt: SupportedFormat = fmt or _detect_format(train_path)

    def load(self) -> DataSchema:
        """Загружает схему данных с применением timeout.

        Returns:
            DataSchema с заполненными splits, target, features, quality, task_hints.

        Raises:
            TimeoutError: загрузка превысила 60 секунд.
        """
        logger.info("Загрузка схемы данных: %s (format=%s)", self.train_path, self.fmt)
        start = time.time()

        def _timeout_handler(signum: int, frame: Any) -> None:
            raise _TimeoutError(f"DataLoader timeout после {_LOAD_TIMEOUT_SECONDS} сек")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_LOAD_TIMEOUT_SECONDS)
        try:
            schema = self._load_internal()
        except _TimeoutError as exc:
            logger.error("DataLoader timeout (%s сек)", _LOAD_TIMEOUT_SECONDS)
            raise TimeoutError(f"Загрузка данных превысила {_LOAD_TIMEOUT_SECONDS} сек") from exc
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        elapsed = time.time() - start
        logger.info("Схема данных загружена за %.1f сек", elapsed)
        return schema

    def _load_internal(self) -> DataSchema:
        """Внутренняя реализация загрузки схемы.

        Returns:
            Заполненная DataSchema.
        """
        import datetime

        schema = DataSchema(generated_at=datetime.datetime.now().isoformat())

        if self.fmt in ("csv", "parquet", "sql_dump"):
            self._load_tabular(schema)
        elif self.fmt == "jsonl":
            self._load_jsonl(schema)
        elif self.fmt == "txt":
            self._load_txt(schema)
        elif self.fmt == "image_dir":
            self._load_image_dir(schema)

        return schema

    def _load_tabular(self, schema: DataSchema) -> None:
        """Загружает tabular данные (CSV/Parquet/SQL Dump).

        Args:
            schema: схема для заполнения.
        """
        train_df = self._read_sample(self.train_path, self.fmt)

        train_split = self._build_split_info("train", self.train_path, self.fmt, train_df)
        schema.splits.append(train_split)

        if self.val_path and self.val_path.exists():
            val_df = self._read_sample(self.val_path, self.fmt)
            schema.splits.append(self._build_split_info("val", self.val_path, self.fmt, val_df))
        else:
            val_df = None

        if self.test_path and self.test_path.exists():
            test_df = self._read_sample(self.test_path, self.fmt)
            schema.splits.append(self._build_split_info("test", self.test_path, self.fmt, test_df))

        # Target
        schema.target = self._analyze_target(train_df)

        # Features
        schema.features = self._analyze_features(train_df)

        # Quality
        schema.quality = self._compute_quality(train_df)

        # Task hints
        schema.task_hints = self._infer_task_hints(schema, train_df)

    def _read_sample(self, path: Path, fmt: SupportedFormat) -> pd.DataFrame:
        """Читает sample данных с ограничением по размеру.

        Args:
            path: путь к файлу.
            fmt: формат файла.

        Returns:
            DataFrame с sample данных.
        """
        file_size = path.stat().st_size
        if file_size <= _SAMPLE_LIMIT_BYTES:
            # Файл помещается целиком
            return self._read_full(path, fmt)
        else:
            # Читаем только первые _SAMPLE_ROWS строк
            logger.info(
                "Файл %.0f МБ > лимит %.0f МБ, загружаем sample %d строк",
                file_size / 1e6,
                _SAMPLE_LIMIT_BYTES / 1e6,
                _SAMPLE_ROWS,
            )
            return self._read_full(path, fmt, nrows=_SAMPLE_ROWS)

    def _read_full(
        self, path: Path, fmt: SupportedFormat, nrows: int | None = None
    ) -> pd.DataFrame:
        """Читает файл полностью или первые nrows строк.

        Args:
            path: путь к файлу.
            fmt: формат.
            nrows: лимит строк (None = всё).

        Returns:
            DataFrame.
        """
        if fmt == "csv":
            return pd.read_csv(path, nrows=nrows)
        elif fmt == "parquet":
            df = pd.read_parquet(path)
            return df.head(nrows) if nrows else df
        elif fmt == "sql_dump":
            return self._read_sql_dump(path, nrows)
        raise ValueError(f"Неожиданный tabular формат: {fmt}")

    def _read_sql_dump(self, path: Path, nrows: int | None = None) -> pd.DataFrame:
        """Читает SQL dump: извлекает INSERT VALUES в DataFrame.

        Поддерживает только простые INSERT INTO ... VALUES ... дампы.

        Args:
            path: путь к .sql файлу.
            nrows: лимит строк.

        Returns:
            DataFrame с данными из INSERT statements.
        """
        import re

        rows = []
        columns: list[str] = []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not columns:
                    col_match = re.search(
                        r"CREATE TABLE\s+\S+\s*\((.*?)\)", line, re.IGNORECASE | re.DOTALL
                    )
                    if col_match:
                        col_defs = col_match.group(1).split(",")
                        for col_def in col_defs:
                            col_def = col_def.strip()
                            if col_def and not col_def.upper().startswith(
                                ("PRIMARY", "KEY", "UNIQUE", "INDEX", "CONSTRAINT")
                            ):
                                col_name = col_def.split()[0].strip("`\"'")
                                columns.append(col_name)
                if line.upper().startswith("INSERT INTO"):
                    values_match = re.search(r"VALUES\s*\((.*)\)", line, re.IGNORECASE)
                    if values_match:
                        raw = values_match.group(1)
                        vals = [v.strip().strip("'\"") for v in raw.split(",")]
                        rows.append(vals)
                        if nrows and len(rows) >= nrows:
                            break
        if not rows:
            logger.warning("SQL dump: INSERT VALUES не найдены в %s", path)
            return pd.DataFrame()
        if not columns:
            columns = [f"col_{i}" for i in range(len(rows[0]))]
        return pd.DataFrame(rows, columns=columns[: len(rows[0])])

    def _build_split_info(
        self, name: str, path: Path, fmt: SupportedFormat, df: pd.DataFrame
    ) -> SplitInfo:
        """Строит SplitInfo из загруженного DataFrame.

        Args:
            name: имя сплита.
            path: путь к файлу.
            fmt: формат файла.
            df: загруженный sample.

        Returns:
            Заполненный SplitInfo.
        """
        size = path.stat().st_size if path.is_file() else _dir_size(path)
        # Реальное число строк: для CSV пересчитываем по размеру если sample обрезан
        n_rows: int | None = len(df)
        if size > _SAMPLE_LIMIT_BYTES and fmt == "csv":
            try:
                n_rows = sum(1 for _ in path.open()) - 1  # минус header
            except Exception:
                n_rows = None

        return SplitInfo(
            name=name,
            path=str(path),
            fmt=fmt,
            n_rows=n_rows,
            n_cols=len(df.columns),
            size_bytes=size,
            columns=list(df.columns),
            dtypes={col: str(df[col].dtype) for col in df.columns},
            sample_loaded=True,
        )

    def _analyze_target(self, df: pd.DataFrame) -> TargetInfo:
        """Анализирует целевую переменную.

        Args:
            df: тренировочный DataFrame.

        Returns:
            TargetInfo с характеристиками таргета.
        """
        if self.target_column not in df.columns:
            logger.warning("Целевая колонка '%s' не найдена в данных", self.target_column)
            return TargetInfo(column=self.target_column)

        col = df[self.target_column]
        n_unique = int(col.nunique())
        null_frac = float(col.isna().mean())
        dtype = str(col.dtype)

        # Определение типа задачи
        if n_unique == 2:
            task_hint = "binary_classification"
        elif (
            (n_unique <= 20
            and col.dtype in (object, "category"))
            or (n_unique <= 20 and n_unique > 2)
        ):
            task_hint = "multiclass_classification"
        elif col.dtype in (np.float64, np.float32, "float64", "float32") or (
            col.dtype in (np.int64, np.int32, "int64", "int32") and n_unique > 20
        ):
            task_hint = "regression"
        else:
            task_hint = "classification"

        class_balance: dict[str, float] = {}
        if task_hint in ("binary_classification", "multiclass_classification", "classification"):
            vc = col.dropna().value_counts(normalize=True)
            class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}

        return TargetInfo(
            column=self.target_column,
            dtype=dtype,
            n_unique=n_unique,
            class_balance=class_balance,
            null_fraction=null_frac,
            task_type_hint=task_hint,
        )

    def _analyze_features(self, df: pd.DataFrame) -> list[FeatureInfo]:
        """Анализирует все признаки (кроме целевой и id колонки).

        Args:
            df: тренировочный DataFrame.

        Returns:
            Список FeatureInfo для каждого признака.
        """
        exclude = {self.target_column}
        if self.id_column:
            exclude.add(self.id_column)

        features = []
        for col in df.columns:
            if col in exclude:
                continue
            series = df[col]
            n_unique = int(series.nunique())
            null_frac = float(series.isna().mean())
            is_constant = n_unique <= 1
            # ID-like: уникальность > 95% и много строк
            is_id_like = n_unique / max(len(df), 1) > 0.95 and len(df) > 100

            features.append(
                FeatureInfo(
                    name=col,
                    dtype=str(series.dtype),
                    null_fraction=round(null_frac, 4),
                    n_unique=n_unique,
                    is_constant=is_constant,
                    is_id_like=is_id_like,
                )
            )
        return features

    def _compute_quality(self, df: pd.DataFrame) -> QualityReport:
        """Вычисляет метрики качества данных.

        Args:
            df: тренировочный DataFrame.

        Returns:
            QualityReport с показателями качества.
        """
        null_cols = [col for col in df.columns if df[col].isna().any()]
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        high_null_cols = [col for col in df.columns if df[col].isna().mean() > 0.5]
        dup_frac = float(df.duplicated().mean())

        # Схема хэш: колонки + типы
        schema_str = "|".join(f"{c}:{df[c].dtype}" for c in df.columns)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()

        return QualityReport(
            total_rows=len(df),
            total_cols=len(df.columns),
            null_cols=null_cols,
            constant_cols=constant_cols,
            high_null_cols=high_null_cols,
            duplicate_rows_fraction=round(dup_frac, 4),
            schema_hash=schema_hash,
        )

    def _infer_task_hints(self, schema: DataSchema, df: pd.DataFrame) -> TaskHints:
        """Выводит подсказки для моделирования на основе анализа данных.

        Args:
            schema: текущая схема.
            df: тренировочный DataFrame.

        Returns:
            TaskHints с рекомендациями.
        """
        task_type = schema.target.task_type_hint if schema.target else "unknown"

        has_datetime = any(
            str(f.dtype).startswith("datetime")
            or "date" in f.name.lower()
            or "time" in f.name.lower()
            for f in schema.features
        )
        has_text = any(
            str(f.dtype) == "object" and df[f.name].dropna().str.split().str.len().median() > 5
            for f in schema.features
            if f.name in df.columns and str(f.dtype) == "object"
        )
        has_high_cardinality = any(
            f.n_unique > 100 and str(f.dtype) in ("object", "category") for f in schema.features
        )

        class_imbalance = False
        if schema.target and schema.target.class_balance:
            min_balance = min(schema.target.class_balance.values())
            class_imbalance = min_balance < 0.1

        # Рекомендация baseline
        if task_type == "regression":
            recommended = "ridge_regression"
        elif task_type == "binary_classification":
            recommended = "logistic_regression"
        else:
            recommended = "logistic_regression"

        warnings = []
        if class_imbalance:
            warnings.append("Дисбаланс классов — рассмотри class_weight='balanced' или SMOTE")
        if schema.quality.high_null_cols:
            warnings.append(f"Высокий процент пропусков в: {schema.quality.high_null_cols[:5]}")
        if schema.quality.constant_cols:
            warnings.append(
                f"Постоянные колонки (можно удалить): {schema.quality.constant_cols[:5]}"
            )

        return TaskHints(
            task_type=task_type,
            has_datetime=has_datetime,
            has_text=has_text,
            has_high_cardinality=has_high_cardinality,
            class_imbalance=class_imbalance,
            recommended_baseline=recommended,
            data_warnings=warnings,
        )

    def _load_jsonl(self, schema: DataSchema) -> None:
        """Загружает JSONL файл (NLP задачи).

        Args:
            schema: схема для заполнения.
        """
        size = self.train_path.stat().st_size
        rows_read = 0
        col_names: set[str] = set()
        with self.train_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    col_names.update(record.keys())
                    rows_read += 1
                except json.JSONDecodeError:
                    continue
                if rows_read >= _SAMPLE_ROWS:
                    break

        schema.splits.append(
            SplitInfo(
                name="train",
                path=str(self.train_path),
                fmt="jsonl",
                n_rows=rows_read,
                n_cols=len(col_names),
                size_bytes=size,
                columns=sorted(col_names),
                sample_loaded=True,
            )
        )
        schema.target = TargetInfo(column=self.target_column, task_type_hint="text_classification")
        schema.task_hints = TaskHints(
            task_type="text_classification",
            has_text=True,
            recommended_baseline="tf_idf_logreg",
        )

    def _load_txt(self, schema: DataSchema) -> None:
        """Загружает TXT файл (NLP).

        Args:
            schema: схема для заполнения.
        """
        size = self.train_path.stat().st_size
        n_lines = 0
        with self.train_path.open(encoding="utf-8", errors="replace") as fh:
            for _ in fh:
                n_lines += 1
                if n_lines >= _SAMPLE_ROWS:
                    break

        schema.splits.append(
            SplitInfo(
                name="train",
                path=str(self.train_path),
                fmt="txt",
                n_rows=n_lines,
                n_cols=1,
                size_bytes=size,
                columns=["text"],
                sample_loaded=True,
            )
        )
        schema.target = TargetInfo(column=self.target_column, task_type_hint="nlp")
        schema.task_hints = TaskHints(
            task_type="nlp",
            has_text=True,
            recommended_baseline="tf_idf_logreg",
        )

    def _load_image_dir(self, schema: DataSchema) -> None:
        """Загружает метаданные директории с изображениями (CV).

        Args:
            schema: схема для заполнения.
        """
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        class_dirs: dict[str, int] = {}
        total_size = 0
        total_files = 0

        for child in self.train_path.iterdir():
            if child.is_dir():
                imgs = [f for f in child.iterdir() if f.suffix.lower() in img_extensions]
                class_dirs[child.name] = len(imgs)
                total_files += len(imgs)
                total_size += sum(f.stat().st_size for f in imgs)
            elif child.suffix.lower() in img_extensions:
                class_dirs["root"] = class_dirs.get("root", 0) + 1
                total_files += 1
                total_size += child.stat().st_size

        schema.splits.append(
            SplitInfo(
                name="train",
                path=str(self.train_path),
                fmt="image_dir",
                n_rows=total_files,
                n_cols=1,
                size_bytes=total_size,
                columns=["image_path"],
                sample_loaded=False,
            )
        )
        schema.target = TargetInfo(
            column=self.target_column,
            n_unique=len(class_dirs),
            task_type_hint="image_classification" if len(class_dirs) > 1 else "image",
        )
        n_classes = len(class_dirs)
        schema.task_hints = TaskHints(
            task_type="image_classification" if n_classes > 1 else "image",
            recommended_baseline="resnet18_pretrained" if n_classes > 1 else "cnn",
        )


def save_data_schema(schema: DataSchema, path: Path) -> None:
    """Сохраняет DataSchema в data_schema.json.

    Args:
        schema: схема данных.
        path: путь к файлу.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    data = _to_dict(schema)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("data_schema.json записан: %s", path)


def load_data_schema(path: Path) -> DataSchema:
    """Читает DataSchema из data_schema.json.

    Args:
        path: путь к файлу.

    Returns:
        Восстановленная DataSchema.
    """
    raw = json.loads(path.read_text())
    # Простая реконструкция — используем только для чтения в других компонентах
    schema = DataSchema(generated_at=raw.get("generated_at", ""))
    for sp in raw.get("splits", []):
        schema.splits.append(
            SplitInfo(**{k: v for k, v in sp.items() if k in SplitInfo.__dataclass_fields__})
        )
    if raw.get("target"):
        schema.target = TargetInfo(
            **{k: v for k, v in raw["target"].items() if k in TargetInfo.__dataclass_fields__}
        )
    schema.quality = QualityReport(
        **{
            k: v
            for k, v in raw.get("quality", {}).items()
            if k in QualityReport.__dataclass_fields__
        }
    )
    schema.task_hints = TaskHints(
        **{
            k: v
            for k, v in raw.get("task_hints", {}).items()
            if k in TaskHints.__dataclass_fields__
        }
    )
    schema.leakage_audit = raw.get("leakage_audit", {})
    schema.adversarial_validation = raw.get("adversarial_validation", {})
    return schema


def _dir_size(path: Path) -> int:
    """Размер директории рекурсивно в байтах.

    Args:
        path: путь к директории.

    Returns:
        Суммарный размер в байтах.
    """
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total
