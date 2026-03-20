# Стадия 05: Data

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-problem (COMPLETE), 02-research (COMPLETE),
03-design-doc v2.0 (COMPLETE), 04-metrics (COMPLETE)

---

## 0. Позиция данных в UAF

UAF — не ML-модель. UAF не обучается на данных и не делает выводы из данных
напрямую. Данные проходят через UAF и передаются Claude Code для экспериментов.

**Ответственность UAF по отношению к данным:**

| Что UAF делает | Что UAF не делает |
|----------------|-------------------|
| Читает метаданные входных данных | Не модифицирует данные (antigoal 4) |
| Проверяет формат и доступность | Не предобрабатывает данные |
| Передаёт информацию о данных в program.md | Не обучает feature engineering |
| Версионирует данные через DVC (symlink) | Не создаёт копии данных |
| Выполняет leakage audit перед запуском | Не владеет данными |
| Включает data quality report в PDF | Не удаляет и не архивирует данные |

**Ключевой принцип:** UAF работает с данными в режиме read-only.
Единственное исключение — DVC `.dvc` файлы (метаданные), не сами данные.

---

## 1. Два пути работы с данными

UAF использует два принципиально разных пути работы с данными. Их необходимо
различать — они не пересекаются и имеют разные ограничения.

**Путь A: UAF DataLoader** — читает только sample/схему для построения
`data_schema.json`. Выполняется один раз при инициализации сессии.
Ограничение 100 МБ применяется ТОЛЬКО здесь. Результат: типы колонок,
размеры, базовые статистики, leakage audit, adversarial validation.

**Путь B: Claude Code** — читает полный датасет с диска напрямую при обучении
моделей. UAF не накладывает никакого лимита на объём данных здесь.
Claude Code загружает столько данных, сколько позволяет RAM и GPU машины.

```
task.yaml (data paths)
       |
       +---> [A] UAF DataLoader -----> data_schema.json (только метаданные)
       |           читает sample        100 МБ лимит, 60 сек timeout
       |           не весь датасет
       |
       +---> [B] Claude Code ----------> полное обучение модели
                   читает data/train.parquet целиком     нет лимита UAF
                   pd.read_parquet() / torch DataLoader
                   ограничение: RAM/GPU машины
```

Лимит 100 МБ — это ограничение DataLoader (Путь A) на чтение метаданных,
не ограничение на обучение. UAF не ограничивает размер данных для обучения.

---

## 2. DataLoader: форматы и механизм чтения

### 2.1 Поддерживаемые форматы входных данных

UAF поддерживает три формата во всех трёх типах задач:

```
CSV   (.csv)           — табличные данные, текстовые датасеты (token lists)
Parquet (.parquet)     — табличные данные, feature stores
SQL Dump (.sql)        — дамп PostgreSQL/SQLite: данные и схема
```

Дополнительные форматы для NLP и CV:

```
NLP:
  .jsonl                — строка JSON на одну запись (тексты, диалоги, QA пары)
  .txt                  — чистый текст, один документ на файл или разделённый \n\n
  HuggingFace datasets  — директория с dataset_info.json (Arrow формат внутри)

CV:
  директория изображений — структура {split}/{class}/{image.jpg}
  COCO JSON + images/    — {annotations.json, images/}
  manifest .csv          — колонки: filepath, label (или bbox_json)
```

Нестандартные форматы (NumPy .npy, PyTorch .pt, Pickle .pkl) принимаются
без автоматического парсинга метаданных: UAF запишет формат и размер файла,
но не будет пытаться прочитать содержимое для metadata schema.

### 2.2 Схема task.yaml для описания данных

Пользователь описывает данные в секции `data` файла `task.yaml`:

```yaml
data:
  # Обязательные поля
  train_path: data/train.parquet       # путь к train split (абсолютный или от task.yaml)
  format: parquet                      # csv | parquet | sql | jsonl | txt | hf_dataset | image_dir | coco | manifest_csv

  # Опциональные поля (target и task_type достаточно для tabular)
  val_path: data/val.parquet           # если null — UAF делает split из train автоматически
  test_path: data/test.parquet         # если указан — не трогается до финального eval
  target_column: label                 # имя целевой колонки (tabular)
  text_column: text                    # для NLP задач
  image_column: filepath               # для CV manifest
  task_type: tabular_classification    # см. список ниже

  # Split параметры (применяются если val_path = null)
  split:
    val_fraction: 0.2
    stratify: true                     # стратификация по target_column
    random_seed: 42                    # фиксируется в MLflow и DVC

  # Опциональные метаданные для отчёта
  description: "Датасет кредитного скоринга, 2024 год"
  source: "internal"                   # internal | kaggle | huggingface | custom
  known_issues: null                   # любые известные проблемы данных
```

**Поддерживаемые task_type:**

```
Tabular:    tabular_classification, tabular_regression, tabular_ranking
NLP:        nlp_classification, nlp_generation, nlp_seq2seq, nlp_language_modeling
CV:         cv_classification, cv_detection, cv_segmentation
RecSys:     recsys_ranking
RL:         rl_policy
```

### 2.3 DataLoader: реализация

DataLoader — компонент DVCSetup/SessionSetup, запускается перед генерацией
program.md. Его задача (Путь A): прочитать минимальный объём данных для
построения Metadata Schema и передать путь к данным в program.md.
Полный датасет DataLoader не загружает — это задача Claude Code (Путь B).

**Принцип DataLoader: читаем только то, что нужно для схемы — sample или метаданные.**

```
Для CSV / Parquet:
  pandas read_csv(nrows=1000) или read_parquet(engine='fastparquet')
  -> dtype inference, null counts, shape, column names

Для SQL Dump:
  sqlite3 / psycopg2 в read-only режиме (PRAGMA query_only=ON для SQLite)
  -> PRAGMA table_info() + COUNT(*) + SELECT * LIMIT 1000

Для JSONL:
  построчное чтение первых 1000 строк
  -> schema inference из keys первых строк

Для HuggingFace dataset:
  datasets.load_dataset(..., split='train[:1%]')
  -> features dict + dataset_info.json

Для image_dir / COCO:
  os.walk() для получения списка файлов + PIL.Image.open() на 5 случайных образцов
  -> shape, channels, dtype; class distribution из директорий

Для manifest CSV:
  read_csv() + проверка существования первых 10 filepath
```

**Ограничение по памяти DataLoader (Путь A):** при чтении sample для построения
схемы DataLoader не загружает более 100 МБ в RAM. Это ограничение применяется
исключительно к фазе извлечения метаданных. Если файл больше 100 МБ,
используется sampling достаточный для надёжного вывода схемы.
Обучение модели (Путь B) этим лимитом не ограничено.

**Timeout DataLoader:** 60 секунд. Если DataLoader не завершился за 60 секунд,
UAF записывает ошибку в metadata и продолжает с тем, что успел прочитать.

### 2.4 Передача данных в Claude Code (Путь B)

UAF не передаёт данные Claude Code напрямую. UAF передаёт:

1. **Путь к данным** в program.md секция Task Description:
   ```
   Data: data/train.parquet (Parquet, 150k rows x 45 cols, target: label)
   Val split: data/val.parquet (38k rows)
   ```

2. **Metadata Schema** (полная, см. раздел 2) как артефакт в MLflow Planning Run
   и как ссылку в program.md:
   ```
   Metadata: .uaf/sessions/{id}/metadata/data_schema.json
   ```

3. **Execution Instructions** в program.md содержат инструкцию Claude Code
   по работе с данными: использовать path из task.yaml, не модифицировать.

Claude Code получает доступ к данным через файловую систему (путь из program.md).
settings.json Claude Code ограничивает write-доступ: разрешена запись только
в `.uaf/sessions/{id}/` — данные пользователя находятся вне этой директории.

**Путь B — чтение данных для обучения:** Claude Code читает полный датасет
напрямую с диска стандартными инструментами (`pd.read_parquet`,
`torch.utils.data.DataLoader`, `datasets.load_dataset` и т.д.).
UAF не накладывает никаких ограничений на объём данных при обучении.
Единственные ограничения — RAM и GPU машины, на которой запускается Claude Code.

---

## 3. Metadata Schema: что UAF знает о данных

### 3.1 Структура data_schema.json

Полная схема, сохраняемая в `.uaf/sessions/{id}/metadata/data_schema.json`:

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-03-19T10:30:00Z",
  "session_id": "abc123",
  "task_type": "tabular_classification",

  "splits": {
    "train": {
      "path": "data/train.parquet",
      "format": "parquet",
      "rows": 150000,
      "columns": 45,
      "size_bytes": 52428800,
      "checksum_md5": "d41d8cd98f00b204e9800998ecf8427e"
    },
    "val": {
      "path": "data/val.parquet",
      "format": "parquet",
      "rows": 38000,
      "columns": 45,
      "size_bytes": 13107200,
      "checksum_md5": "7215ee9c7d9dc229d2921a40e899ec5f"
    },
    "test": null
  },

  "target": {
    "column": "label",
    "dtype": "int64",
    "num_classes": 2,
    "class_distribution": {"0": 0.82, "1": 0.18},
    "imbalance_ratio": 4.56,
    "null_count": 0,
    "null_fraction": 0.0
  },

  "features": {
    "total_count": 44,
    "numeric": {
      "count": 32,
      "columns": ["age", "income", "credit_score", "..."],
      "null_fraction_max": 0.03,
      "null_fraction_mean": 0.005
    },
    "categorical": {
      "count": 10,
      "columns": ["region", "product_type", "..."],
      "high_cardinality": ["region"],
      "null_fraction_max": 0.12
    },
    "datetime": {
      "count": 2,
      "columns": ["application_date", "last_payment_date"]
    },
    "text": {
      "count": 0,
      "columns": []
    },
    "constant_columns": [],
    "duplicate_columns": []
  },

  "quality": {
    "total_null_fraction": 0.008,
    "duplicate_rows_fraction": 0.0,
    "outlier_fraction_numeric": 0.023,
    "constant_columns_count": 0,
    "duplicate_columns_count": 0
  },

  "leakage_audit": {
    "status": "passed",
    "checks_run": [...],
    "warnings": [],
    "errors": []
  },

  "adversarial_validation": {
    "status": "passed",
    "roc_auc": 0.53,
    "threshold": 0.6,
    "warning": null
  },

  "split_params": {
    "val_fraction": 0.2,
    "stratified": true,
    "random_seed": 42,
    "split_generated_by_uaf": false
  },

  "task_hints": {
    "recommended_metric": "roc_auc",
    "imbalance_warning": "class imbalance ratio 4.56, consider class_weight or oversampling",
    "high_null_columns": [],
    "leakage_suspects": []
  }
}
```

### 3.2 Как metadata schema попадает в program.md

ProgramMdGenerator получает `data_schema.json` как часть контекста для LLM-вызова.
В prompt вставляется краткая версия схемы:

```
=== DATA CONTEXT ===
Task type: tabular_classification
Train: 150,000 rows x 44 features, target: label (binary, imbalance 4.56:1)
Val:    38,000 rows (same structure, provided externally)
Feature types: 32 numeric, 10 categorical (high-card: region), 2 datetime
Data quality: 0.8% nulls, no duplicates, no constant columns
Leakage audit: PASSED
Adversarial validation: PASSED (AUC=0.53, train/val indistinguishable)
Warnings: imbalance ratio 4.56 -> consider balanced metrics (AP, F1)
===================
```

На основании этого LLM генерирует правильные исследовательские гипотезы:
например, предлагает попробовать `class_weight='balanced'` для первого baseline,
или указывает что текстовые фичи отсутствуют — значит BERT не нужен.

В секции Task Description program.md фиксируется:

```markdown
## Task Description

**Task:** Binary classification
**Metric:** roc_auc (maximize) — см. Execution Instructions
**Data:**
  - Train: `data/train.parquet` (150k rows, 44 features, imbalance 4.56:1)
  - Val: `data/val.parquet` (38k rows, same schema)
  - Schema: `.uaf/sessions/abc123/metadata/data_schema.json`
**Known issues:** class imbalance (handle explicitly in baseline)
**Constraints:** read-only access to data/
```

### 3.3 Metadata Schema для разных типов задач

**NLP (nlp_classification):**

```json
{
  "features": {
    "text_column": "text",
    "avg_token_count": 128,
    "max_token_count": 512,
    "tokenizer_hint": "bert-base-uncased",
    "language": "ru",
    "unique_texts_fraction": 0.99
  },
  "target": {
    "column": "label",
    "num_classes": 5,
    "class_distribution": {...}
  }
}
```

**CV (cv_classification, image_dir формат):**

```json
{
  "splits": {
    "train": {
      "image_count": 10000,
      "classes": ["cat", "dog", "bird"],
      "class_distribution": {"cat": 0.4, "dog": 0.35, "bird": 0.25},
      "image_shape_sample": [224, 224, 3],
      "formats": ["jpg", "png"],
      "corrupt_images_count": 0
    }
  }
}
```

**NLP Language Modeling (nlp_language_modeling):**

```json
{
  "corpus": {
    "total_tokens_estimate": 50000000,
    "vocab_size_hint": 32000,
    "files_count": 120,
    "total_size_bytes": 1073741824,
    "format": "txt",
    "encoding": "utf-8"
  }
}
```

---

## 4. DVC интеграция: версионирование данных и артефактов

### 4.1 Что версионируется через DVC

DVC версионирует крупные бинарные файлы и артефакты, которые не подходят для git.
В UAF DVC используется в двух контекстах: данные пользователя и артефакты сессии.

**Данные пользователя (только регистрация, без копирования):**

```
Действие:  dvc add data/train.parquet data/val.parquet
Результат: data/train.parquet.dvc  -> коммитится в git
           .gitignore              -> data/train.parquet добавляется в ignore
Хранилище: данные остаются на месте (local DVC cache в .dvc/cache/)
```

UAF выполняет `dvc add` только если пользователь явно указал:
```yaml
dvc:
  track_input_data: true   # default: true
  remote: null             # null = только локальный кэш
```

Если `track_input_data: false` — UAF только логирует checksum_md5 в metadata,
DVC add не вызывается. Данные не трогаются в обоих случаях (antigoal 4).

**Артефакты сессии (автоматически, Claude Code по инструкции):**

```
.uaf/sessions/{id}/
  program.md               -> git (текстовый, небольшой)
  program_approved.md      -> git
  metadata/
    data_schema.json       -> git (небольшой)
  experiments/
    {step_id}/
      {step_id}.py         -> git
      requirements.txt     -> git
      outputs/
        model.pkl          -> DVC (если > 1 МБ)
        predictions.csv    -> DVC (если > 1 МБ)
        feature_importance.csv -> git (обычно < 1 МБ)
  reports/
    report.tex             -> git
    report.pdf             -> DVC (бинарный, > 1 МБ)
    figures/*.png          -> DVC
```

Порог "крупный артефакт": > 1 МБ -> DVC. <= 1 МБ -> git напрямую.
Этот порог задаётся в `uaf_config.yaml`:

```yaml
dvc:
  artifact_size_threshold_mb: 1
  auto_push: false           # не пушить в remote автоматически
  remote: null               # пользователь настраивает remote сам
```

### 4.2 DVC commit protocol (для Claude Code)

Claude Code получает DVC протокол в Execution Instructions секции program.md:

```markdown
## Execution Instructions > DVC Protocol

After completing each experiment step:
1. Run: `dvc add .uaf/sessions/{session_id}/experiments/{step_id}/outputs/`
   (only if outputs/ contains files > 1 MB)
2. Run: `git add .uaf/sessions/{session_id}/experiments/{step_id}/`
3. Run: `git commit -m "experiment: {step_id} mlflow_run_id={run_id}"`
4. Log the git commit sha in MLflow tag: `dvc_commit = {sha}`

For small outputs (< 1 MB total), skip step 1.
Do not run dvc push unless explicitly instructed.
```

Соответствие между MLflow run_id и DVC commit sha — ключевой механизм
воспроизводимости (M-ONLINE-03, M-ONLINE-04 из стадии 04).

### 4.3 DVCSetup: инициализация

DVCSetup выполняет следующую последовательность при старте сессии:

```
1. Проверить что .git существует (UAF требует git-репозиторий)
2. Проверить что dvc установлен: `dvc --version`
3. Если .dvc/ не существует: `dvc init`
4. Если track_input_data=true:
   a. `dvc add {train_path} {val_path} {test_path}`  (если не уже в DVC)
   b. `git add *.dvc .gitignore`
   c. `git commit -m "uaf: track input data session={session_id}"`
5. Сохранить input data checksums в data_schema.json
6. Записать в MLflow Planning Run:
   tag: dvc_initialized = true
   tag: input_data_tracked = true | false
   param: data_checksum_train = {md5}
   param: data_checksum_val = {md5}
```

**Обработка ошибок DVCSetup:**

```
dvc не установлен     -> Warning в терминале, сессия продолжается без DVC
                         M-UAF-12 (dvc_commit_rate) будет = 0
git не инициализован  -> Error, сессия не запускается.
                         UAF выводит: "UAF requires a git repository. Run: git init"
dvc add failed        -> Warning, продолжаем без DVC tracking входных данных
                         Фиксируем checksum вручную в metadata
```

### 4.4 DVC remote (опционально)

DVC remote не обязателен для локальной работы. Конфигурируется пользователем
самостоятельно через `dvc remote add`. UAF не трогает DVC remote конфигурацию.

Если пользователь настроил remote, Claude Code получает инструкцию:
```
Note: DVC remote is configured. You MAY run `dvc push` after key experiments
(baseline, best result). Not required after every step.
```

---

## 5. Adversarial Validation

### 5.1 Назначение и принцип

Adversarial Validation — проверка того, что train и val split происходят из
одного распределения. Если разделение некорректное (временной сдвиг, выборка
из разных источников), модель может показывать хорошие метрики на val,
которые не воспроизводятся на реальных данных.

**Метод:**
1. Создать бинарный целевой признак: `is_val = 0` для train, `is_val = 1` для val
2. Обучить простой классификатор (LightGBM, 100 деревьев) предсказывать `is_val`
3. Оценить ROC-AUC на held-out части объединённого датасета (20% для проверки)

**Интерпретация:**
```
ROC-AUC ~ 0.5  ->  train и val неотличимы -> разделение корректное
ROC-AUC > 0.6  ->  умеренное предупреждение: есть сдвиг распределения
ROC-AUC > 0.7  ->  сильное предупреждение: val существенно отличается от train
ROC-AUC > 0.85 ->  критическое: возможен временной или пространственный сдвиг,
                   результаты на val ненадёжны
```

Порог `> 0.6` фиксируется в `uaf_config.yaml` как `adversarial_val_warning_threshold`.
Порог `> 0.85` — как `adversarial_val_error_threshold`.

### 5.2 Реализация Adversarial Validation

Запускается как часть DataLoader после построения metadata schema.
Только для форматов с таблично-структурированными данными (tabular, NLP с feature
таблицами). Для CV пропускается (нет простого feature set для quick classifier).

**Алгоритм:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def run_adversarial_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    sample_size: int = 10000,
) -> dict:
    """Проверка что train и val из одного распределения."""
    # Убираем целевую переменную из фич
    feature_cols = [c for c in train_df.columns if c != target_column]

    train_sample = train_df[feature_cols].sample(
        min(sample_size, len(train_df)), random_state=42
    )
    val_sample = val_df[feature_cols].sample(
        min(sample_size, len(val_df)), random_state=42
    )

    # Создаём adversarial dataset
    train_sample["_is_val"] = 0
    val_sample["_is_val"] = 1
    combined = pd.concat([train_sample, val_sample], ignore_index=True)

    X = combined.drop("_is_val", axis=1)
    y = combined["_is_val"]

    # Быстрый train/test split внутри adversarial dataset
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM без тюнинга: нам нужна только грубая оценка
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        verbose=-1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    # Топ-10 признаков, по которым train/val различимы
    feature_importance = sorted(
        zip(feature_cols, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return {
        "roc_auc": round(auc, 4),
        "status": (
            "passed" if auc < 0.6
            else "warning" if auc < 0.85
            else "critical"
        ),
        "discriminative_features": [
            {"feature": f, "importance": round(float(i), 4)}
            for f, i in feature_importance
        ],
        "sample_size": min(sample_size, len(train_df)) + min(sample_size, len(val_df)),
    }
```

**Что попадает в metadata schema:**

```json
"adversarial_validation": {
  "status": "warning",
  "roc_auc": 0.68,
  "threshold_warning": 0.6,
  "threshold_critical": 0.85,
  "discriminative_features": [
    {"feature": "application_date", "importance": 0.42},
    {"feature": "quarter", "importance": 0.18}
  ],
  "warning": "Train/val distributions differ (AUC=0.68). Top discriminative feature: application_date. Possible temporal split issue.",
  "recommendation": "Check that val split is not from a different time period."
}
```

### 5.3 Поведение UAF при adversarial validation warning/critical

**status = "passed" (AUC < 0.6):** продолжаем без предупреждений.

**status = "warning" (0.6 <= AUC < 0.85):**
- Предупреждение в терминале при старте
- Предупреждение в program.md секция Task Description
- Включается в data_schema.json и в Data Quality Report (PDF)
- Claude Code получает инструкцию: `Adversarial validation warning — be cautious
  interpreting val metrics, mention this in Final Conclusions`
- Сессия продолжается нормально

**status = "critical" (AUC >= 0.85):**
- Жёсткое предупреждение в терминале с объяснением
- HumanOversightGate показывает предупреждение перед планом program.md
- В program.md: `WARNING: Adversarial validation critical (AUC=0.XX)`
- Пользователь может ввести `y` для продолжения несмотря на предупреждение,
  или `n` для отмены
- Если `y`: все experiment runs получают тег `adversarial_val_critical=true`

**Пропуск adversarial validation:**

Пользователь может отключить:
```yaml
validation:
  adversarial_validation: false   # default: true
```

Если train_path = val_path (один файл, split будет делать UAF) — adversarial
validation запускается на сгенерированном split после его создания.

---

## 6. Leakage Audit

### 6.1 Назначение

Leakage Audit — набор автоматических проверок перед стартом экспериментов.
Цель: выявить наиболее распространённые формы утечки данных (data leakage),
которые приводят к завышенным метрикам на val.

UAF не гарантирует отсутствие утечки — задача слишком широкая для автоматического
обнаружения. UAF проверяет наиболее частые и легко детектируемые паттерны.

### 6.2 Перечень проверок

Все проверки выполняются автоматически. Каждая имеет ID, описание, тяжесть.

**LA-01: Target Column in Features**

```
Описание: целевая колонка присутствует в feature set
Как проверяется: target_column в списке колонок train (после удаления)
Тяжесть: CRITICAL
Автофикс: нет — UAF выбрасывает ошибку, сессия не запускается
Пример: dataset содержит колонку 'label' и её copy 'target' с теми же значениями
```

**LA-02: Perfect Predictor Detection**

```
Описание: одна фича имеет > 0.99 корреляцию с target
Как проверяется: point-biserial correlation (binary target) или
                 Spearman correlation (regression target) для каждой числовой фичи
                 Chi-squared для категориальных (cramers_v > 0.95)
Тяжесть: WARNING (не ошибка — может быть легитимная фича)
Автофикс: нет — записывается в leakage_suspects в task_hints
Пример: 'approved_amount' почти всегда = 0 для rejected, > 0 для approved
```

**LA-03: ID / Index Column Detection**

```
Описание: колонка является идентификатором (user_id, row_id) и попала в features
Как проверяется: unique_fraction > 0.99 для колонки
Тяжесть: WARNING
Автофикс: нет — Claude Code получает hint в program.md
Пример: 'customer_id' с 98% уникальных значений — модель может запомнить IDs
```

**LA-04: Future Information Detection**

```
Описание: datetime колонки с датой в будущем относительно других дат в train
Как проверяется: для datetime колонок — сравнение max(val datetime) с max(train datetime)
                 Если val max > train max: возможная утечка из будущего
Тяжесть: WARNING (требует контекст — не всегда ошибка)
Автофикс: нет
Пример: 'last_transaction_date' в val содержит даты после max даты в train
```

**LA-05: Train/Val Row Overlap**

```
Описание: одни и те же строки присутствуют в train и val
Как проверяется: hash каждой строки (pandas.util.hash_pandas_object)
                 intersection count
Тяжесть: CRITICAL если overlap > 0
Автофикс: нет — ошибка, сессия не запускается если overlap > 1%
         Warning если 0 < overlap <= 1% (может быть статистический артефакт)
```

**LA-06: Target Statistics Leakage (Target Encoding риск)**

```
Описание: категориальные фичи с прямым mapping к target (если они уже target-encoded
          во входных данных)
Как проверяется: для каждой категориальной фичи: уникальных значений фичи == уникальных
                 значений target group means -> похоже на target encoding
Тяжесть: WARNING
Автофикс: нет — hint в task_hints
Пример: 'region_encoded' с float значениями уже target-encoded из external data
```

**LA-07: Temporal Ordering Check**

```
Описание: проверка что val данные не перемешаны с train по времени
          (temporal leakage при случайном split на временных рядах)
Как проверяется: только если есть datetime колонка
                 min(val datetime) >= max(train datetime) - delta?
                 Если нет — случайный split на temporal данных
Тяжесть: WARNING (информационная — не всегда ошибка)
Автофикс: нет
Пример: временной ряд сплитован случайно, не по времени
```

**LA-08: Constant After Split**

```
Описание: фича становится константной в val или train (0 дисперсия)
Как проверяется: std == 0 для numeric, nunique == 1 для categorical — в каждом split
Тяжесть: WARNING
Автофикс: нет — hint в task_hints, constant_columns_by_split
Пример: редкая категория присутствует только в train, в val фича = константа
```

**LA-09: Data Checksum Consistency**

```
Описание: проверка что данные не изменились с момента регистрации в DVC
Как проверяется: MD5 файла сейчас vs checksum в data_schema.json
                 (только если track_input_data=true и dvc add был выполнен ранее)
Тяжесть: WARNING
Автофикс: нет — предупреждение что данные могли измениться
Пример: пользователь перезаписал train.parquet между сессиями
```

**LA-10: Schema Mismatch Train/Val**

```
Описание: train и val имеют разные колонки или dtypes
Как проверяется: set(train.columns) == set(val.columns)
                 для общих колонок: dtype совпадает
Тяжесть: CRITICAL если разные колонки
         WARNING если разные dtypes (может быть автоматически сконвертировано)
Автофикс: нет — ошибка, сессия не запускается
```

### 6.3 Результат Leakage Audit

Результат всех проверок сохраняется в data_schema.json:

```json
"leakage_audit": {
  "status": "passed_with_warnings",
  "checks_run": [
    "LA-01", "LA-02", "LA-03", "LA-04", "LA-05",
    "LA-06", "LA-07", "LA-08", "LA-09", "LA-10"
  ],
  "passed": ["LA-01", "LA-05", "LA-09", "LA-10"],
  "warnings": [
    {
      "check_id": "LA-03",
      "column": "user_id",
      "message": "Possible ID column: unique_fraction=0.997. Verify this is not in feature set.",
      "recommendation": "Exclude user_id from features in experiments."
    },
    {
      "check_id": "LA-07",
      "column": "application_date",
      "message": "Temporal ordering: val contains dates overlapping with train (random split on temporal data).",
      "recommendation": "Consider time-based split if task requires temporal generalization."
    }
  ],
  "errors": [],
  "skipped": ["LA-04"]
}
```

**Поведение UAF по итогу Leakage Audit:**

```
status = "passed"               -> нет предупреждений, продолжаем
status = "passed_with_warnings" -> предупреждения в терминале + в program.md
                                   Claude Code получает hints в Execution Instructions
                                   Сессия продолжается
status = "failed"               -> сессия не запускается
                                   UAF выводит список CRITICAL ошибок
                                   Пользователь должен исправить данные
```

**Инструкция для Claude Code при наличии warnings:**

```markdown
## Execution Instructions > Data Warnings

Leakage audit warnings detected:
- LA-03: Column 'user_id' may be an ID — exclude from feature set
- LA-07: Temporal ordering issue — results on val may not reflect real-world performance

When writing experiments:
1. Exclude 'user_id' from feature set explicitly
2. Mention temporal split caveat in Final Conclusions
```

---

## 7. Data Quality Report

### 7.1 Что входит в Data Quality Report

Data Quality Report — секция LaTeX/PDF отчёта, генерируемая автоматически
из data_schema.json. Это статическая секция (не LLM-генерируемая).
ReportGenerator строит её по шаблону.

Секция называется "Data Overview" и размещается после "Task Description",
перед "Experiment Results".

### 7.2 Структура секции Data Overview в LaTeX

**Блок 1: Dataset Summary Table**

```
\subsection{Dataset Summary}
\begin{tabular}{lrrrr}
Split  & Rows    & Features & Size (MB) & Format   \\
Train  & 150,000 & 44       & 50.0      & Parquet  \\
Val    & 38,000  & 44       & 12.5      & Parquet  \\
Test   & --      & --       & --        & --       \\
\end{tabular}
```

**Блок 2: Feature Type Distribution**

```
\subsection{Feature Types}
\begin{tabular}{lr}
Type       & Count \\
Numeric    & 32    \\
Categorical & 10   \\
Datetime   & 2    \\
Text       & 0     \\
\textit{Constant} & 0 \\
\textit{Duplicate} & 0 \\
\end{tabular}
```

**Блок 3: Target Distribution**

Для классификации:
```
Target column: label (binary)
Class distribution: {0: 82.0%, 1: 18.0%}
Imbalance ratio: 4.56:1
\textcolor{orange}{Note: significant class imbalance — consider weighted metrics}
```

Для регрессии:
```
Target column: price (continuous)
Range: [1200, 89000], Mean: 12450, Std: 8200, Skewness: 2.4
Percentiles: p25=6000, p50=9500, p75=16000
```

**Блок 4: Data Quality Indicators**

```
\subsection{Data Quality}
\begin{tabular}{lr}
Indicator                    & Value   \\
Total null fraction          & 0.8\%   \\
Max null fraction (column)   & 12.0\%  \\
Duplicate rows fraction      & 0.0\%   \\
Outlier fraction (numeric)   & 2.3\%   \\
Constant columns             & 0       \\
Duplicate columns            & 0       \\
\end{tabular}
```

**Блок 5: Leakage Audit Summary**

```
\subsection{Leakage Audit}
Status: \textcolor{orange}{PASSED WITH WARNINGS} \\
Checks run: 10 / 10 \\
Errors (blocking): 0 \\
Warnings: 2

\begin{itemize}
  \item \textbf{LA-03} — Column \texttt{user\_id}: possible ID column (unique\_fraction=0.997)
  \item \textbf{LA-07} — Column \texttt{application\_date}: temporal overlap between train and val
\end{itemize}
```

**Блок 6: Adversarial Validation**

```
\subsection{Train/Val Distribution Check (Adversarial Validation)}
Classifier ROC-AUC: 0.68 \quad Threshold warning: 0.60 \\
Status: \textcolor{orange}{WARNING} \\
Top discriminative features: application\_date (0.42), quarter (0.18) \\
\textit{Val distribution differs from train. Exercise caution interpreting
val metrics — possible temporal distribution shift.}
```

**Блок 7: DVC Tracking**

```
\subsection{Data Versioning}
\begin{tabular}{ll}
Input data tracked in DVC: & Yes \\
Train checksum (MD5):  & d41d8cd98f00b204e9800998ecf8427e \\
Val checksum (MD5):    & 7215ee9c7d9dc229d2921a40e899ec5f \\
DVC commit:            & a3f1b2c (initial data tracking) \\
\end{tabular}
\textit{To restore exact input data: \texttt{git checkout a3f1b2c \&\& dvc checkout}}
```

### 7.3 Figure: Feature Null Distribution (опциональный)

Если `null_fraction_max > 0.05` (есть колонки с > 5% пропусков):

```python
# Matplotlib bar chart
# X: колонки с > 1% пропусков, отсортированы по убыванию
# Y: null_fraction
# Включается в отчёт как Figure 0 (перед experiment figures)
```

### 7.4 Figure: Class Distribution (для классификации)

Если task_type == tabular_classification или nlp_classification:

```python
# Pie chart или bar chart
# Классы с долями
# Annotated если imbalance_ratio > 3.0
```

### 7.5 Что не входит в Data Quality Report

- Полный EDA (гистограммы каждой фичи) — это задача Claude Code в экспериментах
- Корреляционная матрица полного датасета — может быть огромной, не нужна на уровне UAF
- Примеры строк из датасета — данные не должны появляться в отчёте без контроля
  пользователя (конфиденциальность)
- Интерактивные visualizations — PDF-формат, только статика

---

## 8. Специфика DataLoader по типу задачи

### 8.1 Tabular (CSV, Parquet, SQL)

**SQL Dump обработка:**

```
1. Определить СУБД по заголовку файла (-- PostgreSQL dump / SQLite format)
2. Для SQLite: sqlite3.connect(':memory:') + executescript()
3. Для PostgreSQL dump: создать временную SQLite БД, преобразовать CREATE TABLE
   (убрать PostgreSQL-специфичный синтаксис), загрузить INSERT statements
4. Выбрать целевую таблицу: если одна — автоматически,
   если несколько — использовать task.yaml поле data.table_name
5. pd.read_sql() -> DataFrame -> стандартный tabular путь
```

Ограничение: SQL dumps > 500 МБ обрабатываются через sampling (LIMIT 100000).

### 8.2 NLP

**JSONL формат:**

Ожидаемая структура:
```json
{"text": "...", "label": 0}
{"text": "...", "label": 1}
```

DataLoader проверяет:
- Все строки валидный JSON
- Поля `text_column` и `target_column` из task.yaml присутствуют
- Нет пустых text полей

**HuggingFace datasets:**

```python
from datasets import load_dataset
ds = load_dataset("path/to/dataset", split="train[:1%]")
# metadata из ds.info: features, num_rows, splits
```

**Language modeling (txt):**

DataLoader не читает весь корпус. Оценивает размер:
```
total_tokens_estimate = total_bytes / avg_bytes_per_token
avg_bytes_per_token = 4 (rough estimate for BPE)
```

### 8.3 CV

**image_dir структура:**

```
data/
  train/
    cat/ [jpg, png, ...]
    dog/ [jpg, png, ...]
  val/
    cat/ [...]
    dog/ [...]
```

DataLoader:
1. `os.walk()` — считает файлы по классам
2. `PIL.Image.open()` на 5 случайных — проверяет shape, channels
3. Проверяет corrupt images (try/except при открытии)
4. Нет adversarial validation (нет простого feature set для LightGBM)

**COCO формат:**

```python
import json
with open("annotations.json") as f:
    coco = json.load(f)
# images count, categories, annotations count
```

**Leakage Audit для CV:**

Из 10 проверок применяются только:
- LA-05 (row overlap): сравнение имён файлов в train и val
- LA-10 (schema mismatch): не применимо для CV, пропускается

---

## 9. Взаимодействие компонентов: Data Flow

```
task.yaml (data paths)
    |
    +============ ПУТЬ A: UAF DataLoader (метаданные) ==============+
    |                                                               |
    v                                                               |
DVCSetup.init()                                                     |
    |-- dvc add {train_path, val_path}                              |
    |-- git commit "uaf: track input data"                          |
    |                                                               |
    v                                                               |
DataLoader.load_metadata()   <-- только sample, max 100 MB, 60 sec |
    |-- read sample (nrows=1000 / 1% / 5 images)                    |
    |-- compute checksums MD5                                        |
    |-- infer schema (dtypes, nulls, shape)                         |
    |                                                               |
    v                                                               |
LeakageAudit.run()                                                  |
    |-- 10 проверок (LA-01..LA-10)                                  |
    |-- [CRITICAL] -> stop session                                  |
    |-- [WARNING]  -> записать в schema                             |
    |                                                               |
    v                                                               |
AdversarialValidation.run()  (tabular only)                        |
    |-- sample до 10k строк из train+val                            |
    |-- LightGBM adversarial classifier                             |
    |-- [CRITICAL AUC] -> warn at HumanOversightGate               |
    |-- [WARNING AUC]  -> записать в schema                        |
    |                                                               |
    v                                                               |
data_schema.json -> .uaf/sessions/{id}/metadata/                   |
    |-- MLflow Planning Run (checksums, leakage_status)             |
    |-- ProgramMdGenerator (краткая summary схемы в prompt)         |
    |                                                               |
    v                                                               |
program.md:                                                         |
    |-- Task Description (data paths, shape, issues)               |
    |-- Execution Instructions (read-only, DVC protocol, hints)    |
    |                                                               |
    +===============================================================+
    |
    +============ ПУТЬ B: Claude Code (обучение) ====================+
    |                                                               |
    v                                                               |
Claude Code читает program.md + data_schema.json                   |
    |                                                               |
    v                                                               |
Claude Code читает ПОЛНЫЙ датасет с диска напрямую                 |
    |-- pd.read_parquet("data/train.parquet")   <- без лимитов UAF  |
    |-- torch.utils.data.DataLoader(...)        <- вся память GPU   |
    |-- datasets.load_dataset(...)              <- весь корпус      |
    |   ограничение: RAM/GPU машины, не UAF                        |
    |                                                               |
    v                                                               |
эксперименты, обучение моделей, сохранение outputs                 |
    не модифицирует data/                                           |
    |                                                               |
    +===============================================================+
    |
    v
ReportGenerator:
    |-- data_schema.json -> секция "Data Overview" в LaTeX
    |-- DVC commits -> Reproducibility секция
```

---

## 10. Ограничения и явно нерешённые задачи

**Что UAF не делает с данными (явные ограничения):**

1. **Нет автоматического imputation.** UAF обнаруживает пропуски и сообщает
   о них. Заполнение пропусков — задача Claude Code в экспериментах.

2. **Нет автоматического feature engineering.** DataLoader только читает
   существующие фичи. Создание новых фич — задача Claude Code.

3. **Нет data versioning для производных датасетов.** Если Claude Code
   создаёт preprocessed версию данных в сессии, она трекается через DVC
   только как output артефакт, не как отдельный датасет.

4. **DataLoader не поддерживает streaming для извлечения метаданных.**
   Лимит 100 МБ — это ограничение DataLoader (Путь A) на чтение sample
   для построения схемы, не ограничение на обучение. Claude Code (Путь B)
   читает полный датасет без ограничений UAF. Однако если датасет > 10 ГБ,
   DataLoader всё равно прочитает только sample — рекомендуется указать
   `data.sample_path` — отдельный семплированный файл для metadata,
   чтобы schema была более репрезентативной.

5. **SQL dumps > 500 МБ.** Обрабатываются через sampling, полная загрузка
   не гарантируется. Рекомендуется экспортировать в Parquet заранее.

6. **Adversarial validation для CV** не реализована.
   Для CV пользователь должен проверить balance вручную.

7. **Leakage в feature engineering коде.** Если Claude Code пишет
   target encoding без разделения train/val, UAF не обнаружит это.
   Это задача code review при просмотре отчёта (секция Code Quality).

---

## STAGE COMPLETE

Стадия 05-data завершена.

**Зафиксировано:**

- Два пути работы с данными зафиксированы явно: Путь A (UAF DataLoader —
  читает только sample/схему, лимит 100 МБ применяется только здесь,
  timeout 60 сек) и Путь B (Claude Code — читает полный датасет напрямую,
  UAF не ограничивает объём для обучения).

- DataLoader: три основных формата (CSV, Parquet, SQL Dump) + расширенные
  для NLP (JSONL, TXT, HF datasets) и CV (image_dir, COCO, manifest CSV).
  Read-only, чтение sample для построения схемы, timeout 60 сек.

- Metadata Schema: полная структура data_schema.json. 7 разделов:
  splits, target, features, quality, leakage_audit, adversarial_validation,
  task_hints. Передаётся в program.md как краткая summary (не полный JSON).

- DVC интеграция: track_input_data для входных данных (dvc add).
  Артефакты сессии (> 1 МБ -> DVC, <= 1 МБ -> git). DVC commit protocol
  для Claude Code в Execution Instructions. Порог 1 МБ настраивается.

- Adversarial Validation: LightGBM классификатор train vs val.
  Три уровня: passed (AUC < 0.6), warning (0.6-0.85), critical (>= 0.85).
  При critical: блокировка на HumanOversightGate с возможностью override.

- Leakage Audit: 10 автоматических проверок (LA-01..LA-10). Два уровня:
  CRITICAL (блокирует сессию), WARNING (предупреждение + hints в program.md).
  CRITICAL: target leakage, train/val row overlap, schema mismatch.

- Data Quality Report: статическая LaTeX секция в PDF отчёте.
  7 блоков: dataset summary, feature types, target distribution, quality
  indicators, leakage audit summary, adversarial validation, DVC tracking.
  Два опциональных figure: null distribution и class distribution.

- Data Flow: зафиксирован порядок выполнения компонентов (DVCSetup ->
  DataLoader -> LeakageAudit -> AdversarialValidation -> ProgramMdGenerator
  -> Claude Code -> ReportGenerator).

- Ограничения: 7 явных ограничений, которые UAF не решает.

Antigoal 4 (не модифицирует данные) соблюдается: UAF только читает данные
пользователя и регистрирует их в DVC через .dvc файлы (метаданные, не копии).

Переход к стадии 06-validation разрешён.
