# Стадия 10: Features

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-09 (COMPLETE или SKIPPED/EMBEDDED)

---

## 0. Что такое "feature engineering" в контексте UAF

Важно зафиксировать контекст, чтобы не путать два разных уровня.

**Уровень 1 — фичи UAF как системы** (не являются предметом этой стадии):
UAF не имеет собственной ML-модели, которая обучается. У UAF нет feature engineering
в классическом смысле. "Фичи UAF" — это конфигурационные параметры и свойства системы,
которые задаются в стадиях 01-09.

**Уровень 2 — фичи пользовательских экспериментов** (предмет этой стадии):
Feature engineering для задач пользователя, которые Claude Code решает внутри
исследовательских сессий UAF. Здесь UAF выступает как оркестратор: подсказывает
Claude Code какие признаки пробовать, как безопасно тестировать новые фичи,
как отслеживать их влияние на метрику.

Таким образом, стадия 10 описывает механизмы, которые UAF предоставляет Claude Code
для работы с признаками внутри сессии. Это не фичи самой UAF.

---

## 1. Shadow Feature Trick: как UAF инструктирует Claude Code

### 1.1 Проблема

Стандартный подход к feature engineering — добавить новый признак в датасет
и обучить модель. Проблема: если новая фича ухудшает метрику, неясно почему.
Дополнительно: при итеративном добавлении нескольких фич влияние каждой отдельной
размывается. Baseline для сравнения теряется.

В контексте UAF проблема усиливается: Claude Code работает итерациями, и при
внесении изменений в preprocessing секцию experiment.py baseline итерации больше
не воспроизводится без возврата к предыдущей версии experiment.py.

### 1.2 Shadow Feature Trick — определение

Shadow Feature Trick — это инструкция Claude Code строить feature experiments
таким образом, что новые признаки добавляются в датасет как дополнительные колонки
(shadow columns), но базовый набор признаков (baseline feature set) остаётся
нетронутым и воспроизводимым.

Модель обучается дважды или с контрольным условием:
- Версия A: baseline features (те, что работали в лучшей предыдущей итерации)
- Версия B: baseline features + shadow features

Версии A и B в одном experiment run логируются в MLflow как отдельные nested runs
или как отдельные метрики с суффиксами `_baseline` и `_candidate`.

### 1.3 Как это реализовано в program.md

ProgramMdGenerator включает в шаблон program.md секцию Research Phases
специальный тип фазы: `phase_type: feature_engineering`. Это обязательная фаза —
выполняется после Phase 1 всегда (см. раздел 3 об автоматических фичах).

Пример записи фазы в program.md:

```markdown
## Phase 2: Feature Engineering

### Step 2.1 — Shadow Feature Experiment: time features
- hypothesis: "Временные признаки (hour_of_day, day_of_week) улучшат ROC-AUC"
- method: shadow_feature_trick
- shadow_features: ["hour_of_day", "day_of_week", "is_weekend"]
- baseline_run_id: "mlflow_run_id_from_phase1_best"
- metric: roc_auc
- critical: false
- status: pending
- mlflow_run_id: null
- result: null
- conclusion: null

### Step 2.2 — Shadow Feature Experiment: interaction features
- hypothesis: "Произведение age * credit_limit добавляет информацию"
- method: shadow_feature_trick
- shadow_features: ["age_x_credit_limit", "income_to_debt_ratio"]
- baseline_run_id: "mlflow_run_id_from_phase1_best"
- metric: roc_auc
- critical: false
- status: pending
```

Поля `shadow_features` и `baseline_run_id` — специфичные для `phase_type: feature_engineering`.
Остальные поля совпадают со стандартной структурой Research Phase.

### 1.4 Execution Instructions для Shadow Feature Trick

ProgramMdGenerator включает в секцию Execution Instructions программы следующие
инструкции для Claude Code при наличии phase_type: feature_engineering:

```
## Feature Engineering Instructions (Shadow Feature Trick)

При реализации шага с method: shadow_feature_trick:

1. PREPROCESSING секция эксперимента должна строить ДВА датасета:
   - X_baseline: признаки из baseline_run_id (список из data_schema.json features.selected_baseline)
   - X_candidate: X_baseline + shadow_features (новые признаки добавляются здесь)

2. Обучи модель ДВА раза с одинаковыми гиперпараметрами:
   - model_baseline.fit(X_baseline, y_train)
   - model_candidate.fit(X_candidate, y_train)

3. Залогируй в MLflow как nested runs ИЛИ как суффиксные метрики в одном run:
   - Вариант A (nested, предпочтительный): два child run в parent run
     parent: step_id, hypothesis, shadow_features (список)
     child "baseline": roc_auc_val, feature_count=N
     child "candidate": roc_auc_val, feature_count=N+K, shadow_features=K
   - Вариант B (flat): один run с метриками roc_auc_baseline и roc_auc_candidate

4. ARTIFACT-SAVING: сохрани оба препроцессора и feature importance для обеих версий.

5. Решение о принятии shadow features:
   - delta = roc_auc_candidate - roc_auc_baseline
   - Если delta > min_feature_delta (дефолт: 0.002): shadow features принимаются
   - Если delta <= 0: shadow features отвергаются, baseline не меняется
   - Если 0 < delta <= min_feature_delta: записать в conclusion "marginal improvement,
     include if no regression on test data"
   - Запиши решение в program.md step conclusion: "accepted" / "rejected" / "marginal"

6. features.selected_baseline в data_schema.json НЕ обновляется автоматически.
   Обновление происходит только через явную инструкцию в program.md.
   Antigoal 4 (не модифицировать данные/конфиг пользователя) соблюдается:
   data_schema.json — артефакт сессии, не пользовательские данные.
```

### 1.5 Параметр min_feature_delta

`min_feature_delta` задаётся в task.yaml как `research_preferences.min_feature_delta`.
Дефолт: 0.002 (0.2% от целевой метрики). Слишком маленькие улучшения от новых
признаков могут быть шумом или переобучением.

Если пользователь не задал значение — UAF использует дефолт и логирует факт
использования дефолта в MLflow Planning Run как `feature_delta_threshold: default`.

### 1.6 Почему shadow, а не ablation

Ablation (удаление признаков из полного набора) — альтернативный подход.
Shadow предпочтительнее для итеративного добавления признаков потому что:
- Baseline остаётся фиксированным и воспроизводимым (один baseline_run_id)
- Проще интерпретировать: дельта = вклад конкретных shadow features
- Ablation уместен на этапе финальной очистки после принятия нескольких фич
  (см. раздел 1.7)

Ablation включается как отдельный шаг в program.md если итоговый набор принятых
фич > 2x от baseline feature count (эвристика: слишком много новых признаков —
нужна обратная проверка).

### 1.7 Итоговый Feature Set — фиксация

После завершения feature engineering фазы Claude Code обязан:

1. Записать в program.md секцию `## Accepted Features`:
   ```
   baseline_features: [список из data_schema.json]
   accepted_shadow_features: [перечень принятых фич]
   rejected_shadow_features: [перечень отклонённых]
   final_feature_count: N
   final_baseline_run_id: {mlflow_run_id лучшей candidate модели}
   ```

2. Обновить `data_schema.json` секцию `features.selected_baseline` с итоговым
   набором. Это артефакт сессии — UAF не трогает оригинальные данные.

3. Залогировать в MLflow Session Summary Run:
   - `feature_engineering_iterations`: число shadow экспериментов
   - `features_accepted`: число принятых shadow features
   - `features_rejected`: число отклонённых
   - `final_feature_count`: итоговое число признаков

---

## 2. Feature Store: нужен ли он и какой минимальный

### 2.1 Решение: Feature Store не нужен как отдельный компонент

UAF не реализует Feature Store (Feast, Hopsworks, Vertex AI Feature Store и т.п.).
Обоснование:

- UAF работает на локальной машине, один пользователь (из стадии 01)
- Feature Store решает проблему sharing фич между командами и online/offline
  consistency — этих проблем нет в контексте одиночного ML-исследователя
- Antigoal 1: не AutoML для production-деплоя — Feature Store нужен именно там
- Введение Feature Store как компонента нарушает принцип "тонкой оболочки":
  UAF станет значительно сложнее без реального выигрыша

Вместо Feature Store UAF использует минималистичный Feature Registry.

### 2.2 Минимальный Feature Registry

Feature Registry в UAF — это:
1. Секция `features` в `data_schema.json` (уже определена в стадии 05)
2. Дополнительный файл `feature_registry.json` в SESSION_DIR (создаётся при наличии
   feature engineering фазы в program.md)

`feature_registry.json` не является полноценным Feature Store. Это артефакт
одной сессии. Он используется для:
- Отслеживания какие фичи были созданы в сессии
- Передачи этой информации в следующую сессию через --resume
- Формирования секции в PDF отчёте

Структура `feature_registry.json`:

```json
{
  "registry_version": "1.0",
  "session_id": "sess_20260319_abc123",
  "generated_at": "2026-03-19T15:00:00Z",
  "task_type": "tabular_classification",

  "original_features": {
    "count": 44,
    "names": ["age", "income", "credit_limit", "..."],
    "source": "data_schema.json"
  },

  "engineered_features": [
    {
      "name": "hour_of_day",
      "type": "temporal",
      "source_columns": ["transaction_timestamp"],
      "transformation": "hour(transaction_timestamp)",
      "iteration_introduced": 3,
      "mlflow_run_id": "abc123def456",
      "delta_on_acceptance": 0.0047,
      "status": "accepted"
    },
    {
      "name": "age_x_credit_limit",
      "type": "interaction",
      "source_columns": ["age", "credit_limit"],
      "transformation": "age * credit_limit",
      "iteration_introduced": 4,
      "mlflow_run_id": "ghi789jkl012",
      "delta_on_acceptance": -0.0012,
      "status": "rejected"
    }
  ],

  "final_feature_set": {
    "count": 47,
    "names": ["age", "income", "...", "hour_of_day", "day_of_week", "is_weekend"],
    "selected_baseline_run_id": "mno345pqr678"
  },

  "feature_groups": {
    "numeric_original": ["age", "income", "credit_limit"],
    "categorical_original": ["region", "occupation"],
    "temporal_engineered": ["hour_of_day", "day_of_week", "is_weekend"],
    "interaction_engineered": [],
    "text_derived": [],
    "embedding_derived": []
  }
}
```

### 2.3 Жизненный цикл Feature Registry

Feature Registry создаётся в момент первого shadow feature experiment.
Если feature engineering фазы не было — `feature_registry.json` не создаётся.

При `--resume` сессии UAF передаёт `feature_registry.json` предыдущей сессии
в контекст через `improvement_context.md`. Claude Code видит какие фичи уже были
опробованы и какой был результат — не тратит итерации на повторение.

При финальном отчёте ReportGenerator читает `feature_registry.json`
и включает секцию Feature Engineering в PDF (см. раздел 6).

### 2.4 DVC трекинг Feature Registry

`feature_registry.json` — артефакт сессии, размер обычно < 1 МБ.
По протоколу из стадии 05: <= 1 МБ -> git commit.

Инструкция в Execution Instructions:
```
После обновления feature_registry.json:
  git add .uaf/sessions/{id}/feature_registry.json
  git commit -m "feat: update feature_registry iter_{N}"
```

Таким образом история изменений feature registry сохраняется в git,
а не в DVC. Это достаточно для воспроизводимости одиночного исследователя.

---

## 3. Автоматические фичи на основе data_schema.json

### 3.1 Принцип: UAF предлагает, Claude Code реализует

UAF не генерирует фичи автоматически сам. Это нарушало бы antigoal 4
(не модифицировать данные) и antigoal 5 (читаемость важнее оптимизации).

Вместо этого UAF читает `data_schema.json` и генерирует список
**гипотез о перспективных фичах** при формировании program.md через ProgramMdGenerator.
Гипотезы становятся шагами в Phase 2 (Feature Engineering) program.md.
Claude Code видит гипотезы и решает — реализовывать или пропустить.

### 3.2 Правила генерации гипотез по типам колонок

DataLoader из стадии 05 формирует `data_schema.json` с типами колонок.
ProgramMdGenerator применяет следующие детерминированные правила (без LLM):

#### 3.2.1 Tabular

```
Тип колонки: datetime / timestamp
  -> Гипотезы:
     FG-T-01: hour_of_day (если временная гранулярность часы или меньше)
     FG-T-02: day_of_week (если гранулярность дни или меньше)
     FG-T-03: month / quarter (если гранулярность месяцы или меньше)
     FG-T-04: is_weekend (если day_of_week доступен)
     FG-T-05: days_since_epoch (числовое представление, полезно для tree-based)
     Условие: применяется если в data_schema.json есть >= 1 колонка типа datetime

Тип колонки: numeric с высоким диапазоном (max/min ratio > 100)
  -> Гипотезы:
     FG-N-01: log1p(col) — логарифмирование скошенных распределений
     FG-N-02: clip + normalize — обработка выбросов через IQR clip
     Условие: skewness > 2.0 в data_schema.json stats

Тип колонки: два numeric
  -> Гипотезы:
     FG-I-01: col_a / col_b — ратио (только если col_b без нулей по stats)
     FG-I-02: col_a * col_b — произведение (если оба без нулей)
     FG-I-03: col_a - col_b — разность (для однотипных признаков)
     Условие: применяется для топ-3 пар по абсолютной корреляции с target
              (Spearman), корреляция доступна в data_schema.json

Тип колонки: categorical с cardinality > 10
  -> Гипотезы:
     FG-C-01: target encoding (mean target по категории на train)
     FG-C-02: count encoding (frequency encoding)
     FG-C-03: binary encoding для high-cardinality
     Условие: cardinality в data_schema.json features[col].unique_count > 10

Тип колонки: categorical + numeric (группировка)
  -> Гипотезы:
     FG-G-01: mean(numeric) group by (categorical) — групповая статистика
     FG-G-02: std(numeric) group by (categorical)
     Условие: cardinality categorical < 20 (иначе переобучение на train)

Тип колонки: id / key (высокая cardinality, без корреляции с target)
  -> Гипотезы: нет (предупреждение в program.md: "id-like column detected, likely not useful")
     Условие: unique_count / row_count > 0.95
```

#### 3.2.2 NLP

```
Тип: text column (определяется как строковая с median_length > 20 символов)
  -> Гипотезы:
     FG-NLP-01: text_length (len(text)) — длина текста как числовой признак
     FG-NLP-02: word_count — число слов
     FG-NLP-03: avg_word_length — средняя длина слова (индикатор технического текста)
     FG-NLP-04: sentence_count — число предложений (если длинные тексты, median > 200)
     FG-NLP-05: uppercase_ratio — доля заглавных букв (сигнал агрессивности/акцента)
     FG-NLP-06: digit_ratio — доля цифр (числовые тексты)
     FG-NLP-07: special_char_ratio — доля спецсимволов
     FG-NLP-08: TF-IDF top-k features + logistic regression (отдельный step)
     FG-NLP-09: language detection feature (если multilingual сигнал в data_schema)

Специфика NLP: FG-NLP-01..07 — статистические фичи, низкая стоимость.
FG-NLP-08 — отдельная итерация (изменяет архитектуру, не только фичи).
FG-NLP-09 включается только если data_schema.features имеет multilingual_detected=true.
```

#### 3.2.3 CV

```
Тип: image (cv_classification / cv_detection / cv_segmentation)
  -> Гипотезы:
     FG-CV-01: image_size normalization варианты (224x224, 384x384, 512x512)
     FG-CV-02: augmentation policy (baseline: RandomHorizontalFlip + ColorJitter)
     FG-CV-03: augmentation policy (strong: + RandomRotation + RandomErasing)
     FG-CV-04: normalization strategy (ImageNet mean/std vs dataset-specific)
     FG-CV-05: multi-scale inference (TTA, если task.type != cv_detection)

Примечание: для CV "feature engineering" = augmentation + preprocessing политики.
Классические фичи неприменимы — пиксели это уже вектор признаков для CNN.
Shadow Feature Trick для CV: две модели обучаются с разными augmentation политиками.
```

### 3.3 Фильтрация гипотез: приоритет и лимит

ProgramMdGenerator не включает все возможные гипотезы в program.md.
Применяется фильтрация:

1. Гипотезы отбираются только для колонок с detected_signal > threshold:
   - Для numeric: если feature importance (из baseline Phase 1) > 0
   - Для categorical: если chi2 с target p-value < 0.05 (из data_schema.json)
   - Для datetime: всегда (временные фичи почти всегда полезны)

2. Максимум 5 shadow feature шагов в одной feature engineering фазе.
   Если генерируется больше гипотез — берутся топ-5 по ожидаемой пользе.
   Приоритет: datetime > interaction (для топ-пар) > categorical encoding >
   numeric transforms > группировые статистики.

3. Если phase_type: feature_engineering уже есть в program.md от предыдущей сессии
   (--resume), то уже опробованные гипотезы не повторяются.
   ProgramMdGenerator читает `feature_registry.json` перед генерацией.

### 3.4 task_hints как дополнительный сигнал

data_schema.json содержит секцию `task_hints` (определена в стадии 05).
ProgramMdGenerator использует task_hints при генерации feature гипотез:

```json
"task_hints": {
  "potential_feature_engineering": ["datetime columns detected", "high cardinality: region"],
  "leakage_warnings": ["column order_date is after transaction_date"],
  "recommended_encoding": "target_encoding for region (cardinality=47)"
}
```

Эти hints дословно включаются в program.md как контекст для Claude Code.

---

## 4. Организация feature engineering в program.md

### 4.1 Feature Engineering как отдельная фаза

Feature engineering в program.md организована как **обязательная Phase 2**.
Выполняется после Phase 1 всегда, независимо от результатов Phase 1.
Пользователь может явно отключить: `research_preferences.skip_feature_engineering: true`.

Почему отдельная от Phase 1 (Baseline):
- Концептуальная ясность: Phase 1 отвечает на "какая архитектура работает",
  Phase 2 — "какие признаки добавляют ценность"
- Разные стратегии бюджета: Phase 1 требует минимум итераций,
  Phase 2 может продолжаться до сходимости
- Отдельная секция в отчёте (см. раздел 6)

### 4.2 Структура шаблона program.md с feature engineering

Полная структура program.md с Phase 2:

```markdown
# Research Program
Session: {session_id}
Task: {task_description}
Metric: {metric_name} ({direction})

## Phase 1: Baseline (MANDATORY)

### Step 1.1 — Constant baseline
- hypothesis: "DummyClassifier задаёт lower bound"
- method: dummy_classifier
- metric: {metric_name}
- critical: true
- status: pending

### Step 1.2 — Linear baseline
- hypothesis: "LogisticRegression с базовыми фичами"
- method: logistic_regression
- metric: {metric_name}
- critical: true
- status: pending

### Step 1.3 — Gradient boosting baseline
- hypothesis: "CatBoost с дефолтами — strong baseline"
- method: catboost_default
- metric: {metric_name}
- critical: true
- status: pending

---

## Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*

### Step 2.1 — Shadow: datetime features
- hypothesis: "{gipoteza из FG-T-01..05}"
- method: shadow_feature_trick
- shadow_features: ["hour_of_day", "day_of_week", "is_weekend"]
- baseline_run_id: {best_run_id_from_phase1}
- metric: {metric_name}
- critical: false
- status: pending
- mlflow_run_id: null
- result: null
- conclusion: null

### Step 2.2 — Shadow: target encoding for region
- hypothesis: "Target encoding для region (cardinality=47)"
- method: shadow_feature_trick
- shadow_features: ["region_target_enc", "region_count_enc"]
- baseline_run_id: {best_run_id_from_phase1}
- metric: {metric_name}
- critical: false
- status: pending

---

## Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

---

## Execution Instructions
[Стандартные инструкции + Feature Engineering инструкции — см. раздел 1.4]

## Current Status
[Claude Code обновляет]

## Iteration Log
[Claude Code обновляет]

## Accepted Features
[Claude Code заполняет после Phase 2]
```

### 4.3 Переход между фазами — правила

Правила перехода Phase 1 -> Phase 2, закреплённые в Execution Instructions:

```
После завершения Phase 1:
1. Запиши best_run_id в Step 2.x.baseline_run_id для каждого шага Phase 2
2. Если skip_feature_engineering: true (из конфига):
   - Пометь Phase 2 как SKIPPED
3. Иначе: выполняй шаги Phase 2 последовательно
```

Правила перехода Phase 2 -> Phase 3:

```
После завершения Phase 2:
1. Запиши Accepted Features секцию (см. раздел 1.7)
2. Обнови feature_registry.json
3. Phase 3 использует final_feature_set из Accepted Features
4. Если ни одна shadow фича не была принята:
   - Phase 3 использует baseline feature set из Phase 1 best run
   - Запиши вывод: "No feature engineering improvement found. Using Phase 1 features."
```

---

## 5. Feature Importance Tracking в MLflow

### 5.1 Что логируется и куда

Feature importance отслеживается на двух уровнях:

**Уровень A — внутри experiment run (Claude Code):**
Каждый experiment.py, обучивший дерево-based модель, должен логировать
feature importance как артефакт MLflow. Инструкция в Execution Instructions:

```
Для tree-based моделей (CatBoost, XGBoost, LightGBM, RandomForest):
1. Получи feature importance: model.feature_importances_ или model.get_feature_importance()
2. Создай DataFrame: pd.DataFrame({"feature": feature_names, "importance": importances})
3. Сохрани как артефакт: mlflow.log_artifact("feature_importance.csv", "features")
4. Логируй топ-5 признаков как отдельные метрики:
   mlflow.log_metric(f"fi_top1_{top1_feature_name}", top1_importance)
   mlflow.log_metric(f"fi_rank_shadow_{shadow_feat}", rank_of_shadow_feat)
   (fi_rank_shadow_* — только для shadow feature экспериментов)
```

Для линейных моделей (LogisticRegression, Ridge):
```
1. Логируй abs(coef_) как feature_coef.csv
2. Для shadow фич логируй: mlflow.log_metric(f"coef_shadow_{feat}", abs_coef)
```

Для нейросетей (PyTorch):
```
1. SHAP (если доступен и задача tabular — из стадии 08):
   mlflow.log_artifact("shap_importance.csv", "features")
2. Иначе: пропустить importance logging. Записать в MLflow тег:
   mlflow.set_tag("feature_importance_available", "false")
```

**Уровень B — ResultAnalyzer (UAF, post-session):**
ResultAnalyzer из стадии 08 агрегирует feature importance по итерациям.
Это отдельный pass после завершения Claude Code.

### 5.2 Отслеживание между итерациями

Задача: понять как важность признаков меняется от итерации к итерации.

ResultAnalyzer при post-session анализе выполняет:

**Шаг FI-01: Сбор данных.**
Для каждого completed experiment run из MLflow:
- Скачать `feature_importance.csv` из artifacts если есть
- Извлечь `fi_rank_shadow_*` метрики
- Извлечь iteration номер из run tags

**Шаг FI-02: Построение importance matrix.**
Строится матрица: rows = features, columns = iterations.
Значение = normalized importance (0..1, sum=1 по iteration).
Если фича отсутствовала в итерации (shadow эксперимент не включал её) — NaN.

**Шаг FI-03: Stability score.**
Для каждой фичи которая присутствовала в >= 3 итерациях:
- `fi_stability = 1 - std(importance) / mean(importance)` (аналог cv_stability)
- Высокий fi_stability (> 0.7) означает устойчивую значимость фичи

**Шаг FI-04: Logирование в Session Summary Run.**
```python
mlflow.log_metric("fi_top1_stability", stability_of_top1_feature)
mlflow.log_metric("fi_shadow_acceptance_rate", accepted / total_shadow_experiments)
mlflow.log_artifact("feature_importance_matrix.csv", "feature_analysis")
mlflow.log_artifact("feature_importance_timeline.pdf", "feature_analysis")
```

**Шаг FI-05: Внесение в session_analysis.json.**
Секция `feature_importance` в session_analysis.json:
```json
{
  "feature_importance": {
    "available": true,
    "iterations_with_fi": 6,
    "top5_stable_features": [
      {"name": "income", "stability": 0.91, "mean_rank": 1.2},
      {"name": "credit_limit", "stability": 0.87, "mean_rank": 2.1},
      {"name": "hour_of_day", "stability": 0.83, "mean_rank": 3.4},
      {"name": "age", "stability": 0.79, "mean_rank": 3.8},
      {"name": "region_target_enc", "stability": 0.71, "mean_rank": 5.1}
    ],
    "unstable_features": [
      {"name": "age_x_credit_limit", "stability": 0.31, "mean_rank": 12.4,
       "note": "High variance — interaction feature, sensitive to data split"}
    ],
    "shadow_acceptance_rate": 0.6,
    "fi_method": "catboost_feature_importances"
  }
}
```

### 5.3 Hypothesis H-09 (из стадии 08) — feature importance сигнал

Стадия 08 зафиксировала правило гипотезы H-09:
"Если feature importance топ-3 изменились между итерациями — исследовать нестабильность."
Это правило применяется ResultAnalyzer при детекции нестабильных фич из шага FI-03.

При fi_stability < 0.5 для фичи из принятого feature set:
- ResultAnalyzer генерирует гипотезу H-09 в session_analysis.json
- Гипотеза передаётся в improvement_context.md для следующей сессии
- Текст гипотезы: "Feature {name} has low importance stability ({stability:.2f}).
  Consider: (a) removing it, (b) investigating data quality for this column,
  (c) checking for interaction effects."

### 5.4 MLflow артефакты по feature importance

Итоговые артефакты в Session Summary Run (путь в MLflow):

```
mlruns/{experiment_id}/{session_summary_run_id}/artifacts/
  feature_analysis/
    feature_importance_matrix.csv    <- матрица importance по итерациям
    feature_importance_timeline.pdf  <- graph: x=iteration, y=importance, line per feature
    top_features_comparison.csv      <- сводная таблица: feature, mean_importance, stability
  features/
    feature_registry.json            <- финальный реестр фич сессии
```

`feature_importance_matrix.csv` также логируется как отдельный MLflow artifact
в каждом experiment run (per-run версия, только для той итерации).

---

## 6. Feature Engineering в LaTeX/PDF отчёте

### 6.1 Условная секция

Секция Feature Engineering появляется в PDF отчёте всегда, так как Phase 2 обязательная.

Если Phase 2 была пропущена через `skip_feature_engineering: true` — секция заменяется одной строкой:
`\textit{Feature engineering phase was skipped (disabled in task.yaml).}`

Если `feature_registry.json` не существует (ни одного shadow шага не выполнено) — секция
отмечается как:
`\textit{Feature engineering phase completed with no shadow experiments (no applicable features detected).}`

### 6.2 Структура секции Feature Engineering в PDF

```latex
\section{Feature Engineering}

\subsection{Feature Engineering Summary}

% Таблица: общая статистика
\begin{tabular}{ll}
\hline
Parameter & Value \\
\hline
Original Feature Count & 44 \\
Shadow Experiments Run & 5 \\
Features Accepted & 3 \\
Features Rejected & 4 \\
Final Feature Count & 47 \\
Shadow Acceptance Rate & 60\% \\
\hline
\end{tabular}

\subsection{Shadow Feature Experiments}

% Таблица: все shadow experiments с результатами
\begin{tabular}{llrrl}
\hline
Step & Shadow Features & Delta & Threshold & Decision \\
\hline
2.1 & hour\_of\_day, day\_of\_week, is\_weekend & +0.0047 & 0.002 & \textcolor{green}{accepted} \\
2.2 & age\_x\_credit\_limit & -0.0012 & 0.002 & \textcolor{red}{rejected} \\
2.3 & region\_target\_enc & +0.0031 & 0.002 & \textcolor{green}{accepted} \\
2.4 & log1p\_income & +0.0011 & 0.002 & \textcolor{orange}{marginal} \\
\hline
\end{tabular}

\subsection{Feature Importance Stability}

% График: timeline важности фич по итерациям
\begin{figure}[h]
\includegraphics[width=\textwidth]{figures/feature_importance_timeline.pdf}
\caption{Feature importance dynamics across iterations. Stable features shown
         with solid lines, unstable with dashed.}
\end{figure}

% Таблица: топ-5 стабильных фич
\begin{tabular}{lrrl}
\hline
Feature & Mean Rank & Stability & Note \\
\hline
income & 1.2 & 0.91 & original \\
credit\_limit & 2.1 & 0.87 & original \\
hour\_of\_day & 3.4 & 0.83 & engineered (temporal) \\
age & 3.8 & 0.79 & original \\
region\_target\_enc & 5.1 & 0.71 & engineered (encoding) \\
\hline
\end{tabular}

\subsection{Accepted Feature Set}

Итоговый набор из 47 признаков (44 оригинальных + 3 принятых shadow):

\textbf{Accepted engineered features:}
\begin{itemize}
\item \texttt{hour\_of\_day}: temporal, source=transaction\_timestamp, $\Delta=+0.0047$
\item \texttt{day\_of\_week}: temporal, source=transaction\_timestamp, $\Delta=+0.0047$
\item \texttt{region\_target\_enc}: encoding, source=region, $\Delta=+0.0031$
\end{itemize}

\textbf{Rejected features (N=4):}
\begin{itemize}
\item \texttt{age\_x\_credit\_limit}: $\Delta=-0.0012$ (negative impact)
\item \texttt{income\_x\_credit\_limit}: $\Delta=+0.0001$ (below threshold)
\item ...
\end{itemize}

\subsection{DVC Tracking}

Feature registry versioned in git:
\begin{verbatim}
.uaf/sessions/sess_abc123/feature_registry.json
  git commit: a1b2c3d "feat: update feature_registry iter_5"
\end{verbatim}
```

### 6.3 Фигуры в секции Feature Engineering

**feature_importance_timeline.pdf:**
- Matplotlib line plot
- X: номер итерации
- Y: normalized importance (0..1)
- Линия per feature для топ-10 фич
- Цвет: оригинальные фичи — синий, принятые shadow — зелёный, отклонённые — красный
- Legend: имена фич
- Размер: 10x5 дюймов, 150 dpi

Генерируется ResultAnalyzer в шаге FI-04.
Сохраняется в `SESSION_DIR/report/figures/feature_importance_timeline.pdf`.

**top_features_bar.pdf** (опционально, если accepted features > 0):
- Horizontal bar chart
- Топ-15 фич по mean importance
- Цвет бара: оригинальная vs engineered
- Annotation: stability score рядом с баром

### 6.4 Что НЕ попадает в отчёт

Не включается в PDF секцию Feature Engineering:
- Полный список всех 44+ оригинальных признаков — вместо этого ссылка на data_schema.json
- Raw feature importance значения по каждой итерации — вместо этого feature_importance_matrix.csv
  как приложение (appendix)
- Код feature transformation — вместо этого ссылка на experiment.py конкретной итерации в MLflow

Отдельный Appendix в PDF содержит feature_importance_matrix.csv в виде таблицы
если итераций <= 10 (иначе — только ссылка на артефакт в MLflow).

---

## 7. Взаимодействие с другими компонентами

### 7.1 data_schema.json (стадия 05) — источник исходных метаданных

Секция `features` в data_schema.json используется как отправная точка для
генерации гипотез в разделе 3. ProgramMdGenerator читает:
- `features[col].type` — тип колонки
- `features[col].unique_count` — cardinality
- `features[col].stats.skewness` — для FG-N-01 гипотезы
- `features[col].correlation_with_target` — приоритизация гипотез
- `task_hints.potential_feature_engineering` — дополнительный контекст

Обратная связь: после feature engineering фазы UAF обновляет
`data_schema.json` секцию `features.selected_baseline`. Это единственный
случай когда UAF записывает в data_schema.json после его первоначального создания.

### 7.2 ValidationChecker (стадия 06) — leakage в engineered features

Новая проверка добавляется в ValidationChecker: **VR-FE-001**.

```
VR-FE-001: Shadow Feature Leakage Check
  Момент: post-run для experiment runs с method=shadow_feature_trick
  Проверка: target encoding должен вычисляться только на train split
             (нет fit_transform на val)
  Источник: MLflow run tags — проверяем наличие тега "target_enc_fit_on_val"
            (Claude Code должен установить этот тег если есть нарушение)
  Действие при нарушении: WARNING в program.md, не блокирует
  Severity: WARNING (не ERROR) — ResultAnalyzer помечает run как suspect_leakage
```

Target encoding — наиболее распространённый источник leakage при feature engineering.
VR-FE-001 не может детектировать leakage автоматически (UAF не видит код),
но создаёт явный сигнальный механизм.

### 7.3 ResultAnalyzer (стадия 08) — H-09 и feature stability

Связь описана в разделе 5.3. Дополнительно:

ResultAnalyzer в шаге param-metric корреляции (шаг 5 из 8-шагового алгоритма)
дополнительно коррелирует `fi_rank_shadow_*` метрики с `{metric_name}_val`:
- Если rank тесно коррелирует с метрикой — shadow фича вероятно полезна
- Если нет корреляции — фича нейтральна или вредна

Результат добавляется в секцию `feature_importance.correlation_analysis`
в `session_analysis.json`.

### 7.4 SmokeTestRunner (стадия 09) — валидация feature engineering кода

Дополнительный smoke test для итераций с `phase_type: feature_engineering`:

```
ST-12 (новый, для feature engineering итераций):
  Shadow baseline run_id валиден
  Проверка: baseline_run_id из program.md существует в MLflow
            (mlflow.get_run(run_id) не бросает исключение)
  Блокирует: всегда
  Мотивация: если baseline_run_id не существует, shadow comparison невозможен,
             вся итерация бессмысленна
```

ST-12 добавляется к списку тестов ST-01..ST-11 только при наличии
`method: shadow_feature_trick` в текущем шаге program.md.

### 7.5 BudgetController — feature engineering итерации

Shadow Feature Trick требует двух обучений (baseline + candidate).
Это влияет на подсчёт итераций в BudgetController.

Решение: shadow experiment считается как **одна итерация**, не как две.
Обоснование: это единый исследовательский шаг с одной гипотезой.
Двойное обучение — внутренняя реализация, не два отдельных эксперимента.

В budget_status.json:
```json
{
  "budget": {
    "completed_iterations": 3,
    "runs_per_iteration": {
      "phase1_steps": 5,
      "phase2_shadow_step": 2,
      "current_phase": "feature_engineering"
    }
  }
}
```

---

## 8. Антипаттерны и ограничения

### 8.1 Что UAF не делает в feature engineering

- **Не генерирует фичи автоматически**: UAF предлагает гипотезы, Claude Code реализует.
  Автоматическая генерация нарушает antigoal 5 (читаемость > оптимизация).

- **Не использует AutoFeatureEngineering библиотеки** (featuretools, tsfresh, автоматические
  polynomial features): они создают десятки/сотни фич неинтерпретируемо.
  Antigoal 5. Исключение: если пользователь явно указал в task.yaml
  `research_preferences.allow_automated_features: true`.

- **Не хранит фичи между сессиями в Feature Store**: feature_registry.json —
  только для текущей и следующей (--resume) сессии.

- **Не применяет target encoding глобально**: только для колонок с cardinality > 10.
  Target encoding при низкой cardinality — переобучение на train.

### 8.2 Известные ограничения

- **Shadow Feature Trick требует стабильного baseline run**: если Phase 1 выдала
  нестабильные результаты (cv_stability < 0.1), то дельта от shadow фич ненадёжна.
  ResultAnalyzer помечает это как предупреждение в session_analysis.json.

- **Feature importance не работает для нейросетей без SHAP**: для PyTorch моделей
  FI-01..FI-05 шаги выдают неполные данные. Секция в отчёте отмечается как partial.

- **Temporal features требуют правильного gap в time-series split**: если схема
  валидации time_series_split, то target encoding и групповые статистики могут
  создавать leakage из будущего. VR-FE-001 не ловит этот случай автоматически.
  Инструкция в program.md: explicit warning для time-series задач.

---

## 9. Ключевые решения и обоснования

### R-10-01: Shadow Feature Trick как основной механизм

**Решение:** shadow (параллельное сравнение baseline vs candidate) вместо
inline добавления признаков.

**Альтернатива:** просто добавлять фичи и смотреть на итоговую метрику.
**Проблема альтернативы:** нельзя изолировать вклад конкретных фич при одновременном
изменении гиперпараметров. При итеративном исследовании это смешивает сигналы.

**Обоснование выбора:** Shadow trick сохраняет воспроизводимый baseline (один run_id),
изолирует вклад фич, даёт явную дельту для принятия решения. Стоимость — двойное
обучение, что приемлемо для tabular задач (дерево обучается за секунды).

### R-10-02: Feature Store не нужен

**Решение:** Feature Registry как минималистичный JSON артефакт сессии.
**Обоснование:** Feature Store решает проблемы масштаба команды и online/offline
consistency. UAF — одиночный исследователь, нет serving, нет команды.
Добавление Feast/Hopsworks — overkill, нарушает принцип "тонкой оболочки".

### R-10-03: ProgramMdGenerator генерирует детерминированные гипотезы без LLM

**Решение:** правила FG-T-*, FG-N-*, FG-I-*, FG-C-*, FG-NLP-*, FG-CV-* реализуются
как код (if/else по типам из data_schema.json), не как LLM prompt.

**Альтернатива:** LLM генерирует гипотезы о фичах на основе описания данных.
**Проблема альтернативы:** LLM генерирует нереализуемые или неуникальные гипотезы,
тратит дополнительный API вызов, результат не воспроизводим.

**Обоснование:** детерминированные правила — надёжнее, воспроизводимо, дешевле.
LLM вызов сохраняется один (ProgramMdGenerator), но контент feature гипотез
генерируется программно по data_schema.json.

### R-10-04: Phase 2 (Feature Engineering) обязательная; максимум 5 shadow steps

**Решение:** Phase 2 выполняется всегда после Phase 1, независимо от результатов Phase 1.
Лимит — 5 шагов в Phase 2.

**Обоснование обязательности:** feature engineering — стандартный этап исследовательского
цикла. Достижение success_metric_threshold в Phase 1 не означает, что feature engineering
не даст дополнительного улучшения. Пропуск Phase 2 должен быть явным решением пользователя
(`skip_feature_engineering: true`), а не автоматическим поведением системы.

**Обоснование лимита 5 шагов:** Парето-принцип — 20% ключевых признаков дают 80% прироста.
Более 5 shadow экспериментов в одной сессии — признак что задача требует
отдельной полноценной feature analysis сессии, а не итеративного исследования.
При необходимости пользователь запускает новую сессию с task.yaml
`research_preferences.max_feature_engineering_steps: 10`.

### R-10-05: feature_registry.json в git, не DVC

**Решение:** git commit для feature_registry.json (< 1 МБ).
**Обоснование:** история изменений реестра важна для понимания хода исследования.
git log на feature_registry.json даёт читаемую историю решений по фичам.
DVC добавляет overhead для мелких JSON файлов без выигрыша.

---

## 10. Antigoal compliance

| Antigoal | Механизм соблюдения в стадии 10 |
|---|---|
| 1 (не AutoML для prod) | feature_registry.json — исследовательский артефакт, не production feature store |
| 2 (одобрение перед запуском) | Phase 2 добавляется в program.md, проходит HumanOversightGate как часть плана |
| 3 (не скрывать failed) | Rejected shadow features логируются в MLflow и попадают в отчёт |
| 4 (не модифицировать данные) | UAF не создаёт новые фичи сам; фичи создаёт Claude Code в SESSION_DIR |
| 5 (читаемость > оптимизация) | Детерминированные правила вместо AutoFeatureEngineering; лимит 5 шагов |
| 6 (бюджет) | Shadow step = 1 итерация; BudgetController учитывает runs_per_iteration=2 |

---

## STAGE COMPLETE

Стадия 10-features завершена.

Следующая стадия: 11-measurement.
Условие разблокировки: выполнено (стадии 01-10 complete или embedded).

Ключевые решения зафиксированы: R-10-01..R-10-05.
Артефакт: `docs/stage-10-features.md`.
