# Стадия 06: Validation

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-problem (COMPLETE), 02-research (COMPLETE),
03-design-doc v2.0 (COMPLETE), 04-metrics (COMPLETE), 05-data (COMPLETE)

---

## 0. Позиция стадии Validation в UAF

Стадия Validation имеет двойное значение в UAF:

**Смысл 1 — Validation как схема разбиения данных для ML-экспериментов.**
Когда Claude Code обучает модель внутри сессии, ему нужна схема валидации:
как делить данные, сколько фолдов, как избежать leakage между фолдами.
Эта схема задаётся пользователем в `task.yaml` и передаётся Claude Code
через `program.md`. UAF проверяет корректность схемы автоматически.

**Смысл 2 — Validation как проверка самого UAF.**
Прежде чем запускать UAF в реальных исследованиях, нужно убедиться, что
сам UAF работает корректно: компоненты инициализируются, данные передаются
правильно, бюджет считается, отчёт генерируется. Это acceptance criteria UAF.

Оба смысла фиксируются в этой стадии и не смешиваются.

---

## 1. Схемы валидации по типам задач

### 1.1 Принцип выбора схемы

UAF не выбирает схему валидации автоматически. Пользователь задаёт её
в `task.yaml`. UAF проверяет корректность выбранной схемы относительно
типа задачи и размера данных. Если схема не задана, применяется дефолт
из таблицы ниже.

Решение о схеме принимается один раз — при инициализации сессии.
Смена схемы валидации в ходе сессии не допускается (нарушает сравнимость runs).

### 1.2 Схемы по типу задачи

#### Tabular Classification / Tabular Regression

| Условие | Схема | Параметры по умолчанию |
|---------|-------|------------------------|
| N < 1 000 строк | StratifiedKFold | k=10, shuffle=True, seed=42 |
| 1 000 <= N < 50 000 | StratifiedKFold | k=5, shuffle=True, seed=42 |
| N >= 50 000 | Holdout | train=0.8 / val=0.1 / test=0.1 |
| Сильный дисбаланс классов (min_class_ratio < 0.05) | StratifiedKFold | k=5, stratify=True обязательно |

Для регрессии: те же пороги, KFold вместо StratifiedKFold, стратификация
по квантилям (pd.qcut, bins=k) если распределение target сильно скошено
(skewness > 2.0 по данным из data_schema.json).

**Обоснование порогов:**
- N < 1 000: holdout даёт нестабильные оценки. k=10 уменьшает дисперсию.
- N >= 50 000: k-fold избыточно дорог при правильном holdout split.
- Граница 50 000 — компромисс между вычислительной стоимостью и надёжностью оценки.

#### NLP задачи

| Тип NLP задачи | Схема | Примечания |
|----------------|-------|------------|
| Text classification | StratifiedKFold, k=5 | По метке класса |
| Sequence labeling (NER, POS) | GroupKFold, k=5 | Groups = document_id |
| Text generation / summarization | Holdout | train=0.9 / val=0.05 / test=0.05 |
| Question Answering | Holdout | train=0.85 / val=0.1 / test=0.05 |
| Sentence embeddings / retrieval | Holdout | train=0.8 / val=0.1 / test=0.1 |

**Ключевое правило для NLP:** если в датасете есть `document_id` или
`source_id` — документный split обязателен. Случайный split при наличии
нескольких предложений/фрагментов одного документа = гарантированный leakage.
UAF проверяет наличие таких колонок через data_schema.json (leakage audit LA-07).

#### CV задачи

| Тип CV задачи | Схема | Примечания |
|---------------|-------|------------|
| Image classification | Holdout | train=0.8 / val=0.1 / test=0.1 |
| Object detection | Holdout | train=0.8 / val=0.1 / test=0.1 |
| Segmentation | Holdout | train=0.8 / val=0.1 / test=0.1 |
| Few-shot (N < 100 samples/class) | StratifiedKFold | k=5, по классам |

**Ключевое правило для CV:** если в датасете есть `patient_id`,
`session_id`, `subject_id` — GroupKFold по этому полю обязателен.
Особенно критично для медицинских изображений.

#### Time-series задачи

Time-series — отдельный класс. Здесь запрещён любой shuffle.
Применяется только walk-forward схема.

| Подтип | Схема | Параметры |
|--------|-------|-----------|
| Single time series | TimeSeriesSplit | n_splits=5, gap=0 |
| Multiple time series (panel) | TimeSeriesSplit по каждому entity | n_splits=5 |
| С gap между train и val (реалистичный) | TimeSeriesSplit | gap = forecast_horizon |
| Expanding window | ExpandingWindowSplit | min_train_size=задаётся в task.yaml |

**Обязательное правило для time-series:**
`gap >= forecast_horizon`. Если `forecast_horizon` задан в task.yaml
и gap < forecast_horizon — UAF выдаёт ошибку валидации VS-TS-001 (блокирующая).

**Запрещено для time-series:** shuffle=True в любом сплиттере.
UAF проверяет это автоматически.

#### RecSys задачи

| Сценарий | Схема | Примечания |
|----------|-------|------------|
| Cold-start evaluation | LeaveOneOut по user_id | Последнее взаимодействие в test |
| Temporal split | Holdout по времени | cutoff_date задаётся в task.yaml |
| Cross-validation для ранжирования | GroupKFold | Groups = user_id |

#### RL задачи

RL задачи не используют традиционный train/val/test split.
Валидация происходит через оценочные эпизоды (`eval_episodes`).
В task.yaml задаётся:
- `eval_episodes`: число эпизодов для оценки
- `eval_frequency`: каждые N шагов обучения
- `test_episodes`: финальная оценка после обучения

---

## 2. Задание схемы валидации в task.yaml

### 2.1 Структура секции validation в task.yaml

```yaml
validation:
  scheme: auto          # auto | holdout | kfold | stratified_kfold |
                        # group_kfold | time_series_split | leave_one_out
  # Параметры зависят от выбранной схемы

  # Для holdout:
  # train_ratio: 0.8
  # val_ratio: 0.1
  # test_ratio: 0.1

  # Для kfold / stratified_kfold / group_kfold:
  # n_splits: 5
  # shuffle: true       # false обязательно для time_series_split
  # seed: 42

  # Для time_series_split:
  # n_splits: 5
  # gap: 0              # int >= 0, в единицах наблюдений
  # forecast_horizon: 1 # UAF проверяет: gap >= forecast_horizon

  # Для leave_one_out (RecSys):
  # group_col: user_id  # колонка для группировки

  # Универсальные параметры:
  seed: 42              # seed для воспроизводимости, передаётся Claude Code
  test_holdout: true    # выделять ли test set (true) или только train/val (false)
  test_ratio: 0.1       # доля test, если test_holdout: true
  stratify_col: null    # колонка для стратификации (null = target)
  group_col: null       # колонка для GroupKFold
```

**Режим `auto`:** UAF выбирает схему по таблицам из раздела 1.2,
исходя из `task.type`, размера датасета и наличия временных колонок.
Выбранная схема логируется в MLflow Planning Run и отображается в отчёте.

### 2.2 Примеры для каждого типа задач

**Tabular classification, небольшой датасет:**
```yaml
task:
  type: tabular_classification
validation:
  scheme: auto
  seed: 42
  test_holdout: true
  test_ratio: 0.1
```

**Time-series с forecast horizon:**
```yaml
task:
  type: time_series
  forecast_horizon: 7    # дней вперёд
validation:
  scheme: time_series_split
  n_splits: 5
  gap: 7                 # >= forecast_horizon
  seed: 42
  test_holdout: true
  test_ratio: 0.15
```

**NLP с document structure:**
```yaml
task:
  type: nlp_classification
validation:
  scheme: group_kfold
  n_splits: 5
  group_col: document_id
  seed: 42
  test_holdout: true
  test_ratio: 0.1
```

**RecSys temporal:**
```yaml
task:
  type: recsys
validation:
  scheme: holdout
  cutoff_date: "2024-01-01"   # train до, val и test после
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42
```

---

## 3. Передача схемы валидации в program.md

### 3.1 Что UAF записывает в program.md

ProgramMdGenerator включает в `program.md` отдельный раздел
`## Validation Scheme` после секции `## Task Description`.

Содержимое раздела:

```markdown
## Validation Scheme

**Scheme:** {scheme_name}
**Resolved by:** {auto | user-specified}
**Parameters:**
- n_splits: {n_splits}        # или train/val/test ratios для holdout
- shuffle: {true|false}
- seed: {seed}
- group_col: {null | column_name}
- stratify_col: {null | column_name}
- test_holdout: {true|false}
- test_ratio: {ratio}
- gap: {gap}                   # только для time_series_split

**Validation constraints (enforced by UAF):**
{список активных ограничений, сформированный UAF автоматически}

**Critical warnings:**
{список WARNING из LeakageAudit и AdversarialValidation, если есть}

**Execution Instructions — Validation:**
- Используй схему ТОЧНО как задано выше. Не меняй схему между runs.
- seed={seed} должен быть зафиксирован в numpy, random, torch, sklearn.
- Логируй в MLflow: `validation_scheme`, `n_splits`, `fold_idx` (для k-fold).
- Для k-fold: логируй метрику каждого фолда отдельно, затем mean ± std.
- Test set НЕ используется в ходе эксперимента. Только в финальной оценке.
- Если group_col задан — GroupKFold обязателен. Случайный split запрещён.
```

### 3.2 Как Claude Code использует схему

Claude Code получает схему в `program.md` как часть Execution Instructions —
то есть как явное требование, а не рекомендацию.

Ожидаемое поведение Claude Code:

1. Реализует указанный сплиттер из sklearn/pytorch (StratifiedKFold, TimeSeriesSplit, и т.д.)
2. Фиксирует seed везде: `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`
3. Для k-fold: каждый фолд = отдельный MLflow run с тегом `fold_idx`
4. Логирует `validation_scheme` как MLflow param в каждом Experiment Run
5. Test set не трогает до финальной оценки

Это поведение проверяется UAF автоматически через ValidationChecker
(см. раздел 4).

---

## 4. Автоматические проверки UAF

### 4.1 Pre-session checks (до запуска Claude Code)

Выполняются после DataLoader, до HumanOversightGate.
Блокирующие ошибки (VS-*) = сессия не запускается.

| Код | Проверка | Тип | Условие блокировки |
|-----|----------|-----|--------------------|
| VS-T-001 | train_ratio + val_ratio + test_ratio == 1.0 | ERROR | Всегда если holdout |
| VS-T-002 | train_ratio >= 0.5 | ERROR | train < 50% |
| VS-T-003 | val_ratio >= 0.05 | ERROR | val < 5% |
| VS-T-004 | Минимальный размер val set: >= 30 строк | ERROR | val_abs < 30 |
| VS-T-005 | Минимальный размер train set: >= 100 строк | ERROR | train_abs < 100 |
| VS-K-001 | n_splits >= 2 | ERROR | k < 2 |
| VS-K-002 | n_splits <= 20 | WARNING | k > 20 (редкость, но допустимо) |
| VS-K-003 | Каждый фолд содержит >= 1 sample каждого класса | ERROR | Только StratifiedKFold |
| VS-S-001 | shuffle=False для time_series_split | ERROR | shuffle=True + time_series_split |
| VS-S-002 | gap >= forecast_horizon | ERROR | gap < forecast_horizon |
| VS-G-001 | group_col присутствует в данных | ERROR | Колонка не найдена |
| VS-G-002 | Количество уникальных групп >= n_splits | ERROR | groups < n_splits |
| VS-L-001 | Нет overlap между train и val строками | ERROR | Любой row_id в обоих |
| VS-L-002 | Нет overlap между train и test строками | ERROR | Любой row_id в обоих |
| VS-L-003 | Target не присутствует в feature columns | ERROR | Уже из LA-01, дублируется |
| VS-A-001 | AdversarialValidation AUC < 0.85 | ERROR | Блокировка с override |
| VS-A-002 | AdversarialValidation AUC 0.6..0.85 | WARNING | Предупреждение в program.md |
| VS-C-001 | seed задан | WARNING | seed=null (воспроизводимость нарушена) |
| VS-C-002 | Схема совместима с task.type | ERROR | Несовместимость |

**Стратификация для classification:**
Если `task.type = tabular_classification` и `min_class_ratio < 0.01`
(менее 1% миноритарного класса): ERROR VS-K-003 даже при k=5, т.к.
некоторые фолды могут не содержать ни одного примера миноритарного класса.

**Матрица совместимости task.type + scheme:**

| task.type | holdout | kfold | stratified_kfold | group_kfold | time_series_split |
|-----------|---------|-------|-----------------|-------------|-------------------|
| tabular_classification | OK | OK | OK (рек.) | OK | WARN |
| tabular_regression | OK | OK | OK | OK | WARN |
| nlp_* | OK | OK | OK | OK (рек.) | WARN |
| cv_* | OK | OK | OK | OK | ERROR |
| time_series | ERROR | ERROR | ERROR | ERROR | OK (обяз.) |
| recsys | OK | WARN | WARN | OK (рек.) | OK |
| rl | N/A | N/A | N/A | N/A | N/A |

### 4.2 Post-run checks (после каждого Experiment Run)

ValidationChecker читает MLflow после завершения каждого Experiment Run.
Нарушения не блокируют сессию — пишутся как WARNING в Session Summary.

| Код | Что проверяется | Источник данных |
|-----|-----------------|-----------------|
| VR-001 | `validation_scheme` залогирован как MLflow param | MLflow Experiment Run |
| VR-002 | `fold_idx` залогирован для k-fold runs | MLflow тег |
| VR-003 | `seed` залогирован | MLflow param |
| VR-004 | Метрика залогирована на каждом фолде отдельно | MLflow metrics |
| VR-005 | Для k-fold: логирован mean ± std | MLflow metrics: `{metric}_mean`, `{metric}_std` |
| VR-006 | Test metric залогирована отдельно (не смешана с val) | MLflow metric key prefix |
| VR-007 | `n_samples_train`, `n_samples_val` залогированы | MLflow params |

### 4.3 Leakage между фолдами

Leakage между фолдами — наиболее критичный тип ошибки валидации.
Он не всегда очевиден и не всегда обнаруживается простой проверкой.

**Что UAF проверяет автоматически (pre-session):**

1. **Row overlap** (VS-L-001, VS-L-002): индексы строк train и val/test
   не пересекаются. Проверяется через set intersection по row indices.

2. **Group overlap** (если group_col задан): проверяется что группы в val
   не присутствуют в train. Именно это и решает GroupKFold.

3. **Temporal ordering** (для time_series_split): проверяется монотонность
   временного индекса. Если данные не отсортированы — ERROR VS-S-003.

**Что UAF обнаружить не может (передаётся как hints в program.md):**

- Leakage через feature engineering (например, mean encoding целевой переменной
  по всему датасету до сплита). Это обнаруживается только анализом кода.
  UAF добавляет hint: "Применяй fit на train, transform на val/test отдельно."

- Leakage через preprocessing (например, StandardScaler fit на полном датасете).
  UAF добавляет hint: "Все preprocessors должны быть fit только на train fold."

- Leakage через внешние данные (например, использование future знаний из
  внешней таблицы). UAF добавляет hint только если обнаружил timestamp колонки.

### 4.4 Стратификация: когда и как

Стратификация обязательна при:
- `task.type = tabular_classification` и `min_class_ratio < 0.2`
- NLP classification с дисбалансом классов

Стратификация запрещена при:
- Любом time_series split (нарушает временной порядок)
- Задачах регрессии с непрерывным target (применять только через pd.qcut)

UAF проверяет корректность через VS-K-003 и VS-S-001.

---

## 5. Связь схемы валидации с BudgetController

### 5.1 Влияние схемы на оценку бюджета

Схема валидации прямо влияет на стоимость каждой итерации в BudgetController.
BudgetController должен знать о схеме, чтобы корректно считать "одну итерацию".

**Определение одной итерации:**

| Схема | Что считается одной итерацией |
|-------|-------------------------------|
| Holdout | Один train + eval цикл = 1 итерация |
| KFold (k=5) | Один полный k-fold round (5 фолдов) = 1 итерация |
| TimeSeriesSplit (n=5) | Один полный walk-forward round (5 фолдов) = 1 итерация |
| LeaveOneOut | Один LOO round = 1 итерация |

Это важно для budget_mode=fixed: пользователь задаёт `max_iterations: 10`.
Если схема = KFold(k=5), то 10 итераций = 50 MLflow runs.
BudgetController считает iterations, не runs.

### 5.2 Обновление budget_status.json

Поле `validation_scheme` добавляется в budget_status.json:

```json
{
  "session_id": "sess_20260319_001",
  "iterations_used": 3,
  "iterations_limit": 10,
  "budget_exhausted": false,
  "hard_stop": false,
  "validation_scheme": "stratified_kfold",
  "n_splits": 5,
  "runs_per_iteration": 5,
  "total_runs_used": 15,
  "convergence_status": "running",
  "last_updated": "2026-03-19T12:00:00Z"
}
```

Claude Code читает `runs_per_iteration` чтобы понимать, сколько MLflow runs
генерирует каждая итерация (и не считать фолды отдельными итерациями).

### 5.3 Первичная оценка бюджета

При инициализации сессии UAF делает первичную оценку времени на итерацию:

```
estimated_iter_time = base_time_estimate * scheme_multiplier

scheme_multiplier:
  holdout:             1.0
  kfold (k=5):         4.8  (не ровно 5, т.к. train меньше)
  stratified_kfold:    4.8
  group_kfold:         4.8
  time_series_split:   4.5  (walk-forward, train растёт)
  leave_one_out:       N    (N = число уникальных групп)
```

Оценка базового времени (`base_time_estimate`) берётся из `task.yaml`:
```yaml
budget:
  estimated_iter_time_minutes: 5  # пользователь задаёт вручную
```

Если не задано — UAF пишет WARNING в лог: "estimated_iter_time не задано,
BudgetController не может корректно оценить время сессии."

Эта оценка отображается пользователю на HumanOversightGate:
"Ожидаемое время сессии: ~N минут (10 итераций x 5 минут x 4.8 = 240 минут)".

### 5.4 Dynamic mode и validation

В dynamic mode (convergence-based stop) BudgetController проверяет сходимость
по метрике после каждой ИТЕРАЦИИ, не после каждого Run.

Для k-fold: метрика итерации = `{metric}_mean` из MLflow.
Только mean используется для convergence check. Std не проверяется
(но логируется для информации).

---

## 6. Логирование в MLflow

### 6.1 Planning Run

Схема валидации логируется в Planning Run при инициализации сессии.

**Обязательные поля:**

```python
mlflow.log_params({
    "validation.scheme": scheme,           # "stratified_kfold"
    "validation.n_splits": n_splits,       # 5
    "validation.shuffle": shuffle,         # True/False
    "validation.seed": seed,               # 42
    "validation.test_holdout": test_holdout,  # True/False
    "validation.test_ratio": test_ratio,   # 0.1
    "validation.gap": gap,                 # 0 (для time_series)
    "validation.group_col": group_col,     # null или "document_id"
    "validation.stratify_col": stratify_col,  # null или "target"
    "validation.resolved_by": resolved_by, # "auto" или "user-specified"
})

mlflow.log_dict(validation_checks_result, "validation_checks.json")
```

`validation_checks.json` содержит результат всех VS-* проверок:
статус (PASS/WARN/ERROR), код, сообщение.

**Тег:**
```python
mlflow.set_tag("validation.scheme", scheme)
```

### 6.2 Experiment Run (каждый run Claude Code)

Claude Code обязан логировать в каждом Experiment Run:

```python
# Обязательные params (проверяются VR-001, VR-003, VR-007)
mlflow.log_params({
    "validation_scheme": scheme,       # дублирует Planning Run для поиска
    "seed": seed,
    "n_samples_train": n_train,
    "n_samples_val": n_val,
})

# Для k-fold: дополнительно
mlflow.set_tag("fold_idx", fold_idx)          # "0", "1", "2", ...
mlflow.set_tag("is_cv_fold", "true")

# Метрики фолдов (для k-fold, k=5):
mlflow.log_metric("roc_auc_fold_0", 0.82)
mlflow.log_metric("roc_auc_fold_1", 0.84)
...
mlflow.log_metric("roc_auc_mean", 0.83)
mlflow.log_metric("roc_auc_std", 0.009)

# Метрика val vs test — разные ключи
mlflow.log_metric("roc_auc_val", val_score)   # val = используется для выбора
mlflow.log_metric("roc_auc_test", test_score) # test = только финальная оценка
```

**Критичное правило:** `roc_auc_test` логируется ТОЛЬКО в финальном run
наилучшей модели. Не в каждом run. Это инструкция в Execution Instructions
program.md.

### 6.3 Session Summary Run

После завершения сессии ReportGenerator читает MLflow и логирует в
Session Summary Run:

```python
mlflow.log_params({
    "validation.scheme_final": scheme,
    "validation.iterations_completed": n_iter,
    "validation.total_runs": n_runs,
    "validation.runs_per_iteration": n_splits if kfold else 1,
})

mlflow.log_metrics({
    "validation.best_val_score": best_val,
    "validation.best_test_score": best_test,
    "validation.val_test_delta": abs(best_val - best_test),  # overfitting proxy
    "validation.cv_stability": cv_stability,  # std/mean для k-fold, иначе null
    "validation.checks_passed": n_checks_passed,
    "validation.checks_failed": n_checks_failed,
    "validation.checks_warned": n_checks_warned,
})

mlflow.log_dict(validation_summary, "validation_summary.json")
```

**`validation.val_test_delta`** — прокси для overfitting. Если delta > 0.05
(5 процентных пунктов) — WARNING в отчёт.

**`validation.cv_stability`** — только для k-fold. std/mean метрики по фолдам.
Если > 0.1 (10% вариации) — WARNING в отчёт.

---

## 7. LaTeX/PDF отчёт: секция Validation

### 7.1 Структура секции

Секция `Validation Strategy` в PDF отчёте располагается после
`Data Overview` и до `Experiment Results`.

Содержимое (всегда присутствует):

```
Section 3: Validation Strategy

3.1 Validation Scheme
  Таблица: scheme, parameters, resolved_by
  Текст: "Схема выбрана {автоматически / пользователем} на основании..."

3.2 Pre-Session Validation Checks
  Таблица: код проверки, описание, статус (PASS/WARN/ERROR)
  Если были WARN или ERROR с override: цветовая пометка

3.3 Cross-Validation Results (только для k-fold)
  Таблица: fold_idx, n_train, n_val, {metric}, лучший
  Строка "Mean ± Std" внизу

3.4 Val vs Test Comparison
  Таблица: best model, val_score, test_score, delta
  Если delta > 0.05: footnote "Возможный overfitting"

3.5 Validation Stability (только для k-fold)
  cv_stability metric с интерпретацией
```

### 7.2 Генерация через ReportGenerator

ReportGenerator использует три LLM-вызова (из стадии 03).
Секция Validation генерируется в первом вызове (структурная часть отчёта)
как шаблонный LaTeX, не через LLM. LLM участвует только в трактовке
аномалий (если delta > 0.05 или cv_stability > 0.1).

Шаблон секции — Jinja2, данные из:
- `validation_checks.json` из MLflow Planning Run
- `validation_summary.json` из MLflow Session Summary Run
- k-fold метрики из Experiment Runs

### 7.3 Правила оформления в LaTeX

```latex
% Таблица проверок (пример):
\begin{tabular}{llll}
\toprule
Code & Description & Status \\
\midrule
VS-T-001 & train+val+test == 1.0 & \textcolor{green}{PASS} \\
VS-K-003 & Min samples per class & \textcolor{orange}{WARN} \\
\bottomrule
\end{tabular}

% Цветовая схема:
% PASS  -> \textcolor{green}{PASS}
% WARN  -> \textcolor{orange}{WARN}
% ERROR -> \textcolor{red}{ERROR}
% (переопределять xcolor не нужно, tectonic поддерживает dvipsnames)
```

---

## 8. Граничные случаи

### 8.1 Очень маленький датасет (N < 200)

N < 200 при holdout:
- train_ratio=0.8 -> 160 строк train, 20 val. Если val < 30 -> ERROR VS-T-004.
- Рекомендация: переключиться на k-fold.

N < 200 при k-fold:
- k=5: по 40 строк на fold. Критично только при дисбалансе классов.
- UAF пишет WARNING: "Малый датасет, cv_stability может быть высоким."

N < 50 — блокирующая проверка: ERROR VS-T-005 ("train set < 100 строк").
Сессия не запускается.

### 8.2 Одноклассовые данные

Если в данных только один класс (target.n_unique == 1):
ERROR VS-K-003 (невозможна стратификация). Сессия не запускается.
Это также должен был обнаружить DataLoader (стадия 05).

### 8.3 Сильный дисбаланс

min_class_ratio < 0.01: возможно, что в некоторых фолдах StratifiedKFold
не даст ни одного примера миноритарного класса.
UAF проверяет через симуляцию: создаёт объект StratifiedKFold, проверяет
что split() не вызывает ValueError. Если вызывает — ERROR VS-K-003.

### 8.4 Огромный датасет и k-fold

N >= 5 000 000 и k-fold: каждый фолд = 4 000 000 строк train.
UAF пишет WARNING VS-K-002-LARGE:
"k-fold на датасете > 5М строк может быть очень медленным.
Рекомендуется holdout или n_splits=3."
Это не блокирует сессию.

### 8.5 Отсутствие test set

Если `test_holdout: false` (пользователь явно отключил):
UAF логирует WARNING VS-C-003: "Test set отключён. Финальная оценка
будет на val set. Возможен optimistic bias."
Это допустимо для исследовательских задач где test set не нужен.

### 8.6 time-series без явного временного индекса

Если `task.type = time_series` но `scheme = auto` и в данных нет явной
datetime колонки: ERROR VS-TS-002.
UAF не умеет определить порядок строк без временного индекса.
Пользователь должен явно задать или `time_col` в task.yaml, или схему holdout.

---

## 9. ValidationChecker — компонент UAF

### 9.1 Позиция в архитектуре

ValidationChecker — новый компонент, не упомянутый в стадии 03.
Реализован как часть ResearchSessionController (не отдельный класс верхнего уровня).
Он встроен в DataFlow стадии 05:

```
DVCSetup -> DataLoader -> LeakageAudit -> AdversarialValidation
         -> [ValidationChecker pre-session] -> ProgramMdGenerator
         -> HumanOversightGate -> ClaudeCodeRunner
         -> [ValidationChecker post-run, каждый run]
         -> BudgetController -> RuffEnforcer
         -> [ValidationChecker post-session] -> ReportGenerator
```

### 9.2 Ответственность ValidationChecker

| Момент | Что делает |
|--------|-----------|
| Pre-session | Все VS-* проверки. Блокирует или предупреждает. Пишет validation_checks.json |
| Post-run | Все VR-* проверки. Читает MLflow. Добавляет записи в validation_warnings.log |
| Post-session | Агрегирует VR-* нарушения. Логирует в Session Summary Run. Передаёт данные ReportGenerator |

### 9.3 Вывод ValidationChecker

Pre-session вывод на терминал (до HumanOversightGate):

```
=== Validation Checks ===
[PASS] VS-T-001: train+val+test == 1.0
[PASS] VS-T-004: val set >= 30 rows (val_abs=1234)
[WARN] VS-A-002: AdversarialValidation AUC=0.72 (threshold 0.6-0.85)
       Data distributions differ between train and val.
       Hint added to program.md.
[PASS] VS-L-001: No row overlap train/val

Scheme: stratified_kfold(k=5, seed=42)
Resolved by: auto (N=12340, task=tabular_classification)

Proceed to HumanOversightGate? [y/n]:
```

---

## 10. Antigoal-проверка

Схема валидации не нарушает antigoals из стадии 01:

| Antigoal | Проверка |
|----------|---------|
| 1. Не AutoML для production | Схема валидации — исследовательская. Test set = офлайн оценка, не A/B тест |
| 2. Не запускает без одобрения | ValidationChecker выполняется ДО HumanOversightGate. Результаты отображаются при одобрении |
| 3. Не скрывает неудачные эксперименты | VR-* логируются в Session Summary даже для failed runs |
| 4. Не модифицирует данные | ValidationChecker только читает данные (split indices только в памяти) |
| 5. Не оптимизирует под конкретный датасет | Схема задана один раз и не меняется в ходе сессии |
| 6. Не превышает бюджет | BudgetController знает о runs_per_iteration (kfold) |

---

## Ключевые решения стадии 06

### R-06-01: Схема валидации задаётся пользователем, UAF проверяет корректность

UAF не выбирает схему автоматически в production — он предлагает auto-режим
как дефолт, но пользователь может переопределить. Это соответствует antigoal 5
(не оптимизирует под конкретный датасет) и antigoal 2 (человек контролирует).

### R-06-02: "Одна итерация" = полный round, не один fold

BudgetController считает итерации (research cycles), не MLflow runs.
Для k-fold одна итерация = k runs. Это позволяет сравнивать бюджет
между сессиями с разными схемами валидации.

### R-06-03: Test set изолирован до финальной оценки

Test metric логируется ТОЛЬКО для наилучшей модели в финальном run.
Это архитектурное решение, не просто best practice. Нарушение = potential
antigoal 5 violation (оптимизация под test = утечка в итеративной сессии).

### R-06-04: ValidationChecker встроен в ResearchSessionController

Не отдельный компонент верхнего уровня. Это сохраняет архитектуру стадии 03
(6 основных компонентов) и не раздувает список.

### R-06-05: Нет автоматического repair схемы

Если пользователь задал невалидную схему, UAF не исправляет её
автоматически. Он выдаёт ERROR с объяснением и требует исправления task.yaml.
Автоматический repair нарушал бы antigoal 2.

---

## STAGE COMPLETE

Стадия 06-validation завершена 2026-03-19.

Следующая стадия: **07-baseline** (разблокирована).

Артефакт: `docs/stage-06-validation.md`
