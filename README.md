# UAF — Universal AutoML Framework

Автоматизированная система ML-исследований на базе Claude Code. UAF запускает агента, который самостоятельно проходит полный цикл: от baseline до production-ready пайплайна, логирует всё в MLflow и при необходимости выходит за рамки стандартных методов.

## Что умеет

- **5-фазный исследовательский цикл**: baseline → feature engineering → Optuna HPO → свободное исследование → архитектурная инновация
- **Мониторинг бюджета в реальном времени**: 14 типов алертов, hard stop при исчерпании или CRITICAL событии
- **Leakage detection**: soft_warning + sanity threshold, автоматический Leakage Investigation Protocol
- **Chain continuation**: каждая сессия продолжает лучший пайплайн предыдущей через `pipeline.pkl`
- **MLflow + DVC**: полная воспроизводимость экспериментов
- **PDF отчёты**: автоматическая генерация через LaTeX

## Установка

```bash
# Требует Python 3.10+, uv, Claude Code CLI
pip install uv
git clone https://github.com/mxsfog/automldevelopment.git
cd automldevelopment
uv sync
```

Claude Code CLI: [claude.ai/code](https://claude.ai/code)

## Быстрый старт

**1. Создай task.yaml** (пример в `data/example/task.yaml`):

```yaml
task:
  name: my_classification
  type: tabular_classification

data:
  files:
    - path: data/train.csv
      role: main
  target_column: target

metric:
  name: roc_auc
  direction: maximize
  leakage_soft_warning: 0.95    # триггер investigation protocol
  leakage_sanity_threshold: null # hard stop (null = выключен)

budget:
  mode: fixed
  max_iterations: 20
  max_time_hours: 4.0
```

**2. Запусти сессию:**

```bash
uaf run --task data/my_task/task.yaml --autonomous
```

**3. Или цепочку сессий:**

```bash
./chain_run.sh data/my_task/task.yaml 5 1.5
# 5 сессий по 1.5 часа, каждая продолжает предыдущую
```

## CLI

### `uaf run`

```bash
uaf run \
  --task data/example/task.yaml \    # путь к task.yaml (обязательно)
  --session-id my-session \          # ID сессии (автогенерируется)
  --budget-iterations 20 \           # макс. итераций
  --time 4.0 \                       # макс. часов
  --autonomous \                     # без ручного одобрения
  --model claude-opus-4 \            # модель Claude Code
  --prev-session prev-session-id     # chain continuation
```

### `uaf status`

```bash
uaf status [--session SESSION_ID]
# Выводит: итерации, бюджет, алерты, hints, последние метрики
```

### `uaf resume`

```bash
uaf resume --session 20260320-143022-a1b2c3d4
# Возобновляет прерванную сессию с того же состояния
```

### `uaf analyze`

```bash
uaf analyze --session SESSION_ID
# Запускает ResultAnalyzer + SystemErrorAnalyzer, генерирует session_analysis.json
```

### `uaf report`

```bash
uaf report --session SESSION_ID
# Компилирует PDF отчёт из MLflow данных и session_analysis.json
```

### `uaf stop`

```bash
uaf stop [SESSION_ID]
# Graceful остановка: устанавливает hard_stop=true в budget_status.json
```

## task.yaml — конфигурация

```yaml
task:
  name: название_задачи
  description: >
    Описание задачи для Claude Code.
  type: tabular_classification  # см. поддерживаемые типы ниже

data:
  files:
    - path: data/train.csv
      role: main       # main | test | join | reference
    - path: data/extra.csv
      role: reference
  target_column: target
  target_positive_class: 1   # для бинарной классификации

validation:
  scheme: stratified_kfold   # stratified_kfold | time_series | group_kfold | holdout
  n_splits: 5
  seed: 42
  test_holdout: true
  test_ratio: 0.2

metric:
  name: roc_auc
  direction: maximize          # maximize | minimize
  target_value: 0.85           # целевое значение (для отчёта)
  leakage_sanity_threshold: null   # метрика > X → CRITICAL, hard stop
  leakage_soft_warning: null       # метрика > X → WARNING + investigation hint

budget:
  mode: fixed                  # fixed | dynamic
  max_iterations: 20           # для fixed
  max_time_hours: 4.0

# Опционально
research_preferences:
  shadow_feature_trick: true   # Phase 2 feature engineering
  segment_analysis: true
  segment_columns: [category, region]
```

### Поддерживаемые типы задач

| `task.type` | Описание |
|-------------|---------|
| `tabular_classification` | Бинарная и многоклассовая классификация |
| `tabular_regression` | Регрессия на табличных данных |
| `nlp_classification` | Классификация текста |
| `time_series` | Прогнозирование временных рядов |

## Исследовательские фазы

Claude Code автономно проходит 5 фаз, описанных в `program.md`:

### Phase 1: Baseline (обязательная)
Constant → Linear → Non-linear baseline. Устанавливает нижнюю границу метрики.

### Phase 2: Feature Engineering (обязательная)
Shadow Feature Trick: до 5 гипотез о новых признаках. Принимается только если delta > 0.2%.

### Phase 3: Hyperparameter Optimization
Optuna TPE на лучшей конфигурации из Phase 1-2.

### Phase 4: Free Exploration
Ансамбли, threshold optimization, калибровка, сегментация, WebSearch за новыми идеями. Продолжается до `hard_stop: true`.

### Phase 5: Architecture Innovation
**Trigger**: 3+ итерации Phase 4 без улучшения >0.5%, или >60% бюджета использовано.

Агент делает:
1. WebSearch: `"{task_type} {metric} novel approach 2025 arxiv"`
2. Sequential Thinking (5+ шагов) про inductive bias задачи
3. Формулирует 2 нестандартные архитектурные гипотезы
4. Реализует и сравнивает с baseline

Примеры нестандартных подходов: custom PyTorch loss, graph representation, LSTM/Transformer для временных паттернов, Bayesian uncertainty.

## Leakage Detection

Двухуровневая система в `BudgetController`:

| Порог | Действие |
|-------|---------|
| `leakage_soft_warning` | WARNING алерт + hint `investigate_leakage: true` в `budget_status.json` |
| `leakage_sanity_threshold` | Проверяет MLflow на `leakage_verdict=clean`. Если нет — CRITICAL + hard stop |

Когда агент видит `investigate_leakage: true`, он запускает **Leakage Investigation Protocol (STEP L.1–L.5)**:
- L.1: SHAP analysis — топ фичи и подозрительные выбросы важности
- L.2: Temporal consistency check — корреляция с будущим target
- L.3: Permutation test — реально ли фича несёт информацию
- L.4: Retrain без подозрительных фич
- L.5: Вердикт в MLflow (`leakage_verdict=clean/confirmed/suspected`)

## Мониторинг бюджета

`BudgetController` работает в отдельном потоке, каждые 30 сек читает MLflow и пишет `budget_status.json`:

```json
{
  "iterations_used": 12,
  "iterations_limit": 20,
  "budget_fraction_used": 0.60,
  "hard_stop": false,
  "investigate_leakage": true,
  "leakage_investigated": false,
  "alerts": [
    {"code": "MQ-LEAKAGE-SUSPECT", "level": "WARNING", "message": "..."}
  ],
  "hints": [
    "investigate_leakage: true — запусти Leakage Investigation Protocol"
  ],
  "metrics_history": [0.72, 0.74, 0.75, 0.76]
}
```

### 14 типов алертов

| Код | Уровень | Условие |
|-----|---------|---------|
| SW-HANG | CRITICAL | Нет вывода > 2 часа |
| SW-DISK-FULL | CRITICAL | < 1 ГБ свободно |
| MQ-NAN-CASCADE | CRITICAL | 3+ NaN метрики подряд |
| DQ-DATA-MODIFIED | CRITICAL | Входные данные изменились |
| MQ-LEAKAGE-SUSPECT | CRITICAL/WARNING | Метрика > sanity/soft threshold |
| MQ-DEGRADATION | WARNING | 3 последних < 95% best |
| MQ-CONSECUTIVE-FAILS | WARNING | 3+ failed runs подряд |
| BQ-BUDGET-80PCT | WARNING | 80% итерационного бюджета |
| BQ-TIME-80PCT | WARNING | 80% временного бюджета |
| SW-MLFLOW-DOWN | WARNING | MLflow сервер недоступен |
| DQ-SCHEMA-DRIFT | WARNING | Схема данных изменилась |
| MQ-NEW-BEST | INFO | Новый лучший результат |
| MQ-CONVERGENCE | INFO | Метрика сошлась (dynamic mode) |
| BQ-ITER-COMPLETE | INFO | Итерация завершена |

При CRITICAL алерте → hard stop → grace period (60 сек) → SIGTERM → SIGKILL.

## Chain Continuation

Каждая сессия сохраняет лучший пайплайн в `models/best/`:

```
.uaf/sessions/{session_id}/models/best/
├── pipeline.pkl    # полный пайплайн (feature engineering + model + predict)
├── model.cbm       # нативный файл модели (fallback)
└── metadata.json   # метрика, feature_names, threshold, framework
```

Следующая сессия загружает `pipeline.pkl`, воспроизводит точную метрику и переходит сразу к Phase 4. Фазы 1-3 пропускаются.

`chain_run.sh` автоматически управляет цепочкой:

```bash
./chain_run.sh data/task/task.yaml 10 1
# 10 сессий по 1 часу
# каждая продолжает последнюю сессию с метрикой ≤ LEAKAGE_THRESHOLD
```

## MLflow Логирование

Каждый эксперимент логирует:

```python
with mlflow.start_run(run_name="phase4/step_4.3_ensemble") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")           # experiment | architecture_innovation | chain_verify
    mlflow.set_tag("step", "4.3")
    mlflow.set_tag("hypothesis", "VotingClassifier CatBoost+LGBM")
    mlflow.set_tag("status", "success")
    mlflow.set_tag("convergence_signal", "0.3")   # 0.0-1.0, для dynamic mode
    mlflow.log_params({"n_estimators": 500, ...})
    mlflow.log_metrics({"roc_auc": 0.84, "f1": 0.76})
    mlflow.log_artifact("experiments/run.py")
```

Эксперименты хранятся в `uaf/{session_id}` на MLflow UI (http://127.0.0.1:5000).

## Структура проекта

```
uaf/
├── budget/
│   ├── controller.py      # BudgetController — мониторинг в отдельном потоке
│   ├── convergence.py     # детекция конвергенции
│   └── status_file.py     # BudgetStatusV21, атомарная запись JSON
├── core/
│   ├── program_generator.py   # генерация program.md (Jinja2 шаблон)
│   ├── session_controller.py  # оркестратор UAF сессии
│   ├── oversight.py           # HumanOversightGate
│   ├── validation.py          # совместимость validation scheme
│   └── ruff_enforcer.py       # контроль качества кода агента
├── data/
│   ├── loader.py              # DataLoader (CSV, Parquet, JSONL, ...)
│   ├── leakage_audit.py       # 10 проверок leakage (LA-01..LA-10)
│   └── adversarial_validation.py  # distribution shift detection
├── runner/
│   └── claude_runner.py       # ClaudeCodeRunner + settings.json генерация
├── analysis/
│   ├── result_analyzer.py     # 8-шаговый анализ runs после сессии
│   └── system_error_analyzer.py  # анализ ошибок и health score
├── reporting/
│   ├── report_generator.py    # PDF через LaTeX
│   └── latex_templates.py     # шаблоны разделов
├── integrations/
│   ├── mlflow_setup.py        # инициализация MLflow сервера
│   └── dvc_setup.py           # DVC трекинг данных
└── cli.py                     # Click CLI (run, status, resume, ...)

data/
└── example/
    └── task.yaml              # шаблон конфигурации задачи

chain_run.sh                   # скрипт запуска цепочки сессий
```

## Требования

- Python 3.10+
- [Claude Code CLI](https://claude.ai/code)
- MLflow (запускается автоматически)
- DVC (опционально)
- LaTeX: tectonic или pdflatex (для PDF отчётов)

## Лицензия

MIT
