# Стадия 09: Pipeline

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-08 (COMPLETE или SKIPPED/EMBEDDED)

---

## 0. Что такое "пайплайн" в контексте UAF

В классическом ML-проекте стадия 09-pipeline — это построение train.py,
воспроизводимой цепочки data loading -> preprocessing -> training -> evaluation.

В UAF эта роль разделена на два уровня:

**Уровень 1 — UAF Pipeline (инвариантный):**
ResearchSessionController управляет сессией: DataLoader, ValidationChecker,
ProgramMdGenerator, HumanOversightGate, ClaudeCodeRunner, BudgetController,
ResultAnalyzer, ReportGenerator. Этот пайплайн одинаков для любой задачи.

**Уровень 2 — Experiment Pipeline (вариативный):**
Claude Code строит для конкретной задачи. UAF предоставляет scaffold —
шаблон `experiment.py` с обязательными секциями. Claude Code заполняет
его специфичным кодом (модели, препроцессинг, архитектура).

Данный документ описывает оба уровня, с акцентом на уровень 2.

---

## 1. Структура experiment.py — универсальный scaffold

### 1.1 Назначение

`experiment.py` — это не один файл, а шаблон, который UAF генерирует
в начале каждой итерации сессии. Claude Code модифицирует его:
добавляет код модели, препроцессинга, логику специфичную для задачи.
UAF контролирует обязательную структуру через smoke tests.

Аналог karpathy/autoresearch: Claude Code пишет experiment script.
Отличие: UAF обеспечивает scaffold с контрактными секциями вместо
полностью свободного написания. Это гарантирует MLflow compliance,
DVC commit protocol, budget awareness.

### 1.2 Обязательные секции scaffold

Каждый `experiment.py` должен содержать следующие секции в следующем
порядке. Секции помечены комментариями `# UAF-SECTION: <name>` —
это машиночитаемые маркеры для SmokeTestRunner.

```
# UAF-SECTION: IMPORTS
# UAF-SECTION: CONFIG
# UAF-SECTION: MLFLOW-INIT
# UAF-SECTION: DATA-LOADING
# UAF-SECTION: PREPROCESSING
# UAF-SECTION: MODEL-DEFINITION
# UAF-SECTION: TRAINING
# UAF-SECTION: EVALUATION
# UAF-SECTION: MLFLOW-LOGGING
# UAF-SECTION: ARTIFACT-SAVING
# UAF-SECTION: BUDGET-CHECK
```

### 1.3 Полная структура scaffold (базовый шаблон)

```python
"""
Experiment: {{ experiment_name }}
Session: {{ session_id }}
Iteration: {{ iteration }}
Hypothesis: {{ hypothesis }}
Generated: {{ timestamp }}
"""

# UAF-SECTION: IMPORTS
import logging
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

# Импорты добавляет Claude Code под задачу

logger = logging.getLogger(__name__)


# UAF-SECTION: CONFIG
# Конфиг читается из experiment_config.yaml — не хардкод.
# Claude Code добавляет специфичные параметры в experiment_config.yaml.
# Seed фиксируется здесь и логируется в MLflow.

EXPERIMENT_CONFIG_PATH = Path("{{ config_path }}")
SESSION_DIR = Path("{{ session_dir }}")
BUDGET_STATUS_PATH = SESSION_DIR / "budget_status.json"

def load_config() -> dict[str, Any]:
    """Загрузка конфига эксперимента."""
    import yaml
    with open(EXPERIMENT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# UAF-SECTION: MLFLOW-INIT
# Обязательная инициализация. Run ID логируется в program.md.
# experiment_name и run_name — из конфига сессии.
# parent_run_id устанавливается если это nested run (kfold fold).

def init_mlflow(config: dict[str, Any]) -> mlflow.ActiveRun:
    """Инициализация MLflow run для эксперимента."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run = mlflow.start_run(
        run_name=config["mlflow"]["run_name"],
        tags={
            "session_id": config["session"]["id"],
            "iteration": str(config["iteration"]),
            "hypothesis": config["hypothesis"],
            "task_type": config["task"]["type"],
            "uaf_run_type": "experiment",
        },
    )
    # Немедленно логируем seed и ключевые параметры конфига
    mlflow.log_param("random_seed", config["random_seed"])
    mlflow.log_param("iteration", config["iteration"])
    mlflow.log_param("hypothesis", config["hypothesis"])
    return run


# UAF-SECTION: DATA-LOADING
# Claude Code загружает данные из путей в config["data"].
# Данные не модифицируются — только чтение (antigoal 4).
# Логируется: n_train, n_val, n_test (если есть), feature_count.

def load_data(config: dict[str, Any]) -> tuple[Any, ...]:
    """Загрузка данных по путям из конфига.

    Returns:
        Кортеж (X_train, y_train, X_val, y_val) или расширенный
        вариант в зависимости от task.type.
    """
    # Claude Code заполняет эту секцию
    raise NotImplementedError("Claude Code fills this section")


# UAF-SECTION: PREPROCESSING
# Fit только на train, transform на val/test.
# Preprocessor сохраняется как артефакт MLflow.
# Никакого fit_transform на val — это leakage (antigoal 5).

def build_preprocessor(config: dict[str, Any]) -> Any:
    """Построение и возврат preprocessor pipeline."""
    # Claude Code заполняет под задачу
    raise NotImplementedError("Claude Code fills this section")


def apply_preprocessing(
    preprocessor: Any,
    X_train: Any,
    X_val: Any,
    fit: bool = True,
) -> tuple[Any, Any]:
    """Применение preprocessor. fit=True только для train."""
    if fit:
        X_train_proc = preprocessor.fit_transform(X_train)
    else:
        X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    return X_train_proc, X_val_proc


# UAF-SECTION: MODEL-DEFINITION
# Параметры модели — только из config, не хардкод в коде.
# Все параметры логируются в MLflow params.

def build_model(config: dict[str, Any]) -> Any:
    """Построение модели по параметрам конфига."""
    # Claude Code заполняет под задачу
    raise NotImplementedError("Claude Code fills this section")


# UAF-SECTION: TRAINING
# Чекпоинты каждые checkpoint_interval эпох (если применимо).
# Mixed precision если GPU доступен (PyTorch: torch.cuda.amp).
# Early stopping через параметры конфига.

def train_model(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    config: dict[str, Any],
) -> Any:
    """Обучение модели. Возвращает обученную модель."""
    # Claude Code заполняет под задачу
    raise NotImplementedError("Claude Code fills this section")


# UAF-SECTION: EVALUATION
# Метрика из config["task"]["metric"]. Не хардкод.
# predictions.csv сохраняется для ResultAnalyzer (сегментация, SHAP).
# Формат predictions.csv: id, y_true, y_pred, [y_prob] — зависит от задачи.

def evaluate_model(
    model: Any,
    X_val: Any,
    y_val: Any,
    config: dict[str, Any],
) -> dict[str, float]:
    """Вычисление метрик на val/test. Возвращает dict метрика -> значение."""
    # Claude Code заполняет под задачу
    raise NotImplementedError("Claude Code fills this section")


# UAF-SECTION: MLFLOW-LOGGING
# Обязательные поля: все метрики из evaluate_model + params модели.
# predictions.csv как артефакт — обязательно для ResultAnalyzer.
# Preprocessor как артефакт — обязательно для воспроизводимости.

def log_to_mlflow(
    model: Any,
    preprocessor: Any,
    metrics: dict[str, float],
    config: dict[str, Any],
    predictions_path: Path,
) -> None:
    """Логирование всех артефактов и метрик в MLflow."""
    # Параметры модели
    for key, value in config.get("model_params", {}).items():
        mlflow.log_param(key, value)

    # Метрики
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Артефакты
    mlflow.log_artifact(str(predictions_path), artifact_path="predictions")

    # Preprocessor (joblib или torch)
    # Claude Code добавляет сохранение под свою библиотеку

    logger.info(
        "MLflow logged: %d metrics, run_id=%s",
        len(metrics),
        mlflow.active_run().info.run_id,
    )


# UAF-SECTION: ARTIFACT-SAVING
# Артефакты > 1 МБ -> DVC (инструкция в program.md Execution Instructions).
# Артефакты <= 1 МБ -> git commit в SESSION_DIR.
# DVC commit protocol: dvc add <path> && git add <path>.dvc && git commit.

def save_artifacts(
    model: Any,
    preprocessor: Any,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Сохранение артефактов. Возвращает dict имя -> путь."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    # Claude Code заполняет: joblib.dump / torch.save / model.save_model
    # и следует DVC commit protocol из program.md
    return artifacts


# UAF-SECTION: BUDGET-CHECK
# Обязательная проверка перед КАЖДЫМ следующим шагом.
# Читает budget_status.json синхронно (не блокирует).
# Если hard_stop=True -> sys.exit(0) с корректным завершением MLflow run.

def check_budget() -> bool:
    """Проверка budget_status.json. Возвращает True если можно продолжать."""
    try:
        with open(BUDGET_STATUS_PATH) as f:
            status = json.load(f)
        if status.get("hard_stop", False):
            logger.warning("BudgetController hard_stop detected. Stopping experiment.")
            return False
    except FileNotFoundError:
        # budget_status.json ещё не создан — продолжаем
        pass
    except json.JSONDecodeError:
        # Файл пишется в данный момент — продолжаем (следующая проверка поймает)
        pass
    return True


# Точка входа
def main() -> None:
    """Главная функция эксперимента."""
    config = load_config()

    # Фиксируем seed везде
    seed = config["random_seed"]
    np.random.seed(seed)
    # Claude Code добавляет: random.seed, torch.manual_seed и т.д.

    with init_mlflow(config):
        try:
            # Budget check перед каждым тяжёлым шагом
            if not check_budget():
                mlflow.set_tag("status", "budget_stopped")
                return

            X_train, y_train, X_val, y_val = load_data(config)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_val", len(X_val))

            if not check_budget():
                mlflow.set_tag("status", "budget_stopped")
                return

            preprocessor = build_preprocessor(config)
            X_train_proc, X_val_proc = apply_preprocessing(
                preprocessor, X_train, X_val, fit=True
            )

            model = build_model(config)

            if not check_budget():
                mlflow.set_tag("status", "budget_stopped")
                return

            model = train_model(model, X_train_proc, y_train, X_val_proc, y_val, config)

            metrics = evaluate_model(model, X_val_proc, y_val, config)

            # Сохранение predictions.csv для ResultAnalyzer
            predictions_path = SESSION_DIR / f"predictions_{config['iteration']}.csv"
            # Claude Code заполняет сохранение predictions

            artifacts = save_artifacts(model, preprocessor, config, SESSION_DIR / "artifacts")
            log_to_mlflow(model, preprocessor, metrics, config, predictions_path)

            mlflow.set_tag("status", "completed")
            logger.info(
                "Experiment completed. Metrics: %s",
                {k: f"{v:.4f}" for k, v in metrics.items()},
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            logger.exception("Experiment failed")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

### 1.4 experiment_config.yaml — структура

Claude Code создаёт или получает из сессии конфиг вида:

```yaml
session:
  id: "sess_20260319_abc123"
  dir: ".uaf/sessions/sess_20260319_abc123"

iteration: 3
hypothesis: "XGBoost с SMOTE лучше справляется с дисбалансом классов"
random_seed: 42

task:
  type: "tabular_classification"
  metric:
    name: "roc_auc"
    direction: "maximize"

data:
  train_path: "data/train.parquet"
  val_path: "data/val.parquet"
  test_path: "data/test.parquet"       # только для финального run
  target_col: "label"
  feature_cols: null                   # null = все кроме target

validation:
  scheme: "stratified_kfold"
  k: 5
  seed: 42

model_params:
  # Claude Code заполняет под конкретный эксперимент

mlflow:
  tracking_uri: ".uaf/mlruns"
  experiment_name: "sess_20260319_abc123"
  run_name: "iter_003_xgb_smote"
```

---

## 2. Адаптация scaffold под типы задач

UAF использует параметризованный scaffold, не отдельные шаблоны.
Тип задачи управляет поведением через `task.type` в конфиге.
Claude Code читает task.type и заполняет секции соответственно.

### 2.1 Tabular (classification / regression)

Секции DATA-LOADING, PREPROCESSING, MODEL-DEFINITION, TRAINING,
EVALUATION заполняются с pandas/sklearn/CatBoost/XGBoost.

Специфика:
- Preprocessing: ColumnTransformer (StandardScaler + OneHotEncoder + OrdinalEncoder)
- Fit preprocessor на train, transform val/test
- predictions.csv содержит: id, y_true, y_pred, [y_prob_class_1]
- SHAP доступен (tree-based: TreeExplainer)
- Early stopping: CatBoost/XGBoost native через eval_set

Пример заполнения MODEL-DEFINITION для tabular classification:
```python
from catboost import CatBoostClassifier

def build_model(config: dict[str, Any]) -> CatBoostClassifier:
    return CatBoostClassifier(
        **config["model_params"],
        random_seed=config["random_seed"],
        verbose=False,
    )
```

### 2.2 NLP (text classification / NER / generation)

Специфика:
- DATA-LOADING: jsonlines или HuggingFace datasets
- PREPROCESSING: tokenizer fit не нужен (pretrained) — трансформ tokenize
- MODEL-DEFINITION: AutoModel из transformers или sklearn TF-IDF + LR
- TRAINING: PyTorch training loop или HF Trainer
- Mixed precision: torch.cuda.amp.autocast если CUDA доступен
- predictions.csv: id, y_true, y_pred, text_length (для сегментации)
- Чекпоинты: каждые `checkpoint_interval` шагов -> session artifacts

Пример заполнения TRAINING для NLP (HF Trainer путь):
```python
from transformers import Trainer, TrainingArguments

def train_model(model, X_train, y_train, X_val, y_val, config):
    training_args = TrainingArguments(
        output_dir=str(SESSION_DIR / "checkpoints"),
        seed=config["random_seed"],
        **config["training_params"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_val,
    )
    trainer.train()
    return model
```

### 2.3 CV (image classification / detection / segmentation)

Специфика:
- DATA-LOADING: image_dir с manifest CSV или COCO JSON
- PREPROCESSING: torchvision.transforms — нет fit, только конфигурация
- TRAINING: PyTorch DataLoader, batch size из конфига, pin_memory=True
- Mixed precision: torch.cuda.amp (GradScaler)
- Чекпоинты: `torch.save(model.state_dict(), ...)` каждые N эпох
- predictions.csv: id, y_true, y_pred, img_width, img_height (для сегментации)
- Модель: torchvision.models или timm (через config["model_name"])

### 2.4 Параметризация vs отдельные шаблоны

Решение: один параметризованный scaffold с task.type-зависимым
заполнением секций Claude Code.

Обоснование: отдельные шаблоны усложняют поддержку (DRY нарушается).
Параметризация через task.type — стандартный подход, читаемый Claude Code.
Claude Code понимает условную логику лучше, чем множество специализированных файлов.

Исключение: если задача требует принципиально иного потока (например,
reinforcement learning с environment loop) — Claude Code создаёт
дополнительные файлы, но experiment.py остаётся точкой входа.

---

## 3. Smoke tests и unit tests перед запуском

### 3.1 Назначение SmokeTestRunner

SmokeTestRunner — компонент UAF (логически часть ClaudeCodeRunner),
запускается ПЕРЕД каждым experiment run. Не замена тестам Claude Code —
контрактная проверка UAF-специфичных требований.

Если smoke test не пройден -> эксперимент не запускается.
Claude Code уведомляется через program.md (строка `smoke_test_status: FAILED`
с описанием провала). Claude Code исправляет и перезапускает.

### 3.2 Список smoke tests

```
ST-01  Секции scaffold присутствуют
       Проверка: grep "# UAF-SECTION:" experiment.py -> все 11 секций
       Блокирует: всегда

ST-02  Синтаксис Python корректен
       Проверка: python -m py_compile experiment.py
       Блокирует: всегда

ST-03  ruff lint без ошибок
       Проверка: ruff check experiment.py --select E,F,W
       Блокирует: всегда (не warnings — ошибки)

ST-04  MLflow init присутствует в коде
       Проверка: grep "mlflow.start_run" experiment.py
       Блокирует: всегда

ST-05  budget_status.json check присутствует
       Проверка: grep "check_budget" experiment.py
       Блокирует: всегда

ST-06  Seed фиксируется
       Проверка: grep "random_seed" experiment.py
       Блокирует: всегда

ST-07  NotImplementedError отсутствует в заполненных секциях
       Проверка: статический анализ — секции DATA-LOADING, MODEL-DEFINITION
                 и TRAINING не должны содержать raise NotImplementedError
       Блокирует: всегда

ST-08  experiment_config.yaml существует и валиден
       Проверка: yaml.safe_load, проверка обязательных ключей
                 (session.id, iteration, task.type, task.metric, random_seed)
       Блокирует: всегда

ST-09  MLFLOW_TRACKING_URI доступен
       Проверка: mlflow.get_tracking_uri() + проверка наличия директории
       Блокирует: всегда

ST-10  Нет хардкода путей за пределами SESSION_DIR
       Проверка: regex поиск абсолютных путей кроме SESSION_DIR и данных
       Блокирует: предупреждение в program.md, не блокирует запуск

ST-11  Dry run (только для tabular)
       Проверка: запуск experiment.py с n_rows=10 (флаг --dry-run)
                 должен завершиться за 30 секунд без ошибок
       Блокирует: всегда для tabular. Для NLP/CV — опционально
                  (трансформеры на 10 строках могут требовать CUDA)
```

### 3.3 Dry run механизм

ST-11 требует поддержки `--dry-run` флага в experiment.py.
ProgramMdGenerator включает инструкцию в Execution Instructions:

```
После написания experiment.py добавь поддержку --dry-run:
- import argparse + parser.add_argument("--dry-run", action="store_true")
- если --dry-run: n_rows=10, epochs=1 (или max_iter=1), no_mlflow_log=True
- dry-run не логирует в MLflow, не сохраняет артефакты
- dry-run завершается за < 60 секунд
```

SmokeTestRunner запускает:
```bash
python experiment.py --dry-run
```
с таймаутом 90 секунд. Если зависает -> SIGTERM -> ST-11 FAILED.

### 3.4 Результат smoke tests

SmokeTestRunner пишет `smoke_test_report.json` в SESSION_DIR:

```json
{
  "session_id": "sess_20260319_abc123",
  "iteration": 3,
  "timestamp": "2026-03-19T14:23:45",
  "passed": true,
  "tests": [
    {"id": "ST-01", "status": "passed", "message": "All 11 sections found"},
    {"id": "ST-02", "status": "passed", "message": ""},
    {"id": "ST-07", "status": "failed", "message": "DATA-LOADING contains NotImplementedError"},
    ...
  ],
  "blocking_failures": ["ST-07"]
}
```

Если `passed: false` -> ClaudeCodeRunner не запускает эксперимент.
Вместо этого пишет в program.md секцию `## Smoke Test Failures` с описанием.

---

## 4. uv как менеджер зависимостей

### 4.1 Почему uv

uv (Astral) выбран как менеджер зависимостей по трём причинам:
- Скорость: разрешение зависимостей на порядок быстрее pip
- Изоляция: встроенный venv management без virtualenv
- Lock-файлы: `uv.lock` воспроизводим (pinned версии)

pip остаётся как fallback если uv недоступен.

### 4.2 Структура зависимостей в UAF сессии

Каждая сессия работает в одном из двух режимов:

**Режим A — shared environment (дефолт):**
Одна глобальная venv для всех сессий. Зависимости кумулятивны.
`use_venv: false` в task.yaml (или не указано).
Подходит для однотипных задач на одной машине.

**Режим B — session-isolated venv:**
`use_venv: true` в task.yaml.
UAF создаёт `.uaf/sessions/{id}/venv/` через uv.
Каждая сессия — изолированная среда.
Медленнее запуск (установка зависимостей), но воспроизводимость выше.

### 4.3 requirements.txt в сессии

UAF требует наличия `requirements.txt` или `pyproject.toml` в SESSION_DIR.
ProgramMdGenerator включает в Execution Instructions:

```
Если добавляешь новую библиотеку:
1. Добавь в SESSION_DIR/requirements.txt с pinned версией
2. Запусти: uv pip install -r requirements.txt
3. Зафиксируй: dvc add SESSION_DIR/requirements.txt
   (requirements.txt <= 1 МБ -> git, не dvc)
```

Формат requirements.txt:
```
# UAF base (не модифицировать)
mlflow==2.11.1
dvc==3.49.0
pyyaml==6.0.1
numpy==1.26.4

# Session-specific (Claude Code добавляет)
catboost==1.2.5
scikit-learn==1.4.2
```

### 4.4 Разрешение конфликтов зависимостей

Если `uv pip install` завершается с конфликтом:
- Claude Code логирует конфликт как WARNING в program.md
- Пробует `uv pip install --resolution=lowest-direct`
- Если не помогает — пишет `dependency_conflict: true` в experiment_config.yaml
- SmokeTestRunner (ST-08) помечает предупреждение, не блокирует

### 4.5 uv.lock и воспроизводимость

После успешной установки:
```bash
uv pip freeze > SESSION_DIR/requirements.lock
```
`requirements.lock` попадает в git (мелкий файл).
MLflow Planning Run содержит `requirements_lock_sha` как param.
Это обеспечивает точную воспроизводимость среды через месяц.

---

## 5. Таймауты и kill logic — BudgetController

### 5.1 Уровни таймаутов

Три независимых уровня защиты от зависания:

```
Уровень 1 — Experiment Timeout (per-experiment):
  Источник: task.yaml -> budget.experiment_timeout_seconds (дефолт: 3600)
  Механизм: BudgetController polling, проверяет время старта run
  Действие при превышении: пишет hard_stop=true в budget_status.json
            с reason="experiment_timeout"
  Claude Code должен остановить текущий эксперимент

Уровень 2 — Session Timeout (total):
  Источник: task.yaml -> budget.max_time_hours (дефолт: нет лимита)
  Механизм: BudgetController, wall clock с момента старта сессии
  Действие: hard_stop=true с reason="session_timeout"

Уровень 3 — Session Budget (iteration count):
  Источник: task.yaml -> budget.max_iterations (fixed mode)
  Механизм: BudgetController, счётчик завершённых runs / runs_per_iteration
  Действие: hard_stop=true с reason="budget_exhausted"
```

### 5.2 Алгоритм kill logic

BudgetController polling thread (каждые 30 сек):

```
1. Читает MLflow runs для текущей сессии
2. Считает completed_iterations = completed_runs / runs_per_iteration
3. Проверяет wall_clock vs max_time_hours
4. Проверяет активный run vs experiment_timeout_seconds
5. Проверяет convergence (если dynamic mode)
6. Если любое условие stop:
   a. Устанавливает hard_stop=true в budget_status.json
   b. Записывает stop_reason и timestamp
   c. Запускает grace period таймер (5 минут)
7. По истечении grace period:
   a. Проверяет: Claude Code всё ещё работает?
   b. Если да: отправляет SIGTERM claude процессу
   c. Ждёт 30 секунд
   d. Если процесс не завершился: SIGKILL
   e. Логирует forced_kill=true в MLflow Session tag
```

### 5.3 Grace period — что происходит за 5 минут

За 5 минут grace period Claude Code обязан:
- Завершить текущий experiment.py (сохранить частичные результаты)
- Вызвать `mlflow.end_run(status="KILLED")` для активного run
- Записать `status: budget_stopped` в program.md Current Status
- Сохранить predictions.csv (даже частичный)

Инструкция в Execution Instructions program.md:
```
При обнаружении hard_stop=true в budget_status.json:
1. Заверши текущий шаг обучения (не прерывай mid-epoch)
2. Сохрани model checkpoint в SESSION_DIR/checkpoints/
3. Сохрани predictions.csv с доступными предсказаниями
4. Вызови mlflow.end_run(status="KILLED") если run активен
5. Запиши в program.md: "## Stop Reason: {reason}"
6. sys.exit(0)
```

### 5.4 budget_status.json полная схема

```json
{
  "session_id": "sess_20260319_abc123",
  "updated_at": "2026-03-19T14:30:00",
  "hard_stop": false,
  "stop_reason": null,
  "grace_period_started_at": null,

  "budget": {
    "mode": "fixed",
    "max_iterations": 10,
    "completed_iterations": 3,
    "runs_per_iteration": 5,
    "remaining_iterations": 7
  },

  "timing": {
    "session_started_at": "2026-03-19T13:00:00",
    "max_time_hours": null,
    "elapsed_hours": 1.5,
    "current_experiment_started_at": "2026-03-19T14:15:00",
    "experiment_timeout_seconds": 3600
  },

  "convergence": {
    "mode": "dynamic",
    "patience": 3,
    "no_improvement_count": 1,
    "min_delta": 0.001,
    "best_metric": 0.847,
    "converged": false
  },

  "metrics_history": [
    {"iteration": 1, "metric": 0.812, "run_id": "abc"},
    {"iteration": 2, "metric": 0.831, "run_id": "def"},
    {"iteration": 3, "metric": 0.847, "run_id": "ghi"}
  ]
}
```

### 5.5 Защита от зависания внутри эксперимента

Проблема: Claude Code запускает experiment.py, тот зависает
(бесконечный цикл, deadlock, OOM swap).

BudgetController не имеет прямого доступа к subprocess эксперимента —
Claude Code запускает его через встроенный Bash tool.

Решение — двухуровневое:

**Уровень 1 (инструкция):** ProgramMdGenerator включает в Execution Instructions:
```
Запускай experiment.py с таймаутом:
  timeout {experiment_timeout}s python experiment.py
Если завершился по таймауту (exit code 124) — логируй как timeout_error.
```

**Уровень 2 (внешний):** BudgetController отслеживает MLflow:
если активный run не обновлял метрики > experiment_timeout секунд -> hard_stop.
Это работает только если experiment.py логирует step-метрики в процессе
обучения (не только финальные).

Инструкция Claude Code для PyTorch:
```python
# Логировать шаговые метрики в training loop:
for step, batch in enumerate(train_loader):
    loss = compute_loss(...)
    if step % log_every_n_steps == 0:
        mlflow.log_metric("train_loss_step", loss.item(), step=step)
```

---

## 6. Содержимое LaTeX/PDF отчёта по пайплайну

### 6.1 Секция в отчёте: "Experiment Pipeline"

ReportGenerator формирует секцию `\section{Experiment Pipeline}` с подсекциями.

### 6.2 Подсекция: Pipeline Overview

Таблица: параметры сессии, тип задачи, схема валидации.

```latex
\subsection{Pipeline Overview}

\begin{tabular}{ll}
\hline
Parameter & Value \\
\hline
Session ID & sess\_20260319\_abc123 \\
Task Type & tabular\_classification \\
Validation Scheme & stratified\_kfold (k=5) \\
Total Iterations & 10 \\
Completed Iterations & 8 \\
Stop Reason & metric\_convergence \\
Total Runs & 40 \\
Failed Runs & 3 \\
\hline
\end{tabular}
```

### 6.3 Подсекция: Smoke Test Results

Таблица всех smoke test результатов для каждой итерации.
Если ST-* провалился в какой-то итерации — выделяется красным (\textcolor{red}).

```latex
\subsection{Smoke Test Results}

\begin{tabular}{lccl}
\hline
Test & Pass Rate & Blocking & Notes \\
\hline
ST-01 Scaffold sections & 8/8 & Yes & \\
ST-07 No NotImplementedError & 7/8 & Yes & iter 2: DATA-LOADING incomplete \\
ST-11 Dry run & 8/8 & Yes & \\
\hline
\end{tabular}
```

### 6.4 Подсекция: Dependencies

Таблица установленных зависимостей (из requirements.lock).
Выделяются пакеты, отличные от UAF base.

Если use_venv=false — добавляется примечание:
"Experiments run in shared environment. Exact versions from uv freeze."

### 6.5 Подсекция: Budget Utilization

График (matplotlib -> .pdf figure) timeline сессии:
- X: время
- Y: значение метрики
- Точки: completed runs
- Вертикальная линия: момент остановки (если принудительная)

Текст: "Budget utilized: N/M iterations (X%). Stop reason: convergence/timeout/exhausted."

Если был forced_kill (SIGKILL) — добавляется предупреждение:
\textcolor{red}{Warning: Session terminated by SIGKILL. Last run may be incomplete.}

### 6.6 Подсекция: Code Quality (RuffReport)

Интегрируется из RuffEnforcer post-processing:
- ruff_clean_rate = число файлов без ошибок / всего файлов
- Таблица: файл -> число нарушений -> топ-3 правила
- Если ruff_clean_rate < 0.95 — предупреждение (M-UAF-03)

### 6.7 Подсекция: Reproducibility Checklist

Чеклист воспроизводимости в формате таблицы Yes/No:

```
Random seed fixed:            Yes (seed=42)
MLflow run IDs logged:        Yes
DVC tracked artifacts:        Yes
requirements.lock present:    Yes
Git commit hash logged:       Yes
Test set isolated:            Yes
```

reproducibility_score = число Yes / 6. Логируется в Session Summary Run.

---

## 7. UAF Pipeline (уровень 1) — полная диаграмма потока

```
task.yaml + данные
        |
        v
[DVCSetup] --- dvc init если нужно, dvc add input data
        |
        v
[DataLoader] --- CSV/Parquet/JSONL/images -> сырые данные
        |
        v
[LeakageAudit + AdversarialValidation] --- data_schema.json
        |
        v
[ValidationChecker pre-session] --- VS-* проверки (18)
        |  CRITICAL? -> EXIT с описанием в logs
        v
[ProgramMdGenerator] --- 1 Anthropic API call
        |  input: task.yaml + data_schema.json summary + validation.scheme
        |  output: program.md (Phases + Execution Instructions)
        v
[HumanOversightGate] --- y/n/edit (стандартный режим)
        |  n -> EXIT
        |  edit -> открыть редактор -> повторить gate
        v
[SessionSetup] --- mkdir SESSION_DIR, создать settings.json,
        |           создать scaffold experiment.py (базовый),
        |           создать experiment_config.yaml,
        |           создать budget_status.json (initial)
        v
[BudgetController] --- запуск polling thread
        |
        v
[ИТЕРАЦИОННЫЙ ЦИКЛ]:
        |
        +---> [ClaudeCodeRunner] --- subprocess: claude --settings settings.json
        |          |
        |          |  Claude Code:
        |          |    читает program.md + Execution Instructions
        |          |    пишет experiment.py (заполняет секции)
        |          |    пишет experiment_config.yaml (параметры)
        |          |    запускает SmokeTestRunner -> если FAIL: исправляет
        |          |    запускает: timeout N python experiment.py
        |          |    логирует в MLflow (metrics, params, artifacts)
        |          |    сохраняет predictions.csv
        |          |    следует DVC commit protocol
        |          |    проверяет budget_status.json
        |          |    обновляет program.md (status, result, conclusion)
        |
        +---> [BudgetController polling] --- читает MLflow, обновляет budget_status.json
        |                                    hard_stop? -> grace period -> SIGTERM/SIGKILL
        |
        +---> [ValidationChecker post-run] --- VR-* проверки (7)
        |
        +---> [ResultAnalyzer real-time] --- обновляет metrics_history, convergence
        |
        +--- hard_stop OR budget_exhausted OR convergence OR claude_exit
        |
        v
[RuffEnforcer] --- ruff check + ruff format на всех .py в SESSION_DIR
        |           создаёт RuffReport
        v
[ResultAnalyzer post-session] --- session_analysis.json (полный анализ)
        |
        v
[ValidationChecker post-session] --- val_test_delta, cv_stability
        |
        v
[ReportGenerator] --- 3 Anthropic API calls
        |   Секции: Executive Summary, Task, Research Program,
        |           Experiment Pipeline (эта стадия), Experiment Results,
        |           Error Analysis, Recommendations, Code Quality,
        |           Reproducibility, Appendix
        |   Компилятор: tectonic -> pdflatex -> .tex fallback
        v
report.pdf + session_analysis.json + budget_status.json (final)
```

---

## 8. Взаимодействие пайплайна с компонентами стадий 05-08

### 8.1 Связь с DataLoader (стадия 05)

DataLoader запускается до пайплайна. Результат — `data_schema.json`.
experiment_config.yaml содержит пути к уже подготовленным данным.
experiment.py не вызывает DataLoader — только читает файлы по путям.
Antigoal 4 соблюдён: данные только читаются.

### 8.2 Связь с ValidationChecker (стадия 06)

Pre-session проверки (VS-*) выполнены до генерации program.md.
Post-run проверки (VR-*) выполняются после каждого experiment run:
- ClaudeCodeRunner вызывает ValidationChecker.check_post_run(run_id)
- Если VR-* failure: добавляет предупреждение в program.md
- Не останавливает сессию (только CRITICAL VS-* блокируют)

### 8.3 Связь с ResultAnalyzer (стадия 08)

ResultAnalyzer — часть пайплайна, работает в двух режимах:
- Real-time: обновляет budget_status.json.metrics_history каждые 30 сек
- Post-session: полный анализ -> session_analysis.json -> ReportGenerator

Feedback loop к следующей итерации: ResultAnalyzer пишет
`improvement_context.md` в SESSION_DIR. При --resume ClaudeCodeRunner
передаёт его как дополнительный контекст Claude Code.

### 8.4 Связь с baseline logic (стадия 07/EMBEDDED)

ProgramMdGenerator всегда генерирует Phase 1 с baseline методами.
experiment.py Phase 1 итерации всегда проще Phase 2+:
- iter 1: DummyClassifier -> минимальный scaffold, нет preprocessor
- iter 2: LogisticRegression -> ColumnTransformer + скейлер
- iter 3+: сложные модели -> полный scaffold

---

## 9. Решения и обоснования

### R-09-01: Scaffold с контрактными секциями vs полностью свободный experiment.py

Решение: scaffold с маркерами `# UAF-SECTION:`.
Альтернатива: Claude Code пишет полностью свободно (как karpathy/autoresearch).

Обоснование: свободный подход нарушает antigoal 3 (нет гарантии MLflow logging)
и antigoal 6 (нет budget check). Scaffold минимален — не ограничивает свободу
Claude Code, но гарантирует контрактные требования через smoke tests.

### R-09-02: Один параметризованный scaffold vs отдельные шаблоны

Решение: один scaffold, task.type управляет заполнением.
Обоснование: DRY, одна точка сопровождения. Claude Code понимает условную
логику из task.type без дополнительных инструкций. Отдельные шаблоны
увеличивают сложность ProgramMdGenerator без выгоды.

### R-09-03: uv как менеджер зависимостей

Решение: uv с requirements.lock.
Обоснование: скорость установки критична при session-isolated venv.
requirements.lock -> воспроизводимость среды. Conda не используется
(antigoal 4: не усложнять окружение пользователя).

### R-09-04: Grace period 5 минут

Решение: 5 минут до SIGTERM после hard_stop.
Обоснование: достаточно для сохранения checkpoint и завершения MLflow run.
Меньше — риск потери данных. Больше — нарушение antigoal 6 (бюджет продолжает
тратиться). SIGKILL только если SIGTERM не помог за 30 секунд.

### R-09-05: SmokeTestRunner блокирует при ST-07 (NotImplementedError)

Решение: ST-07 блокирует запуск.
Обоснование: experiment.py с незаполненными секциями завершится с
NotImplementedError через секунды, потратив MLflow run ID и итерацию бюджета.
Лучше поймать до запуска. Claude Code получает чёткий сигнал.

---

## 10. Файловая структура SESSION_DIR

```
.uaf/sessions/sess_20260319_abc123/
├── program.md                       # план (Claude Code обновляет)
├── experiment_config.yaml           # конфиг текущей итерации
├── experiment.py                    # scaffold (Claude Code заполняет)
├── requirements.txt                 # зависимости сессии
├── requirements.lock                # pinned (uv pip freeze)
├── budget_status.json               # BudgetController пишет
├── smoke_test_report.json           # SmokeTestRunner пишет
├── session_analysis.json            # ResultAnalyzer пишет (post-session)
├── improvement_context.md           # ResultAnalyzer -> следующая сессия
│
├── artifacts/                       # модели, препроцессоры (> 1 МБ -> DVC)
│   ├── iter_001_model.pkl
│   ├── iter_001_preprocessor.pkl
│   └── iter_001_model.pkl.dvc
│
├── checkpoints/                     # DL чекпоинты (всегда > 1 МБ -> DVC)
│   └── iter_003_epoch_10.pt.dvc
│
├── predictions/                     # predictions.csv каждой итерации
│   ├── predictions_001.csv
│   └── predictions_003.csv
│
└── report/                          # ReportGenerator выход
    ├── report.tex
    ├── report.pdf
    └── figures/
        ├── metric_timeline.pdf
        └── feature_importance.pdf
```

settings.json Claude Code хранится в `.uaf/sessions/{id}/settings.json` —
не в SESSION_DIR напрямую (изолированно от артефактов).

---

## 11. Antigoal compliance

| Antigoal | Механизм соблюдения в стадии 09 |
|---|---|
| 1 (не AutoML для prod) | experiment.py — исследовательский скрипт, не serving код |
| 2 (одобрение перед запуском) | HumanOversightGate до SessionSetup и запуска пайплайна |
| 3 (не скрывать failed runs) | Smoke test failures, failed experiments — в отчёт |
| 4 (не модифицировать данные) | experiment.py только читает по путям из конфига |
| 5 (читаемость > оптимизация) | scaffold структура, конфиг через YAML, не хардкод |
| 6 (бюджет) | BudgetController kill logic, grace period, SIGTERM/SIGKILL |

---

## STAGE COMPLETE

Стадия 09-pipeline завершена.

Следующая стадия: 10-features.
Условие разблокировки: выполнено (стадии 01-09 complete или embedded).

Ключевые решения зафиксированы: R-09-01..R-09-05.
Артефакт: `docs/stage-09-pipeline.md`.
