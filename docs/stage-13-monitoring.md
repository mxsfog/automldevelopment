# Стадия 13: Monitoring

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-10 (COMPLETE или SKIPPED/EMBEDDED), 11-12 (SKIPPED/antigoal-1)

---

## 0. Контекст: мониторинг сессии, не продакшн-модели

В классическом ML продакшн-мониторинг следит за поведением deployed модели
во внешней среде: data drift, concept drift, deградация precision в production.
UAF — исследовательская система, работающая локально, без деплоя.

Мониторинг UAF — это наблюдение за тем, что происходит внутри одной сессии
в реальном времени. Объекты наблюдения:

- UAF-процессы и Claude Code (программная среда)
- входные данные и DVC-чексуммы между итерациями (качество данных)
- метрики экспериментов по мере поступления (качество модели)
- расход бюджета и прогресс к цели (бизнес-цель сессии)

Пирамида мониторинга адаптирована к этому контексту. Каждый уровень
добавляется поверх предыдущего — нижний уровень является предусловием
для значимости верхнего.

---

## 1. Пирамида мониторинга UAF (4 уровня)

```
                [Business KPIs]          <- бюджет, прогресс, цель сессии
              [Model Quality]            <- метрики, NaN, OOM, деградация
            [Data Quality]              <- DVC checksums, schema, drift между итерациями
          [Software Health]             <- процессы, окружение, MLflow доступность
```

### Уровень 1 — Software Health (фундамент)

Что мониторится:

| Объект | Что проверяется | Источник данных |
|--------|----------------|-----------------|
| Claude Code процесс | PID жив, не завис (CPU > 0% или запустился < N секунд назад) | psutil / subprocess |
| MLflow tracking server | HTTP 200 на `/api/2.0/mlflow/experiments/list` | requests.get(timeout=5) |
| uv окружение | `uv pip check` — нет конфликтующих зависимостей | subprocess |
| SESSION_DIR | Доступна запись, не переполнен диск (> 1 ГБ свободно) | shutil.disk_usage |
| budget_status.json | Файл существует, валидный JSON, не устарел (mtime < 120 сек) | os.stat + json.loads |
| Логи experiment.py | stdout/stderr не зависли (последняя строка < 300 сек назад) | ClaudeCodeRunner.last_output_at |

Период опроса: каждые 30 секунд (совпадает с основным polling циклом BudgetController).

Критичность: Software Health — единственный уровень, где сбой немедленно
прерывает оценку остальных уровней. Если Claude Code завис — данные о метриках
из MLflow ненадёжны.

### Уровень 2 — Data Quality

Что мониторится между итерациями:

| Объект | Что проверяется | Источник данных |
|--------|----------------|-----------------|
| DVC checksums входных данных | md5/sha256 из .dvc файлов не изменились с начала сессии | dvc status --json |
| data_schema.json | Файл не модифицирован после начала сессии | sha256 файла при старте vs текущий |
| feature_registry.json | Нет неожиданных изменений вне итерационного цикла | sha256 при конце каждой итерации |
| Схема MLflow runs | Новые runs содержат обязательные params (VR-001..007) | MLflow API polling |
| predictions.csv | Появились в текущей итерации (не пустые) | os.path.exists + os.path.getsize |

Логика: данные мониторятся только на изменения между итерациями — не в рамках
одной итерации. Если DVC checksums изменились в ходе сессии, это нарушение
antigoal 4 (UAF не модифицирует данные пользователя) и серьёзный сигнал.

Период опроса: конец каждой итерации + ленивая проверка каждые 60 секунд.

### Уровень 3 — Model Quality

Что мониторится в реальном времени:

| Объект | Что проверяется | Источник данных |
|--------|----------------|-----------------|
| Task metric | Значение из MLflow по каждому завершённому run | MLflow API: `mlflow.search_runs` |
| NaN/Inf в метриках | metric value is NaN или Inf | math.isnan + math.isinf |
| OOM события | `torch.cuda.OutOfMemoryError` или `MemoryError` в stderr | ClaudeCodeRunner stderr stream |
| Деградация метрики | Метрика на итерации N+1 хуже min по всем предыдущим > degradation_threshold | metrics_history |
| Loss расходимость | Training loss растёт монотонно 3 итерации подряд (если логируется) | MLflow: `train_loss` |
| Failed runs подряд | 3 и более failed MLflow runs без единого успешного | MLflow run status |
| Timeout runs | Runs, завершённые по experiment_timeout, а не нормально | MLflow tag: `uaf.stop_reason` |

Деградация считается мягкой (warning) при degradation_threshold = 0.05 от best_metric,
жёсткой (alert) при > 0.15.

Период опроса: как только MLflow run меняет статус RUNNING -> FINISHED/FAILED.
Реализация: MLflow API polling каждые 30 секунд, фильтр по `experiment_id`.

### Уровень 4 — Business KPIs

Что мониторится для оценки прогресса сессии:

| KPI | Что проверяется | Порог |
|-----|----------------|-------|
| Бюджет итераций | completed_iterations / max_iterations | > 0.8 = warning, >= 1.0 = hard stop |
| Бюджет времени | elapsed_hours / max_time_hours | > 0.8 = warning, >= 1.0 = hard stop |
| Прогресс метрики | best_metric достиг success_metric_threshold | достиг = можно ранней остановке |
| Сходимость | Критерий no_improvement(patience) + delta < min_delta | выполнен = convergence stop |
| LLM-сигнал | Claude Code тег `uaf.convergence_signal` в MLflow | "CONVERGED" = учитывается |
| Расход API | Оценочные токены (из ClaudeCodeRunner метаданных если доступны) | информационно |
| Smoke test pass rate | passed_smoke_tests / total_smoke_tests по сессии | < 0.8 = проблема в scaffold |

Business KPIs опрашиваются каждые 30 секунд вместе с основным циклом.
Единственный KPI, ведущий к немедленному hard stop: бюджет итераций и бюджет времени.

---

## 2. BudgetController: реальное время мониторинга

### 2.1 Архитектура BudgetController

BudgetController — это daemon thread, который запускается параллельно с
ClaudeCodeRunner. Он не блокирует основной поток UAF.

```
ResearchSessionController (main thread)
    |
    +---> ClaudeCodeRunner (subprocess: claude process)
    |
    +---> BudgetController (daemon thread)
              |
              +---> MLflow API polling (каждые 30 сек)
              |         поиск runs по experiment_id
              |         разбор статусов, метрик, параметров
              |
              +---> Software Health checks (каждые 30 сек)
              |         psutil.pid_exists(claude_pid)
              |         MLflow healthcheck
              |         disk space check
              |
              +---> Data Quality checks (каждые 60 сек)
              |         dvc status
              |         schema file hashes
              |
              +---> budget_status.json (запись при каждом poll)
              |
              +---> session.log (запись при каждом событии)
```

### 2.2 Polling interval: 30 секунд

Обоснование выбора 30 секунд:
- Эксперименты длятся минуты, не секунды -> 30 сек даёт < 1% overhead
- NaN loss обнаруживается максимум через 30 сек после завершения run
- Слишком частый polling нагружает MLflow tracking server и файловую систему
- При experiment_timeout >= 60 сек (минимальный разумный) мы успеваем среагировать

Исключение: stderr Claude Code читается потоково (subprocess PIPE), без задержки.
OOM и критические ошибки обнаруживаются немедленно из stderr.

### 2.3 Алгоритм одного polling цикла

```
Каждые 30 секунд:

1. Software Health (приоритет 0 — проверяется первым)
   a. psutil.pid_exists(claude_pid) AND cpu_percent(interval=1) > 0
      OR process started < hang_timeout_seconds ago
   b. MLflow healthcheck (GET /health, timeout=5 сек)
   c. disk_free > 1 ГБ
   d. budget_status.json mtime < 120 сек
   -> если (a) FAIL: алерт SW-HANG, запуск hang_recovery
   -> если (b) FAIL: алерт SW-MLFLOW-DOWN, пауза polling (не hard stop)
   -> если (c) FAIL: алерт SW-DISK-FULL, hard stop
   -> если (d) FAIL: алерт SW-STATUS-STALE (warning)

2. MLflow runs query
   runs = mlflow.search_runs(
       experiment_ids=[session_experiment_id],
       filter_string="tags.`mlflow.runName` LIKE 'iter_%'",
       order_by=["start_time DESC"]
   )

3. Budget calculation
   completed_iterations = count runs WHERE status IN ("FINISHED", "FAILED")
       / runs_per_iteration
   elapsed_hours = (now - session_start) / 3600
   -> обновить budget.iterations_used, budget.elapsed_hours в budget_status.json

4. Model Quality checks (по последнему завершённому run)
   last_run = first FINISHED run from query
   IF last_run exists:
       metric_val = last_run.data.metrics.get(metric_name)
       check_nan_inf(metric_val)
       check_degradation(metric_val, metrics_history)
       check_consecutive_failures(runs)

5. Convergence check (только dynamic mode)
   no_improvement = best_metric не менялся >= patience итераций
   delta_ok = abs(current - prev_best) < min_delta
   min_iter_ok = completed_iterations >= min_iterations
   llm_signal = "CONVERGED" IN [r.data.tags.get("uaf.convergence_signal") for r in runs[-2:]]
   -> convergence = (no_improvement AND delta_ok AND min_iter_ok) OR (llm_signal x2)

6. Hard stop decision
   причины для hard_stop=true:
   - completed_iterations >= max_iterations (antigoal 6)
   - elapsed_hours >= max_time_hours (antigoal 6)
   - disk_free < 1 ГБ (SW-DISK-FULL)
   - consecutive_nan_runs >= 3
   - convergence = True (корректная остановка)

7. Запись budget_status.json (атомарная через tmp + rename)

8. Запись в session.log
```

### 2.4 Hang detection и recovery

Claude Code считается зависшим если:
- PID существует (psutil.pid_exists = True)
- НО cpu_percent(interval=2) == 0 в течение hang_timeout_seconds = 300 сек
- И последняя строка в stdout/stderr появилась > 300 сек назад

При обнаружении зависания:

```
Шаг 1: алерт SW-HANG в session.log (уровень CRITICAL)
Шаг 2: запись в budget_status.json: alerts.append({type: "SW-HANG", ...})
Шаг 3: ожидание 60 секунд (возможно, Claude Code работает над сложной задачей без вывода)
Шаг 4: повторная проверка cpu_percent и stdout mtime
Шаг 5: если всё ещё завис -> hard_stop=true с reason="agent_hang"
Шаг 6: grace period 5 минут -> SIGTERM -> 30 сек -> SIGKILL
```

Консервативный подход: лучше ложный алерт (FP), чем потеря работающего Claude Code (FN).
FP: Claude Code перезапускается, теряем текущую итерацию.
FN: Claude Code завис навсегда, бюджет исчерпан без результата.

---

## 3. Алерты и реакции

### 3.1 Реестр алертов

Каждый алерт имеет: код, уровень серьёзности, триггер, немедленная реакция,
запись в session.log, запись в budget_status.json.alerts.

#### Уровень CRITICAL (немедленные действия, возможен hard stop)

**SW-HANG** — Claude Code завис
- Триггер: cpu=0% + no stdout в течение hang_timeout_seconds (300 сек)
- Реакция: 60 сек ожидание -> повторная проверка -> hard_stop + SIGTERM
- Запись: session.log CRITICAL, budget_status.json alerts[]

**SW-DISK-FULL** — диск заполнен
- Триггер: disk_free < 1 ГБ
- Реакция: немедленный hard_stop=true с reason="disk_full"
- Причина: запись артефактов и checkpoint провалится, теряем эксперимент полностью

**MQ-NAN-CASCADE** — NaN в метриках 3 итерации подряд
- Триггер: последние 3 завершённых run имеют NaN в primary metric
- Реакция: hard_stop=true с reason="nan_cascade", не ждём grace period
- Причина: Claude Code не исправляет проблему сам -> нужно вмешательство человека

**MQ-OOM** — Out of Memory
- Триггер: "CUDA out of memory" ИЛИ "MemoryError" в stderr Claude Code
- Реакция: записать алерт, ждать завершения текущего run (не прерывать)
  После завершения run: добавить в budget_status.json hints: ["reduce_batch_size", "use_gradient_checkpointing"]
  Claude Code читает hints при следующей проверке budget_status.json
- НЕ hard_stop: Claude Code может сам уменьшить batch_size и продолжить

**DQ-DATA-MODIFIED** — входные данные изменились в ходе сессии
- Триггер: sha256(input_data_dvc_hash) != hash_at_session_start
- Реакция: hard_stop=true с reason="data_integrity_violation"
  Запись в session.log: WARNING antigoal-4 violation
- Причина: воспроизводимость нарушена, продолжать нельзя

#### Уровень WARNING (предупреждение, без hard stop)

**MQ-DEGRADATION** — деградация основной метрики
- Триггер: metric_value < best_metric * (1 - degradation_threshold), threshold=0.05
- Реакция: запись в budget_status.json.alerts[], добавить в metrics_history.degradation_flag=true
  Hint в budget_status.json для Claude Code: смотри Execution Instructions
- Не прерывает: Claude Code видит деградацию в следующей итерации и реагирует самостоятельно

**MQ-CONSECUTIVE-FAILS** — серия failed runs
- Триггер: 2 подряд FAILED MLflow runs (не OOM, не timeout)
- Реакция: WARNING в session.log, alert в budget_status.json
  При 3 подряд FAILED: hard_stop=true с reason="consecutive_failures"
- Причина: что-то системно сломано, Claude Code не справляется самостоятельно

**SW-MLFLOW-DOWN** — MLflow недоступен
- Триггер: HTTP != 200 на MLflow healthcheck
- Реакция: пауза мониторинга MLflow API, повторная попытка через 60 сек
  Не hard_stop: эксперимент продолжается, MLflow может временно быть недоступен
  После 5 неудачных попыток (5 мин): WARNING в session.log, BQ-MLFLOW-UNAVAILABLE alert
  После 10 неудачных попыток (10 мин): hard_stop с reason="mlflow_unavailable"
- Причина: без MLflow теряем воспроизводимость (antigoal implicit)

**BQ-BUDGET-80PCT** — использовано 80% бюджета
- Триггер: completed_iterations / max_iterations > 0.8
- Реакция: WARNING в session.log, alert в budget_status.json
  budget_status.json.budget.warning=true
  Claude Code видит warning и может начать "финальные эксперименты"

**BQ-TIME-80PCT** — использовано 80% времени
- Триггер: elapsed_hours / max_time_hours > 0.8
- Аналогично BQ-BUDGET-80PCT

**DQ-SCHEMA-DRIFT** — data_schema.json изменился
- Триггер: sha256(data_schema.json) != hash_at_session_start
- Реакция: WARNING, не hard_stop (Claude Code может корректно обновить schema при feature engineering)
  Если изменение не в рамках итерации feature engineering: escalate to CRITICAL DQ-DATA-MODIFIED

#### Уровень INFO (информационные записи без действий)

**MQ-NEW-BEST** — новый лучший результат
- Триггер: metric_value > best_metric (для maximize) ИЛИ < (для minimize)
- Запись: session.log INFO, budget_status.json.metrics_history[-1].is_best=true

**MQ-CONVERGENCE** — достигнута сходимость
- Триггер: алгоритм сходимости из стадии 04 выполнен
- Запись: session.log INFO, budget_status.json.convergence.converged=true
  hard_stop=true с reason="convergence" (ожидаемое завершение)

**BQ-ITER-COMPLETE** — итерация завершена
- Триггер: completed_iterations увеличился на 1
- Запись: session.log INFO с номером итерации и текущей метрикой

### 3.2 Сводная таблица реакций

| Ситуация | Алерт | Действие | hard_stop |
|----------|-------|----------|-----------|
| Claude Code завис (300 сек) | SW-HANG | 60 сек ожидание -> SIGTERM | да |
| Диск < 1 ГБ | SW-DISK-FULL | немедленно | да |
| NaN 3 итерации подряд | MQ-NAN-CASCADE | немедленно | да |
| OOM в stderr | MQ-OOM | hint в budget_status.json | нет |
| Деградация метрики > 5% | MQ-DEGRADATION | hint, предупреждение | нет |
| 3 подряд FAILED runs | MQ-CONSECUTIVE-FAILS | | да |
| MLflow down > 10 мин | BQ-MLFLOW-UNAVAILABLE | | да |
| 80% бюджета | BQ-BUDGET-80PCT | warning в budget_status | нет |
| Входные данные изменились | DQ-DATA-MODIFIED | antigoal-4 violation | да |
| Сходимость | MQ-CONVERGENCE | корректное завершение | да |
| Бюджет исчерпан | BQ-BUDGET-EXHAUSTED | antigoal-6 | да |

---

## 4. budget_status.json — финальная схема

Файл пишется атомарно: BudgetController пишет во временный файл
`budget_status.json.tmp`, затем `os.replace()` (атомарная операция на POSIX).
Claude Code никогда не видит частично записанный файл.

```json
{
  "schema_version": "2.1",
  "session_id": "sess_20260319_abc123",
  "updated_at": "2026-03-19T14:30:00.123456",
  "updated_at_epoch": 1742393400.123,

  "hard_stop": false,
  "stop_reason": null,

  "budget": {
    "mode": "fixed",
    "max_iterations": 10,
    "max_time_hours": 4.0,
    "iterations_used": 3,
    "iterations_remaining": 7,
    "elapsed_hours": 1.2,
    "elapsed_seconds": 4320,
    "budget_pct_iterations": 0.30,
    "budget_pct_time": 0.30,
    "warning": false,
    "runs_per_iteration": 5,
    "experiment_timeout_seconds": 3600
  },

  "timing": {
    "session_start": "2026-03-19T13:10:00",
    "session_start_epoch": 1742389800,
    "last_run_start": "2026-03-19T14:25:00",
    "last_run_start_epoch": 1742393100,
    "last_run_id": "run_abc123def456",
    "estimated_completion": "2026-03-19T16:30:00",
    "estimated_iterations_per_hour": 2.5
  },

  "convergence": {
    "enabled": false,
    "converged": false,
    "patience": 3,
    "min_delta": 0.001,
    "min_iterations": 3,
    "no_improvement_count": 1,
    "current_delta": 0.005,
    "llm_signal_count": 0,
    "llm_signal_consecutive": 0
  },

  "metrics_history": [
    {
      "iteration": 1,
      "run_id": "run_xxx",
      "metric_name": "roc_auc",
      "metric_value": 0.782,
      "metric_direction": "maximize",
      "is_best": false,
      "is_nan": false,
      "is_failed": false,
      "degradation_flag": false,
      "timestamp": "2026-03-19T13:45:00",
      "runs_in_iteration": 5,
      "failed_runs_in_iteration": 0,
      "stop_reason_in_iteration": null
    },
    {
      "iteration": 2,
      "run_id": "run_yyy",
      "metric_name": "roc_auc",
      "metric_value": 0.801,
      "metric_direction": "maximize",
      "is_best": true,
      "is_nan": false,
      "is_failed": false,
      "degradation_flag": false,
      "timestamp": "2026-03-19T14:15:00",
      "runs_in_iteration": 5,
      "failed_runs_in_iteration": 0,
      "stop_reason_in_iteration": null
    },
    {
      "iteration": 3,
      "run_id": "run_zzz",
      "metric_name": "roc_auc",
      "metric_value": 0.795,
      "metric_direction": "maximize",
      "is_best": false,
      "is_nan": false,
      "is_failed": false,
      "degradation_flag": true,
      "timestamp": "2026-03-19T14:28:00",
      "runs_in_iteration": 5,
      "failed_runs_in_iteration": 1,
      "stop_reason_in_iteration": null
    }
  ],

  "best_metric": {
    "value": 0.801,
    "iteration": 2,
    "run_id": "run_yyy",
    "timestamp": "2026-03-19T14:15:00"
  },

  "software_health": {
    "claude_pid": 12345,
    "claude_alive": true,
    "claude_last_output_at": "2026-03-19T14:29:55",
    "hang_detected": false,
    "mlflow_available": true,
    "mlflow_last_check": "2026-03-19T14:29:58",
    "disk_free_gb": 42.3,
    "disk_ok": true,
    "uv_env_ok": true,
    "session_dir_writable": true
  },

  "data_quality": {
    "input_data_hash_ok": true,
    "data_schema_hash_ok": true,
    "feature_registry_hash_ok": true,
    "dvc_status_clean": true,
    "last_dq_check": "2026-03-19T14:28:00"
  },

  "alerts": [
    {
      "alert_id": "alert_001",
      "code": "MQ-DEGRADATION",
      "level": "WARNING",
      "timestamp": "2026-03-19T14:29:00",
      "iteration": 3,
      "message": "Metric roc_auc degraded: 0.795 vs best 0.801 (delta=-0.006, threshold=0.05)",
      "resolved": false
    }
  ],

  "hints": [],

  "phase": {
    "current_phase": "phase_1",
    "phase_step": 3,
    "baseline_run_id": null,
    "shadow_mode": false
  }
}
```

### 4.1 Поля, добавленные в стадии 13 (расширение схемы из стадии 09)

Стадия 09 зафиксировала базовую схему budget_status.json. Стадия 13 добавляет:

- `schema_version` — версия схемы для backward compatibility
- `updated_at_epoch` — unix timestamp для точных вычислений интервалов
- `budget.budget_pct_iterations` и `budget.budget_pct_time` — проценты для Claude Code
- `timing.estimated_completion` и `timing.estimated_iterations_per_hour` — ETA
- `convergence` — полный блок с детальным состоянием сходимости
- `metrics_history[].runs_in_iteration`, `failed_runs_in_iteration`, `stop_reason_in_iteration`
- `best_metric` — отдельный блок для быстрого доступа
- `software_health` — полный блок состояния SW
- `data_quality` — блок DQ checksums
- `alerts[]` — история алертов (не только текущие)
- `hints[]` — подсказки Claude Code (например, после OOM)
- `phase` — текущая фаза program.md и shadow mode

### 4.2 Поле hints — механизм обратной связи BudgetController -> Claude Code

`hints` — это список строк, которые BudgetController добавляет при определённых
событиях. Claude Code читает budget_status.json при каждой итерации (BUDGET-CHECK секция)
и может реагировать на hints.

```json
"hints": [
  "OOM detected in iteration 3: consider reducing batch_size or enabling gradient_checkpointing",
  "Metric degraded 2 consecutive iterations: try higher regularization or different architecture"
]
```

Hints не являются командами — Claude Code следует им по своему усмотрению.
Это единственный канал прямой коммуникации BudgetController -> Claude Code
(помимо hard_stop и warnings).

---

## 5. session.log — структура и уровни логирования

### 5.1 Расположение и ротация

```
SESSION_DIR/
├── session.log          <- основной лог сессии
└── session.log.1        <- предыдущий (если ротация сработала)
```

Ротация: при достижении 50 МБ. Максимум 2 файла (session.log + session.log.1).
Кодировка: UTF-8. Формат: структурированный текст (не JSON — для удобства чтения человеком).

### 5.2 Формат строки лога

```
TIMESTAMP LEVEL [COMPONENT] MESSAGE [KEY=VALUE ...]
```

Пример:
```
2026-03-19T14:30:00.123 INFO  [BudgetController] Poll cycle #42: iterations=3/10, elapsed=1.2h/4.0h
2026-03-19T14:30:00.456 INFO  [BudgetController] Run run_zzz FINISHED: roc_auc=0.795 (best=0.801)
2026-03-19T14:30:00.789 WARN  [BudgetController] ALERT MQ-DEGRADATION: metric=0.795, best=0.801, delta=-0.006
2026-03-19T14:30:01.012 INFO  [BudgetController] budget_status.json written (atomic)
```

Компоненты в []:
- `BudgetController` — polling, алерты, budget calculations
- `ClaudeCodeRunner` — stdout/stderr Claude Code, процессные события
- `SmokeTestRunner` — результаты smoke tests
- `ValidationChecker` — VS-* и VR-* результаты
- `ResultAnalyzer` — post-iteration обновления
- `ReportGenerator` — финальный отчёт
- `ResearchSessionController` — переходы состояний сессии
- `DVC` — dvc add, dvc push, dvc status

### 5.3 Что пишется на каждом уровне

**DEBUG** (только при `--debug` флаге UAF):
- Полный JSON каждого MLflow run при polling
- Raw stdout Claude Code построчно
- Детали атомарной записи budget_status.json (tmp path, os.replace())
- Timing каждого шага polling цикла

**INFO** (стандартный уровень):
- Старт и завершение polling цикла (с номером)
- Каждый завершённый MLflow run: run_id, status, metric_value
- Новый лучший результат (MQ-NEW-BEST)
- Завершение итерации с итоговыми показателями
- Переходы фаз (phase_1 -> phase_2 -> phase_3)
- Software health OK (раз в 5 циклов, не каждый)
- Начало и конец сессии

**WARNING**:
- Все алерты уровня WARNING: MQ-DEGRADATION, MQ-CONSECUTIVE-FAILS (2), BQ-BUDGET-80PCT, BQ-TIME-80PCT, SW-MLFLOW-DOWN, DQ-SCHEMA-DRIFT
- MLflow повторные попытки подключения
- Failed run с traceback excerpt (первые 10 строк)
- Grace period начало (перед SIGTERM)

**ERROR**:
- MLflow search_runs упал с исключением
- budget_status.json не удалось записать (os.replace() упал)
- dvc status завершился с ненулевым кодом
- Unexpectedly отсутствующий файл (predictions.csv не создан после FINISHED run)

**CRITICAL**:
- hard_stop=true запись (любая причина)
- SW-HANG обнаружен
- DQ-DATA-MODIFIED нарушение antigoal-4
- MQ-NAN-CASCADE
- SIGTERM отправлен Claude Code
- SIGKILL отправлен Claude Code

### 5.4 Стандартные блоки в session.log

Начало сессии:
```
2026-03-19T13:10:00.000 INFO  [ResearchSessionController] === UAF Session sess_20260319_abc123 started ===
2026-03-19T13:10:00.001 INFO  [ResearchSessionController] task=tabular_classification metric=roc_auc direction=maximize
2026-03-19T13:10:00.002 INFO  [ResearchSessionController] budget=fixed max_iterations=10 max_time=4.0h
2026-03-19T13:10:00.003 INFO  [ResearchSessionController] validation=stratified_kfold k=5 runs_per_iteration=5
2026-03-19T13:10:00.100 INFO  [BudgetController] Polling thread started interval=30s hang_timeout=300s
2026-03-19T13:10:00.200 INFO  [ClaudeCodeRunner] Claude Code process started pid=12345
```

Завершение сессии:
```
2026-03-19T16:15:00.000 INFO  [BudgetController] Convergence criterion met: no_improvement=3, delta=0.0003 < 0.001
2026-03-19T16:15:00.001 CRITICAL [BudgetController] hard_stop=true reason=convergence
2026-03-19T16:15:00.002 INFO  [BudgetController] Grace period started: 300 seconds
2026-03-19T16:17:30.000 INFO  [ResearchSessionController] Claude Code exited normally (grace period)
2026-03-19T16:17:30.001 INFO  [ResultAnalyzer] Post-session analysis started
2026-03-19T16:20:00.000 INFO  [ResultAnalyzer] session_analysis.json written
2026-03-19T16:20:00.001 INFO  [ReportGenerator] LaTeX generation started
2026-03-19T16:25:00.000 INFO  [ReportGenerator] report.pdf compiled successfully (tectonic)
2026-03-19T16:25:00.001 INFO  [ResearchSessionController] === Session completed: best roc_auc=0.812 (iter 5/10) ===
```

---

## 6. Интеграция с MLflow

### 6.1 Способ интеграции: файловая система MLflow + API

BudgetController использует два канала доступа к MLflow:

**Канал 1: MLflow Python API** (приоритетный)
```python
import mlflow

# Поиск runs по session experiment
runs = mlflow.search_runs(
    experiment_ids=[session.mlflow_experiment_id],
    filter_string=f"tags.`uaf.session_id` = '{session_id}'",
    order_by=["attribute.start_time DESC"],
    max_results=100,
)
```

Используется для: получения метрик, статусов, тегов, параметров.

**Канал 2: Файловая система MLflow** (fallback при недоступности API)
```
SESSION_DIR/.mlruns/{experiment_id}/{run_id}/
    ├── meta.yaml          <- status, start_time, end_time
    ├── metrics/           <- файлы с именем метрики, содержат timestamp + value
    │   └── roc_auc        <- "1742389800.123 0.782 0"
    ├── params/            <- файлы с именем параметра
    └── tags/              <- файлы с именем тега
```

При `mlflow_tracking_uri = "file:///path"` (локальный MLflow) оба канала доступны.
При недоступности API (SW-MLFLOW-DOWN) BudgetController переходит на прямое
чтение файлов. Это менее надёжно (файлы могут быть незаконченными при активном run),
поэтому используется только как fallback.

### 6.2 MLflow теги, которые мониторинг читает

BudgetController читает следующие теги из MLflow runs:

| Тег | Кто пишет | Что означает |
|-----|-----------|--------------|
| `uaf.session_id` | UAF при старте run | принадлежность к сессии |
| `uaf.iteration` | UAF или Claude Code | номер итерации |
| `uaf.phase` | Claude Code | phase_1 / phase_2 / phase_3 |
| `uaf.stop_reason` | Claude Code | причина завершения run |
| `uaf.convergence_signal` | Claude Code | "CONVERGED" если Claude Code считает сходимость |
| `uaf.run_type` | UAF или Claude Code | experiment / shadow / analysis / planning |
| `uaf.is_baseline` | Claude Code | true для baseline run в shadow experiments |

### 6.3 MLflow метрики, которые мониторинг читает

- `{metric.name}` — основная task metric (roc_auc, rmse, etc.)
- `train_loss` — если логируется (для NaN/расходимость detection)
- `{metric.name}_val` — валидационная метрика (если логируется отдельно)
- `fi_rank_shadow_*` — feature importance ranks в shadow experiments

### 6.4 Что BudgetController пишет в MLflow

BudgetController НЕ создаёт новые MLflow runs. Он только читает.
Исключение: при hard_stop BudgetController добавляет тег к последнему
активному run через MLflow API:

```python
mlflow.set_tag("uaf.forced_stop", "true")
mlflow.set_tag("uaf.forced_stop_reason", reason)
```

Это позволяет отличить нормально завершённый run от прерванного.

### 6.5 Session Summary Run и мониторинг

После завершения Claude Code ResearchSessionController создаёт Session Summary Run.
В него включаются данные мониторинга:

```python
mlflow.log_metrics({
    "monitoring.total_alerts": len(alerts),
    "monitoring.critical_alerts": len([a for a in alerts if a["level"] == "CRITICAL"]),
    "monitoring.warning_alerts": len([a for a in alerts if a["level"] == "WARNING"]),
    "monitoring.hang_events": hang_count,
    "monitoring.oom_events": oom_count,
    "monitoring.nan_runs": nan_run_count,
    "monitoring.degradation_events": degradation_count,
    "monitoring.budget_pct_iterations_final": budget_pct_iter,
    "monitoring.budget_pct_time_final": budget_pct_time,
})
mlflow.log_param("monitoring.stop_reason", stop_reason)
mlflow.log_artifact(str(session_dir / "session.log"))
mlflow.log_artifact(str(session_dir / "budget_status.json"))
```

---

## 7. Мониторинг в LaTeX/PDF отчёте

### 7.1 Секция "Session Monitoring" в отчёте

ReportGenerator добавляет секцию мониторинга после секции "Experiment Pipeline"
и перед "Experiment Results". Секция всегда присутствует (даже если алертов не было).

Структура секции:

```latex
\section{Session Monitoring}

\subsection{Session Health Summary}
% Таблица: Software Health, Data Quality, Model Quality, Business KPIs
% Каждая строка: компонент, статус (OK/WARNING/CRITICAL), событий, примечание

\subsection{Budget Utilization}
% Горизонтальная гистограмма: использованные итерации vs максимум
% Строка: elapsed time vs max_time
% stop_reason с объяснением (convergence / budget_exhausted / user_interrupt / etc.)

\subsection{Metric Progression}
% Линейный график: metric_value по итерациям
% Горизонтальная линия: success_metric_threshold (если задан)
% Точки: цветные (зелёные=best, красные=degradation, жёлтые=failed)
% Подпись: best_metric, итерация где достигнут

\subsection{Alerts Log}
% Таблица всех алертов уровня WARNING и выше:
% iteration | timestamp | code | level | message
% Если алертов нет: "No alerts recorded during session"

\subsection{Software Health Events}
% Только если были SW-* алерты или hang events
% Описание событий с временными метками

\subsection{Data Integrity}
% DVC checksums: OK / VIOLATION
% Schema changes: none / N changes
% Antigoal-4 status: not violated / VIOLATED (красный \textcolor{red})
```

### 7.2 Графики мониторинга (matplotlib -> PDF)

**График 1: Metric Progression (обязательный)**
- x-ось: номер итерации
- y-ось: значение primary metric
- Серия: metric_value по итерациям из metrics_history
- Маркеры: звезда для is_best=true, крест для is_failed=true, треугольник для degradation_flag=true
- Горизонтальная линия: best_metric value (пунктир)
- Заголовок: f"Metric Progression: {metric.name} ({metric.direction})"

**График 2: Budget Burndown (обязательный)**
- Двойная шкала: итерации (левая) и время в часах (правая)
- Заполненная область: использованный бюджет
- Вертикальная линия: момент hard_stop или convergence
- Легенда: stop_reason

**График 3: Alert Timeline (только если warnings >= 1)**
- Временная ось с маркерами алертов
- Цвет: WARNING=жёлтый, ERROR=оранжевый, CRITICAL=красный
- Подписи: код алерта

### 7.3 Текстовая часть (генерируется Claude Code в рамках сессии)

Claude Code в финальном этапе сессии генерирует секцию "Monitoring Conclusions".
Входные данные: budget_status.json финальный, список алертов, metrics_history, stop_reason.

Генерируется и сохраняется в SESSION_DIR/report/sections/monitoring_conclusions.md:
- что прошло штатно
- что вызвало проблемы (если были алерты)
- рекомендации по следующей сессии исходя из паттерна мониторинга

Объём: 150-300 слов. UAF ReportGenerator читает файл и встраивает в LaTeX
как \subsection{Monitoring Conclusions}.

---

## 8. Новые компоненты и изменения в существующих

### 8.1 Изменения в BudgetController

BudgetController v1.0 (стадия 09): polling MLflow, budget check, hard_stop, grace period.

BudgetController v2.0 (стадия 13): добавляются:
- Software Health checks в цикл polling
- Data Quality checks каждые 60 сек
- Реестр алертов + запись в budget_status.json.alerts[]
- Hints механизм -> budget_status.json.hints[]
- Hang detection с hang_timeout_seconds
- stderr streaming от ClaudeCodeRunner для OOM detection

### 8.2 Изменения в ClaudeCodeRunner

ClaudeCodeRunner добавляет:
- `last_output_at` — timestamp последней строки stdout/stderr
- stderr stream -> BudgetController через queue (non-blocking)
- Метод `get_stderr_buffer(n_lines=50)` для BudgetController

### 8.3 Новый компонент: MonitoringDashboard (опциональный)

При запуске с флагом `--monitor` UAF запускает простой терминальный dashboard
(Rich library) рядом с основным процессом. Показывает в реальном времени:

```
UAF Session sess_20260319_abc123  [RUNNING]  13:10 elapsed

Software Health    [OK]
Data Quality       [OK]
Model Quality      [OK]  best roc_auc=0.801 (iter 2)
Business KPIs      [WARNING]  3/10 iterations (30%), 1.2h/4.0h (30%)

Last iteration: 3  |  metric=0.795  |  degradation WARNING
Active alerts: 1   |  MQ-DEGRADATION (iter 3)

[BudgetController] Poll #42 completed (0.2s)
```

Обновление: каждые 30 секунд (синхронно с polling циклом).
Реализация: Rich Live + Layout + Table. Читает budget_status.json.
Не блокирует основной поток — отдельный thread.

---

## 9. Antigoals и мониторинг

Мониторинг должен соблюдать antigoals из стадии 01:

**Antigoal 1** (не AutoML для production): мониторинг не следит за деградацией
в production — только за экспериментами в сессии. Данный артефакт это фиксирует.

**Antigoal 2** (не запускает без одобрения): мониторинг не может инициировать
новые итерации или изменять program.md самостоятельно. Он только останавливает.

**Antigoal 3** (не скрывает неудачные эксперименты): все failed runs,
all алерты включая CRITICAL — видны в отчёте. Нет фильтрации "плохих" данных.

**Antigoal 4** (не модифицирует данные): DQ-DATA-MODIFIED триггерит hard_stop.
Мониторинг сам не пишет в директории данных пользователя.

**Antigoal 6** (не превышает бюджет): BQ-BUDGET-EXHAUSTED -> hard_stop немедленно.
FN дороже FP для budget enforcement.

---

## 10. Взаимодействие мониторинга с остальными компонентами

```
ClaudeCodeRunner ----stderr/stdout----> BudgetController.stderr_queue
                                              |
MLflow API <--------search_runs()------  BudgetController.polling_loop()
                                              |
                                        budget_status.json (атомарно)
                                              |
                       Claude Code <---reads--- (BUDGET-CHECK секция)
                       Claude Code ---hints--> реакция в следующей итерации
                                              |
                                        session.log (append)
                                              |
                      MonitoringDashboard <---reads--- (optional --monitor)
                                              |
                      ResultAnalyzer <--metrics_history--- (post-session)
                                              |
                      ReportGenerator <--budget_status.json (final), session.log---
```

---

## 11. Решения стадии 13 (сводка)

**R-13-01: Мониторинг = 4-уровневая пирамида, адаптированная к сессионному контексту**
SW Health -> DQ -> Model Quality -> Business KPIs. Нижний уровень — предусловие
для значимости верхнего. Software Health проверяется первым в каждом цикле.

**R-13-02: Polling 30 сек — единый интервал для всех уровней кроме DQ (60 сек)**
Обоснование: overhead < 1%, OOM читается из stderr потоково без задержки.

**R-13-03: budget_status.json — единая точка коммуникации BudgetController -> Claude Code**
Атомарная запись (tmp + os.replace). Поле hints[] — механизм подсказок.
Расширение схемы v2.1: software_health, data_quality, alerts[], hints[], phase.

**R-13-04: session.log — структурированный текст для человека, не JSON**
Удобен для `tail -f` и grep во время сессии. JSON только в budget_status.json.

**R-13-05: MLflow интеграция через API (приоритет) + файловая система (fallback)**
BudgetController только читает MLflow. Исключение: тег forced_stop при hard_stop.

**R-13-06: Алерты реестром (код + уровень + итерация + resolved)**
Все алерты хранятся в budget_status.json.alerts[] и session.log.
В LaTeX отчёт попадают все WARNING и выше.

**R-13-07: Hang detection консервативный (FP допустим)**
300 сек CPU=0% -> 60 сек ожидание -> повторная проверка -> hard_stop.
FP (ложное завершение) стоит одной итерации. FN (реальный hang) стоит всего бюджета.

**R-13-08: MonitoringDashboard опциональный (--monitor флаг)**
Rich-based, читает budget_status.json. Не блокирует BudgetController.

**R-13-09: Metrics Progression и Budget Burndown — обязательные графики в отчёте**
Alert Timeline — только при наличии WARNING+. Claude Code генерирует Monitoring Conclusions в рамках сессии.

---

## STAGE COMPLETE

Стадия 13-monitoring завершена 2026-03-19.

Артефакт: `docs/stage-13-monitoring.md`

Следующая стадия: **14-serving** (SKIPPED/antigoal-1 — нет production деплоя).
После: **15-ownership**.
