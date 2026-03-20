# Стадия 04: Metrics

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-problem (COMPLETE), 02-research (COMPLETE), 03-design-doc v2.0 (COMPLETE)

---

## 0. Контекст: почему метрики UAF отличаются от стандартной схемы

UAF — не ML-модель. У неё нет trainable параметров, нет градиентного спуска, нет
классической loss function. Трёхуровневая схема (loss / offline / online) применима,
но требует переосмысления:

| Уровень в классическом ML | Аналог в UAF |
|--------------------------|--------------|
| Loss function | Метрика качества конкретного ML-эксперимента — задаётся в task.yaml, разная для каждой задачи. UAF не оптимизирует её сам — её оптимизирует Claude Code в сессии |
| Offline metric | Метрика качества UAF как системы — насколько хорошо UAF управляет сессией, генерирует планы, экономит бюджет |
| Online metric | Метрика ценности для ML-инженера в реальном использовании — время, воспроизводимость, качество отчёта |

Принципиальное следствие: метрика уровня 1 (task metric) не фиксирована в UAF.
Она задаётся пользователем через task.yaml и меняется от задачи к задаче.
UAF обязан правильно передать эту метрику Claude Code и правильно отразить
её в MLflow и отчёте — это его ответственность, а не сама оптимизация.

---

## 1. Уровень 1: Task Metric (аналог Loss Function)

### 1.1 Определение

Task metric — метрика, которую Claude Code оптимизирует в ходе ML-экспериментов
внутри сессии. Определяется пользователем в `task.yaml` под ключом `metric`.

Это единственная метрика, которую UAF передаёт Claude Code как цель.
UAF не выбирает метрику, не меняет её в ходе сессии, не агрегирует между задачами.

### 1.2 Схема task.yaml для метрики

```yaml
metric:
  name: roc_auc               # имя метрики (логируется в MLflow как ключ)
  direction: maximize         # maximize | minimize
  primary: true               # флаг: эта метрика — основная для сравнения runs
  threshold: null             # опционально: минимальный приемлемый результат
  secondary_metrics:          # опционально: дополнительные метрики для логирования
    - precision
    - recall
    - f1
```

Claude Code логирует `metric.name` в MLflow как `mlflow.log_metric(name, value)`.
BudgetController читает эту же метрику из MLflow для проверки сходимости.

### 1.3 Стандартные метрики по типу задачи

UAF не навязывает метрику. Приведённые значения — дефолты в task.yaml шаблонах
(примеры из `examples/` директории), а также подсказки для ProgramMdGenerator.

**Tabular Classification:**

| Задача | Рекомендуемая metric.name | direction |
|--------|--------------------------|-----------|
| Бинарная классификация | `roc_auc` | maximize |
| Мультиклассовая классификация | `macro_f1` | maximize |
| Сильный дисбаланс классов | `average_precision` | maximize |
| Задача с бизнес-порогом | `f1` | maximize |

Дополнительно логируются по умолчанию (если tabular + classification):
`accuracy`, `precision`, `recall`, `log_loss`.

**Tabular Regression:**

| Задача | Рекомендуемая metric.name | direction |
|--------|--------------------------|-----------|
| Общий случай | `rmse` | minimize |
| Устойчивость к выбросам | `mae` | minimize |
| Относительная ошибка | `mape` | minimize |
| Объяснение дисперсии | `r2` | maximize |

Дополнительно: `mse`, `median_ae`.

**NLP / Language Modelling:**

| Задача | Рекомендуемая metric.name | direction |
|--------|--------------------------|-----------|
| Языковое моделирование | `val_bpb` (bits per byte) | minimize |
| Языковое моделирование (альт.) | `val_loss` (cross-entropy) | minimize |
| Классификация текста | `roc_auc` или `macro_f1` | maximize |
| Sequence labeling (NER) | `span_f1` | maximize |
| Машинный перевод | `bleu` | maximize |
| Суммаризация | `rouge_l` | maximize |
| Semantic similarity | `pearson_r` | maximize |

**CV / Computer Vision:**

| Задача | Рекомендуемая metric.name | direction |
|--------|--------------------------|-----------|
| Image classification | `top1_accuracy` | maximize |
| Object detection | `map50` (mAP@0.5) | maximize |
| Semantic segmentation | `miou` | maximize |
| Instance segmentation | `map50_95` | maximize |
| Image generation quality | `fid` | minimize |

**Ranking / RecSys:**

| Задача | Рекомендуемая metric.name | direction |
|--------|--------------------------|-----------|
| Рекомендации | `ndcg_at_10` | maximize |
| Поиск | `mrr` | maximize |
| Ранжирование | `map_at_k` | maximize |

**Reinforcement Learning:**

| Задача | Рекомендуемая metric.name | direction |
|--------|--------------------------|-----------|
| Общий случай | `mean_episode_reward` | maximize |
| Стабильность обучения | `episode_reward_std` | minimize |

### 1.4 Ответственность UAF по отношению к task metric

UAF обеспечивает следующее (не оптимизацию, а инфраструктуру):

1. **Передача метрики в Claude Code:** `metric.name` и `metric.direction`
   прописаны в секции Task Description файла program.md и в Execution Instructions.

2. **Унифицированное логирование:** Claude Code логирует метрику в MLflow
   по имени из `metric.name`. Имя метрики фиксировано на всё время сессии.
   Это обеспечивает сравнимость runs.

3. **Использование в BudgetController (dynamic mode):** BudgetController
   читает значения метрики из MLflow и проверяет критерий сходимости.
   Направление (`direction`) учитывается при вычислении delta.

4. **Отображение в отчёте:** ReportGenerator использует `metric.name` как
   основную ось в таблицах и графиках. Лучший run определяется по этой метрике
   с учётом `direction`.

5. **Не валидирует:** UAF не проверяет, что Claude Code действительно
   оптимизирует нужную метрику. Это задача пользователя при review program.md.

---

## 2. Уровень 2: UAF System Quality Metrics (аналог Offline Metric)

Метрики, которые отвечают на вопрос: насколько хорошо UAF выполняет свою роль
как системы управления экспериментами. Измеряются на тестовых сессиях и
при ретроспективном анализе реальных сессий.

### 2.1 Метрики качества plan generation (ProgramMdGenerator)

**M-UAF-01: Plan Structural Completeness**

```
plan_completeness = (секции присутствующие) / (секции обязательные)

Обязательные секции: Metadata, Task Description, Research Phases,
Execution Instructions, Current Status, Iteration Log, Final Conclusions.

Целевое значение: 1.0 (все секции)
Измерение: автоматически при валидации после генерации program.md
Где логируется: MLflow planning run, тег `plan_completeness`
```

**M-UAF-02: Plan Phase Coverage**

```
plan_phase_coverage = (фазы в program.md) / (ожидаемые фазы по типу задачи)

Ожидаемые фазы минимум: baseline + improvement_iterations.
Целевое значение: >= 2 фазы для любой задачи
Где логируется: MLflow planning run, тег `plan_phases_count`
```

**M-UAF-03: Plan Generation Retry Rate**

```
retry_rate = попытки_генерации > 1 / всего_сессий

Допустимое значение: < 10% сессий требуют retry при валидации структуры.
Если > 10%: пересматривать prompt в ProgramMdGenerator.
Где логируется: MLflow planning run, тег `generation_attempts`
```

**M-UAF-04: Approval Wait Time**

```
approval_wait_time_seconds: время от показа program.md до y/n/edit

Это не метрика качества UAF в чистом виде — это характеристика взаимодействия.
Логируется для понимания: если время < 30 секунд, человек не читал plan.
Если время > 3600 секунд — сессия была отложена или заброшена.

Целевое значение: 120–900 секунд (читает и принимает осознанно)
Где логируется: MLflow planning run, тег `approval_wait_time_seconds`
```

### 2.2 Метрики эффективности бюджета (BudgetController)

**M-UAF-05: Budget Utilization Rate (fixed mode)**

```
budget_utilization = consumed_iterations / max_iterations

Целевое значение для fixed mode: > 0.5 (система работает, не останавливается рано)
Если < 0.3: Claude Code остановился досрочно — возможно convergence_signal слишком ранний
Если = 1.0: бюджет исчерпан без сходимости — рассмотреть увеличение бюджета

Где логируется: MLflow session summary run (финальный run с типом `session_summary`)
Тег: `budget_utilization`
```

**M-UAF-06: Convergence Detection Rate (dynamic mode)**

```
convergence_detected = True если сессия завершилась по критерию сходимости
                       False если завершилась по safety cap

Целевое значение: convergence в >= 60% сессий в dynamic mode.
Если < 60%: safety cap срабатывает слишком часто — либо задачи слишком трудные,
либо min_delta слишком маленький.

Где логируется: MLflow session summary run, тег `convergence_detected`
```

**M-UAF-07: Metric Progress Rate**

```
progress_rate = (best_metric_final - best_metric_baseline) / best_metric_baseline

Только для maximize задач (для minimize: инвертировать знаменатель).
Показывает: насколько Claude Code улучшил метрику относительно baseline в рамках сессии.

Целевое значение: зависит от задачи. Тревожный сигнал: < 0.01 (1%) — сессия
не улучшила baseline, возможно budget слишком мал или задача невозможна без
дополнительных данных.

Где логируется: MLflow session summary run, метрика `metric_progress_rate`
```

**M-UAF-08: Hard Stop Rate**

```
hard_stop_rate = сессии завершённые через SIGTERM / всего сессий

Целевое значение: < 5%
Если > 5%: Claude Code регулярно игнорирует budget_status.json или grace period
мал. Пересмотреть grace period или усилить инструкцию в program.md.

Где логируется: MLflow session summary run, тег `terminated_by_sigterm`
```

### 2.3 Метрики качества кода (RuffEnforcer)

**M-UAF-09: Ruff Clean Rate (post-processing)**

```
ruff_clean_rate = файлы без unfixable нарушений / всего .py файлов в сессии

Целевое значение: >= 0.95 (95% файлов чистые после уровня 1 enforcement в Claude Code)
Если < 0.95: инструкция в program.md недостаточна — усилить.

Где логируется: MLflow session summary run, метрика `ruff_clean_rate`
Также в: ReportGenerator, секция Code Quality Report
```

**M-UAF-10: Ruff Violations Per File (post-processing)**

```
violations_per_file = суммарные unfixable нарушения / всего файлов

Целевое значение: < 1.0 (в среднем меньше одного unfixable нарушения на файл)
Где логируется: MLflow session summary run, метрика `ruff_violations_per_file`
```

### 2.4 Метрики воспроизводимости (MLflow + DVC)

**M-UAF-11: MLflow Logging Compliance**

```
mlflow_compliance = runs с обязательными полями / всего experiment runs

Обязательные поля: тег `session_id`, тег `type=experiment`, тег `status`,
                   хотя бы одна метрика, хотя бы один параметр, файл кода как артефакт.

Целевое значение: 1.0
Где проверяется: ResultCollector после завершения сессии
Где логируется: MLflow session summary run, тег `mlflow_compliance`
```

**M-UAF-12: DVC Commit Rate**

```
dvc_commit_rate = шаги с DVC коммитом / всего завершённых шагов

Целевое значение: >= 0.95
Если < 0.95: Claude Code не выполняет DVC протокол — усилить инструкцию или
добавить принудительный DVC коммит в ClaudeCodeRunner.

Где проверяется: DVCSetup проверяет git log на наличие commit'ов с mlflow_run_id
Где логируется: MLflow session summary run, метрика `dvc_commit_rate`
```

### 2.5 Метрики качества сессии в целом

**M-UAF-13: Failed Experiment Rate**

```
failed_experiment_rate = runs с тегом status=failed / всего experiment runs

Информационная метрика — не плохо и не хорошо само по себе.
Если = 0: либо задача тривиальна, либо Claude Code не пробует рискованные подходы.
Если > 0.5: задача слишком сложная или бюджет слишком мал — нет времени исправить ошибки.

Целевое значение: 0.1–0.3 (здоровый уровень риска в исследовании)
Где логируется: MLflow session summary run, метрика `failed_experiment_rate`
```

**M-UAF-14: Iteration Efficiency**

```
iteration_efficiency = уникальные_гипотезы_протестированы / всего_runs

Близкое к 1.0 значение: каждый run проверяет новую гипотезу.
Близкое к 0.5: много повторных runs (перезапуски, дебаг того же кода).

Целевое значение: >= 0.7
Где логируется: MLflow session summary run, метрика `iteration_efficiency`
```

---

## 3. Уровень 3: Online Metrics (ценность для ML-инженера)

Метрики реального использования. Измеряются по итогам нескольких сессий.
Часть из них — субъективные (опрос пользователя или self-assessment).

### 3.1 Временная экономия

**M-ONLINE-01: Active Time Saved**

```
Определение: время, которое ML-инженер потратил бы на те же эксперименты вручную
             минус активное время, потраченное с UAF.

Активное время с UAF = время на написание task.yaml + review program.md
                       + review отчёта.
Оценка ручного времени = из task.yaml поле `manual_baseline_hours` (опционально,
заполняется пользователем для сравнения).

Целевое значение: > 5x экономия для типичной сессии из 10+ итераций.
Как измерять: UAF логирует `approval_wait_time_seconds` и `total_session_hours`.
Активное время = approval_wait_time + report_review_time (вручную оценивается).

Где логируется: не в MLflow автоматически (субъективная компонента).
Фиксируется в финальном отчёте в секции Reproducibility как информация о сессии.
```

**M-ONLINE-02: Session Wall Clock Time**

```
total_session_hours: время от запуска uaf run до получения report.pdf

Логируется автоматически: MLflow session summary run, метрика `session_wall_clock_hours`
Также: в терминальном summary при завершении сессии.

Информационная метрика. Позволяет планировать следующие сессии.
```

### 3.2 Воспроизводимость

**M-ONLINE-03: Reproducibility Score**

```
Бинарная метрика — можно ли воспроизвести лучший run из данной сессии.

Критерии воспроизводимости (все должны выполняться):
  1. MLflow run_id лучшего run существует
  2. DVC commit для этого run существует
  3. Код эксперимента (артефакт в MLflow) существует
  4. Зависимости зафиксированы (requirements.txt или pyproject.toml в артефактах)
  5. seed зафиксирован в MLflow params (тег `seed`)

reproducibility_score = выполненные_критерии / 5

Целевое значение: = 1.0 (все 5 критериев)
Где логируется: MLflow session summary run, метрика `reproducibility_score`
Также в: ReportGenerator, секция Reproducibility
```

**M-ONLINE-04: Cross-Reference Integrity**

```
cross_ref_integrity = runs с валидным dvc_commit / всего experiment runs

Проверяется: для каждого MLflow run с тегом `dvc_commit` выполнить
             git cat-file -t {sha} — существует ли этот commit.

Целевое значение: = 1.0
Где проверяется: ResultCollector после сессии
Где логируется: MLflow session summary run, тег `cross_ref_integrity`
```

### 3.3 Качество отчёта

**M-ONLINE-05: Report Generation Success Rate**

```
Бинарная метрика на сессию: был ли сгенерирован report.pdf.

Причины неудачи:
  - tectonic и pdflatex не установлены (падback: только .tex)
  - LLM API недоступен (fallback: статические секции без LLM-текста)
  - Нет данных в MLflow (пустая сессия)

report_generated = True если report.pdf создан
tex_generated = True если report.tex создан (слабый fallback)

Целевое значение: report_generated = True в >= 95% сессий
Где логируется: MLflow session summary run, теги `report_pdf_generated`, `report_tex_generated`
```

**M-ONLINE-06: Failed Experiments Coverage in Report**

```
failed_coverage = failed runs отображённых в отчёте / всего failed runs в MLflow

Целевое значение: = 1.0 (antigoal 3: не скрывать неудачи)
Где проверяется: ReportGenerator проверяет что секция Failed Experiments содержит
                 все runs с тегом status=failed
Где логируется: MLflow session summary run, тег `failed_coverage`
```

### 3.4 Операционные метрики

**M-ONLINE-07: UAF Process Stability**

```
Бинарная метрика на сессию: завершился ли UAF-процесс без unhandled exception.

Причины сбоев UAF (не Claude Code):
  - MLflow server не запустился
  - Anthropic API timeout при генерации program.md
  - DVC init failed
  - ReportGenerator unhandled exception

uaf_stable = True если ResearchSessionController завершился в состоянии DONE
             или REPORTING_PARTIAL (не через unhandled exception)

Целевое значение: >= 0.99
Где логируется: статус UAF-процесса в stderr при завершении + MLflow session summary
                тег `uaf_exit_status` = success | partial | error
```

**M-ONLINE-08: Antigoal Violation Rate**

```
Проверка что система не нарушила antigoals в ходе сессии.

antigoal_violations: список нарушений. Каждое — отдельная проверка:
  AG2: Claude Code запустился без approval? (проверяется по MLflow tag approval_status)
  AG3: failed runs скрыты в отчёте? (failed_coverage < 1.0)
  AG4: файлы данных модифицированы? (проверяется через git diff data/)
  AG6: бюджет превышен? (consumed > limit + 10% погрешности)

Целевое значение: 0 нарушений на сессию
Где логируется: MLflow session summary run, тег `antigoal_violations_count`
При ненулевом значении: предупреждение в терминале при завершении сессии
```

---

## 4. MLflow: что логируется обязательно (framework level)

Это то, что UAF логирует независимо от типа задачи и содержимого task.yaml.
Claude Code логирует task-specific метрики сам по инструкции в program.md.

### 4.1 Planning Run (тип: planning)

Создаётся MLflowSetup при инициализации сессии.

```
Run name: planning/initial
Tags:
  type = planning
  session_id = {uuid}
  approval_status = pending | approved | rejected
  approval_mode = standard | fully_autonomous
  approval_wait_time_seconds = {float}
  generation_attempts = {int}
  plan_completeness = {float}
  plan_phases_count = {int}
Params:
  session_id = {uuid}
  task_title = {str}
  budget_mode = fixed | dynamic
  budget_max_iterations = {int | null}
  budget_max_cost_usd = {float | null}
  budget_max_time_hours = {float | null}
  budget_patience = {int | null}
  budget_min_delta = {float | null}
  claude_model = {str}
  uaf_version = {str}
Artifacts:
  program.md (draft)
  program_approved.md (после одобрения, перезаписывается)
  task.yaml
  budget.yaml
```

### 4.2 Experiment Run (тип: experiment)

Создаётся Claude Code в каждом эксперименте (по инструкции из program.md).
UAF не контролирует этот run напрямую, но проверяет его наличие через M-UAF-11.

```
Run name: {phase_id}/{step_id}
Tags (обязательные, проверяются M-UAF-11):
  type = experiment
  session_id = {uuid}
  step_id = {str}
  status = running | success | failed
  critical = true | false
  convergence_signal = {float 0.0-1.0}
  dvc_commit = {git_sha}
Params (обязательные):
  step_id = {str}
  session_id = {uuid}
  seed = {int}
  + task-specific hyperparameters
Metrics:
  {metric.name} = {float}   <- основная метрика из task.yaml
  + secondary_metrics (если указаны в task.yaml)
Artifacts:
  {step_id}.py              <- код эксперимента
  stdout.log
  stderr.log
  traceback.txt             <- только если status=failed
```

### 4.3 Session Summary Run (тип: session_summary)

Создаётся UAF (ResultCollector) после завершения сессии. Содержит агрегированные
метрики уровней 2 и 3.

```
Run name: summary/final
Tags:
  type = session_summary
  session_id = {uuid}
  uaf_exit_status = success | partial | error
  terminated_by_sigterm = true | false
  convergence_detected = true | false
  report_pdf_generated = true | false
  report_tex_generated = true | false
  antigoal_violations_count = {int}
  failed_coverage = {float}
  mlflow_compliance = {float}
  cross_ref_integrity = {float}
Metrics:
  budget_utilization = {float}
  metric_progress_rate = {float}
  ruff_clean_rate = {float}
  ruff_violations_per_file = {float}
  failed_experiment_rate = {float}
  iteration_efficiency = {float}
  reproducibility_score = {float}
  session_wall_clock_hours = {float}
Params:
  total_runs = {int}
  successful_runs = {int}
  failed_runs = {int}
  best_run_id = {str}
  best_metric_value = {float}
  best_metric_name = {str}
  baseline_metric_value = {float}
Artifacts:
  session_summary.json      <- все поля в машиночитаемом формате
```

---

## 5. Что попадает в LaTeX/PDF отчёт из метрик

ReportGenerator читает данные из MLflow и строит отчёт. Ниже — точное
соответствие метрик и секций отчёта.

### 5.1 Executive Summary (LLM-генерируемая секция)

Входные данные для LLM:
- Лучший run: `best_run_id`, `best_metric_value`, `best_metric_name`
- Прогресс: `metric_progress_rate` (насколько улучшили baseline)
- Статистика сессии: `total_runs`, `successful_runs`, `failed_runs`
- Режим бюджета и факт сходимости/исчерпания

Что выводится в секции:
- Одно-двух абзацный текст: главный результат + рекомендация
- Таблица: best result vs baseline
- Флаг: converged | budget exhausted

### 5.2 Experiment Results — Overview Table

Таблица для каждого experiment run (включая failed):

| Поле в таблице | Источник в MLflow |
|----------------|-------------------|
| Step ID | тег `step_id` |
| Phase | часть run name |
| Status | тег `status` |
| Primary Metric | `{metric.name}` |
| Key Params | параметры run (топ-3 по важности) |
| MLflow Run ID | run.info.run_id |
| Notes | тег `failure_reason` (если failed) |

Failed runs отображаются серым шрифтом в LaTeX, с текстом failure_reason.
Это прямое следствие antigoal 3.

### 5.3 Analysis and Findings (LLM-генерируемая секция)

Входные данные для LLM:
- Все experiment runs: параметры + метрики + статусы
- Финальный program.md (секция Final Conclusions, написанная Claude Code)
- `iteration_efficiency`, `failed_experiment_rate`, `metric_progress_rate`

Что выводится:
- Паттерны в данных: какие параметры влияли на метрику
- Анализ неудачных экспериментов: что не сработало и почему
- Достигнута ли цель из task.yaml (threshold, если был задан)

### 5.4 Code Quality Report (автоматическая секция)

Источник: RuffReport от RuffEnforcer.

Выводится:
- `ruff_clean_rate`: X% файлов чисты
- `ruff_violations_per_file`: среднее нарушений на файл
- Таблица: файл -> violations count -> unfixable violations
- Для unfixable: список с кодом правила и описанием

### 5.5 Reproducibility (автоматическая секция)

Источник: MLflow session summary run + DVC git log.

Выводится:
- `session_id`
- `best_run_id` и `dvc_commit` для лучшего run
- Версии зависимостей (из requirements.txt если есть)
- `claude_model` (из planning run params)
- `seed` значения для каждого run
- `reproducibility_score`: X/5 критериев
- Инструкция воспроизведения: `git checkout {sha} && dvc checkout`

### 5.6 Figures (matplotlib -> PDF)

Три стандартных графика, генерируются автоматически:

**Figure 1: Metric Over Iterations**
```
Ось X: номер iteration (порядок запуска runs)
Ось Y: значение metric.name
Линия: best so far (накопленный максимум/минимум)
Точки: отдельные runs (зелёные = success, красные = failed)
Аннотация: budget hard stop line если применимо
```

**Figure 2: Hyperparameter Importance** (только если было HPO)
```
Если в сессии были runs с вариативными гиперпараметрами:
Bar chart: параметр -> корреляция с metric.name (Spearman)
Показывает какие параметры наиболее влияли на результат
```

**Figure 3: Runs Comparison Heatmap**
```
Строки: runs (только successful)
Колонки: основные параметры + metric
Heatmap: нормализованные значения
Позволяет визуально сравнить runs по всем измерениям сразу
```

---

## 6. Критерии остановки в динамическом режиме

Формализация TR-5 через метрики. Критерии остановки — это decision function
BudgetController в dynamic mode.

### 6.1 Три класса критериев

**Класс A: Safety Cap (жёсткий, всегда проверяется первым)**

```python
def is_safety_cap_exceeded(status: BudgetStatus, config: BudgetConfig) -> bool:
    return (
        status.consumed_iterations >= config.safety_cap_iterations
        or status.consumed_cost_usd >= config.safety_cap_cost_usd
        or status.consumed_time_hours >= config.safety_cap_time_hours
    )
```

Если True — немедленный hard stop. Не зависит ни от каких метрик.

**Класс B: Metric Convergence (алгоритмический)**

```python
def is_metric_converged(
    metrics_history: list[float],
    direction: Literal["maximize", "minimize"],
    patience: int,
    min_delta: float,
    min_iterations: int,
) -> bool:
    """Все три условия должны выполняться одновременно."""
    if len(metrics_history) < min_iterations:
        return False

    # Нормализованные дельты за последние patience итераций
    recent = metrics_history[-patience:]
    deltas = [
        abs(recent[i] - recent[i - 1]) / (abs(recent[i - 1]) + 1e-10)
        for i in range(1, len(recent))
    ]

    # Условие 1: нет улучшения в последние patience итераций
    no_improvement = all(d < min_delta for d in deltas)

    # Условие 2: абсолютная проверка — лучший результат не менялся
    best = max(metrics_history) if direction == "maximize" else min(metrics_history)
    recent_best = max(recent) if direction == "maximize" else min(recent)
    no_best_improvement = abs(best - recent_best) / (abs(best) + 1e-10) < min_delta

    # Условие 3: минимальное число итераций пройдено
    min_iter_met = len(metrics_history) >= min_iterations

    return no_improvement and no_best_improvement and min_iter_met
```

Примечание: `metrics_history` — список значений основной метрики (metric.name)
из всех successful experiment runs, в хронологическом порядке.

**Класс C: LLM Convergence Signal (эвристический)**

```python
def is_llm_signal_convergence(
    convergence_signals: list[float],
    threshold: float,
    min_iterations: int,
    consecutive_required: int = 2,
) -> bool:
    """
    Срабатывает если Claude Code несколько раз подряд сигнализировал о сходимости.
    convergence_signal: float 0.0-1.0, пишется Claude Code в MLflow tag.
    """
    if len(convergence_signals) < min_iterations:
        return False

    recent = convergence_signals[-consecutive_required:]
    return all(s >= threshold for s in recent)
```

Дефолтные значения: `threshold=0.9`, `consecutive_required=2`.

### 6.2 Итоговый критерий остановки

```python
def should_stop(
    status: BudgetStatus,
    config: BudgetConfig,
    metrics_history: list[float],
    convergence_signals: list[float],
    direction: Literal["maximize", "minimize"],
) -> tuple[bool, str]:
    """
    Возвращает (stop: bool, reason: str).
    Порядок проверки зафиксирован — важен.
    """
    # 1. Safety cap всегда первый
    if is_safety_cap_exceeded(status, config):
        return True, "safety_cap_exceeded"

    # 2. Метрическая сходимость (алгоритмическая)
    if is_metric_converged(
        metrics_history,
        direction,
        config.patience,
        config.min_delta,
        config.min_iterations,
    ):
        return True, "metric_converged"

    # 3. LLM-сигнал (дополнительный, не заменяет метрическую)
    if is_llm_signal_convergence(
        convergence_signals,
        threshold=0.9,
        min_iterations=config.min_iterations,
    ):
        return True, "llm_signal_convergence"

    return False, "continue"
```

### 6.3 Конфигурация критериев по умолчанию

```yaml
budget:
  mode: dynamic
  convergence:
    patience: 3               # минимум 3 итерации без улучшения
    min_delta: 0.001          # 0.1% относительного изменения — порог "нет улучшения"
    min_iterations: 3         # не останавливаться до 3-й итерации в любом случае
    llm_signal_threshold: 0.9
    llm_signal_consecutive: 2
  safety_cap:
    max_iterations: 50
    max_cost_usd: 20.0
    max_time_hours: 24.0
```

**Обоснование значений по умолчанию:**

- `patience=3`: меньше 3 — слишком агрессивная остановка (случайный plateau).
  Больше 5 — долго ждать если сходимость уже очевидна.

- `min_delta=0.001`: для большинства метрик (ROC-AUC, RMSE) изменение < 0.1%
  практически незначимо. Для NLP (val_bpb) — аналогично.
  Пользователь может уменьшить до 0.0001 для тонкой оптимизации.

- `min_iterations=3`: нельзя останавливаться раньше чем после baseline + 2 итераций.
  Иначе сходимость на "baseline" — не информативный результат.

- `safety_cap_iterations=50`: для одиночной задачи на локальной машине 50 итераций
  — разумный верхний предел. Claude Code unlikely нужно больше для исследовательской
  задачи (не production HPO sweep).

### 6.4 Метрики для мониторинга критериев остановки

Следующие значения логируются в `budget_status.json` при каждом polling
(BudgetController обновляет файл каждые 30 секунд):

```json
{
  "hard_stop": false,
  "reason": null,
  "consumed_iterations": 7,
  "consumed_cost_usd": 1.23,
  "consumed_time_hours": 2.1,
  "remaining_fraction": 0.65,
  "metrics_history": [0.72, 0.74, 0.751, 0.758, 0.761, 0.762, 0.762],
  "convergence_signals": [0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95],
  "metric_converged": false,
  "llm_signal_convergence": true,
  "safety_cap_fraction": 0.14,
  "warning_triggered": false
}
```

Это то, что Claude Code видит при проверке budget_status.json.
Логика остановки на стороне UAF, Claude Code только читает `hard_stop`.

---

## 7. Сводная таблица метрик

| ID | Название | Уровень | Тип | Целевое значение | Где логируется |
|----|---------|---------|-----|-----------------|----------------|
| M-UAF-01 | plan_completeness | UAF System | авто | = 1.0 | MLflow planning run |
| M-UAF-02 | plan_phases_count | UAF System | авто | >= 2 | MLflow planning run |
| M-UAF-03 | generation_attempts | UAF System | авто | < 1.1 avg | MLflow planning run |
| M-UAF-04 | approval_wait_time_seconds | UAF System | авто | 120–900 | MLflow planning run |
| M-UAF-05 | budget_utilization | UAF System | авто | > 0.5 (fixed) | MLflow session summary |
| M-UAF-06 | convergence_detected | UAF System | авто | >= 60% сессий | MLflow session summary |
| M-UAF-07 | metric_progress_rate | UAF System | авто | > 0.01 | MLflow session summary |
| M-UAF-08 | hard_stop_rate | UAF System | авто | < 5% сессий | MLflow session summary |
| M-UAF-09 | ruff_clean_rate | UAF System | авто | >= 0.95 | MLflow session summary |
| M-UAF-10 | ruff_violations_per_file | UAF System | авто | < 1.0 | MLflow session summary |
| M-UAF-11 | mlflow_compliance | UAF System | авто | = 1.0 | MLflow session summary |
| M-UAF-12 | dvc_commit_rate | UAF System | авто | >= 0.95 | MLflow session summary |
| M-UAF-13 | failed_experiment_rate | UAF System | авто | 0.1–0.3 | MLflow session summary |
| M-UAF-14 | iteration_efficiency | UAF System | авто | >= 0.7 | MLflow session summary |
| M-ONLINE-01 | active_time_saved | Online | частично субъект. | > 5x | report.pdf |
| M-ONLINE-02 | session_wall_clock_hours | Online | авто | — (инфо) | MLflow session summary |
| M-ONLINE-03 | reproducibility_score | Online | авто | = 1.0 | MLflow session summary |
| M-ONLINE-04 | cross_ref_integrity | Online | авто | = 1.0 | MLflow session summary |
| M-ONLINE-05 | report_pdf_generated | Online | авто | >= 0.95 | MLflow session summary |
| M-ONLINE-06 | failed_coverage | Online | авто | = 1.0 | MLflow session summary |
| M-ONLINE-07 | uaf_exit_status | Online | авто | = success | MLflow session summary |
| M-ONLINE-08 | antigoal_violations_count | Online | авто | = 0 | MLflow session summary |

Task metrics (уровень 1): определяются пользователем в task.yaml (не фиксированы).

---

## 8. Стоимость ошибок: FP vs FN для ключевых метрик

UAF как система управления содержит несколько мест где возможны ошибочные решения.

### 8.1 BudgetController: ранняя остановка vs продолжение

```
FP (ложное срабатывание): BudgetController решил что сходимость достигнута,
                           остановил сессию, но реально оставалось ещё 20%
                           потенциального улучшения.
Цена FP: недополученное качество модели. Субъективно неприятно, но некритично.
Пользователь может перезапустить сессию с --resume.

FN (пропуск остановки): сходимость реально достигнута, но BudgetController
                        продолжает сессию, тратя бюджет впустую.
Цена FN: потраченные деньги (API токены) и время. При fixed mode нарушение antigoal 6.

Вывод: FN дороже FP для fixed mode (antigoal). Для dynamic mode — примерно одинаково.
Дизайн: `patience=3` и требование трёх условий одновременно делают алгоритм
консервативным (избегает FP). Для fixed mode предпочтительнее.
```

### 8.2 HumanOversightGate: пропуск некорректного плана

```
FP (одобрение плохого плана): пользователь сказал y, Claude Code исполнил план
                               который содержал ошибки в гипотезах или методах.
Цена FP: потраченный бюджет на неправильные эксперименты. Потеря времени.

FN: в данном случае не применимо — gate либо открыт (одобрен), либо закрыт (отклонён).
    Технический FN: gate ошибочно отклонил хороший план. Цена: пользователь
    перезапускает. Некритично.

Вывод: UAF не может предотвратить FP (это задача пользователя при review).
UAF обеспечивает: план всегда показывается, редактирование доступно, timeout 24 часа.
```

### 8.3 MLflow Logging Compliance: потеря данных

```
FP (ложная тревога): mlflow_compliance < 1.0, но данные реально в порядке.
Цена FP: предупреждение, которое пользователь проигнорирует.

FN (пропуск реальной потери): Claude Code не залогировал метрику в MLflow,
                               UAF не обнаружил это при проверке.
Цена FN: BudgetController не видит метрику -> не может проверить сходимость ->
         остановится только по safety cap. Потеря воспроизводимости run.
         Частичное нарушение M-ONLINE-03.

Вывод: FN дороже. Проверка mlflow_compliance выполняется строго.
При compliance < 0.9: предупреждение в терминале + тег в MLflow.
```

---

## 9. Ограничения метрик и что не измеряется

**Что намеренно не измеряется:**

1. **Качество гипотез в program.md** (M-UAF-02 только считает количество фаз).
   Оценить качество гипотезы автоматически нельзя — это суждение ML-инженера.
   Фиксируется в review program.md при HumanOversightGate.

2. **Правильность выбора метрики пользователем** (UAF принимает metric.name на веру).
   Пользователь мог указать accuracy вместо ROC-AUC на дисбалансированной задаче.
   Это вне контроля UAF.

3. **Качество кода с точки зрения алгоритмической корректности** (RuffEnforcer
   проверяет стиль, не логику). Проверяет ли код то, что должен — задача code review
   при просмотре отчёта.

4. **Стоимость API Claude Code в сессии** (UAF не имеет прямого доступа к биллингу
   Claude Code). `consumed_cost_usd` в BudgetConfig — оценочная, основана на
   токенах из MLflow tags если Claude Code их логирует, иначе не заполняется.

5. **Субъективная полезность отчёта** (M-ONLINE-01 частично субъективна).
   Self-assessment пользователя после получения отчёта. UAF не опрашивает
   пользователя автоматически.

---

## STAGE COMPLETE

Стадия 04-metrics завершена.

**Зафиксировано:**

- Трёхуровневая схема метрик адаптирована для UAF как системы управления экспериментами
- Task metrics (уровень 1): определяются в task.yaml, не фиксированы в UAF, 11 категорий
- UAF System Quality Metrics (уровень 2): 14 метрик, все автоматически логируются в MLflow
- Online Metrics (уровень 3): 8 метрик, часть субъективных
- MLflow schema: Planning Run, Experiment Run (обязательные поля), Session Summary Run
- LaTeX report: точное соответствие метрик и секций, три стандартных figure
- Критерии остановки (dynamic mode): три класса, порядок проверки зафиксирован,
  алгоритм `should_stop()` с обоснованием дефолтных значений
- FP vs FN: BudgetController (FN дороже), HumanOversightGate (FP неизбежен),
  MLflow compliance (FN дороже)
- Что не измеряется: зафиксировано явно

Переход к стадии 05-data разрешён.
