# Стадия 15: Ownership

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-10, 13 (COMPLETE), 07/11/12/14 (SKIPPED)

---

## 0. Контекст и ограничения ownership

UAF — одиночный проект. Один ML-инженер, локальная машина, нет команды, нет
CI/CD, нет production. Это делает стадию ownership не формальностью ("зафиксировать
ответственность между командами"), а практическим инструментом выживания системы
при длительном перерыве, смене машины или потере контекста.

Bus Factor = 1 зафиксирован в стадии 01 как риск. В стадии 15 мы его не
устраняем (это невозможно при одном человеке), но митигируем через документацию,
воспроизводимость и явные runbooks.

Все решения ниже принимаются с учётом этого контекста. Разделы, применимые только
к командам (escalation paths, Consulted/Informed с другими людьми), адаптированы
к реальности одиночного инженера.

---

## 1. RACI по компонентам UAF

Легенда (адаптированная к одному человеку):
- **R (Responsible)** — компонент или действие, которое выполняется. Всегда один исполнитель.
- **A (Accountable)** — кто принимает итоговое решение и несёт ответственность за результат. При одном человеке = всегда ML-инженер.
- **C (Consulted)** — внешние системы/инструменты, у которых запрашивается информация до действия.
- **I (Informed)** — что уведомляется о результате (логи, артефакты, MLflow).

При Bus Factor = 1: A всегда = ML-инженер. Поэтому таблица сосредоточена на R (кто/что
исполняет) и C/I (что задействовано), чтобы фиксировать зависимости и точки отказа.

### 1.1 Основные компоненты

| # | Компонент | R (Исполнитель) | C (Консультируется с) | I (Уведомляет) | Критичность |
|---|-----------|-----------------|----------------------|----------------|-------------|
| 1 | **ResearchSessionController** | UAF (Python процесс) | — | session.log, budget_status.json | Критичный — state machine, сбой = нет сессии |
| 2 | **ProgramMdGenerator** | UAF через Anthropic API | Anthropic API (claude-3-opus/sonnet) | program.md (черновик), session.log | Критичный — без него нет плана |
| 3 | **HumanOversightGate** | ML-инженер (интерактивный ввод) | program.md (для чтения), data_schema.json | session.log, approved_program.md | Критичный — antigoal 2, единственная точка ручного контроля |
| 4 | **BudgetController** | UAF (polling thread) | MLflow API, budget_status.json, psutil | budget_status.json, session.log, SIGTERM к Claude Code | Критичный — antigoal 6 |
| 5 | **RuffEnforcer** | UAF (post-processing) | uv/ruff CLI | ruff_report.json, session.log | Некритичный — сессия продолжается при сбое |
| 6 | **ReportGenerator** | UAF через Anthropic API | MLflow API, session_analysis.json, Anthropic API | report.tex, report.pdf, session.log | Важный — артефакт сессии, но сессия не зависит |
| 7 | **MLflowSetup** | UAF при инициализации | mlflow CLI/API (локальный сервер) | session.log, MLflow experiment ID | Критичный — без MLflow нет трекинга |
| 8 | **DVCSetup** | UAF при инициализации | git CLI, dvc CLI | .dvc файлы, session.log | Важный — без DVC нет версионирования артефактов |
| 9 | **ClaudeCodeRunner** | UAF (subprocess управление) | claude CLI, settings.json | Claude Code stdout/stderr, session.log | Критичный — это агент, исполняющий эксперименты |
| 10 | **ValidationChecker** | UAF (встроен в RSC) | data_schema.json, task.yaml | validation_checks.json, session.log | Критичный — блокирует сессию при CRITICAL |
| 11 | **DataLoader** | UAF | файловая система (CSV/Parquet/SQL/JSONL/images) | data_schema.json, session.log | Критичный — без данных нет сессии |
| 12 | **ResultAnalyzer** | UAF (post-session) | MLflow API, predictions.csv, session.log | session_analysis.json, session.log | Важный — без него нет гипотез для --resume |
| 13 | **SmokeTestRunner** | UAF (перед запуском Claude Code) | experiment.py, MLflow API, uv | smoke_test_report.json, session.log | Критичный — блокирует запуск при провале |

### 1.2 Интеграционные и агентные компоненты

| Компонент | R | C | I | Примечание |
|-----------|---|---|---|------------|
| **Claude Code (агент)** | Claude Code process (управляет Anthropic) | program.md, experiment_config.yaml, budget_status.json | MLflow runs, DVC commits, session.log | Внешний агент. UAF не управляет им в loop, только мониторит снаружи |
| **MonitoringDashboard** | UAF (опциональный --monitor) | budget_status.json | Rich terminal output | Не влияет на сессию, только отображение |
| **LeakageAudit** | UAF (часть DataLoader) | data_schema.json, task.yaml | data_schema.json (leakage_audit секция), session.log | Встроен в DataLoader, не отдельный процесс |
| **AdversarialValidation** | UAF (часть DataLoader) | train/val splits, LightGBM | data_schema.json (adversarial_validation секция) | Результат передаётся в HumanOversightGate при critical |

### 1.3 Хранилища данных

| Хранилище | Владелец | Что хранит | Критичность потери |
|-----------|----------|------------|-------------------|
| MLflow (локальный) | ML-инженер | Все runs, метрики, параметры, артефакты < 1 МБ | Критичная — потеря = нет результатов |
| DVC remote (локальный путь) | ML-инженер | Артефакты > 1 МБ, входные данные (.dvc метаданные) | Высокая — потеря = нет моделей и больших артефактов |
| git репозиторий | ML-инженер | Код UAF, program.md версии, .dvc файлы, конфиги | Критичная — потеря = нет истории |
| SESSION_DIR | ML-инженер | session.log, budget_status.json, smoke_test_report.json, report/ | Средняя — восстанавливаемо из MLflow + git |
| Входные данные пользователя | ML-инженер | Исходные CSV/Parquet/images | Критичная — UAF не трогает, ответственность пользователя |

### 1.4 Внешние зависимости в RACI

| Зависимость | Что делает UAF | Что даёт зависимость | RACI роль |
|-------------|---------------|---------------------|-----------|
| Anthropic API | ProgramMdGenerator + ReportGenerator | LLM inference | C — консультируется перед записью результата |
| Claude Code CLI | ClaudeCodeRunner | Агент-исполнитель экспериментов | C + I — главный исполнитель research plan |
| MLflow | BudgetController, ReportGenerator, ValidationChecker | Трекинг экспериментов | C — читается; I — пишется экспериментами |
| DVC | DVCSetup, DataLoader | Версионирование артефактов | C + I — консультируется git |
| uv | SmokeTestRunner, RuffEnforcer | Управление зависимостями | C — вызывается перед запуском |
| ruff | RuffEnforcer, SmokeTestRunner | Линтинг и форматирование | C — вызывается post-processing |
| tectonic/pdflatex | ReportGenerator | LaTeX -> PDF компиляция | C — fallback цепочка |

---

## 2. Bus Factor анализ

### 2.1 Текущее состояние

Bus Factor = 1 по всем компонентам без исключений. Единственный человек, знающий:
- архитектурные решения и их обоснования
- нестандартные решения (например, почему stage 07 embedded, почему Feature Store не нужен)
- состояние MLflow экспериментов и что означают конкретные run_id
- текущие открытые вопросы и технический долг

Это нормально для одиночного проекта, но требует явного плана митигации.

### 2.2 Критические точки знаний

Критические точки — те, где потеря контекста приводит к невозможности продолжить
работу или высокому риску повторения ошибок:

| Точка знаний | Риск потери | Митигация |
|--------------|-------------|-----------|
| Архитектурные решения v2.0 (почему выброшены компоненты v1.0) | Высокий — риск возврата к v1.0 антипаттернам | Зафиксированы в docs/stage-03-design-doc.md и progress.md |
| Связка MLflow run_id <-> git sha <-> DVC sha | Критичный — без неё нельзя воспроизвести эксперимент | requirements.lock SHA в Planning Run, DVC auto-commits |
| Порядок компонентов в flow (DataLoader -> LeakageAudit -> AdversarialValidation -> ...) | Средний — неправильный порядок = молчаливые ошибки | Зафиксировано в stage-05 и stage-09 |
| Antigoals как ограничения проектирования | Высокий — нарушение antigoal может проявиться поздно | Antigoals в каждом stage документе + progress.md |
| Budget схема и критерии остановки | Средний — неправильная остановка = antigoal 6 нарушение | Зафиксированы в stage-04 и stage-09 |
| settings.json как механизм безопасности Claude Code | Высокий — без него Claude Code имеет неограниченный доступ | Зафиксировано в stage-03 TR-2 v2.0 |
| Shadow Feature Trick и baseline_run_id | Средний — без знания можно сломать feature engineering | Зафиксировано в stage-10 |
| Реестр алертов (12 кодов) и их действия | Средний — неправильная реакция на алерт | Зафиксировано в stage-13 |

### 2.3 Стратегия митигации через документацию

При Bus Factor = 1 единственная реальная митигация — это качество документации.
Принцип: система должна быть восстанавливаема новым человеком за 4 часа чтения.

**Три уровня документации:**

1. **Обзорный уровень (docs/progress.md):** все стадии, все ключевые решения,
   antigoals, архитектура v2.0. Один файл. Читается за 30 минут.

2. **Детальный уровень (docs/stage-NN-*.md):** полный контекст каждого решения
   с обоснованием. Читается при необходимости понять конкретный компонент.

3. **Операционный уровень (этот документ, секция 3):** пошаговые runbooks.
   Не требуют понимания архитектуры. Работают как чеклист.

**Что должно быть в каждом компоненте кода** (когда будет реализован):
- Docstring с ссылкой на stage документ (например, `# Реализует: stage-09, R-09-01`)
- Комментарий к нестандартным решениям с объяснением "почему"
- Ссылка на antigoal если компонент обеспечивает его соблюдение

### 2.4 Риски специфичные для UAF

**Риск 1: Anthropic API изменит поведение Claude Code**
Вероятность: средняя (Claude Code активно развивается).
Последствие: program.md Execution Instructions могут перестать работать.
Митигация: зафиксировать версию Claude Code CLI в requirements. Тест smoke_test
ST-04 проверяет что MLflow интеграция работает — косвенно покрывает.

**Риск 2: Сессия накопила много MLflow runs, и поиск нужного run_id затруднён**
Вероятность: высокая после нескольких месяцев использования.
Митигация: структурированное именование экспериментов в MLflowSetup
(uaf/{task_name}/{date}), уникальный session_id в Planning Run tags.

**Риск 3: DVC remote и MLflow разъехались (артефакт есть в MLflow, но нет в DVC)**
Вероятность: средняя при ручных манипуляциях вне UAF.
Митигация: Reproducibility Checklist в отчёте + runbook восстановления (секция 3.5).

**Риск 4: Потеря SESSION_DIR при переезде на другую машину**
Вероятность: низкая (явное действие).
Последствие: потеря session.log, smoke_test_report.json.
Митигация: ReportGenerator включает ключевые данные в report.pdf — главный
артефакт сессии переносим без SESSION_DIR.

---

## 3. Runbook

Runbook написан как последовательность шагов. Никаких предположений о знании
архитектуры — только конкретные команды и действия.

### 3.1 Новая сессия: запуск с нуля

**Предусловие:** task.yaml заполнен, данные доступны по пути из task.yaml.

**Шаги:**

```
Шаг 1. Проверить предусловия (см. секцию 4 — Чеклист перед запуском)

Шаг 2. Запустить UAF
  uaf run --task task.yaml

Шаг 3. Дождаться генерации program.md
  UAF вызывает ProgramMdGenerator (1-2 мин).
  В терминале: "[INFO] [ProgramMdGenerator] program.md готов к review"

Шаг 4. Прочитать program.md
  cat .uaf/sessions/{session_id}/program.md
  Особое внимание: Research Phases (Phase 1 обязательно начинается с baseline),
  Execution Instructions (проверить что metric.name совпадает с task.yaml).

Шаг 5. HumanOversightGate: одобрить / отредактировать / отклонить
  UAF покажет: "Одобрить план? [y/n/e (edit)]"
  y  — запуск без изменений
  e  — открывает $EDITOR с program.md, UAF ждёт сохранения
  n  — сессия прерывается, program.md сохранён в SESSION_DIR для анализа

Шаг 6. Наблюдение во время сессии
  Вариант A (пассивный): подождать завершения, читать session.log:
    tail -f .uaf/sessions/{session_id}/session.log

  Вариант B (активный дашборд):
    uaf run --task task.yaml --monitor
    Rich-дашборд показывает budget_status.json в реальном времени.

Шаг 7. Получение результатов
  По завершении: "Session complete. Report: .uaf/sessions/{session_id}/report/report.pdf"
  Открыть report.pdf — главный артефакт сессии.
  MLflow UI для детального анализа runs: mlflow ui --port 5000
```

### 3.2 Возобновление сессии: --resume

**Когда использовать:** предыдущая сессия завершилась досрочно (hard_stop,
crash, ручная остановка Ctrl+C) и есть незавершённые гипотезы или плохие метрики.

**Шаги:**

```
Шаг 1. Найти session_id предыдущей сессии
  ls -lt .uaf/sessions/  — последняя директория по времени

Шаг 2. Проверить что осталось
  cat .uaf/sessions/{prev_session_id}/budget_status.json | python -m json.tool
  Смотреть: stop_reason, iterations_done, best_metric_value

  cat .uaf/sessions/{prev_session_id}/session_analysis.json | python -m json.tool
  Смотреть: hypotheses (H-01..H-09), systemic_failure_category

  cat .uaf/sessions/{prev_session_id}/program.md
  Смотреть: незавершённые шаги (status: pending/in_progress), Iteration Log

Шаг 3. Решить: resume или новая сессия?
  Resume подходит если:
  - Были успешные runs (есть базовые результаты для comparison)
  - Есть конкретные гипотезы для проверки из session_analysis.json
  - Данные не изменились (dvc status показывает clean)

  Новая сессия подходит если:
  - session_analysis.json не создан (сессия упала очень рано)
  - Данные изменились
  - Задача переформулирована

Шаг 4. Запустить resume
  uaf resume --session {prev_session_id}

Шаг 5. UAF автоматически:
  - Читает session_analysis.json -> создаёт improvement_context.md
  - improvement_context.md передаётся в ProgramMdGenerator как контекст
  - Создаётся новый program.md с учётом результатов прошлой сессии
  - Проходит HumanOversightGate снова (план должен быть одобрен)
  - Новая сессия стартует с session_id = {prev_session_id}-resume-{N}

Шаг 6. MLflow связь между сессиями
  Новые runs создаются в том же experiment (uaf/{task_name}/{date}).
  Planning Run новой сессии содержит тег: parent_session_id = {prev_session_id}
  Это позволяет строить цепочку сессий в MLflow UI.
```

### 3.3 Что делать при крашах UAF

Классифицировать краш до принятия действий.

**Тип A: Краш до HumanOversightGate**

Признак: SESSION_DIR создан, program.md отсутствует или неполный.

```
Действия:
1. Проверить session.log:
   cat .uaf/sessions/{session_id}/session.log | grep ERROR

2. Типичные причины:
   - Anthropic API недоступен -> проверить: curl https://api.anthropic.com/health
   - task.yaml невалиден -> проверить: python -c "import yaml; yaml.safe_load(open('task.yaml'))"
   - Данные недоступны -> проверить путь в task.yaml

3. Исправить причину -> uaf run --task task.yaml (новая сессия)
   Старый SESSION_DIR можно удалить — он неполный.
```

**Тип B: Краш во время Claude Code сессии**

Признак: SESSION_DIR существует, program.md одобрен, budget_status.json есть,
MLflow содержит некоторые runs.

```
Действия:
1. Проверить статус:
   cat .uaf/sessions/{session_id}/budget_status.json | python -c \
     "import json,sys; d=json.load(sys.stdin); print(d.get('stop_reason'), d.get('iterations_done'))"

2. Проверить последние алерты:
   cat .uaf/sessions/{session_id}/session.log | grep "CRITICAL\|ERROR" | tail -20

3. Если stop_reason = "hw_oom" или "disk_full":
   - Освободить ресурсы
   - Рассмотреть уменьшение batch_size в task.yaml
   - uaf resume --session {session_id}

4. Если stop_reason = "claude_hang":
   - Проверить что Claude Code процесса нет: ps aux | grep claude
   - Если есть — убить: kill -9 {pid}
   - uaf resume --session {session_id}

5. Если stop_reason = "mlflow_down":
   - Проверить MLflow: mlflow experiments list
   - Если не работает: mlflow server --host 127.0.0.1 --port 5000 &
   - Дождаться старта (5-10 сек), затем: uaf resume --session {session_id}

6. Если причина непонятна:
   - Запустить ResultAnalyzer вручную (если есть хотя бы 1 completed run):
     uaf analyze --session {session_id}
   - Это создаст session_analysis.json и позволит использовать --resume
```

**Тип C: Краш ReportGenerator**

Признак: сессия завершена (session.log содержит "Claude Code finished"),
но report.pdf не создан.

```
Действия:
1. Проверить session.log:
   grep "ReportGenerator\|ERROR\|tectonic\|pdflatex" .uaf/sessions/{session_id}/session.log

2. Если проблема с LaTeX компилятором:
   - Проверить наличие: which tectonic || which pdflatex
   - Если нет: установить tectonic (https://tectonic-typesetting.github.io)
   - Перегенерировать отчёт вручную:
     uaf report --session {session_id}

3. Если проблема с Anthropic API (LLM вызовы ReportGenerator):
   - Подождать и повторить: uaf report --session {session_id}
   - Fallback: report.tex будет создан даже если LLM недоступен,
     но секции "Executive Summary" и "Monitoring Conclusions" будут пустыми.
     Скомпилировать вручную: cd .uaf/sessions/{session_id}/report && tectonic report.tex

4. Все данные сессии в MLflow — отчёт восстанавливаем всегда.
```

**Тип D: Краш SmokeTestRunner (провал smoke tests)**

Признак: session.log содержит "SmokeTestRunner FAILED", сессия не запустилась.

```
Действия:
1. Читать smoke_test_report.json:
   cat .uaf/sessions/{session_id}/smoke_test_report.json | python -m json.tool

2. По коду теста:
   ST-01: в experiment.py отсутствуют секции # UAF-SECTION -> Claude Code не заполнил scaffold
          -> запустить Claude Code вручную для заполнения scaffold, или --regenerate-scaffold
   ST-02: синтаксическая ошибка в experiment.py -> python -m py_compile experiment.py
   ST-03: ruff нашёл ошибки -> cd SESSION_DIR && ruff check experiment.py
   ST-04: mlflow.start_run отсутствует -> добавить в # UAF-SECTION: MLFLOW_INIT
   ST-09: MLflow недоступен -> см. Тип B пункт 5
   ST-11: dry-run > 90 сек -> уменьшить n_rows в task.yaml dry_run.n_rows

3. После исправления: uaf run --task task.yaml --skip-plan-generation
   (переиспользует существующий program.md, пропускает ProgramMdGenerator)
```

### 3.4 Как читать результаты

**Основной артефакт: report.pdf**

```
Структура отчёта (порядок чтения):
1. Executive Summary (страница 1-2)
   - Что исследовалось, сколько итераций, лучшая метрика
   - Ключевые выводы и рекомендации

2. Experiment Results (таблица всех runs)
   - best_run: лучший run с метрикой и параметрами
   - Таблица: run_id, метрика, статус (success/failed/partial)
   - ВАЖНО: failed runs включены обязательно (antigoal 3)

3. Failed Experiments (отдельная секция)
   - Категории ошибок, systemic failure если >= 50%
   - Это самая ценная секция для понимания что не сработало

4. Feature Engineering (если Phase 2 выполнялась)
   - Принятые/отклонённые признаки, shadow deltas
   - feature_importance_timeline.pdf

5. Session Monitoring
   - Budget Burndown, Metric Progression графики
   - Alert Log (если были WARNING/CRITICAL)

6. Recommendations
   - Гипотезы H-01..H-09 для следующей итерации
   - Конкретные действия (не "попробуйте другой подход")
```

**MLflow UI для детального анализа:**

```
Запуск: mlflow ui --port 5000
Открыть: http://localhost:5000

Навигация:
- Experiment: uaf/{task_name}/{date}
  - Planning Run: параметры сессии, program.md как артефакт
  - Experiment Runs: все итерации, параметры, метрики
  - Session Summary Run: агрегированные метрики M-UAF-01..14
  - Analysis Run (если SHAP): shap_bar_chart.pdf, shap_importance.csv

Полезные фильтры в MLflow UI:
  - status = FINISHED -> только успешные runs
  - tags.uaf.forced_stop = true -> runs с принудительной остановкой
  - tags.run_type = experiment -> основные runs (исключить planning/analysis)

Для воспроизведения конкретного run:
  - Открыть run -> вкладка Artifacts -> requirements.lock
  - Тег git_sha показывает коммит с кодом этого run
```

**session_analysis.json для понимания гипотез:**

```
cat .uaf/sessions/{session_id}/session_analysis.json | python -c "
import json, sys
d = json.load(sys.stdin)
print('=== Best Run ===')
print(f\"  run_id: {d['best_run']['run_id']}\")
print(f\"  metric: {d['best_run']['metric_value']}\")
print()
print('=== Hypotheses ===')
for h in d.get('hypotheses', []):
    print(f\"  {h['id']}: {h['description']} [priority={h['priority']}]\")
print()
print('=== Systemic Failures ===')
sf = d.get('systemic_failure')
if sf:
    print(f\"  Category: {sf['category']}, Count: {sf['count']}\")
else:
    print('  None')
"
```

### 3.5 Восстановление эксперимента из MLflow + DVC

Сценарий: SESSION_DIR утерян (другая машина, случайное удаление), нужно
восстановить конкретный эксперимент до воспроизводимого состояния.

**Шаги восстановления:**

```
Шаг 1. Найти run в MLflow
  mlflow runs search --experiment-name "uaf/{task_name}/{date}" \
    --filter "tags.run_type = 'experiment'" --order-by "metrics.{metric_name} DESC"

  Записать: run_id, tags.git_sha, tags.session_id

Шаг 2. Восстановить код
  git checkout {git_sha}
  # В репозитории будут: experiment.py, program.md, task.yaml для этого run

Шаг 3. Восстановить зависимости
  mlflow artifacts download --run-id {run_id} --artifact-path requirements.lock \
    --dst-path ./recovered/
  uv pip install -r ./recovered/requirements.lock

Шаг 4. Восстановить данные (если нужны)
  # .dvc файлы из git checkout содержат checksums
  dvc pull
  # Если DVC remote недоступен — данные нужно восстановить вручную

Шаг 5. Получить параметры для воспроизведения
  mlflow runs get --run-id {run_id}
  # Все параметры эксперимента в разделе params

Шаг 6. Запустить воспроизведение
  # Параметры из MLflow -> experiment_config.yaml вручную
  python experiment.py --config experiment_config.yaml --seed {seed_from_mlflow}

Шаг 7. Сверить результат
  # Ожидаемая метрика из MLflow run должна совпасть с допуском +-epsilon
  # Если не совпадает: проверить versions зависимостей (hardware differences OK)
```

**Важно:** полная воспроизводимость возможна только при:
1. git_sha присутствует в MLflow тегах (ST-12 это проверяет)
2. requirements.lock залогирован как артефакт Planning Run
3. DVC checksums не нарушены (dvc status чист)
4. random seed зафиксирован в параметрах MLflow run

---

## 4. Dependency Map: внешние зависимости

### 4.1 Карта зависимостей

```
UAF
├── КРИТИЧЕСКИЕ (сессия невозможна без них)
│   ├── Anthropic API (claude-3-*) ──── ProgramMdGenerator, ReportGenerator
│   ├── Claude Code CLI ─────────────── ClaudeCodeRunner (основной агент)
│   ├── MLflow (локальный) ──────────── BudgetController, ValidationChecker, ReportGenerator
│   └── Python 3.10+ + uv ───────────── все компоненты UAF
│
├── ВАЖНЫЕ (сессия деградирует, но продолжается)
│   ├── DVC ──────────────────────────── DVCSetup (версионирование артефактов)
│   ├── ruff ─────────────────────────── RuffEnforcer, SmokeTestRunner ST-03
│   └── git ──────────────────────────── DVC, воспроизводимость (git_sha в MLflow)
│
└── НЕКРИТИЧЕСКИЕ (только для отдельных features)
    ├── tectonic / pdflatex ──────────── ReportGenerator LaTeX -> PDF
    ├── LightGBM ─────────────────────── AdversarialValidation
    ├── Rich ─────────────────────────── MonitoringDashboard (--monitor)
    └── psutil ───────────────────────── BudgetController hang detection
```

### 4.2 Действия при недоступности каждой зависимости

#### Anthropic API недоступен

**Симптом:** ProgramMdGenerator завершается с ошибкой, сессия не стартует.
**Также:** ReportGenerator не может сгенерировать Executive Summary и Conclusions.

**Действия:**

```
1. Проверить статус: curl -s https://api.anthropic.com/health
   Если timeout -> нет сети. Если 503 -> API downtime.

2. Временный fallback для program.md:
   uaf run --task task.yaml --program program_template.md
   Флаг --program позволяет передать вручную написанный program.md.
   Система переходит сразу к HumanOversightGate.

3. Для ReportGenerator:
   При недоступности API отчёт генерируется без LLM-секций.
   Секции данных, таблицы runs, графики — всё есть.
   LLM-секции (Executive Summary, Monitoring Conclusions, Recommendations)
   заменяются заглушкой: "[LLM analysis unavailable — API unreachable]"
   report.pdf создаётся, в него включены все данные экспериментов.

4. Мониторинг: подписаться на https://status.anthropic.com
```

#### MLflow недоступен

**Симптом:** ST-09 (smoke test) провален, или BudgetController пишет алерт SW-MLFLOW-DOWN.

```
1. Проверить статус локального MLflow:
   curl http://127.0.0.1:5000/api/2.0/mlflow/experiments/list 2>/dev/null \
     && echo "OK" || echo "DOWN"

2. Запустить MLflow сервер:
   mlflow server --host 127.0.0.1 --port 5000 \
     --backend-store-uri ./mlruns --default-artifact-root ./mlruns &

3. Подождать старта (5-10 сек), повторить curl проверку.

4. Если mlruns директория отсутствует — MLflow не был инициализирован.
   Это означает что DVCSetup/MLflowSetup не выполнился.
   Запустить: uaf init --task task.yaml (только инициализация, без запуска сессии)

5. Если MLflow упал во время сессии:
   Алерт SW-MLFLOW-DOWN залогирован в session.log.
   BudgetController переключается на filesystem fallback:
   читает ./mlruns/{experiment_id}/{run_id}/metrics/ напрямую.
   Это покрывает 80% функциональности. Сессия продолжается.
   После восстановления MLflow — fallback отключается автоматически.

6. Данные экспериментов не теряются: MLflow пишет в ./mlruns/ файловой системой
   (не только через HTTP API). При рестарте сервера все данные доступны.
```

#### DVC недоступен или сломан

**Симптом:** DVCSetup завершается с ошибкой, session.log содержит ошибку dvc CLI.

```
1. Проверить: dvc --version

2. Установить если отсутствует:
   uv tool install dvc

3. Если DVC репозиторий не инициализирован в проекте:
   cd {project_root} && dvc init && git commit -m "init dvc"

4. Degraded mode: запустить сессию без DVC трекинга:
   uaf run --task task.yaml --no-dvc
   Что теряется:
   - Артефакты > 1 МБ не версионируются (только в MLflow если < 1 МБ)
   - Связка git sha <-> data checksums нарушена
   - Воспроизводимость данных не гарантирована

5. dvc status показывает "modified" на входных данных:
   Это WARNING — данные изменились после dvc add.
   ОСТАНОВИТЬСЯ. Проверить не нарушен ли antigoal 4 (UAF не должен менять данные).
   Если данные менял пользователь вручную — это нормально.
   Обновить: dvc add {data_path} && git add *.dvc && git commit -m "update data checksums"
```

#### uv недоступен

**Симптом:** SmokeTestRunner ST-11 (dry-run) не может выполнить pip install.

```
1. Проверить: uv --version
   Если отсутствует: curl -LsSf https://astral.sh/uv/install.sh | sh

2. Временный fallback на pip:
   В task.yaml: use_venv: false (использовать системный Python)
   uaf run --task task.yaml --pip-fallback
   Зависимости ставятся через pip install -r requirements.lock
   Это менее воспроизводимо (нет lock resolution), но работает.

3. Если uv есть но конфликт зависимостей:
   uv pip install --resolution=lowest-direct -r requirements.lock
   Если не помогает: удалить виртуальное окружение и создать заново:
   uv venv --python 3.10 .venv && uv pip install -r requirements.lock
```

#### Claude Code CLI недоступен

**Симптом:** ClaudeCodeRunner не может запустить подпроцесс `claude`.

```
1. Проверить: claude --version
   Если отсутствует: npm install -g @anthropic-ai/claude-code
   Или через pip: pip install claude-code-cli (уточнить актуальный пакет)

2. Проверить аутентификацию:
   claude auth status
   Если не авторизован: claude auth login

3. Если Claude Code есть но не отвечает (зависает на старте):
   - Проверить ~/.claude/settings.json — невалидный JSON блокирует старт
   - Удалить или восстановить settings.json:
     cp .uaf/sessions/{session_id}/settings.json ~/.claude/settings.json

4. Нет fallback для Claude Code — это основной агент UAF.
   Без Claude Code эксперименты не выполняются.
   Единственный вариант: написать experiment.py вручную и запустить без UAF.
```

---

## 5. Чеклист перед запуском сессии

Выполнять последовательно. Каждый пункт — проверяемое условие.

### 5.1 Данные и задача

```
[ ] task.yaml существует и валиден
    python -c "import yaml; d=yaml.safe_load(open('task.yaml')); print(d['task']['type'], d['metric']['name'])"

[ ] Данные доступны по пути из task.yaml
    python -c "import yaml,os; d=yaml.safe_load(open('task.yaml')); \
      print('OK' if os.path.exists(d['data']['train_path']) else 'MISSING')"

[ ] Данные в ожидаемом формате (хотя бы head)
    python -c "import pandas as pd, yaml; d=yaml.safe_load(open('task.yaml')); \
      df=pd.read_csv(d['data']['train_path'], nrows=5); print(df.dtypes)"

[ ] target_col присутствует в данных
    (проверяется автоматически в DataLoader, но лучше убедиться вручную)

[ ] budget задан явно (итерации или время)
    python -c "import yaml; d=yaml.safe_load(open('task.yaml')); print(d['budget'])"
```

### 5.2 Окружение

```
[ ] Python 3.10+ доступен
    python --version

[ ] uv установлен и работает
    uv --version

[ ] MLflow сервер запущен или будет запущен UAF автоматически
    curl http://127.0.0.1:5000/api/2.0/mlflow/experiments/list 2>/dev/null \
      && echo "MLflow UP" || echo "MLflow DOWN (UAF запустит автоматически)"

[ ] Claude Code CLI доступен и авторизован
    claude --version && claude auth status

[ ] Anthropic API доступен
    curl -s https://api.anthropic.com/health | python -m json.tool

[ ] Свободное место на диске (минимум 5 ГБ рекомендуется)
    df -h .

[ ] DVC инициализирован в проекте
    dvc status 2>/dev/null && echo "DVC OK" || echo "DVC NOT INITIALIZED"
```

### 5.3 Предыдущие сессии (если есть)

```
[ ] Нет незавершённых Claude Code процессов от предыдущих сессий
    ps aux | grep claude | grep -v grep

[ ] MLflow не содержит зависших runs (RUNNING без активного процесса)
    mlflow runs search --experiment-name "uaf/{task_name}" \
      --filter "status = 'RUNNING'" --order-by "start_time DESC"
    Если есть — завершить вручную:
    mlflow runs delete --run-id {stuck_run_id}  # или пометить как FAILED

[ ] Если использовался --resume: improvement_context.md из прошлой сессии прочитан
    cat .uaf/sessions/{prev_session_id}/improvement_context.md
```

### 5.4 Специфичные проверки по типу задачи

```
Для time-series задач:
[ ] В task.yaml задан time_col и forecast_horizon
[ ] В task.yaml задан gap >= forecast_horizon
[ ] Данные отсортированы по времени (нет перемешивания)

Для NLP задач с многодокументными выборками:
[ ] В task.yaml задан document_id_col (обязателен для GroupKFold)

Для CV задач:
[ ] image_dir доступна и содержит изображения в ожидаемом формате
[ ] manifest CSV (если используется) валиден: image_path и label колонки присутствуют

Для задач с DL моделями:
[ ] GPU доступен (если требуется): nvidia-smi
[ ] Checkpoint директория имеет достаточно места (модели занимают много)
[ ] experiment_timeout в task.yaml достаточен для одной эпохи
```

### 5.5 Финальная проверка

```
[ ] task.yaml прочитан целиком — нет случайных значений от прошлой задачи
[ ] metric.direction задан корректно (maximize/minimize)
[ ] budget.max_iterations не чрезмерно мал (минимум 5 для осмысленного исследования)
[ ] Текущая рабочая директория — корень проекта (не SESSION_DIR)
    pwd
```

Если все пункты прошли — можно запускать.

---

## 6. LaTeX/PDF отчёт: секция Ownership

Секция Ownership включается в отчёт как часть раздела "Reproducibility & Ownership".
Это не отдельный раздел — это краткий блок в конце отчёта, автоматически
генерируемый из данных сессии без LLM вызова.

### 6.1 Что включается в отчёт

**Блок 1: Session Identification**

```latex
\subsection*{Session Identification}
\begin{tabular}{ll}
\textbf{Session ID:} & uaf-{session_id} \\
\textbf{Date:} & {session_start_datetime} \\
\textbf{Task:} & {task.name} \\
\textbf{UAF Version:} & {uaf.__version__} \\
\textbf{Git SHA:} & \texttt{{git_sha}} \\
\textbf{MLflow Experiment:} & {mlflow_experiment_name} \\
\textbf{Planning Run ID:} & \texttt{{planning_run_id}} \\
\end{tabular}
```

**Блок 2: Reproducibility Checklist**

Автоматически проставляются галочки (OK / MISSING) на основе реальных данных сессии.

```latex
\subsection*{Reproducibility Checklist}
\begin{itemize}
  \item[\checkedbox] Git commit фиксирован: \texttt{{git_sha}}
  \item[\checkedbox] requirements.lock залогирован в MLflow (run: {planning_run_id})
  \item[\checkedbox] DVC checksums входных данных зафиксированы
  \item[\checkedbox] Random seed во всех runs: {seed}
  \item[\optionalbox] DVC remote настроен: {dvc_remote_url или "not configured"}
\end{itemize}
```

Где checkedbox = зелёный квадрат если OK, красный если MISSING.
optionalbox = серый (информационно, не обязательно).

**Блок 3: Component Versions**

Таблица версий всех ключевых зависимостей из requirements.lock Planning Run.

```latex
\subsection*{Component Versions}
\begin{tabular}{ll}
\textbf{Python:} & {python_version} \\
\textbf{UAF:} & {uaf_version} \\
\textbf{Claude Code:} & {claude_code_version} \\
\textbf{MLflow:} & {mlflow_version} \\
\textbf{DVC:} & {dvc_version} \\
\textbf{uv:} & {uv_version} \\
\end{tabular}
```

**Блок 4: Recovery Instructions (краткий)**

```latex
\subsection*{Recovery Instructions}
\begin{enumerate}
  \item Восстановить окружение: \texttt{uv pip install -r requirements.lock}
  \item Восстановить данные: \texttt{dvc pull} (требует DVC remote)
  \item Восстановить код: \texttt{git checkout {git\_sha}}
  \item MLflow данные доступны в: \texttt{./mlruns/{experiment\_id}/}
  \item Полная документация: \texttt{docs/stage-15-ownership.md}
\end{enumerate}
```

**Блок 5: Bus Factor Statement**

Фиксируется состояние на момент сессии:

```latex
\subsection*{Ownership Statement}
\textbf{Bus Factor:} 1 (одиночный проект). \\
\textbf{Ответственный:} {user_name или "ML-инженер"} \\
\textbf{Документация:} docs/stage-15-ownership.md содержит полный runbook. \\
\textbf{Примечание:} все артефакты сессии воспроизводимы через MLflow run \texttt{{planning\_run\_id}}.
```

### 6.2 Что не включается в отчёт

Следующее зафиксировано здесь, в stage-15 документе, но не дублируется в PDF:
- Полный RACI (избыточен в отчёте об исследовании)
- Детальные runbooks (есть в stage-15, в PDF только краткие recovery instructions)
- Dependency Map полная версия (в PDF только таблица версий)
- Чеклист перед запуском (операционный документ, не артефакт исследования)

Принцип: report.pdf — артефакт исследования, не операционная документация.
Секция Ownership в нём — это минимально необходимое для воспроизведения результатов.

---

## 7. Решения и обоснования

**R-15-01: Нет формального RACI с несколькими людьми**
При Bus Factor = 1 классическая RACI матрица избыточна. Адаптирована под реальность:
R = кто/что исполняет, C = что задействовано, I = что уведомляется. A = всегда
ML-инженер (не указывается в таблице как константа).

**R-15-02: Runbook как первичный митигатор Bus Factor**
Принцип: новый человек должен уметь запустить сессию и интерпретировать результаты
за 4 часа без чтения кода. Runbook проектируется под этот критерий. Типовые сценарии
крашей покрываются явными шагами, не общими советами.

**R-15-03: Dependency Map содержит действия при недоступности**
Недостаточно знать что зависимость существует. Нужно знать что делать при отказе.
Каждая зависимость имеет: degraded mode (если возможен), восстановление, fallback.

**R-15-04: Чеклист — исполняемые команды, не описания**
Каждый пункт чеклиста содержит конкретную команду для проверки. "Убедиться что
MLflow работает" — плохо. `curl http://127.0.0.1:5000/api/2.0/mlflow/experiments/list`
с ожидаемым результатом — хорошо.

**R-15-05: Секция Ownership в PDF — минимально необходимое**
report.pdf — исследовательский артефакт. Операционная документация в нём
избыточна и мешает читаемости. Только: session_id, git_sha, reproducibility checklist,
component versions, краткие recovery instructions. Полный runbook остаётся в stage-15.

---

## STAGE COMPLETE

**Стадия 15 (ownership) завершена.**

Созданные артефакты:
- `docs/stage-15-ownership.md` — настоящий документ

Ключевые результаты:
- RACI адаптирован к одиночному проекту (Bus Factor = 1): 13 компонентов с R/C/I
- Критические точки знаний задокументированы, митигации зафиксированы
- Runbook: 5 сценариев (новая сессия, resume, 4 типа крашей), восстановление из MLflow+DVC
- Dependency Map: 7 внешних зависимостей с действиями при отказе каждой
- Чеклист: 5 групп проверок, все с исполняемыми командами
- Секция Ownership в PDF: 5 блоков, автоматически генерируется без LLM

Следующая стадия: **16-postmortem**
