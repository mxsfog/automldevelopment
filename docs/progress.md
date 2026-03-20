# Progress: Universal AutoResearch Framework (UAF)

**Проект:** Universal AutoResearch Framework
**Начало:** 2026-03-19
**Методология:** Babushkin & Kravchenko (2025), 16 стадий

---

## Статус стадий

| # | Стадия | Статус | Дата | Артефакт |
|---|--------|--------|------|----------|
| 01 | problem | **COMPLETE** | 2026-03-19 | `docs/stage-01-problem.md` |
| 02 | research | **COMPLETE** | 2026-03-19 | `docs/stage-02-research.md` |
| 03 | design-doc | **COMPLETE v2.0** | 2026-03-19 | `docs/stage-03-design-doc.md` |
| 04 | metrics | **COMPLETE** | 2026-03-19 | `docs/stage-04-metrics.md` |
| 05 | data | **COMPLETE** | 2026-03-19 | `docs/stage-05-data.md` |
| 06 | validation | **COMPLETE** | 2026-03-19 | `docs/stage-06-validation.md` |
| 07 | baseline | **SKIPPED/EMBEDDED** | 2026-03-19 | baseline logic встроена в шаблон program.md |
| 08 | error-analysis | **COMPLETE** | 2026-03-19 | `docs/stage-08-error-analysis.md` |
| 09 | pipeline | **COMPLETE** | 2026-03-19 | `docs/stage-09-pipeline.md` |
| 10 | features | **COMPLETE** | 2026-03-19 | `docs/stage-10-features.md` |
| 11 | measurement | **SKIPPED/antigoal-1** | 2026-03-19 | нет production, нет A/B тестов — измерение встроено в стадию 04 |
| 12 | integration | **SKIPPED/antigoal-1** | 2026-03-19 | нет production интеграции — UAF сам является системой |
| 13 | monitoring | **COMPLETE** | 2026-03-19 | `docs/stage-13-monitoring.md` |
| 14 | serving | **SKIPPED/antigoal-1** | 2026-03-19 | нет production serving — UAF локальная исследовательская система |
| 15 | ownership | **COMPLETE** | 2026-03-19 | `docs/stage-15-ownership.md` |
| 16 | postmortem | **COMPLETE** | 2026-03-19 | `docs/stage-16-postmortem.md` |

Легенда: **COMPLETE** — завершена | **SKIPPED/antigoal-1** — не применима (antigoal 1: нет production) | **SKIPPED/EMBEDDED** — логика встроена в другой компонент

---

## Правила перехода

- Нельзя перейти к следующей стадии без явного `STAGE COMPLETE` в текущей
- Нельзя писать код модели до завершения стадий 01-06
- Нельзя выбирать архитектуру до завершения стадии 07 (baseline)
- Нельзя оптимизировать inference до стадии 13 (мониторинг работает)
- Нельзя игнорировать antigoals из стадии 01

---

## Ключевые решения

### Решения стадии 03 v2.0 (2026-03-19) — АРХИТЕКТУРНЫЙ ПЕРЕСМОТР

**Причина пересмотра:**
Принципиальное изменение агентной архитектуры: Claude Code как агент вместо
собственного LLM-клиента. UAF перестаёт управлять LLM в loop — он управляет
сессией вокруг Claude Code. Karpathy делал autoresearch именно так.

**Что выброшено:**
- LLMClient (Protocol + OpenAI/Anthropic/Ollama реализации): Claude Code сам управляет LLM
- PlanningAgent: заменён ProgramMdGenerator (один LLM вызов, без агентного loop)
- ExperimentRunner: Claude Code сам пишет и запускает код через встроенные tools
- FailureRecovery: Claude Code сам обрабатывает traceback и исправляет код
- AST-validator: безопасность через settings.json Claude Code (allowedTools, deny list)

**TR-2 v2.0: Claude Code Execution Environment**
settings.json определяет что Claude Code может и не может делать.
`Write(.uaf/sessions/{id}/**)` — пишет только в директорию сессии.
Deny list: rm -rf, curl, wget, ssh, sudo.
venv опционален: `use_venv: true` в task.yaml -> инструкция в program.md.

**TR-3 v2.0: Claude Code как единый агент**
Было: Plan-and-Execute с PlanningAgent + ExperimentRunner управляемыми UAF.
Стало: Claude Code — единый агент. UAF создаёт program.md + settings.json,
запускает Claude Code, мониторит снаружи. Claude Code сам реализует
Plan-and-Execute внутри своей сессии.
HumanOversightGate сохраняется: одобрение plan перед запуском Claude Code.

**TR-4 v2.0: program.md как контракт с Claude Code**
program.md содержит не только план, но и Execution Instructions —
явные инструкции для Claude Code: MLflow API, ruff, DVC commit protocol,
budget awareness (читать budget_status.json). Claude Code следует им как
части задания.

**TR-5: Convergence Criterion — без изменений**
Комбинированный критерий: no_improvement(patience) AND delta < min_delta AND min_iterations.
LLM-сигнал: Claude Code пишет convergence_signal в MLflow tags.
BudgetController читает при polling.

**TR-7: ruff Enforcement (новое)**
Двухуровневый: (1) инструкция в program.md -> Claude Code применяет ruff сам
после каждого .py файла; (2) RuffEnforcer post-processing после завершения сессии.
Конфиг: line-length=99, Python 3.10+, полный набор правил.

**TR-8: UAF не делает прямых LLM вызовов (обновлено 2026-03-20)**
UAF — чистая Python утилита без LLM вызовов и API ключей.
ProgramMdGenerator подготавливает context/ пакет, Claude Code генерирует program.md.
ReportGenerator компилирует PDF из секций, которые Claude Code генерирует в конце сессии.
Вся работа LLM (план + эксперименты + отчёт) происходит внутри одной Claude Code сессии.

**Компоненты v2.0 (6 основных, было 9):**
- ResearchSessionController — state machine сессии
- ProgramMdGenerator — подготовка context/ пакета (без LLM вызовов)
- HumanOversightGate — approval checkpoint
- BudgetController — polling MLflow, hard stop через budget_status.json
- RuffEnforcer — post-processing ruff на всех .py файлах сессии
- ReportGenerator — LaTeX + PDF из MLflow данных и секций от Claude Code

Плюс интеграционные: MLflowSetup, DVCSetup, ClaudeCodeRunner.

**BudgetController v2.0:**
Работает в отдельном thread с polling интервалом 30 секунд.
Читает MLflow runs (считает итерации).
Пишет budget_status.json (Claude Code читает).
Hard stop: budget_status.json{hard_stop=true} -> grace period 5 минут -> SIGTERM.

---

### Решения стадии 03 v1.0 (2026-03-19) — ЗАМЕНЕНЫ В v2.0

~~**TR-1: LLM Provider Strategy — provider-agnostic Protocol**~~
Заменён TR-8. LLMClient Protocol и три реализации выброшены.

~~**TR-2 v1.0: venv isolation через UAF**~~
Заменён TR-2 v2.0. Claude Code settings.json вместо UAF-managed venv.

~~**TR-3 v1.0: Plan-and-Execute с UAF в loop**~~
Заменён TR-3 v2.0. Claude Code как единый агент.

~~**TR-6: Failure Recovery retry x3**~~
Удалён. Claude Code сам обрабатывает ошибки.

TR-4, TR-5, TR-7 (ReportGenerator LaTeX): сохранены, частично обновлены.

---

### Решения стадии 02 (2026-03-19)

**R-02-01: Build vs Buy зафиксирован**
Buy: MLflow, DVC, Optuna, Anthropic API, AutoGluon/FLAML как callable tools внутри экспериментов.
Build: ResearchSessionController, ProgramMdGenerator, HumanOversightGate,
BudgetController, RuffEnforcer, ReportGenerator.
Claude Code: используем как внешний агент (не Buy, не Build — Deploy).

**R-02-02: Ближайший аналог — karpathy/autoresearch (март 2026)**
Покрывает ~30% нужной функциональности. Форкать не нужно.
Берём: концепт program.md, iteration budget, git трекинг.
Karpathy использовал Claude Code как агент — теперь и мы.

**R-02-03: Plan-and-Execute как агентная архитектура**
Обоснование: явный plan-артефакт необходим для human approval gate.
В v2.0: Claude Code реализует Plan-and-Execute внутри своей сессии.

**R-02-04: Шесть технических решений**
TR-1..TR-6 из v1.0 -> TR-2..TR-5, TR-7..TR-8 в v2.0.

**R-02-05: Degree of Innovation — умеренная, инженерная**
UAF — правильная инженерная композиция для незакрытого практического gap.

### Решения стадии 01 (2026-03-19)

**R-01-01: Два режима бюджета** — фиксированный и динамический.
**R-01-02: Обязательное одобрение program.md** — стандартный режим.
**R-01-03: Гибридная генерация program.md** — LLM черновик + human oversight.
**R-01-04: MLflow + DVC** — обязательные компоненты.
**R-01-05: Scope ограничен исследованием** — нет production, нет serving.

---

## Antigoals (из стадии 01, неизменны)

1. Не является AutoML для production-деплоя
2. Не запускает эксперименты без одобрения плана (в стандартном режиме)
3. Не скрывает неудачные эксперименты
4. Не модифицирует данные и код пользователя
5. Не оптимизирует под конкретный датасет в ущерб читаемости
6. Не превышает фиксированный бюджет

---

## Архитектура v2.0 (зафиксирована в стадии 03 v2.0)

**Концепция:** UAF — тонкая оболочка вокруг Claude Code.
UAF не управляет LLM в loop. UAF управляет сессией вокруг Claude Code.

**Flow:**
```
Пользователь
  -> ProgramMdGenerator (context/ пакет, без LLM)
  -> HumanOversightGate (y/n/edit) -> одобрение
  -> SessionSetup (settings.json, dirs, MLflow, DVC)
  -> ClaudeCodeRunner (запуск subprocess)
       Claude Code: читает context/, генерирует program.md
       Claude Code: пишет код, запускает, логирует в MLflow, применяет ruff
       Claude Code: обновляет program.md, делает DVC commits
       Claude Code: читает budget_status.json, останавливается при hard_stop
       Claude Code: генерирует текстовые секции отчёта в report/sections/
  -> BudgetController (polling thread, hard stop)
  -> RuffEnforcer (post-processing)
  -> ReportGenerator (читает sections/ + MLflow -> LaTeX -> PDF)
  -> Пользователь получает report.pdf
```

**6 компонентов + интеграционные:**

Основные:
- ResearchSessionController — state machine
- ProgramMdGenerator — подготовка context/ пакета, Jinja2 шаблон (без LLM)
- HumanOversightGate — approval checkpoint (antigoal 2)
- BudgetController — polling thread, budget_status.json, hard stop
- RuffEnforcer — ruff post-processing + RuffReport
- ReportGenerator — LaTeX/PDF из MLflow данных и секций Claude Code

Интеграционные:
- MLflowSetup — init, experiment creation, cross-referencing
- DVCSetup — init, auto commits
- ClaudeCodeRunner — subprocess управление Claude Code

Агент (внешний):
- Claude Code — автономно исполняет research plan

Storage (Buy):
- MLflow — метрики, параметры, артефакты runs
- DVC — версионирование program.md, кода, отчётов

---

## Контекст проекта

**Пользователь:** ML-инженер, одиночный, локальная машина
**Цель:** автоматизация цикла ML-исследований (гипотеза -> эксперимент -> вывод)
**Ожидаемый выигрыш:** 5-10x сокращение активного времени на рутинные исследования
**Bus Factor:** 1 (решается в стадии 15)
**Claude Code как агент:** ключевое архитектурное решение v2.0

---

## Решения стадии 04 (2026-03-19)

**Трёхуровневая схема метрик адаптирована для UAF:**

**Уровень 1 — Task Metrics (аналог Loss):**
Задаётся пользователем в task.yaml (metric.name + direction). Не фиксирована в UAF.
11 категорий задач: tabular classification/regression, NLP, CV, RecSys, RL.
UAF отвечает за передачу метрики Claude Code и корректное логирование в MLflow.

**Уровень 2 — UAF System Quality (14 метрик, Offline):**
M-UAF-01..14: quality plan generation, budget efficiency, code quality, reproducibility.
Все автоматически логируются в MLflow Session Summary Run.
Ключевые целевые значения: ruff_clean_rate >= 0.95, mlflow_compliance = 1.0,
reproducibility_score = 1.0, antigoal_violations_count = 0.

**Уровень 3 — Online Metrics (8 метрик):**
M-ONLINE-01..08: active_time_saved (> 5x), report_pdf_generated >= 95%,
failed_coverage = 1.0 (antigoal 3), uaf_exit_status = success.

**MLflow Schema (framework level):**
- Planning Run: tags + params сессии + program.md артефакт
- Experiment Run: обязательные поля, проверяются через M-UAF-11
- Session Summary Run: все агрегированные метрики UAF уровня 2 и 3

**Критерии остановки (dynamic mode):**
Три класса: Safety Cap (первый, жёсткий) -> Metric Convergence (алгоритмический,
три условия одновременно) -> LLM Signal (эвристический, consecutive=2).
Дефолты: patience=3, min_delta=0.001, min_iterations=3, safety_cap=50 iter.

**FP vs FN:**
BudgetController: FN дороже (нарушение antigoal 6) -> консервативный алгоритм.
MLflow compliance: FN дороже (потеря воспроизводимости) -> строгая проверка.

### Решения стадии 05 (2026-03-19)

**Форматы входных данных:**
CSV, Parquet, SQL Dump (основные) + JSONL, TXT, HF datasets (NLP) +
image_dir, COCO, manifest CSV (CV). DataLoader: read-only, sampling 100 МБ, timeout 60 сек.

**Metadata Schema (data_schema.json):**
7 разделов: splits, target, features, quality, leakage_audit,
adversarial_validation, task_hints. Полный JSON хранится в сессии,
краткая summary передаётся в ProgramMdGenerator как LLM-контекст.

**DVC интеграция:**
track_input_data: dvc add для входных данных (метаданные, не копии).
Артефакты сессии: > 1 МБ -> DVC, <= 1 МБ -> git.
DVC commit protocol закреплён в Execution Instructions program.md.
Связка MLflow run_id <-> git sha обеспечивает воспроизводимость.

**Adversarial Validation:**
LightGBM (100 деревьев) train vs val, ROC-AUC.
Пороги: passed < 0.6, warning 0.6-0.85, critical >= 0.85.
Critical: блокировка на HumanOversightGate, override требует явного y.
Только для tabular и NLP. CV не поддерживается.

**Leakage Audit (10 проверок, LA-01..LA-10):**
CRITICAL (блокируют): LA-01 (target in features), LA-05 (row overlap), LA-10 (schema mismatch).
WARNING (предупреждение + hints): LA-02..LA-04, LA-06..LA-09.
Результат фиксируется в data_schema.json и передаётся Claude Code как hints.

**Data Quality Report (LaTeX секция):**
7 блоков: dataset summary, feature types, target distribution, data quality,
leakage audit summary, adversarial validation, DVC tracking.
2 опциональных figure: null distribution (если max null > 5%), class distribution.

**Data Flow (порядок выполнения):**
DVCSetup -> DataLoader -> LeakageAudit -> AdversarialValidation ->
ProgramMdGenerator (с data context) -> HumanOversightGate ->
Claude Code (читает program.md) -> ReportGenerator (секция Data Overview).

**Antigoal 4 соблюдается:** UAF читает данные, не модифицирует.
dvc add создаёт .dvc файлы (метаданные), данные остаются на месте.

### Решения стадии 06 (2026-03-19)

**R-06-01: Схема валидации задаётся пользователем, UAF проверяет корректность**
scheme=auto как дефолт (по таблицам task.type + N строк), но пользователь переопределяет.
Antigoal 2 и 5 соблюдены.

**R-06-02: "Одна итерация" = полный round, не один fold**
BudgetController считает iterations (research cycles), не MLflow runs.
Для kfold(k=5): 1 итерация = 5 runs. budget_status.json содержит runs_per_iteration.

**R-06-03: Test set изолирован до финального run**
roc_auc_test логируется ТОЛЬКО для наилучшей модели в финальном run.
Нарушение = antigoal 5 violation (утечка test в итеративной оптимизации).

**R-06-04: ValidationChecker встроен в ResearchSessionController**
Не отдельный компонент — часть RSC. 3 момента: pre-session (VS-*), post-run (VR-*), post-session.

**R-06-05: Нет автоматического repair схемы**
ERROR = требует исправления task.yaml. Автоматический repair нарушает antigoal 2.

**Ключевые проверки (18 VS-*, 7 VR-*):**
- VS-T-001..005: train/val размеры и пропорции (holdout)
- VS-K-001..003: kfold параметры и стратификация
- VS-S-001..003: time-series (shuffle=False, gap >= forecast_horizon, монотонность)
- VS-G-001..002: group_col присутствует и достаточно групп
- VS-L-001..003: row overlap и target leakage
- VS-A-001..002: AdversarialValidation пороги
- VS-C-001..003: seed, совместимость scheme+task.type, test_holdout
- VR-001..007: post-run compliance (MLflow params и metrics)

**Метрики в Session Summary Run:**
val_test_delta (порог 0.05), cv_stability (порог 0.1 = std/mean).

**Схема time-series: gap >= forecast_horizon обязателен.**
Нарушение = ERROR VS-S-002 (блокирует сессию).

**NLP с document_id: GroupKFold обязателен.**
Случайный split при наличии нескольких фрагментов одного документа = leakage.

### Решения стадии 07 (2026-03-19) — SKIPPED/EMBEDDED

**R-07-01: Baseline logic встроена в шаблон program.md**
UAF — не ML-модель, отдельный baseline эксперимент для системы не нужен.
Порядок Phase 1: Constant/Dummy -> Rule-based -> Linear -> Simple non-linear.
ProgramMdGenerator всегда включает Phase 1 как обязательную первую фазу.
Отдельный артефакт не создаётся.

---

### Решения стадии 08 (2026-03-19)

**R-08-01: ResultAnalyzer — два слоя**
Слой A (ResultAnalyzer): анализ ML-экспериментов из MLflow, session_analysis.json.
Слой B (SystemErrorAnalyzer): самодиагностика UAF, SE-01..SE-09 категории.

**R-08-02: Post-session анализ — 8 шагов**
Разделение runs -> ранжирование -> метрический профиль -> анализ failures ->
param-metric корреляции (Spearman, >= 5 runs) -> сегментация -> гипотезы -> сборка.

**R-08-03: Feedback loop — 3 канала**
Канал 1 (основной): program.md, внутренний Claude Code loop.
Канал 2: budget_status.json с metrics_history.
Канал 3: improvement_context.md при --resume.

**R-08-04: 9 детерминированных правил гипотез (H-01..H-09)**
Без LLM. Максимум 5 по приоритету. H-09 — специфично для SHAP.

**R-08-05: SHAP — только tabular + tree-based по умолчанию**
Отдельный MLflow run type=analysis. Артефакты: shap_values.npy, shap_importance.csv,
shap_bar_chart.pdf. Включение через research_preferences.feature_importance в task.yaml.

**R-08-06: Сегментация требует prediction_csv в MLflow**
Инструкция в program.md через ProgramMdGenerator. Без predictions_available=True
сегментация пропускается без ошибки.

**R-08-07: failed_categories — 7 категорий по keyword matching**
import_error, oom_error, data_error, timeout_error, assertion_error, runtime_error, other.
Systemic failure если одна категория >= 50% failed runs.
Antigoal 3 соблюдён: partial runs включаются в отчёт с пометкой "(partial)".

**R-08-08: Отчёт — 6 секций от error analysis**
Analysis and Findings (LLM), Failed Experiments, Feature Importance/SHAP,
Segment Analysis, Improvement Hypotheses, UAF System Health.

---

### Решения стадии 09 (2026-03-19)

**R-09-01: Scaffold с контрактными секциями (# UAF-SECTION: <name>)**
experiment.py — параметризованный шаблон с 11 обязательными секциями.
Claude Code заполняет секции, не пишет с нуля. Smoke tests проверяют
наличие секций и их корректность до запуска.

**R-09-02: Один параметризованный scaffold, не отдельные шаблоны**
task.type управляет заполнением (tabular/NLP/CV). DRY. Claude Code
понимает условную логику без дополнительных файлов.

**R-09-03: uv + requirements.lock для воспроизводимости**
uv pip freeze -> requirements.lock в SESSION_DIR. sha в MLflow Planning Run.
Два режима: shared environment (дефолт) и session-isolated venv (use_venv: true).
Конфликты: uv --resolution=lowest-direct как fallback.

**R-09-04: Kill logic — grace period 5 минут, SIGTERM -> SIGKILL**
hard_stop=true -> 5 минут grace period (Claude Code сохраняет checkpoint,
закрывает MLflow run) -> SIGTERM -> 30 секунд -> SIGKILL.
Три уровня таймаутов: experiment_timeout, session_timeout, budget iterations.

**R-09-05: SmokeTestRunner — 11 тестов (ST-01..ST-11)**
ST-01: scaffold секции, ST-02: синтаксис, ST-03: ruff lint,
ST-04: mlflow.start_run, ST-05: check_budget, ST-06: seed,
ST-07: NotImplementedError отсутствует в заполненных секциях,
ST-08: experiment_config.yaml валиден, ST-09: MLflow доступен,
ST-10: нет хардкода путей (warning), ST-11: dry-run (tabular, < 90 сек).
smoke_test_report.json в SESSION_DIR. Провал блокирует запуск.

**Файловая структура SESSION_DIR зафиксирована:**
program.md, experiment_config.yaml, experiment.py, requirements.lock,
budget_status.json, smoke_test_report.json, session_analysis.json,
artifacts/ (модели, > 1 МБ -> DVC), checkpoints/ (DL, всегда DVC),
predictions/ (predictions.csv каждой итерации), report/ (LaTeX/PDF).

**Уровень 1 (UAF Pipeline) и Уровень 2 (Experiment Pipeline) разделены:**
UAF Pipeline — инвариантный (ResearchSessionController, компоненты).
Experiment Pipeline — вариативный (scaffold + заполнение Claude Code).

---

## Следующие шаги

**Текущая задача:** стадия 16-postmortem

---

### Решения стадии 13 (2026-03-19)

**R-13-01: Пирамида мониторинга — 4 уровня, адаптированные к сессионному контексту**
SW Health -> DQ -> Model Quality -> Business KPIs.
SW Health — предусловие для остальных. Проверяется первым в каждом цикле.

**R-13-02: Единый polling interval 30 сек (DQ — 60 сек)**
OOM читается из stderr потоково без задержки — не зависит от polling.

**R-13-03: budget_status.json v2.1 — единая точка коммуникации**
Атомарная запись (os.replace). Новые поля: software_health, data_quality,
alerts[], hints[], phase. Схема расширяет v1.0 из стадии 09 (обратно совместима).

**R-13-04: session.log — структурированный текст (не JSON)**
TIMESTAMP LEVEL [COMPONENT] MESSAGE формат. Удобен для tail -f.
JSON только в budget_status.json.

**R-13-05: MLflow интеграция через API (приоритет) + файловая система (fallback)**
BudgetController только читает. Исключение: тег uaf.forced_stop при hard_stop.
Session Summary Run включает monitoring.* метрики.

**R-13-06: Реестр алертов — 12 кодов, 3 уровня**
CRITICAL (hard_stop): SW-HANG, SW-DISK-FULL, MQ-NAN-CASCADE, DQ-DATA-MODIFIED.
WARNING (предупреждение): MQ-DEGRADATION, MQ-CONSECUTIVE-FAILS, BQ-BUDGET-80PCT, BQ-TIME-80PCT, SW-MLFLOW-DOWN, DQ-SCHEMA-DRIFT.
INFO: MQ-NEW-BEST, MQ-CONVERGENCE, BQ-ITER-COMPLETE.

**R-13-07: Hang detection консервативный (FP допустим, FN недопустим)**
300 сек CPU=0% + no stdout -> 60 сек ожидание -> повторная проверка -> hard_stop.

**R-13-08: MonitoringDashboard опциональный (--monitor)**
Rich-based, читает budget_status.json, не блокирует BudgetController.

**R-13-09: Отчёт содержит обязательные графики мониторинга**
Metric Progression (всегда), Budget Burndown (всегда), Alert Timeline (если WARNING+).
Claude Code генерирует Monitoring Conclusions в рамках сессии, ReportGenerator вставляет в отчёт.

**Стадии 11, 12 — SKIPPED/antigoal-1:**
11-measurement: нет production, нет A/B тестов -> онлайн-метрики определены в стадии 04.
12-integration: нет production интеграции -> UAF сам является системой.

---

### Решения стадии 15 (2026-03-19)

**R-15-01: RACI адаптирован к одному человеку**
A = всегда ML-инженер (константа, не дублируется). R = кто/что исполняет,
C = что задействовано, I = что уведомляется. 13 компонентов покрыты.
Критические компоненты: RSC, ProgramMdGenerator, BudgetController, SmokeTestRunner,
ValidationChecker, MLflowSetup, ClaudeCodeRunner — без них сессия невозможна.
Некритические: RuffEnforcer, MonitoringDashboard — сессия продолжается при сбое.

**R-15-02: Bus Factor = 1 митигируется документацией, не устраняется**
Три уровня: docs/progress.md (обзор, 30 мин), docs/stage-NN-*.md (детальный),
docs/stage-15-ownership.md (runbook, операционный). Критерий: новый человек
запускает сессию и интерпретирует результаты за 4 часа без чтения кода.
Критические точки знаний: архитектура v2.0, связка MLflow+DVC+git, antigoals,
settings.json как механизм безопасности, shadow feature trick.

**R-15-03: Runbook — 5 сценариев с исполняемыми командами**
Новая сессия, --resume, 4 типа крашей (до HumanOversightGate, во время Claude Code,
ReportGenerator, SmokeTestRunner провал), восстановление из MLflow+DVC.
Каждый сценарий: диагностика -> причина -> конкретные команды -> действие.

**R-15-04: Dependency Map — 7 зависимостей с degraded mode**
КРИТИЧЕСКИЕ: Anthropic API, Claude Code CLI, MLflow, Python+uv.
ВАЖНЫЕ: DVC, ruff, git. НЕКРИТИЧЕСКИЕ: tectonic/pdflatex, LightGBM, Rich, psutil.
Для каждой: что теряется при отказе, как восстановить, есть ли fallback.
Claude Code — единственная зависимость без fallback (основной агент, замены нет).

**R-15-05: Чеклист — 5 групп, все пункты с командами**
Группы: данные и задача, окружение, предыдущие сессии, специфика task.type,
финальная проверка. Не описание "убедитесь что" — конкретный `python -c` или
`curl` с ожидаемым выводом.

**R-15-06: Секция Ownership в PDF — 5 блоков без LLM вызова**
Session Identification, Reproducibility Checklist (автоматические OK/MISSING),
Component Versions из requirements.lock, краткие Recovery Instructions, Bus Factor Statement.
report.pdf — исследовательский артефакт, операционные runbooks остаются в stage-15.

---

### Решения стадии 10 (2026-03-19)

**R-10-01: Shadow Feature Trick как основной механизм**
Shadow (параллельное сравнение baseline vs candidate) вместо inline добавления признаков.
Baseline run_id фиксируется в program.md. Дельта = единственный критерий принятия.
Двойное обучение считается одной итерацией в BudgetController.

**R-10-02: Feature Store не нужен — Feature Registry как минималистичный JSON**
feature_registry.json в SESSION_DIR: список engineered фич, status accepted/rejected,
delta, source_columns, iteration_introduced. DVC не нужен для JSON < 1 МБ.
При --resume передаётся в improvement_context.md для следующей сессии.

**R-10-03: Детерминированные правила генерации гипотез (без LLM)**
ProgramMdGenerator генерирует feature гипотезы программно по data_schema.json:
FG-T-* (temporal), FG-N-* (numeric transforms), FG-I-* (interactions),
FG-C-* (categorical encoding), FG-NLP-* (text statistics), FG-CV-* (augmentation).
Приоритет: datetime > interaction > encoding > numeric > group stats. Максимум 5 шагов.

**R-10-04: Phase 2 (Feature Engineering) обязательная, выполняется всегда после Phase 1**
Не зависит от результатов Phase 1. Пропускается только при явном skip_feature_engineering: true в task.yaml.
Лимит: 5 shadow шагов в одной фазе.
Переход Phase 2 -> Phase 3: фиксация Accepted Features + обновление data_schema.json features.selected_baseline.

**R-10-05: Feature importance tracking через ResultAnalyzer (post-session)**
MLflow per-run: feature_importance.csv + fi_rank_shadow_* метрики.
Post-session: importance matrix (features x iterations), fi_stability score, top5_stable_features.
H-09 из стадии 08 триггерится при fi_stability < 0.5.
Артефакты Session Summary Run: feature_importance_matrix.csv, feature_importance_timeline.pdf.

**Новые добавления к существующим компонентам:**
- VR-FE-001: новая ValidationChecker проверка — target encoding leakage для shadow experiments
- ST-12: новый SmokeTestRunner тест — baseline_run_id валиден в MLflow
- min_feature_delta: 0.002 дефолт (из task.yaml research_preferences.min_feature_delta)
- budget_status.json: runs_per_iteration для phase2_shadow_step = 2

---

### Решения стадии 16 (2026-03-19)

**R-16-01: Session Postmortem — одновременно шаблон и процесс**
Шаблон без процесса не заполняется. AUTO/MANUAL разделение явное. Критерий
полноты постмортема зафиксирован (иначе заполняется формально).

**R-16-02: improvement_context.md как канал между сессиями**
Секция 4 (гипотезы) + секция 7 (выводы) постмортема -> improvement_context.md.
Передаётся в ProgramMdGenerator при --resume или новой сессии по той же задаче.
Обеспечивает преемственность без повтора ошибок.

**R-16-03: CHANGES.md — трёхшаговый цикл итерации UAF**
Pending -> Validation (2+ постмортема для DESIGN, 1 для P1) -> Applied с эффектом.
Пять типов изменений: BUG/DESIGN/USABILITY/PERFORMANCE/NEW.
P1 изменения блокируют следующую сессию.

**R-16-04: health_history.jsonl — накопленный опыт по сессиям**
Append-only JSONL. Одна строка = одна сессия. 11 Health KPIs с целевыми и тревожными
порогами. uaf health команда для просмотра тренда. Пороги пересматриваются после 5 сессий.

**R-16-05: Session Retrospective Synthesis — опциональный, через Claude Code subprocess**
Активируется при M-UAF-09 < 4.0 или флаге --deep-retrospective. UAF пишет
retrospective_context.md и запускает отдельный Claude Code subprocess для генерации
Synthesis (200-300 слов, один главный следующий шаг). TD-04 закрыт.

**R-16-06: Технический долг проектной фазы (TD-01..TD-05)**
TD-01: нет baseline данных M-UAF-01 (заполнится после первых сессий).
TD-02: шаблон context/ ProgramMdGenerator не протестирован на реальных задачах.
TD-03: дефолты BudgetController требуют эмпирической калибровки.
TD-04: Session Retrospective Synthesis — реализуется через Claude Code subprocess (закрыт архитектурно).
TD-05: uaf health команда не определена в stage-09 pipeline.

---

## ФИНАЛЬНОЕ РЕЗЮМЕ ПРОЕКТА

**Проект завершён.** Все 16 стадий проектирования пройдены.
Дата: 2026-03-19. Методология: Babushkin & Kravchenko (2025).

---

### Что построено

**Universal AutoResearch Framework (UAF) v2.0** — исследовательская система для
автоматизации цикла ML-экспериментов: гипотеза -> эксперимент -> вывод -> следующая гипотеза.

Пользователь: одиночный ML-инженер, локальная машина.
Агент: Claude Code (внешний).
Ожидаемый эффект: 5-10x сокращение активного времени на рутинные исследования.

**Концепция:** UAF — тонкая оболочка вокруг Claude Code. UAF управляет сессией
(plan, oversight, budget, monitoring, reporting). Claude Code управляет экспериментами
(код, MLflow, ruff, DVC, ошибки). Разделение ответственности чёткое.

---

### Компоненты системы

**6 основных компонентов:**

| Компонент | Ответственность | Критичность |
|-----------|----------------|-------------|
| ResearchSessionController | State machine сессии, оркестрация всех компонентов | Критичный |
| ProgramMdGenerator | Подготовка context/ пакета (без LLM); Claude Code генерирует program.md при старте | Критичный |
| HumanOversightGate | Одобрение plan перед запуском (antigoal 2) | Критичный |
| BudgetController | Polling thread 30 сек, budget_status.json, hard stop | Критичный |
| RuffEnforcer | ruff post-processing всех .py файлов сессии | Некритичный |
| ReportGenerator | Компиляция LaTeX/PDF из MLflow данных и секций, сгенерированных Claude Code | Важный |

**3 интеграционных компонента:**

| Компонент | Ответственность |
|-----------|----------------|
| MLflowSetup | Инициализация MLflow, experiment structure, Planning Run |
| DVCSetup | DVC init, auto-commits артефактов |
| ClaudeCodeRunner | Subprocess управление Claude Code, settings.json |

**Встроенные подкомпоненты (не отдельные классы):**

| Компонент | Где встроен |
|-----------|-------------|
| ValidationChecker (18 VS-* + 7 VR-* + VR-FE-001) | ResearchSessionController |
| DataLoader (CSV/Parquet/SQL/JSONL/images) | ResearchSessionController |
| LeakageAudit (10 проверок LA-01..LA-10) | DataLoader |
| AdversarialValidation (LightGBM, AUC пороги) | DataLoader |
| ResultAnalyzer слой A (8 шагов, H-01..H-09) | ReportGenerator + BudgetController |
| SystemErrorAnalyzer слой B (SE-01..SE-09) | ReportGenerator |
| SmokeTestRunner (12 тестов ST-01..ST-12) | ResearchSessionController |
| MonitoringDashboard (Rich, опциональный --monitor) | BudgetController |

**Внешний агент:**
- Claude Code — автономно исполняет research plan, пишет код, логирует в MLflow,
  применяет ruff, делает DVC commits, читает budget_status.json

**Buy-зависимости:**
- MLflow — трекинг всех runs, метрик, параметров, артефактов
- DVC — версионирование данных и артефактов > 1 МБ
- Optuna — гиперпараметры (внутри экспериментов, Claude Code вызывает)
- Claude Code CLI — агент, запускается как subprocess (UAF не вызывает Anthropic API напрямую)
- uv — управление зависимостями, requirements.lock
- ruff — линтинг и форматирование кода

---

### Ключевые архитектурные решения

**TR-3 v2.0: Claude Code как единый агент**
UAF не управляет LLM в loop. UAF создаёт program.md + settings.json, запускает Claude Code
subprocess, мониторит снаружи. Claude Code реализует Plan-and-Execute внутри своей сессии.
Это устранило необходимость в LLMClient, PlanningAgent, ExperimentRunner, FailureRecovery.

**TR-4 v2.0: program.md как контракт**
Один файл содержит план исследования (для человека при HumanOversightGate) и
Execution Instructions (для Claude Code). ML-инженер одобряет и план и инструкции
за один просмотр.

**TR-2 v2.0: settings.json как механизм безопасности**
Claude Code ограничен settings.json: пишет только в SESSION_DIR, deny list для
деструктивных команд. Заменяет AST-валидатор из v1.0.

**BudgetController + budget_status.json v2.1**
Polling thread 30 сек. Пишет budget_status.json (атомарно через os.replace).
Claude Code читает и самостоятельно останавливается при hard_stop=true.
Grace period 5 минут. 12 алерт-кодов (4 CRITICAL, 6 WARNING, 2 INFO).

**Shadow Feature Trick**
Phase 2: параллельное обучение baseline + candidate в одной итерации BudgetController.
Delta > min_feature_delta (дефолт 0.002) = feature accepted. Feature Registry = JSON.
Feature Store не нужен (нет serving, нет команды).

**Validation Scheme auto-select**
scheme=auto по task.type + N строк. Time-series: gap >= forecast_horizon обязателен (VS-S-002).
NLP с document_id: GroupKFold обязателен. Test set изолирован до финального run.

---

### Что пропущено и почему

| Стадия | Статус | Причина |
|--------|--------|---------|
| 07-baseline | SKIPPED/EMBEDDED | UAF — не модель. Baseline logic встроена как Phase 1 шаблона program.md. |
| 11-measurement | SKIPPED/antigoal-1 | Нет production, нет A/B тестов. Онлайн-метрики определены в стадии 04. |
| 12-integration | SKIPPED/antigoal-1 | Нет production интеграции. UAF сам является конечной системой. |
| 14-serving | SKIPPED/antigoal-1 | Нет production serving. UAF — локальная исследовательская система. |

Antigoal 1 ("не является AutoML для production-деплоя") — единственная причина
трёх из четырёх пропущенных стадий. Это правильное следствие scope ограничения,
зафиксированного в стадии 01. Scope не расширялся ни разу за 16 стадий.

---

### Ключевые числа системы

| Параметр | Значение |
|----------|----------|
| Компонентов основных | 6 |
| Компонентов интеграционных | 3 |
| Antigoals | 6 (неизменны с стадии 01) |
| Anthropic API вызовов из UAF за сессию | 0 (UAF не делает прямых LLM вызовов) |
| Smoke тестов | 12 (ST-01..ST-12) |
| Validation проверок | 26 (18 VS-* + 7 VR-* + 1 VR-FE-001) |
| Алерт-кодов мониторинга | 12 (4 CRITICAL + 6 WARNING + 2 INFO) |
| Метрик UAF качества (M-UAF) | 14 |
| Метрик онлайн (M-ONLINE) | 8 |
| Правил гипотез ResultAnalyzer | 9 (H-01..H-09) |
| Системных ошибок SystemErrorAnalyzer | 9 (SE-01..SE-09) |
| Health KPIs UAF | 11 |
| Технический долг проектной фазы | 5 пунктов (TD-01..TD-05) |
| Polling interval BudgetController | 30 сек |
| Grace period при hard_stop | 5 минут |
| Критерий Bus Factor | 4 часа до первого запуска сессии |

---

### Следующий шаг

Реализация компонентов UAF согласно документации.

Рекомендуемая последовательность реализации:
1. MLflowSetup + DVCSetup (инфраструктура)
2. BudgetController (polling, budget_status.json v2.1)
3. DataLoader + ValidationChecker (данные и валидация)
4. ProgramMdGenerator (промпт + Jinja2 шаблон)
5. HumanOversightGate (интерактивный approval)
6. SmokeTestRunner (12 тестов)
7. ClaudeCodeRunner (subprocess, settings.json)
8. ResultAnalyzer + SystemErrorAnalyzer (post-session)
9. RuffEnforcer (post-processing)
10. ReportGenerator (LaTeX + PDF, компиляция из секций Claude Code)
11. ResearchSessionController (state machine, связывает всё)
12. CLI: uaf run, uaf resume, uaf report, uaf health, uaf analyze

Первые 3 сессии следует рассматривать как тестирование промпта ProgramMdGenerator
и калибровку дефолтов BudgetController (TD-01..TD-03).
