# Стадия 16: Postmortem

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-10, 13, 15 (COMPLETE), 07/11/12/14 (SKIPPED)

---

## 0. Контекст: что такое postmortem для UAF

Классический postmortem — это ретроспектива команды после инцидента или завершения
проекта. В контексте UAF это не применимо дословно: нет команды, нет ретроспектив в
классическом смысле, нет production-инцидентов.

Постмортем UAF имеет два уровня:

**Уровень 1 — Session Postmortem:** анализ завершённой исследовательской сессии.
Что сработало, что не сработало, что нужно изменить в следующей сессии.
Выполняется после каждой сессии ML-инженером.

**Уровень 2 — Framework Postmortem:** накопленный анализ нескольких сессий,
направленный на улучшение самого UAF. Выполняется периодически (например,
после 5-10 сессий) или при обнаружении системных проблем.

Этот документ определяет:
- шаблон session postmortem
- механизм итерации над program.md на основе постмортема
- механизм итерации над UAF (компоненты, алгоритмы)
- автоматическую часть session retrospective PDF
- метрики здоровья UAF накопленные по сессиям
- финальный чеклист закрытия сессии

---

## 1. Шаблон Session Postmortem

Заполняется ML-инженером после каждой завершённой сессии. Время на заполнение:
15-30 минут. Хранится в SESSION_DIR как `session_postmortem.md`.

Структура шаблона — ниже. Секции [AUTO] заполняются UAF автоматически из данных
MLflow и session.log. Секции [MANUAL] заполняет ML-инженер.

---

### Шаблон: `.uaf/sessions/{session_id}/session_postmortem.md`

```markdown
# Session Postmortem

**Session ID:** {session_id}
**Task:** {task.name}
**Date:** {session_start_datetime}
**Duration:** {session_duration_min} мин
**Status:** {success | budget_exhausted | hard_stop | crashed}

---

## 1. Результаты сессии [AUTO]

**Метрика цели:** {metric.name} = {metric.direction}
**Лучший результат:** {best_metric_value} (run: {best_run_id})
**Baseline (Phase 1):** {phase1_best_metric_value} (baseline_run_id: {baseline_run_id})
**Улучшение над baseline:** {improvement_over_baseline_pct}%
**Итераций выполнено:** {iterations_done} / {iterations_budget}
**Бюджет использован:** {budget_pct_used}%
**Runs успешных / всего:** {successful_runs} / {total_runs}
**Причина остановки:** {stop_reason}

### Принятые Feature Engineering признаки [AUTO]
{accepted_features_list или "Phase 2 не выполнялась"}

### Системные ошибки UAF [AUTO]
{system_errors_list из SystemErrorAnalyzer или "Нет"}

---

## 2. Анализ программы исследования [MANUAL]

### 2.1 Что сработало по плану?
(Какие гипотезы из program.md подтвердились, какие методы дали ожидаемые результаты)

-

### 2.2 Что не сработало и почему?
(Какие гипотезы провалились, какие методы не дали ожидаемого эффекта)
(Для каждого: гипотеза -> наблюдение -> вероятная причина)

-

### 2.3 Что стало неожиданностью?
(Аномальные результаты, неожиданные паттерны в данных, OOM на неожиданном размере,
hang detection срабатывавший ложно или не срабатывавший когда нужно)

-

### 2.4 Оценка качества program.md
(Насколько хорошо сгенерированный план соответствовал задаче?)

| Аспект | Оценка (1-5) | Комментарий |
|--------|-------------|-------------|
| Релевантность гипотез | | |
| Порядок фаз | | |
| Execution Instructions (MLflow, ruff, budget) | | |
| Достаточность бюджета | | |
| Точность метрики цели | | |

**Средняя оценка program.md:** {среднее}

---

## 3. Конкретные проблемы UAF [MANUAL + AUTO]

Для каждой проблемы: компонент, описание, тип (bug | design | usability | performance).

### 3.1 Проблемы компонентов [MANUAL]

| # | Компонент | Проблема | Тип | Приоритет (P1-P3) |
|---|-----------|----------|-----|-------------------|
| | | | | |

### 3.2 Системные ошибки UAF [AUTO из SystemErrorAnalyzer]

{se_errors_table из session_analysis.json}

### 3.3 Нарушения antigoals [AUTO]
{antigoal_violations из M-UAF-14 в MLflow или "Нарушений нет"}

**Важно:** любое нарушение antigoal = P1 приоритет. Нарушение antigoal значит
что дизайн системы не работает. Это не usability issue — это архитектурный дефект.

---

## 4. Гипотезы для следующей сессии [AUTO из ResultAnalyzer]

{hypotheses_list из session_analysis.json (H-01..H-09, отсортированные по приоритету)}

### Дополнительные гипотезы от ML-инженера [MANUAL]

(Что не поймал ResultAnalyzer, но ML-инженер видит из опыта)

-

---

## 5. Итерация над program.md [MANUAL]

Что конкретно нужно изменить в шаблоне program.md (или в логике ProgramMdGenerator)
чтобы следующая сессия была лучше.

Формат: [ШАБЛОН | ИНСТРУКЦИЯ | ПРОМПТ] -> описание изменения

| # | Где изменить | Текущее поведение | Желаемое поведение | Приоритет |
|---|-------------|------------------|-------------------|-----------|
| | | | | |

---

## 6. Итерация над UAF [MANUAL]

Что нужно изменить в компонентах UAF. Формат: компонент -> изменение -> обоснование.

| # | Компонент | Изменение | Обоснование | Тип | Приоритет |
|---|-----------|-----------|-------------|-----|-----------|
| | | | | bug/design/usability/perf | P1/P2/P3 |

---

## 7. Финальные выводы [MANUAL]

### Основной вывод по задаче:
(Один абзац: что выяснилось о задаче)

### Рекомендация по следующему шагу:
- [ ] Новая сессия с уточнённой задачей: {описание}
- [ ] Resume с гипотезами: {список H-01..H-09}
- [ ] Закрыть задачу: результат {достаточный / недостаточный}
- [ ] Другое: {описание}

---

_Postmortem создан: {datetime}_
_Время на заполнение: ___ мин_
```

---

## 2. Процесс Session Postmortem

### 2.1 Когда выполнять

Session Postmortem выполняется после каждой сессии UAF, независимо от её исхода.
Исключение: сессия упала до HumanOversightGate (нет данных для анализа) — в этом
случае постмортем выполнять нецелесообразно, достаточно краткой записи в session.log.

Оптимальное время — сразу после завершения сессии, пока контекст свеж.
Если откладывается — не дольше чем 24 часа.

### 2.2 Последовательность шагов

```
Шаг 1. UAF автоматически генерирует session_postmortem.md с [AUTO] секциями.
   Это происходит в ReportGenerator после завершения сессии.
   Файл сохраняется в SESSION_DIR/session_postmortem.md.

Шаг 2. ML-инженер открывает session_postmortem.md и заполняет [MANUAL] секции.
   Ориентир: report.pdf уже прочитан (секция Experiment Results, Failed Experiments,
   Monitoring, Recommendations).
   Параллельно открыт MLflow UI для деталей по конкретным runs.

Шаг 3. ML-инженер заполняет секции 2, 3.1, 5, 6, 7.
   Секции 3.2, 3.3, 4 уже заполнены автоматически.

Шаг 4. session_postmortem.md коммитится в git вместе с остальными SESSION_DIR артефактами.
   git add .uaf/sessions/{session_id}/session_postmortem.md
   git commit -m "postmortem: {session_id} - {task_name}"

Шаг 5. Если в секции 5 или 6 есть изменения с приоритетом P1:
   -> немедленно создать issue в CHANGES.md (см. секцию 3 ниже).
   -> не запускать следующую сессию без фикса P1 изменений.

Шаг 6. Обновить UAF Health Dashboard (см. секцию 5 ниже).
   uaf health --update --session {session_id}
   Это добавляет строку в .uaf/health_history.jsonl.
```

### 2.3 Критерии качественного постмортема

Постмортем считается полным если:
- Каждая провалившаяся гипотеза имеет объяснение "почему" (не "не сработало")
- Секция 5 содержит хотя бы одно конкретное изменение program.md
- Секция 7 содержит чёткую рекомендацию по следующему шагу
- Если были нарушения antigoal — они задокументированы с P1 приоритетом

Постмортем считается неполным если:
- Секция 2.2 содержит только "не сработало" без причины
- Секция 5 пуста (даже если сессия была идеальной — это означает программный промпт
  не нужно менять, но это должно быть явно записано)
- Секция 7 не содержит чёткой рекомендации

---

## 3. Итерация над program.md

Постмортем — основной механизм улучшения program.md. Изменения двух типов:

### 3.1 Тип A: изменения конкретной сессии (improvement_context.md)

Если сессия завершается с незавершёнными гипотезами или недостаточным результатом,
ML-инженер запускает следующую сессию через `uaf resume` или новую сессию.

В обоих случаях постмортем передаётся в ProgramMdGenerator как контекст через
`improvement_context.md`. Этот файл создаётся UAF автоматически из секций 4 и 7
session_postmortem.md, дополняется ML-инженером при необходимости.

**Схема improvement_context.md:**

```markdown
# Improvement Context

**Source session:** {prev_session_id}
**Task:** {task.name}

## Результаты предыдущей сессии
- Лучшая метрика: {best_metric_value}
- Улучшение над baseline: {improvement_pct}%
- Причина остановки: {stop_reason}

## Гипотезы для проверки (из ResultAnalyzer)
{hypotheses list H-01..H-09 с описаниями}

## Дополнительный контекст от ML-инженера
{из секции 4 "дополнительные гипотезы" и секции 7}

## Что не нужно повторять
{из секции 2.2 session_postmortem.md — провалившиеся подходы}

## Accepted features из предыдущей сессии
{feature_registry.json summary — только accepted features}
```

ProgramMdGenerator получает этот файл и учитывает его при генерации нового program.md.
Это обеспечивает преемственность между сессиями без повтора ошибок.

### 3.2 Тип B: изменения шаблона program.md (системные)

Если одна и та же проблема с plan повторяется в двух или более сессиях подряд,
это системная проблема шаблона или промпта ProgramMdGenerator.

Изменения шаблона фиксируются в `CHANGES.md` и применяются к следующей сессии:

**`CHANGES.md` в корне проекта:**

```markdown
# UAF Changes Log

## Pending (не применено)
| # | Тип | Компонент | Описание | Source Postmortem | Приоритет |
|---|-----|-----------|----------|-------------------|-----------|

## Applied (применено, дата)
| # | Тип | Компонент | Описание | Source Postmortem | Применено |
|---|-----|-----------|----------|-------------------|-----------|
```

Типы изменений program.md:

| Тип | Описание | Пример |
|-----|----------|--------|
| TEMPLATE | Изменение структуры program.md (добавить/убрать секцию) | Добавить секцию "Known Failures" для передачи в следующую сессию |
| INSTRUCTION | Изменение Execution Instructions в program.md | Добавить инструкцию для сохранения predictions.csv в конкретном формате |
| PROMPT | Изменение системного промпта ProgramMdGenerator | Добавить требование к конкретизации гипотез ("гипотеза должна содержать ожидаемый delta") |
| PHASE | Изменение логики фаз (фаза 1, 2, 3) | Изменить порядок baseline методов в Phase 1 |
| BUDGET | Изменение дефолтных бюджетных параметров | Увеличить дефолт patience с 3 до 5 для NLP задач |

### 3.3 Алгоритм решения: изменить program.md или experiment.py?

Частая ошибка: ML-инженер хочет "зафиксировать" что-то в program.md, хотя
это относится к коду эксперимента. Правило:

- Если проблема в том **что** исследуется -> изменять program.md (шаблон, промпт)
- Если проблема в том **как** выполняется эксперимент -> изменять scaffold experiment.py
- Если проблема в том **как** Claude Code получает инструкции -> изменять Execution Instructions
- Если проблема в компонентах UAF (monitoring, budget, validation) -> CHANGES.md тип COMPONENT

---

## 4. Итерация над UAF

### 4.1 Типы изменений UAF

Изменения UAF классифицируются по компоненту и типу. Все изменения фиксируются в `CHANGES.md`.

**Типы изменений:**

| Тип | Описание | Когда применять |
|-----|----------|-----------------|
| BUG | Компонент ведёт себя не так как задокументировано | Немедленно (P1) |
| DESIGN | Дизайн компонента правильно реализован, но даёт плохой результат | После 2+ подтверждений (P2) |
| USABILITY | Компонент работает, но неудобен в использовании | Накопленный backlog (P3) |
| PERFORMANCE | Компонент работает правильно, но медленно | После профилирования (P3) |
| NEW | Новый компонент или capability | После обоснования в постмортеме (P2) |

**Приоритеты:**

| Приоритет | Описание | Когда фиксировать | Блокирует следующую сессию? |
|-----------|----------|------------------|----------------------------|
| P1 | Нарушение antigoal, потеря данных, crash воспроизводимый | Сразу после постмортема | Да |
| P2 | Серьёзная деградация качества сессии, неверный дизайн компонента | До следующего пятничного review | Нет |
| P3 | Usability, cleanup, minor improvements | Накопленный backlog | Нет |

### 4.2 Как идентифицировать что менять

Каждый компонент UAF имеет конкретные сигналы проблем. Сигналы извлекаются из
session.log, session_analysis.json и постмортема вручную.

**ResearchSessionController:**
- Сигнал проблемы: сессия зависла в промежуточном состоянии без алерта
- Сигнал проблемы: state machine перешла в некорректное состояние (session.log)
- Как детектировать: grep "ERROR\|CRITICAL" session.log | grep "RSC"

**ProgramMdGenerator:**
- Сигнал проблемы: ML-инженер полностью переписал program.md при HumanOversightGate
- Сигнал проблемы: Phase 1 не содержала baseline в правильном порядке
- Сигнал проблемы: Execution Instructions отсутствовали или неполные
- Как детектировать: оценка quality_plan_generation (M-UAF-01) < 3 в постмортеме

**HumanOversightGate:**
- Сигнал проблемы: ML-инженер одобряет план не прочитав его (antigoal 2 bypass по привычке)
- Сигнал проблемы: ML-инженер не может понять структуру plan за 5 минут
- Как детектировать: субъективная оценка в постмортеме секция 2.4

**BudgetController:**
- Сигнал проблемы: hard_stop сработал позже ожидаемого (budget_pct_used > 110%)
- Сигнал проблемы: convergence сработала слишком рано (iterations_done < min_iterations)
- Сигнал проблемы: hang detection — ложные срабатывания > 1 в сессии
- Как детектировать: M-UAF-05 (budget_efficiency), monitoring.total_alerts в MLflow

**SmokeTestRunner:**
- Сигнал проблемы: ST-* провалился по причине несвязанной с экспериментом
- Сигнал проблемы: dry-run > 90 сек на нормальных данных
- Как детектировать: smoke_test_report.json failed_tests список

**ValidationChecker:**
- Сигнал проблемы: VS-*/VR-* ложные срабатывания блокируют корректные конфигурации
- Сигнал проблемы: val_test_delta > 0.05 не было поймано (antigoal 5)
- Как детектировать: M-UAF-06 (validation_coverage) + ручной просмотр val_test_delta

**RuffEnforcer:**
- Сигнал проблемы: ruff_clean_rate < 0.95 по M-UAF-07
- Сигнал проблемы: ruff ошибки одного типа повторяются в нескольких runs
- Как детектировать: ruff_report.json categories

**ReportGenerator:**
- Сигнал проблемы: report.pdf содержит пустые секции без объяснения
- Сигнал проблемы: LLM-генерируемые секции нерелевантны задаче
- Сигнал проблемы: report_generation_time_sec > 300
- Как детектировать: M-UAF-08 (report_pdf_generated), M-UAF-09 (report_quality_score)

**ResultAnalyzer:**
- Сигнал проблемы: гипотезы H-01..H-09 не совпадают с тем что ML-инженер видит вручную
- Сигнал проблемы: fi_stability < 0.5 не было поймано правилом H-09
- Как детектировать: секция 2.3 постмортема ("неожиданности") + сравнение гипотез

### 4.3 Механизм применения изменений UAF

Изменения UAF не применяются немедленно — они проходят через трёхшаговый цикл:

```
Шаг 1. Идентификация
  Постмортем сессии -> секция 6 (итерация над UAF) -> запись в CHANGES.md (Pending)

Шаг 2. Валидация
  После 2+ независимых постмортемов с одной проблемой (или 1 при P1):
  -> Перенести из Pending в активную задачу
  -> Описать ожидаемый эффект фикса (как изменится M-UAF-NN метрика?)

Шаг 3. Применение и проверка
  -> Реализовать изменение в UAF коде
  -> Запустить тестовую сессию
  -> Записать в CHANGES.md Applied с датой и наблюдаемым эффектом
```

Для P1 изменений шаги 1 и 2 объединяются — применять немедленно.

---

## 5. Session Retrospective в PDF

Постмортем порождает отдельную секцию в report.pdf. Эта секция называется
"Session Retrospective" и является последней секцией отчёта.

### 5.1 Автоматическая часть [AUTO]

Генерируется UAF из данных MLflow и session_analysis.json без LLM вызова.
Это четвёртый блок в ReportGenerator (первые три: Executive Summary, Monitoring
Conclusions, Recommendations — уже определены в стадиях 08 и 13).

**Содержание автоматической части:**

```latex
\section{Session Retrospective}

\subsection{Session Summary}
% Таблица: метрика цели, baseline, лучший результат, улучшение, итерации, бюджет, статус
\begin{tabular}{ll}
\textbf{Task:} & {task.name} \\
\textbf{Objective:} & \texttt{{metric.name}} ({metric.direction}) \\
\textbf{Baseline:} & {phase1_best_metric_value} \\
\textbf{Best result:} & {best_metric_value} (+{improvement_pct}\% over baseline) \\
\textbf{Iterations:} & {iterations_done} / {budget} ({budget_pct_used}\%) \\
\textbf{Runs:} & {successful_runs} succeeded, {failed_runs} failed, {partial_runs} partial \\
\textbf{Stop reason:} & {stop_reason} \\
\textbf{Duration:} & {session_duration_min} min \\
\end{tabular}

\subsection{UAF System Quality Metrics}
% Таблица M-UAF-01..14 с фактическими значениями и целевыми порогами
\begin{tabular}{llll}
\textbf{Metric} & \textbf{ID} & \textbf{Value} & \textbf{Target} \\
\hline
Plan quality (1-5) & M-UAF-01 & {val} & >= 3.5 \\
Budget efficiency & M-UAF-05 & {val} & >= 0.8 \\
... (все 14 метрик) ...
\end{tabular}

\subsection{Antigoal Compliance}
% Таблица antigoals 1-6: для каждого — статус (OK / VIOLATION) и evidence
\begin{tabular}{lll}
\textbf{Antigoal} & \textbf{Status} & \textbf{Evidence} \\
\hline
1. No production AutoML & \checkmark OK & {evidence} \\
2. No experiments without approval & \checkmark OK & HumanOversightGate log \\
3. No hidden failures & \checkmark OK & {failed_runs_count} runs in report \\
4. No data modification & \checkmark OK & DVC checksums unchanged \\
5. No test set leakage & \checkmark OK & val\_test\_delta = {val} \\
6. No budget overrun & \checkmark OK & budget\_pct\_used = {val}\% \\
\end{tabular}
% Если есть VIOLATION: \textcolor{red}{VIOLATION} с описанием

\subsection{Improvement Hypotheses (for next session)}
% Список H-01..H-09 из session_analysis.json, отсортированных по приоритету
\begin{enumerate}
  \item \textbf{H-XX} [priority={N}]: {description}
  ...
\end{enumerate}

\subsection{UAF Health Snapshot}
% M-UAF Health как светофор: метрики выше/ниже порога
% Опционально: тренд если есть данные health_history.jsonl
```

### 5.2 Ручная часть [MANUAL + PDF]

После завершения автоматической части, UAF предлагает ML-инженеру добавить ручные
наблюдения. Эти наблюдения включаются в PDF в секцию "Engineer Notes".

Механизм: UAF ждёт пока ML-инженер заполнит `session_postmortem.md` (секции 2, 7).
Секция 7 "Финальные выводы" из session_postmortem.md добавляется в PDF дословно
(без LLM обработки) в отдельный subsection "Engineer Notes".

Если session_postmortem.md не заполнен в течение 24 часов после завершения сессии,
UAF сохраняет report.pdf без Engineer Notes (с пометкой "[Engineer notes not provided]").

Пересборка report.pdf с Engineer Notes выполняется командой:
```
uaf report --session {session_id} --include-postmortem
```
Эта команда читает заполненный session_postmortem.md и пересобирает PDF.

### 5.3 Session Retrospective Synthesis (опциональный, через Claude Code)

Постмортем может включать дополнительный блок Synthesis — краткий (200-300 слов)
анализ основного результата, причин неудач и одного конкретного следующего шага.

**Активация:** M-UAF-09 (report_quality_score) < 4.0 ИЛИ явный флаг
`--deep-retrospective` при вызове `uaf report --session {id} --include-postmortem`.

**Как работает:** UAF записывает данные сессии и постмортем в
`SESSION_DIR/context/retrospective_context.md`, затем запускает Claude Code
с задачей сгенерировать блок Synthesis и сохранить в
`SESSION_DIR/report/sections/retrospective_synthesis.md`.

Инструкция для Claude Code при генерации Synthesis:
```
Данные сессии: {session_summary_json}
Постмортем ML-инженера: {session_postmortem.md секции 2 и 7}

Напиши краткий (200-300 слов) анализ:
1. Основной результат: что удалось выяснить о задаче
2. Главная причина неудач (если были)
3. Один конкретный следующий шаг (не список, один самый важный)

Стиль: технический, без лишних слов. Не повторяй данные из таблиц.
```

Результат — параграф "Synthesis" в секции Session Retrospective PDF.
Компилируется при следующем вызове `uaf report --include-postmortem`.

Технический долг TD-04 закрыт: вызов реализуется через Claude Code subprocess,
не через прямой Anthropic API.

---

## 6. Метрики здоровья UAF

UAF Health — набор метрик, накопленных по нескольким сессиям, позволяющих
оценить тренд качества самого фреймворка.

### 6.1 Хранилище: `.uaf/health_history.jsonl`

Каждая завершённая сессия добавляет одну JSON-строку в `health_history.jsonl`.
Файл хранится в корне проекта UAF (не в SESSION_DIR), версионируется git.

**Схема строки:**

```json
{
  "session_id": "uaf-20260319-142305",
  "task_name": "titanic_survival",
  "task_type": "tabular_classification",
  "date": "2026-03-19T14:23:05",
  "session_duration_min": 87,
  "stop_reason": "convergence",

  "uaf_metrics": {
    "M_UAF_01": 4.2,
    "M_UAF_02": 0.73,
    "M_UAF_03": 12.4,
    "M_UAF_04": 7.1,
    "M_UAF_05": 0.84,
    "M_UAF_06": 1.0,
    "M_UAF_07": 0.97,
    "M_UAF_08": 1.0,
    "M_UAF_09": 3.8,
    "M_UAF_10": 1.0,
    "M_UAF_11": 1.0,
    "M_UAF_12": 1.0,
    "M_UAF_13": 0.0,
    "M_UAF_14": 0.0
  },

  "task_metrics": {
    "metric_name": "roc_auc",
    "metric_direction": "maximize",
    "baseline_value": 0.712,
    "best_value": 0.849,
    "improvement_pct": 19.2
  },

  "session_events": {
    "total_runs": 18,
    "successful_runs": 15,
    "failed_runs": 2,
    "partial_runs": 1,
    "antigoal_violations": 0,
    "oom_events": 1,
    "nan_cascade_events": 0,
    "hang_detections": 0
  },

  "postmortem_completed": true,
  "postmortem_program_changes": 2,
  "postmortem_uaf_changes": 1,
  "postmortem_priority_p1": 0
}
```

### 6.2 Агрегированные метрики здоровья UAF (Health KPIs)

Вычисляются из health_history.jsonl по последним N сессиям (дефолт N=10).

| Метрика | Формула | Целевой порог | Тревожный порог |
|---------|---------|---------------|-----------------|
| UAF Session Success Rate | % сессий с stop_reason in {convergence, budget_exhausted} | >= 80% | < 60% |
| Mean Improvement Over Baseline | mean(improvement_pct) по сессиям | >= 10% | < 5% |
| Mean Budget Efficiency | mean(M_UAF_05) | >= 0.75 | < 0.60 |
| Antigoal Violation Rate | % сессий с M_UAF_14 > 0 | 0% | > 0% |
| Ruff Clean Rate Trend | slope(M_UAF_07) по времени | >= 0 (не деградирует) | < -0.02/сессия |
| MLflow Compliance Rate | mean(M_UAF_11) | = 1.0 | < 0.95 |
| Reproducibility Rate | mean(M_UAF_12) | = 1.0 | < 0.95 |
| Report Generation Rate | mean(M_UAF_08) | >= 0.95 | < 0.85 |
| Mean Crash Rate | % сессий с stop_reason = crashed | 0% | > 5% |
| Postmortem Completion Rate | % сессий с postmortem_completed = true | 100% | < 80% |
| P1 Changes Per Session | mean(postmortem_priority_p1) | 0 | > 0.2/сессия |

### 6.3 UAF Health Dashboard

Просматривается командой `uaf health`:

```
uaf health               -> показывает Health KPIs за последние 10 сессий
uaf health --n 20        -> за последние 20 сессий
uaf health --trend       -> тренд каждой метрики (консольный ASCII-график)
uaf health --full        -> все сессии в табличном виде
```

Вывод: терминальная таблица (Rich) с цветовой индикацией (зелёный/жёлтый/красный
по целевым и тревожным порогам).

Пример вывода:
```
UAF Health Dashboard (last 10 sessions, updated 2026-03-19)

Metric                      Value     Target    Status
Session Success Rate        85%       >=80%     OK
Mean Improvement/Baseline   14.3%     >=10%     OK
Budget Efficiency           0.79      >=0.75    OK
Antigoal Violation Rate     0%        0%        OK
Ruff Clean Rate (avg)       0.96      >=0.95    OK
MLflow Compliance           1.00      1.00      OK
Reproducibility             1.00      1.00      OK
Report Generation           0.90      >=0.95    WARN
Crash Rate                  0%        0%        OK
Postmortem Completion       80%       100%      WARN
P1 Changes/Session          0.1       0         OK

Sessions: 10  |  Last session: 2026-03-19  |  health_history.jsonl: 10 entries
```

### 6.4 Когда пересматривать пороги

Пороги Health KPIs — не абсолютные числа. Они должны пересматриваться:
- После первых 5 сессий: установить реалистичные baseline значения
- При изменении типа задач (новый тип появился в использовании)
- При крупном обновлении UAF (major version) — пороги могут стать устаревшими

Пересмотр порогов — решение ML-инженера, фиксируется в `CHANGES.md` с обоснованием.

---

## 7. Финальный чеклист закрытия сессии

Выполняется после каждой завершённой сессии UAF. Последовательность важна:
последующие шаги зависят от предыдущих.

### Группа 1: Верификация завершения

```
[ ] Сессия завершилась (budget_status.json: hard_stop = false ИЛИ stop_reason задан)
    python -c "import json; d=json.load(open('.uaf/sessions/{id}/budget_status.json')); \
      print(d.get('stop_reason', 'NO STOP REASON'))"

[ ] Claude Code процесс завершён (нет зависших процессов)
    ps aux | grep claude | grep -v grep
    Если есть: kill {pid}

[ ] Все MLflow runs закрыты (нет статуса RUNNING без активного процесса)
    mlflow runs search --experiment-name "uaf/{task_name}" \
      --filter "status = 'RUNNING'" --order-by "start_time DESC"

[ ] session.log содержит финальную строку "Session {session_id} complete"
    tail -5 .uaf/sessions/{session_id}/session.log
```

### Группа 2: Артефакты сессии

```
[ ] report.pdf существует и не пустой (> 50 КБ)
    ls -lh .uaf/sessions/{session_id}/report/report.pdf

[ ] session_analysis.json существует
    ls .uaf/sessions/{session_id}/session_analysis.json

[ ] smoke_test_report.json существует
    ls .uaf/sessions/{session_id}/smoke_test_report.json

[ ] Артефакты > 1 МБ добавлены в DVC
    dvc status
    Если показывает modified: dvc add {artifact_path}

[ ] feature_registry.json существует (если была Phase 2)
    ls .uaf/sessions/{session_id}/feature_registry.json
```

### Группа 3: Воспроизводимость

```
[ ] git commit содержит SESSION_DIR артефакты (program.md, конфиги, session_analysis.json)
    git log --oneline -3

[ ] DVC checksums актуальны
    dvc status
    Ожидаемый результат: "Data and pipelines are up to date."
    Если нет: dvc add {файлы} && git add *.dvc && git commit -m "dvc: update {session_id}"

[ ] MLflow Planning Run содержит git_sha тег
    mlflow runs get --run-id {planning_run_id} | grep git_sha

[ ] requirements.lock залогирован в MLflow Planning Run
    mlflow artifacts list --run-id {planning_run_id} | grep requirements.lock
```

### Группа 4: Постмортем

```
[ ] session_postmortem.md заполнен ML-инженером (секции 2, 5, 6, 7)
    ls .uaf/sessions/{session_id}/session_postmortem.md

[ ] Если есть P1 изменения: добавлены в CHANGES.md
    grep "P1" docs/CHANGES.md

[ ] session_postmortem.md закоммичен в git
    git log --oneline -1 | grep postmortem

[ ] health_history.jsonl обновлён
    tail -1 .uaf/health_history.jsonl | python -c "import json,sys; d=json.load(sys.stdin); \
      print(d['session_id'], d['stop_reason'])"
```

### Группа 5: Следующий шаг (по рекомендации постмортема)

```
[ ] Рекомендация из секции 7 постмортема прочитана и принято решение:
    cat .uaf/sessions/{session_id}/session_postmortem.md | grep -A 10 "## 7\."

[ ] ОДИН из вариантов выбран явно:
    - Resume: uaf resume --session {session_id}
    - Новая сессия: задача переформулирована в task.yaml
    - Закрыть задачу: задокументировать вывод в отдельном .md файле
    - Отложить: добавить в backlog с датой возврата

[ ] Если P1 изменения в CHANGES.md: до следующей сессии реализовать фикс
    (не запускать следующую сессию с известным P1 дефектом)
```

Все пункты выполнены -> сессия официально закрыта. Можно запускать следующую.

---

## 8. Накопленный опыт проектирования UAF (Design Postmortem)

Этот раздел — однократный постмортем проектной фазы UAF (стадии 01-16).
Не является шаблоном для повторения — документирует решения проектной фазы
для будущего сопровождения и итерации.

### 8.1 Что пошло не по первоначальному плану

**Архитектурный пересмотр v1.0 -> v2.0 (стадия 03)**

Первоначально UAF проектировался как самостоятельный агентный фреймворк
с LLMClient (3 провайдера), PlanningAgent (собственный plan-and-execute loop),
ExperimentRunner (управляет запуском кода). Это был классический подход к
"ML automation" системам.

Пересмотр произошёл при анализе karpathy/autoresearch: принципиально более
простая архитектура с лучшим результатом. Ключевой инсайт: Claude Code уже
является plan-and-execute агентом с tool use. UAF не должен дублировать эту
функциональность — он должен управлять сессией вокруг Claude Code.

Это сократило компонентный состав с 9 до 6 основных компонентов и полностью
устранило категорию проблем "агентный loop ненадёжен при длинных сессиях".

**Обоснование SKIPPED стадий**

Стадии 11-12-14 пропущены по antigoal-1 (нет production). Это не упрощение —
это следствие scope ограничения, заложенного в стадии 01. Стадия 07 встроена
в шаблон program.md — UAF не модель, baseline logic применяется к моделям
внутри сессии, а не к самой системе.

### 8.2 Ключевые архитектурные решения в ретроспективе

**TR-3 v2.0: Claude Code как единый агент — верное решение**

Причина: UAF-managed LLM loop требовал бы решения проблем prompting,
context management, tool reliability для каждого нового task type.
Claude Code решает эти проблемы сам. UAF получает надёжный агент
с известным API (program.md) вместо постоянно эволюционирующего внутреннего.

**TR-4: program.md как контракт — правильная абстракция**

program.md содержит и план (для человека) и инструкции (для Claude Code).
Один файл служит двум целям без конфликта. Это важно: если бы инструкции
для Claude Code хранились отдельно, HumanOversightGate не видел бы их при
review. ML-инженер одобряет не только план исследования, но и то, как
именно Claude Code будет его исполнять.

**BudgetController vs Claude Code: граница ответственности**

BudgetController мониторит снаружи (читает MLflow, пишет budget_status.json).
Claude Code читает budget_status.json и сам решает когда остановиться.
Это не идеально — есть delay между hard_stop=true и фактической остановкой
Claude Code. Но альтернатива (SIGTERM немедленно) ещё хуже: Claude Code
не успевает сохранить checkpoint и закрыть MLflow run.

Grace period 5 минут — компромисс. Может потребовать корректировки
после первых реальных сессий (CHANGES.md тип DESIGN, P2).

**Shadow Feature Trick vs Feature Store**

Feature Store избыточен для одиночного исследователя без serving.
Shadow Trick (параллельное обучение baseline + candidate в одной итерации)
даёт то же что Feature Store в части "оценка полезности признака" с нулевым
инфраструктурным оверхедом. Верное решение для данного scope.

**Validation vs Antigoal 2**

ValidationChecker не делает автоматический repair схемы — это следствие
antigoal 2 (не запускает без одобрения). Repair схемы валидации был бы
неявным изменением задачи (shift от пользовательской конфигурации).
Это правило важно сохранить при итерации ValidationChecker.

### 8.3 Что осталось неопределённым (технический долг проектной фазы)

**TD-01: Нет реальных данных о качестве program.md (M-UAF-01)**
Метрика quality_plan_generation определена (1-5, оценивается ML-инженером),
но нет baseline данных о том, как хорошо ProgramMdGenerator работает на
реальных задачах разных типов. Заполнится после 5-10 реальных сессий.

**TD-02: Промпт ProgramMdGenerator не протестирован**
Prompt engineering для ProgramMdGenerator выполнен теоретически.
Реальное качество генерации program.md неизвестно до первого запуска.
Первые 3 сессии будут преимущественно тестированием промпта.

**TD-03: Оптимальные дефолты BudgetController неизвестны**
patience=3, min_delta=0.001, min_iterations=3, safety_cap=50 — разумные
значения на бумаге. Реальные оптимальные значения зависят от типа задачи
и размера датасета. Потребуется эмпирическая калибровка.

**TD-04: Session Retrospective Synthesis — реализуется через Claude Code subprocess**
Определён в секции 5.3 этого документа. При реализации: UAF записывает
retrospective_context.md и запускает отдельный Claude Code subprocess.
Никаких прямых Anthropic API вызовов из UAF. Архитектурно закрыт.

**TD-05: uaf health команда не определена в стадии 09 (pipeline)**
Команда `uaf health` определена здесь, в стадии 16. При реализации CLI
нужно добавить эту команду в ResearchSessionController / отдельный CLI
подмодуль. Ссылка: stage-16, секция 6.3.

---

## 9. Решения и обоснования

**R-16-01: Session Postmortem — одновременно шаблон и процесс**
Шаблон без процесса не заполняется. Процесс без шаблона непоследователен.
Оба закреплены в одном документе. Критерий качественного постмортема (секция 2.3)
необходим чтобы ML-инженер не заполнял формально.

**R-16-02: AUTO vs MANUAL разделение в шаблоне**
Постмортем бесполезен если требует ручного копирования данных из MLflow.
Автоматическая часть [AUTO] должна заполняться без участия ML-инженера.
Ручная часть [MANUAL] требует суждения, которое система не может заменить.
Разделение явное (метки в шаблоне) чтобы ML-инженер понимал что от него ожидается.

**R-16-03: health_history.jsonl как накопленный опыт**
Один session_analysis.json — снапшот. Тренд по нескольким сессиям — информация
о том, улучшается ли фреймворк или деградирует. Это позволяет выявлять DESIGN
проблемы, которые не видны в одной сессии. JSONL формат: append-only, никогда
не перезаписывается, тривиально парсится.

**R-16-04: Ручная часть postmortem -> PDF через "Engineer Notes"**
ML-инженер не должен дважды писать одно и то же. Секция 7 постмортема
включается в PDF дословно — это позволяет иметь один источник правды
(session_postmortem.md) вместо двух (постмортем + секция в отчёте).

**R-16-05: P1 изменения блокируют следующую сессию**
Если ML-инженер знает о P1 дефекте и запускает следующую сессию — он намеренно
принимает риск нарушения antigoal. Правило "P1 блокирует" формализует то,
что должно быть очевидным поведением. Antigoal нарушения — единственный
автоматически генерируемый P1.

**R-16-06: Технический долг проектной фазы задокументирован явно**
TD-01..TD-05 — это не недостатки дизайна. Это нормальная часть проектирования:
некоторые решения могут быть приняты только на основе реальных данных использования.
Явная фиксация TD позволяет не забыть их при итерации UAF.

---

## STAGE COMPLETE

**Стадия 16 (postmortem) завершена.**

Созданные артефакты:
- `docs/stage-16-postmortem.md` — настоящий документ

Ключевые результаты:
- Шаблон session_postmortem.md: [AUTO] и [MANUAL] секции, критерий полноты
- Механизм итерации program.md: improvement_context.md (тип A) + CHANGES.md (тип B)
- Механизм итерации UAF: 5 типов изменений, 3 уровня приоритета, 3-шаговый цикл
- Session Retrospective в PDF: автоматическая часть + Engineer Notes + опциональный Synthesis (через Claude Code)
- Метрики здоровья UAF: 11 Health KPIs, health_history.jsonl, uaf health команда
- Финальный чеклист: 5 групп, все с исполняемыми командами
- Design Postmortem проектной фазы: 5 ключевых решений, 5 пунктов технического долга

**Проект UAF — все 16 стадий завершены.**

Следующий шаг: реализация компонентов согласно docs/stage-03-design-doc.md (v2.0).
Рекомендуемая последовательность: MLflowSetup -> BudgetController -> ProgramMdGenerator ->
HumanOversightGate -> SmokeTestRunner -> ClaudeCodeRunner -> RuffEnforcer -> ReportGenerator.
