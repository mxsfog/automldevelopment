# Стадия 02: Обзор существующих решений (Research)

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Статус:** STAGE COMPLETE
**Предшествующая стадия:** 01-problem (COMPLETE)

---

## 1. Цель стадии

Ответить на три вопроса:

1. Что уже существует в пространстве задачи UAF — и почему этого недостаточно?
2. Что берём готовое (Buy/Use), что пишем сами (Build)?
3. Насколько предлагаемый подход нов (Degree of Innovation)?

Вывод этой стадии питает стадию 03-design-doc: архитектурные решения должны
опираться на доказанные пробелы в существующих решениях, а не на интуицию.

---

## 2. Пространство задачи: четыре категории решений

UAF находится на пересечении четырёх классов инструментов:

```
AutoML-фреймворки      LLM-coding-агенты
   (AutoGluon, H2O,       (SWE-agent, OpenHands,
    FLAML, Optuna)          Devin)
              \              /
               [UAF: нужное]
              /              \
Research Automation      Agent Orchestration
  (AI Scientist,           (LangChain/LangGraph,
   karpathy/autoresearch)    CrewAI, AutoGen)
```

Ни одна из категорий в одиночку не закрывает задачу UAF.
Разбираем каждую подробно.

---

## 3. Категория 1: AutoML-фреймворки

### 3.1 AutoGluon (AWS, open-source)

**Что умеет:**
- Конец-в-конец AutoML для табличных данных, текста, изображений, временных рядов
- Multi-layer model stacking и ensembling из коробки
- Пресеты: `best_quality`, `high_quality`, `medium_quality` — баланс точности/скорости
- Автоматический выбор признаков, предобработка, feature engineering
- Поддержка PyTorch и XGBoost/CatBoost внутри

**Что не умеет (применительно к UAF):**
- Не генерирует и не трекит гипотезы — это чёрный ящик, который выдаёт модель
- Нет концепции "план исследования" (program.md) — нет человеческого oversight
- Нет связи с MLflow: AutoGluon ведёт свой внутренний трекинг, несовместимый
- Нет iterative reasoning: AutoGluon запускается один раз, не адаптирует план
  по результатам промежуточных экспериментов
- Нет объяснения "почему эта архитектура": отчёт есть, вывода "что попробовать
  дальше" нет
- Scope: оптимизация конкретной задачи ML, не исследовательский цикл

**Ограничения:**
- Нет встроенного распределённого вычисления (ограничение для нашей локальной машины
  несущественно, но показывает, что фреймворк заточен под managed-среды)
- Не поддерживает произвольные архитектурные эксперименты — только те, что
  встроены во фреймворк

**Вердикт для UAF:** AutoGluon полезен как инструмент внутри экспериментов
(UAF может вызывать AutoGluon как один из подходов), но не как замена UAF.

### 3.2 FLAML (Microsoft Research, open-source)

**Что умеет:**
- Budget-aware оптимизация: находит хорошую модель при заданном временном бюджете
- Поддержка классификации, регрессии, time series, NLP
- Lightweight: меньше зависимостей чем AutoGluon
- Интеграция с Ray для параллельных запусков
- Экономичен по ресурсам

**Что не умеет (применительно к UAF):**
- Аналогично AutoGluon: нет concept of "исследовательская гипотеза"
- Нет LLM-планировщика, нет iterative reasoning
- Нет MLflow-интеграции из коробки (можно добавить вручную, но не встроено)
- Не объясняет почему — только "лучшая конфигурация найдена за N секунд"

**Вердикт для UAF:** FLAML — кандидат как "executor" для гиперпараметрной
оптимизации внутри UAF-эксперимента. Не замена.

### 3.3 H2O AutoML (H2O.ai, open-source)

**Что умеет:**
- Масштабирование на большие датасеты (распределённый режим через кластер)
- Встроенная интерпретируемость: SHAP, feature importance, partial dependence
- Веб-UI для аналитиков без кода
- Accessible через Python, R, Java
- GBM, Random Forest, DeepLearning, GLM, Stacked Ensembles

**Что не умеет (применительно к UAF):**
- Ресурсоёмкий: падает в constrained environment (несоответствие нашей
  локальной машине)
- Нет LLM-интеграции, нет agent loop
- Не поддерживает произвольный code generation
- UI-ориентирован, не CLI/API-first для автоматизации

**Вердикт для UAF:** Не подходит как компонент. Слишком тяжёлый для
локальной машины, архитектурно несовместим с agent-driven подходом.

### 3.4 Optuna (Preferred Networks, open-source)

**Что умеет:**
- Байесовская оптимизация гиперпараметров (TPE, CMA-ES, GP с v4.4)
- Define-by-run API: динамическое построение search space
- Pruner для раннего отсева плохих trials (MedianPruner, HyperbandPruner)
- Горизонтальное масштабирование: RDB storage для distributed trials
- Dashboard для визуализации optimization history
- Нативная MLflow-интеграция через `optuna-integration`
- В Optuna v5 (roadmap): LLM-powered dashboard queries

**Что не умеет (применительно к UAF):**
- Optuna — инструмент HPO, не agent: не генерирует гипотезы, не пишет код
- Нет концепции "план исследования" с человеческим oversight
- Оптимизирует заданную objective function, не формулирует что проверять дальше
- Не управляет циклом "гипотеза -> код -> запуск -> анализ -> новая гипотеза"

**Вердикт для UAF:** Optuna — обязательный компонент UAF для HPO внутри
экспериментов. Встраивается нативно, MLflow-интеграция есть. Не пишем сами.

### 3.5 Итоговая карта AutoML-фреймворков

| Фреймворк | Hypothesis? | Code Gen? | MLflow? | Agent Loop? | Вердикт |
|-----------|-------------|-----------|---------|-------------|---------|
| AutoGluon | нет | нет | нет | нет | Use inside experiments |
| FLAML | нет | нет | нет | нет | Use inside experiments |
| H2O | нет | нет | нет | нет | Skip (тяжёлый) |
| Optuna | нет | нет | да | нет | Core component (HPO) |

---

## 4. Категория 2: LLM-агенты для кода

### 4.1 Devin (Cognition AI, коммерческий)

**Что умеет:**
- Full-stack software engineering agent: от issue до PR
- Браузер, терминал, редактор кода — полная среда
- Long-horizon planning для сложных задач
- Коммерческий продукт с managed cloud-окружением

**Что не умеет (применительно к UAF):**
- Коммерческий, нет self-hosted варианта — нарушает наше ограничение
  "локальная машина, без облака"
- Не специализирован на ML-исследованиях: знает инженерию, не знает
  "как правильно поставить ML-эксперимент"
- Нет концепции program.md, нет structured research loop
- Нет встроенной MLflow/DVC интеграции
- SWE-bench: 13.86% unassisted — не лучший кодер для ML-специфики

**Вердикт для UAF:** Не подходит. Коммерческий, не ML-специфичный,
не self-hosted.

### 4.2 SWE-agent (Princeton/Stanford, open-source)

**Что умеет:**
- Agent-Computer Interface (ACI): специальный интерфейс для LLM-агентов
- Специализирован на GitHub issues -> code fix
- NeurIPS 2024: академически обоснованная архитектура
- Open-source, можно встраивать
- Хорошо работает с Claude 3.5: ~27-47% на SWE-bench Lite

**Что не умеет (применительно к UAF):**
- Заточен на software engineering, не ML research
- Нет понятия "итерация эксперимента", "метрика качества модели"
- Нет MLflow/DVC интеграции
- Нет structured research planning
- Не знает о научном методе в ML: нет concept of "baseline", "ablation",
  "comparison across runs"

**Вердикт для UAF:** ACI-концепт интересен как паттерн для нашего
Tool Interface. Сам SWE-agent встраивать не нужно.

### 4.3 OpenHands (open-source, бывший OpenDevin)

**Что умеет:**
- Event-stream архитектура: моделирует агент-среда взаимодействие
- Multi-agent delegation: иерархические структуры агентов
- CodeAct: выполнение произвольного кода в sandbox
- 53% resolve rate с Claude Sonnet (SWE-bench Lite)
- 69k+ GitHub stars — самое большое community среди open-source
- Docker sandboxing встроено
- Активное развитие: multi-agent стало table stakes в Feb 2026

**Что не умеет (применительно к UAF):**
- Общий инструмент разработки, не ML-специфичный
- Нет нативной MLflow интеграции
- Нет structured research loop (program.md концепт)
- Нет explicit experiment tracking, reproducibility guarantees
- Overhead для нашего сценария: нам не нужна полная среда разработчика,
  нам нужен специализированный ML-experiment executor

**Вердикт для UAF:** CodeAct-паттерн (выполнение кода как tool call)
берём как архитектурный паттерн. Docker sandboxing — рассматриваем как
опциональный компонент для изоляции. Сам OpenHands встраивать избыточно.

### 4.4 Итоговая карта LLM-агентов

| Агент | Open? | Self-host? | ML Research? | MLflow? | Вердикт |
|-------|-------|------------|--------------|---------|---------|
| Devin | нет | нет | нет | нет | Skip |
| SWE-agent | да | да | нет | нет | Pattern only (ACI) |
| OpenHands | да | да | нет | нет | Pattern only (CodeAct) |

---

## 5. Категория 3: Research Automation

### 5.1 AI Scientist v1/v2 (Sakana AI, open-source)

**Архитектура:**
- v1: LLM генерирует идею -> пишет код -> запускает -> пишет LaTeX-бумагу
- v2: Agentic Tree Search + VLM feedback + parallel experiment execution
- Поиск по Semantic Scholar для novelty check
- Полный pipeline: идея -> код -> эксперименты -> plots -> LaTeX paper

**Достижения:**
- v2: одна из 3 поданных работ прошла peer review на ICLR workshop
- Стоимость одной бумаги: $6-15, 3.5 часа "человеческого участия"
- Первая система, сгенерировавшая peer-reviewed ML paper

**Критические провалы (из независимой оценки, arxiv 2502.14297):**
- 42% экспериментов падали из-за ошибок в сгенерированном коде
- Literature review: плохая оценка новизны, часто мислейблирует известные
  концепты как новые
- Fabricated experimental results и hallucinated methodology — задокументированы
- v2 vs v1: не всегда лучше; v1 по шаблону надёжнее, v2 exploratory но
  с низким success rate
- Не поддерживает structured human oversight в loop: полная автономия
  (противоречит нашему antigoal 2)
- Нет MLflow/DVC: использует собственный ad-hoc трекинг
- Нет concept of "budget control"
- Scope: academic paper generation, не ML engineering research

**Ключевое расхождение с UAF:**
AI Scientist решает задачу "написать академическую статью". UAF решает задачу
"ускорить практическое ML-исследование инженера". Это разные задачи. AI Scientist
не контролируется человеком в loop, скрывает неудачи в финальном paper,
и заточен на academic novelty, а не на engineering quality.

**Вердикт для UAF:** Концепт experiment -> paper pipeline — интересен для
финального отчёта UAF. Архитектурные паттерны (agentic tree search)
рассматриваем. Сам AI Scientist не встраиваем: другой scope, нет human
oversight, нет MLflow.

### 5.2 karpathy/autoresearch (open-source, март 2026)

**Архитектура (630 строк Python):**
- Три файла: `prepare.py` (данные, утилиты — нельзя трогать),
  `train.py` (модель, оптимизатор, цикл — агент меняет только этот файл),
  `program.md` (инструкции для агента на natural language — человек пишет)
- Фиксированный time budget: каждый эксперимент ровно 5 минут
- Метрика: val_bpb (validation bits per byte)
- Цикл: агент предлагает изменение train.py -> запускает -> если val_bpb
  улучшилось, git commit (advance); если нет, git reset
- ~12 экспериментов/час, ~100 за ночь
- Полностью автономный: нет human-in-the-loop

**Достижения:**
- 50 экспериментов за ночь, обнаружен лучший learning rate — автономно
- Shopify CEO: 37 экспериментов, +19% к performance overnight
- Вирусный: Fortune, TechCrunch, MarkTechPost о нём писали

**Критические ограничения (применительно к UAF):**
- Жёстко заточен на LLM training (nanochat): не универсален
- Нет MLflow: только git commits как трекинг (потеря сравнимости метрик)
- Нет DVC: нет версионирования данных/артефактов
- Нет human oversight в loop: полная автономия (antigoal 2 нарушен для нас)
- Нет structured reporting: просто git log как история
- Нет бюджетного контроля по деньгам (только time budget)
- Одна метрика (val_bpb) — не обобщается на произвольную ML-задачу
- Нет self-correction при упавшем коде: если train.py крашится, агент
  продолжает без recovery
- Нет concept of research phases (baseline -> ablation -> final)
- Минималистичен намеренно: Karpathy сделал proof-of-concept, не production

**Ключевое расхождение с UAF:**
autoresearch — это "дай LLM-у менять train.py без ограничений". UAF — это
"структурированное исследование с планом, oversight, MLflow, DVC, reporting".
autoresearch ценен как validation идеи (LLM-агент умеет делать ML-эксперименты)
и как источник вдохновения для program.md концепции.

**Что берём из autoresearch:**
- Концепт program.md как natural language инструкции агенту — прямо наш подход
- Time/iteration budgeting идея
- Git как часть experiment tracking (в нашем случае дополняем MLflow)

**Вердикт для UAF:** autoresearch — ближайший прообраз UAF. Берём концепт
program.md, budget-bounded loop. Не форкаем: архитектура несовместима
(нет MLflow, нет DVC, нет human oversight, нет универсальности).

### 5.3 Итоговая карта Research Automation

| Система | Human in Loop? | MLflow? | DVC? | Universal? | Вердикт |
|---------|---------------|---------|------|------------|---------|
| AI Scientist v2 | нет | нет | нет | нет (academic) | Pattern only |
| karpathy/autoresearch | нет | нет | нет | нет (LLM training) | Concept inspiration |

---

## 6. Категория 4: Agent Orchestration фреймворки

Это инструменты для построения агентов, а не готовые агенты.
Рассматриваем как кандидаты для реализации UAF.

### 6.1 LangChain / LangGraph

**Плюсы:**
- LangGraph: stateful graphs, поддержка re-planning в runtime
- Большая экосистема tool integrations
- Human-in-the-loop через interrupt nodes встроен

**Минусы:**
- Высокий overhead и абстракции, скрывающие что происходит
- Быстро меняющийся API (breaking changes на каждый minor release)
- Для нашего сценария (одиночный пользователь, локальная машина) — overengineered
- Debugging сложен: многоуровневые abstraction layers

### 6.2 CrewAI

**Плюсы:**
- Declarative tool scoping для security
- Multi-agent: удобно для researcher + executor ролей

**Минусы:**
- Ещё более высокий уровень абстракции чем LangChain
- Меньший контроль над prompt engineering
- Production-ориентирован, а не research-ориентирован

### 6.3 AutoGen (Microsoft, open-source)

**Плюсы:**
- Built-in Docker sandboxing для code execution
- Хорошо документирован для multi-agent scenarios
- Поддерживает ConversableAgent паттерн

**Минусы:**
- Оптимизирован для conversational multi-agent, не для batch ML research
- Конфигурация через JSON/YAML — менее гибко для dynamic search spaces
- Microsoft-специфичные паттерны

### 6.4 Vanilla ReAct / Plan-and-Execute (без фреймворка)

**Плюсы:**
- Полный контроль над prompt engineering
- Нет overhead от фреймворков
- Легко дебажить: каждый шаг прозрачен
- Меньше зависимостей, меньше breaking changes

**Минусы:**
- Больше кода писать вручную
- Нет готовых tool integrations

**Вердикт для UAF:** Начинаем с минималистичного подхода — собственная
реализация Plan-and-Execute без тяжёлых фреймворков. LangGraph рассматриваем
как опциональную зависимость в стадии 09-pipeline если понадобится.

---

## 7. Build vs Buy анализ

### 7.1 Матрица решений

| Компонент | Решение | Обоснование |
|-----------|---------|-------------|
| LLM API | **Buy** (OpenAI/Anthropic) | Нет смысла обучать, API достаточно |
| Локальная LLM | **Buy** (Ollama + llama/mistral) | Готовая инфраструктура для local inference |
| Experiment Tracking | **Buy** (MLflow) | Требование из стадии 01; зрелый инструмент |
| Data Versioning | **Buy** (DVC) | Требование из стадии 01; зрелый инструмент |
| HPO | **Buy** (Optuna) | Нативная MLflow интеграция, TPE/GP самплеры |
| Code Execution Sandbox | **Buy** (subprocess + venv) | Для локальной машины достаточно; Docker — опционально |
| AutoML Executor | **Buy** (AutoGluon/FLAML как инструменты внутри) | Используем как callable tools, не как orchestrator |
| LLM Agent Orchestration | **Build** (custom Plan-and-Execute) | Нет подходящего фреймворка без overhead |
| program.md Schema | **Build** | Специфично для UAF, не существует |
| Research Loop Controller | **Build** | Ядро UAF, специфично для нашей задачи |
| Budget Controller | **Build** | Специфично (два режима: fixed/dynamic) |
| Result Analyzer | **Build** (LLM-powered) | Специфично: читает MLflow, формулирует выводы |
| Report Generator | **Build** (LLM + шаблон) | Специфично для UAF отчёта |
| Git Integration | **Buy** (GitPython или subprocess) | Трекинг кода экспериментов |

### 7.2 Ключевой принцип Build vs Buy

Берём готовое везде где оно: (a) решает задачу без изменений, (b) имеет
стабильный API, (c) не создаёт vendor lock-in, (d) соответствует нашим
ограничениям (локально, без облака).

Строим сами то, что является уникальной логикой UAF: planning loop,
human oversight protocol, budget control, MLflow-aware analysis.

---

## 8. Gap Analysis: чего нет ни в одном существующем решении

Это пространство, которое занимает UAF.

**Gap 1: Structured ML Research Loop с Human Oversight**
Ни один из рассмотренных инструментов не реализует цикл:
`план -> одобрение -> эксперименты -> анализ -> следующий план`
с явным checkpoint для человека перед каждым дорогостоящим шагом.
autoresearch идеологически близок, но полностью автономен.

**Gap 2: MLflow-native Research Agent**
Все агенты (AI Scientist, autoresearch, SWE-agent, OpenHands) используют
ad-hoc трекинг (git commits, JSON файлы, LaTeX tables). Никто не интегрирован
с MLflow как primary store для всех метрик и артефактов.

**Gap 3: Universal ML Task (не только LLM training)**
autoresearch заточен на LLM pretraining (nanochat).
AI Scientist заточен на deep learning papers.
AutoML-фреймворки заточены на supervised learning с фиксированным API.
UAF должен работать для произвольной ML-задачи пользователя.

**Gap 4: Budget-aware Research Planning с двумя режимами**
Никто из рассмотренных не поддерживает явно два режима:
фиксированный бюджет (hard stop) и динамический бюджет (convergence-based).
FLAML имеет time budget, но это не то же самое что API-budget или
convergence detection на уровне research loop.

**Gap 5: Transparent Failure Reporting**
AI Scientist скрывает неудачи в финальном paper.
autoresearch делает git reset без отчёта об отклонённых экспериментах.
UAF должен логировать все попытки в MLflow, включая упавшие (antigoal 3).

**Gap 6: Reproducible Code Generation**
Существующие агенты генерируют код, но не гарантируют воспроизводимость:
нет фиксации seed, версии данных, зависимостей. DVC + MLflow + seed fixation
как обязательные компоненты — уникальное требование UAF.

---

## 9. Degree of Innovation

### 9.1 Что полностью ново

- Structured research loop с explicit human approval checkpoint (program.md
  approval gate перед запуском экспериментов)
- MLflow как primary experiment store для LLM-driven research agent
- Convergence-based dynamic budget с формальным критерием остановки
- Full transparency of failures: все попытки в MLflow, включая upавшие

### 9.2 Что является комбинацией существующих идей

- LLM как планировщик исследований (из AI Scientist)
- program.md как natural language инструкция агенту (из autoresearch Karpathy)
- Plan-and-Execute агентная архитектура (устоявшийся паттерн)
- Code generation + execution loop (из SWE-agent, OpenHands)
- Budget-bounded search (из FLAML, autoresearch)

### 9.3 Что мы намеренно не делаем новым

- LLM как backbone: используем API, не обучаем
- MLflow, DVC, Optuna: используем как есть
- Паттерны code execution: subprocess, venv — стандартные инструменты

### 9.4 Оценка новизны

Степень инновации: **умеренная, инженерная**.

UAF не публикует новый алгоритм и не проводит ML research по формуле
AI Scientist. UAF — это правильная инженерная комбинация существующих
компонентов для конкретной практической задачи, которую никто другой
не решает именно так. Это нормально и достаточно.

Аналогия: MLflow не изобрёл эксперимент-трекинг, он правильно упаковал
его в инженерный инструмент. UAF не изобретает research automation,
но правильно упаковывает его для ML-инженера на локальной машине.

---

## 10. Ключевые технические решения для следующих стадий

Решения, которые необходимо принять в стадии 03-design-doc, обоснованные
результатами этого research:

**TR-1: LLM Provider Strategy**
Нужно решить: OpenAI API / Anthropic API / локальная модель через Ollama /
абстракция над несколькими провайдерами.
Данные: OpenHands показывает 53% с Claude Sonnet. Karpathy использует Claude.
Риск: vendor lock-in. Рекомендация: абстракция с provider-agnostic interface.

**TR-2: Code Execution Isolation**
Нужно решить: subprocess в текущем окружении / venv / Docker.
Данные: OpenHands и AutoGen используют Docker. Karpathy использует прямой
subprocess. Для локальной машины Docker добавляет overhead.
Рекомендация: venv isolation как default, Docker как опция.

**TR-3: Agent Architecture**
Нужно решить: ReAct (tight loop) / Plan-and-Execute (explicit plan first).
Данные: для нашей задачи с human approval checkpoint Plan-and-Execute
архитектурно лучше: план — явный артефакт, который человек одобряет.
Рекомендация: Plan-and-Execute с explicit approval gate.

**TR-4: Scope of program.md**
Нужно решить: program.md как статичный файл (Karpathy) или как living document
(агент обновляет по результатам).
Данные: AI Scientist v2 показывает, что pure exploration без структуры
даёт низкий success rate. Структура лучше.
Рекомендация: living document с версионированием через DVC.

**TR-5: Convergence Criterion для динамического бюджета**
Нужно решить: как определить что дальнейшие итерации бесполезны.
Варианты: (a) нет улучшения за N итераций, (b) delta < epsilon,
(c) LLM сам решает (ненадёжно).
Рекомендация: (a) + (b) с конфигурируемыми порогами.

**TR-6: Failure Recovery**
Нужно решить: что делать когда сгенерированный код падает.
Данные: AI Scientist: 42% упавших экспериментов. autoresearch: нет recovery.
Рекомендация: retry с self-correction (LLM читает traceback, предлагает fix),
max 3 попытки, после — skip и log в MLflow.

---

## 11. Итоговые выводы стадии 02

1. **Ни одно существующее решение не закрывает задачу UAF полностью.**
   Ближайший аналог — karpathy/autoresearch — покрывает ~30% нужной
   функциональности (agent loop + program.md концепт), но критически
   расходится в human oversight, MLflow, универсальности.

2. **Buy список зафиксирован:** MLflow, DVC, Optuna, LLM API (абстракция
   над провайдерами), AutoGluon/FLAML как callable tools.

3. **Build список зафиксирован:** Research Loop Controller, Plan-and-Execute
   агент, program.md schema + lifecycle, Budget Controller (fixed/dynamic),
   MLflow-aware Result Analyzer, Report Generator.

4. **Degree of Innovation: умеренная.** UAF не претендует на академическую
   новизну. Задача — правильная инженерная композиция.

5. **Шесть gaps определены** — они же являются обоснованием существования
   UAF. Каждый gap должен быть закрыт конкретным компонентом в design doc.

6. **Шесть технических решений** требуют принятия в стадии 03. Рекомендации
   даны, но финальный выбор — в design doc после полного scope.

---

## STAGE COMPLETE

Стадия 02-research завершена. Артефакт создан.
Переход к стадии 03-design-doc разрешён.
