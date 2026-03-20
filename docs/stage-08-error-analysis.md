# Стадия 08: Error Analysis

**Проект:** Universal AutoResearch Framework (UAF)
**Дата:** 2026-03-19
**Версия:** 1.0
**Статус:** STAGE COMPLETE
**Предшествующие стадии:** 01-06 (COMPLETE), 07 (SKIPPED — см. раздел 0)

---

## 0. Статус стадии 07-baseline и обоснование skip

Стадия 07-baseline в классической методологии Babushkin & Kravchenko предполагает
проведение фактических экспериментов по схеме Constant -> Rule-based -> Linear -> Complex
для измерения baseline перед выбором архитектуры.

UAF — не ML-модель. У UAF нет модели, нет обучения, нет hyperparameter search.
UAF — инструментальная система. Baseline logic в контексте UAF означает иное:
как UAF организует baseline эксперименты внутри сессии Claude Code.

**Принятое решение:** baseline logic зафиксирована как обязательная структура
шаблона program.md. ProgramMdGenerator всегда включает Phase 1: Baseline как
первую фазу любого плана. Порядок методов в Phase 1:
1. Constant/dummy baseline (sklearn DummyClassifier или аналог)
2. Rule-based (пороговые правила, статистические эвристики)
3. Linear model (LogisticRegression, LinearRegression, Ridge)
4. Simple non-linear (CatBoost/XGBoost с дефолтными параметрами)

Это поведение закреплено в шаблоне `execution_instructions.md.jinja2`
как обязательная секция. Claude Code не может пропустить Phase 1.

Отдельный артефакт `docs/stage-07-baseline.md` не создаётся.
Стадия 07 помечается в progress.md как SKIPPED/EMBEDDED.

---

## 1. Контекст: что такое error analysis для UAF

UAF не является ML-моделью — у неё нет confusion matrix, нет FP/FN на примерах.
Error analysis здесь означает два разных слоя:

**Слой A — ResultAnalyzer:** как UAF анализирует результаты ML-экспериментов
из завершённой сессии и формирует feedback для следующей итерации в program.md.
Это инструментальный слой: UAF читает MLflow, извлекает паттерны, формирует
гипотезы об улучшениях.

**Слой B — SystemErrorAnalyzer:** как UAF анализирует собственные системные ошибки
— crash rate, ruff violations, budget overrun, MLflow compliance failures.
Это слой самодиагностики.

Оба слоя производят выходные данные, которые попадают в отчёт.
Слой A влияет на следующую итерацию program.md (если сессия продолжается).
Слой Б не влияет на итерации — только на отчёт и на улучшение UAF.

---

## 2. ResultAnalyzer: алгоритм анализа после эксперимента

ResultAnalyzer — логический компонент внутри ReportGenerator и BudgetController.
Не отдельный Python-класс — набор функций, вызываемых в нужные моменты lifecycle.

### 2.1 Момент вызова

ResultAnalyzer вызывается в двух контекстах:

**Контекст 1 — BudgetController polling (каждые 30 секунд):**
Читает последний завершённый experiment run из MLflow. Обновляет:
- `metrics_history` для проверки сходимости
- `convergence_signals` для LLM-сигнала
- `budget_status.json` для Claude Code

Это минимальный real-time анализ. Цель — не глубокий анализ, а детекция
hard stop conditions. Полный анализ — только по завершении сессии.

**Контекст 2 — Post-session (после завершения Claude Code):**
Полный анализ всех runs. Результат идёт в ReportGenerator и в артефакт
`session_analysis.json` в директории сессии.

### 2.2 Алгоритм post-session анализа

```
Входные данные:
  all_runs: list[MLflowRun]       # все experiment runs сессии
  program_md_final: str           # финальный program.md с выводами Claude Code
  task_config: TaskConfig         # task.yaml (тип задачи, метрика, направление)
  ruff_report: RuffReport         # от RuffEnforcer

Шаг 1: Разделение runs по статусу
  successful_runs = [r for r in all_runs if r.tags["status"] == "success"]
  failed_runs     = [r for r in all_runs if r.tags["status"] == "failed"]
  planning_runs   = [r for r in all_runs if r.tags["type"] == "planning"]

Шаг 2: Ранжирование successful runs
  Сортировка по task_config.metric (direction: maximize/minimize)
  best_run    = successful_runs[0]  # лучший
  worst_run   = successful_runs[-1] # худший из успешных
  baseline_run = первый run с тегом step_id содержащим "baseline" или "phase1"

Шаг 3: Метрический профиль сессии
  baseline_value  = baseline_run.metrics[task_config.metric.name]
  best_value      = best_run.metrics[task_config.metric.name]
  metric_delta    = best_value - baseline_value (с учётом direction)
  metric_delta_pct = metric_delta / (|baseline_value| + 1e-10)

  improvement_trajectory = [r.metrics[metric] for r in successful_runs в хронологии]
  trajectory_slope = линейная регрессия trajectory (slope > 0 = улучшение)

Шаг 4: Анализ failed experiments
  Для каждого failed_run:
    traceback_text = mlflow_client.download_artifact(run_id, "traceback.txt")
    failure_category = classify_failure(traceback_text)
    # failure_category: одна из [import_error, runtime_error, oom_error,
    #                             data_error, timeout_error, assertion_error, other]
    error_summary = extract_error_summary(traceback_text)
    # extract_error_summary: первая строка Exception + последние 3 строки traceback

  failed_categories_counts = Counter(failure_categories)
  systemic_failure = failure_category появляется в >= 50% failed runs

Шаг 5: Param-metric корреляции (если >= 5 successful runs)
  Для каждого числового параметра p:
    values  = [r.params[p] for r in successful_runs]
    metrics = [r.metrics[task_config.metric.name] for r in successful_runs]
    corr_p  = spearmanr(values, metrics).correlation
  top_params_by_correlation = top 3 параметра по |corr_p|
  (это будет Figure 2 в ReportGenerator: Hyperparameter Importance)

Шаг 6: Сегментация ошибок по типу задачи (см. раздел 4)
  segment_analysis = analyze_by_task_type(
      task_type=task_config.task_type,
      successful_runs=successful_runs,
      failed_runs=failed_runs,
  )

Шаг 7: Гипотезы об улучшениях
  hypotheses = generate_improvement_hypotheses(
      metric_delta_pct=metric_delta_pct,
      trajectory_slope=trajectory_slope,
      top_params=top_params_by_correlation,
      failed_categories=failed_categories_counts,
      segment_analysis=segment_analysis,
      program_md_conclusions=extract_conclusions(program_md_final),
  )
  # generate_improvement_hypotheses — детерминированная rule-based функция
  # (не LLM вызов). Продуцирует список структурированных гипотез (см. раздел 3).

Шаг 8: Сборка SessionAnalysis
  session_analysis = SessionAnalysis(
      baseline_value=baseline_value,
      best_value=best_value,
      metric_delta_pct=metric_delta_pct,
      improvement_trajectory=improvement_trajectory,
      trajectory_slope=trajectory_slope,
      top_params_by_correlation=top_params_by_correlation,
      failed_runs_analysis=failed_runs_analysis,
      systemic_failure=systemic_failure,
      segment_analysis=segment_analysis,
      improvement_hypotheses=hypotheses,
      ruff_summary=ruff_report.summary(),
  )
  # Сохраняется как .uaf/sessions/{id}/session_analysis.json
  # Передаётся в ReportGenerator как входные данные для Analysis секций
```

### 2.3 Условия для запуска полного анализа

Полный анализ запускается только при:
- Минимум 1 successful run существует (иначе анализировать нечего)
- MLflow доступен и возвращает runs (иначе ResultCollector fallback)

При 0 successful runs: `session_analysis.json` создаётся с флагом
`no_successful_runs=True`. Claude Code не генерирует секцию Analysis and Findings
(нечего анализировать). ReportGenerator включает только секцию Failed Experiments.

При MLflow недоступен: анализ пропускается, отчёт строится только из
`program_md_final` с предупреждением.

---

## 3. Feedback loop: как анализ влияет на следующую итерацию

### 3.1 Принципиальный момент архитектуры

В UAF v2.0 feedback loop работает не через UAF-компоненты в real-time,
а через два канала:

**Канал 1 — через program.md:** Claude Code сам обновляет program.md после
каждого шага, записывает Result и Conclusion, и планирует следующий шаг на основе
своего анализа предыдущих результатов. Это внутренний feedback loop Claude Code.
UAF не участвует в нём напрямую.

**Канал 2 — через budget_status.json:** BudgetController пишет `metrics_history`
и `convergence_signals` в этот файл. Claude Code видит динамику метрики
и может скорректировать стратегию.

**Канал 3 — ResultAnalyzer для нового цикла (опционально):**
При использовании `--resume {session_id}` с новым бюджетом ResultAnalyzer
формирует `improvement_context.md` из предыдущей сессии, который включается
в контекст следующего вызова ProgramMdGenerator.

### 3.2 Формат improvement_context.md

Этот файл создаётся ResultAnalyzer при `uaf run --resume {session_id}`:

```markdown
# Improvement Context: предыдущая сессия {prev_session_id}

## Результаты предыдущей сессии
- Лучший результат: {metric_name} = {best_value} (run: {best_run_id})
- Baseline: {metric_name} = {baseline_value}
- Прогресс: {metric_delta_pct:.1%}
- Итераций использовано: {n_runs} из {budget}

## Что сработало (top params по корреляции)
| Параметр | Корреляция с {metric_name} | Диапазон |
|----------|--------------------------|----------|
{top_params_table}

## Что не сработало (failed experiments)
| Шаг | Категория ошибки | Краткое описание |
|-----|-----------------|-----------------|
{failed_runs_table}

## Системные паттерны
{systemic_failure_note}

## Гипотезы для следующей сессии
{improvement_hypotheses_list}

## Рекомендации по бюджету
{budget_recommendation}
```

ProgramMdGenerator получает этот файл как дополнительный контекст и учитывает
его при генерации Research Phases. Конкретно: фазы предыдущей сессии, уже
закрытые (Baseline, initial HPO), не дублируются в новой сессии. Claude Code
начинает с более продвинутых гипотез.

### 3.3 Генерация improvement_hypotheses (детерминированная логика)

Функция `generate_improvement_hypotheses` работает по правилам без LLM:

```
Правило H-01: Если metric_delta_pct < 0.02 (улучшение < 2%):
  Добавить гипотезу: "Архитектура baseline оптимальна для данных.
  Рассмотреть feature engineering вместо смены модели."

Правило H-02: Если trajectory_slope > 0 (метрика всё ещё растёт):
  Добавить гипотезу: "Сессия завершилась до выхода на плато.
  Увеличить бюджет или patience."

Правило H-03: Если top_params_by_correlation[0].correlation > 0.7:
  Добавить гипотезу: "Параметр {param} сильно коррелирует с метрикой.
  Провести более детальный grid search по этому параметру."

Правило H-04: Если failed_categories["oom_error"] >= 2:
  Добавить гипотезу: "Несколько OOM. Уменьшить batch_size или использовать
  gradient checkpointing."

Правило H-05: Если failed_categories["import_error"] >= 1:
  Добавить гипотезу: "Import error — зависимость не установлена.
  Добавить в task.yaml: constraints.required_packages."

Правило H-06: Если systemic_failure == True:
  Добавить гипотезу: "Системная ошибка категории {category} повторяется.
  Изолировать в отдельный шаг с явной обработкой ошибки."

Правило H-07: Если segment_analysis.worst_segment.delta < -0.05:
  Добавить гипотезу: "Сегмент {segment} показывает деградацию -{delta:.1%}.
  Рассмотреть отдельную модель для этого сегмента или взвешивание."

Правило H-08: Если n_runs == budget (бюджет исчерпан, convergence_detected=False):
  Добавить гипотезу: "Бюджет исчерпан без сходимости. Увеличить patience или
  safety_cap_iterations для следующей сессии."
```

Правила не взаимоисключающие — все применимые добавляются в список.
Максимум 5 гипотез (по приоритету правил H-01 > H-02 > ... > H-08).

---

## 4. SHAP и feature importance

### 4.1 Когда запускается

SHAP-анализ запускается Claude Code по инструкции в program.md, не UAF.
UAF включает инструкцию через ProgramMdGenerator при выполнении условий:

```
SHAP включается в program.md если:
  task_type in ["tabular_classification", "tabular_regression"]
  AND N_features >= 5
  AND (best_model_type in ["catboost", "xgboost", "lightgbm", "random_forest"]
       OR task_config.research_preferences.feature_importance == True)
```

Для NLP и CV SHAP не применяется (нет смысла для токенов/пикселей как фичей
в классическом понимании). Вместо SHAP для NLP используется attention visualization
(опционально, только если явно указано в task.yaml).

### 4.2 Инструкция в program.md

ProgramMdGenerator добавляет в Phase 2 (Improvement Iterations) отдельный шаг:

```markdown
#### Step 2.X: Feature Importance Analysis

- **Hypothesis:** Выявить наиболее информативные признаки для снижения размерности
  и ускорения итераций.
- **Method:** SHAP TreeExplainer на лучшей модели из Phase 1.
  Логировать: mean(|SHAP values|) для каждого признака.
  Построить bar chart top-20 features.
- **Metric:** ruff_clean + shap_computation_seconds (не основная метрика)
- **Critical:** false
- **MLflow logging required:**
  - mlflow.log_artifact("shap_values.npy")
  - mlflow.log_artifact("shap_importance.csv")
  - mlflow.log_artifact("shap_bar_chart.pdf")
  - mlflow.log_metric("shap_computation_seconds", ...)
  - mlflow.log_param("shap_n_samples", ...)   # сколько примеров для SHAP
```

### 4.3 Что логируется в MLflow (детально)

Все SHAP артефакты логируются в отдельный experiment run с тегом `type=analysis`:

```
Run name: phase2/shap_analysis
Tags:
  type = analysis
  session_id = {uuid}
  step_id = "phase2.shap"
  status = success | failed
  analysis_type = shap
Params:
  shap_n_samples = {int}      # число примеров (обычно min(N, 500))
  shap_explainer_type = tree  # tree / kernel / gradient
  model_run_id = {best_run_id}  # run чьей модели считали SHAP
Metrics:
  shap_computation_seconds = {float}
  shap_n_features_nonzero = {int}  # признаков с ненулевым SHAP
Artifacts:
  shap_values.npy              # массив shap values [N_samples x N_features]
  shap_importance.csv          # feature -> mean(|shap|), отсортировано
  shap_bar_chart.pdf           # top-20 features bar chart
  shap_waterfall_top3.pdf      # waterfall plot для 3 примеров (опционально)
```

### 4.4 Использование SHAP-результатов в ResultAnalyzer

ResultAnalyzer читает `shap_importance.csv` из MLflow артефактов:

```python
# В post-session анализе, если shap run существует:
shap_run = [r for r in all_runs if r.tags.get("analysis_type") == "shap"]
if shap_run:
    shap_csv = download_artifact(shap_run[0], "shap_importance.csv")
    shap_df = pd.read_csv(shap_csv)
    top_shap_features = shap_df.nlargest(10, "mean_abs_shap")["feature"].tolist()
    low_shap_features = shap_df[shap_df["mean_abs_shap"] < shap_threshold]["feature"].tolist()
```

Если `len(low_shap_features) > N_features * 0.3`:
добавляется гипотеза H-09 в improvement_hypotheses:
"Более 30% признаков имеют пренебрежимо малый SHAP (порог: {shap_threshold}).
Рассмотреть удаление: {low_shap_features[:5]}... Это ускорит обучение и
может улучшить генерализацию."

Если SHAP завершился с failure: гипотеза H-09 не добавляется,
в отчёт записывается `shap_status=failed` без блокировки остального анализа.

### 4.5 SHAP в LaTeX отчёте

ReportGenerator включает SHAP данные в секцию Analysis and Findings:

```latex
\subsection{Feature Importance (SHAP)}
% Если shap_run успешен:
%   - Таблица top-10 features с mean(|SHAP|)
%   - Figure: shap_bar_chart.pdf (из MLflow artifact)
%   - Текст: сколько признаков с нулевым вкладом
%   - Рекомендация по feature selection если low_shap_features > 30%
% Если shap_run failed или не запускался:
%   - Примечание "SHAP analysis not available: {reason}"
```

---

## 5. Сегментация ошибок по типам задач

### 5.1 Зачем нужна сегментация

Единая метрика (например, ROC-AUC = 0.82) скрывает неравномерность модели:
на одних подгруппах она работает хорошо, на других — провально. Сегментация
выявляет это и формирует гипотезы о target sub-task.

UAF выполняет базовую сегментацию автоматически (через ResultAnalyzer),
более глубокую — через инструкцию в program.md для Claude Code.

### 5.2 Tabular задачи

**Автоматические сегменты (ResultAnalyzer, без LLM):**

```python
def segment_tabular(
    data_schema: DataSchema,
    best_run: MLflowRun,
    task_config: TaskConfig,
) -> TabularSegmentAnalysis:
    """
    Сегментация только если prediction_csv артефакт доступен в best_run.
    prediction_csv: файл с колонками [id, y_true, y_pred, y_prob, ...features]
    Логируется Claude Code по инструкции в program.md (см. ниже).
    """
    segments = {}

    # Сегмент 1: по основным категориальным признакам (топ-3 по SHAP если есть)
    cat_features = data_schema.features.categorical[:3]
    for feat in cat_features:
        for value in predictions[feat].unique()[:10]:  # не более 10 значений
            mask = predictions[feat] == value
            seg_metric = compute_metric(predictions[mask], task_config)
            segments[f"{feat}={value}"] = seg_metric

    # Сегмент 2: по квантилям числового признака с высоким SHAP (если есть)
    if top_shap_features:
        quant_feat = top_shap_features[0]
        if quant_feat in data_schema.features.numeric:
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                mask = quantile_mask(predictions[quant_feat], q)
                seg_metric = compute_metric(predictions[mask], task_config)
                segments[f"{quant_feat}_{q}"] = seg_metric

    # Сегмент 3: по размеру класса (только классификация)
    if task_config.task_type.endswith("classification"):
        for class_label in predictions["y_true"].unique():
            mask = predictions["y_true"] == class_label
            class_count = mask.sum()
            seg_metric = compute_metric(predictions[mask], task_config)
            segments[f"class={class_label} (N={class_count})"] = seg_metric

    # Найти worst и best сегменты
    overall_metric = best_run.metrics[task_config.metric.name]
    worst = min(segments.items(), key=lambda x: x[1])
    best_seg = max(segments.items(), key=lambda x: x[1])
    worst_delta = worst[1] - overall_metric  # отрицательное = деградация

    return TabularSegmentAnalysis(
        segments=segments,
        overall_metric=overall_metric,
        worst_segment=worst[0],
        worst_segment_delta=worst_delta,
        best_segment=best_seg[0],
    )
```

**Условие запуска:** сегментация только если `prediction_csv` артефакт
присутствует в MLflow артефактах best_run. Если нет — сегментация пропускается.

**Инструкция Claude Code в program.md** (добавляется ProgramMdGenerator
для tabular задач):

```
После обучения финальной модели в каждом шаге:
  Сохранить предсказания на val set:
    predictions_df = pd.DataFrame({
        "id": X_val.index, "y_true": y_val, "y_pred": model.predict(X_val),
        **X_val.to_dict(orient="list")
    })
    # Для классификации добавить y_prob:
    predictions_df["y_prob"] = model.predict_proba(X_val)[:, 1]
    mlflow.log_text(predictions_df.to_csv(index=False), "predictions.csv")
```

### 5.3 NLP задачи

Для NLP сегментация выполняется по:

**Сегмент 1 — по длине текста** (число токенов/слов):
```
short:  длина < 25-й перцентиль
medium: 25-й ... 75-й перцентиль
long:   > 75-й перцентиль
```
Если метрика на `long` значительно ниже (delta < -0.05): гипотеза "модель
плохо работает с длинными текстами, рассмотреть truncation strategy".

**Сегмент 2 — по источнику/домену** (если есть `source` или `domain` колонка
в данных, обнаруживается через data_schema.json):
Метрика отдельно для каждого домена. Большой разброс -> гипотеза о domain adaptation.

**Сегмент 3 — по языку** (если есть `language` колонка или детектирован
multi-language датасет через data_schema):
Раздельная метрика по языкам.

Условие запуска NLP сегментации: те же требования к `predictions.csv`
(для classification) или `eval_results.jsonl` (для generation/seq2seq).

**Для seq2seq/generation задач:** predictions.csv заменяется на:
```
eval_results.jsonl: [{"id": .., "input": .., "reference": .., "prediction": ..,
                      "score": .., "text_length": ..}, ...]
```
Сегментация по `text_length` и по первому токену входа (если это шаблонный промпт).

### 5.4 CV задачи

CV сегментация — наиболее специфична, во многом зависит от задачи.

**Image Classification:**

Сегментация по:
- Размер изображения (small / medium / large — по 25/75 перцентилям H*W)
- Яркость (mean pixel value: dark / normal / bright — три tertile)
- Класс (per-class accuracy — стандартная)

Инструкция Claude Code для CV в program.md:
```
После eval сохранить:
  eval_results.csv с колонками [id, y_true, y_pred, y_prob_max, img_h, img_w, img_mean_brightness]
  mlflow.log_text(eval_results.to_csv(), "eval_results.csv")
```

**Object Detection:**
Сегментация по:
- Размер bbox (small: area < 32^2, medium: 32^2 - 96^2, large: > 96^2) — COCO стандарт
- Количество объектов на изображении (1, 2-5, >5)

**Semantic Segmentation:**
Сегментация по классу (per-class IoU). Стандартная практика — логировать
`per_class_iou.json` как MLflow артефакт.

ResultAnalyzer для CV читает `eval_results.csv` и вычисляет метрику
для каждого сегмента аналогично tabular.

### 5.5 Формат SegmentAnalysis в session_analysis.json

```json
{
  "segment_analysis": {
    "task_type": "tabular_classification",
    "overall_metric": 0.847,
    "segments": {
      "age_Q1": 0.791,
      "age_Q2": 0.852,
      "age_Q3": 0.863,
      "age_Q4": 0.831,
      "class=0 (N=4750)": 0.889,
      "class=1 (N=250)": 0.612
    },
    "worst_segment": "class=1 (N=250)",
    "worst_segment_metric": 0.612,
    "worst_segment_delta": -0.235,
    "best_segment": "age_Q3",
    "best_segment_metric": 0.863,
    "predictions_available": true,
    "shap_available": true
  }
}
```

---

## 6. Анализ failed experiments

### 6.1 Что извлекается из краша

ResultAnalyzer обрабатывает каждый failed run по следующей схеме:

```python
@dataclass
class FailedRunAnalysis:
    run_id: str
    step_id: str
    failure_category: str       # классификация типа ошибки
    error_summary: str          # краткое описание (1-2 строки)
    traceback_tail: str         # последние 5 строк traceback
    recovery_hint: str          # детерминированная подсказка
    systemic: bool              # True если эта же категория >= 50% failed runs
    mlflow_data_partial: bool   # True если run содержит частичные метрики
    code_artifact_present: bool # True если .py файл залогирован в MLflow
```

**Классификация по traceback:**

```python
def classify_failure(traceback_text: str) -> str:
    """Детерминированная классификация по ключевым словам в traceback."""
    text = traceback_text.lower()

    if "modulenotfounderror" in text or "importerror" in text:
        return "import_error"

    if "cuda out of memory" in text or "out of memory" in text:
        return "oom_error"

    if "keyerror" in text or "column" in text or "shape" in text:
        return "data_error"

    if "timeouterror" in text or "timed out" in text:
        return "timeout_error"

    if "assertionerror" in text:
        return "assertion_error"

    if "valueerror" in text or "typeerror" in text:
        return "runtime_error"

    return "other"
```

**Recovery hints (детерминированные):**

| Категория | Recovery hint |
|-----------|--------------|
| `import_error` | Проверить установленные пакеты: добавить в `task.yaml: constraints.required_packages` |
| `oom_error` | Уменьшить `batch_size` или `n_estimators`. Рассмотреть `gradient_checkpointing=True` |
| `data_error` | Проверить названия колонок в data_schema.json. Возможен schema drift между шагами |
| `timeout_error` | Увеличить `task.yaml: constraints.max_training_time_minutes` или упростить модель |
| `assertion_error` | Проверить размеры массивов перед fit(). Добавить assert shapes в эксперимент |
| `runtime_error` | Проверить типы данных (int vs float). Добавить явный приведение типов |
| `other` | Изучить полный traceback в MLflow артефакте `traceback.txt` |

### 6.2 Повторяющиеся ошибки (systemic failure)

Если одна категория ошибки составляет >= 50% всех failed runs, это системная
проблема — не случайная. ResultAnalyzer устанавливает `systemic=True`
и добавляет в `session_analysis.json`:

```json
{
  "systemic_failure": {
    "detected": true,
    "category": "oom_error",
    "occurrence_rate": 0.67,
    "affected_steps": ["phase2/step3", "phase2/step5", "phase2/step7"],
    "recommendation": "OOM происходит в 67% failed runs. Все затронутые шаги связаны с увеличением batch_size. Рекомендация: зафиксировать batch_size=32 для оставшихся экспериментов."
  }
}
```

Это автоматически включается в секцию Failed Experiments LaTeX отчёта
как отдельный блок с рамкой (visually highlighted).

### 6.3 Частично завершённые runs

Если run упал в середине (есть частичные метрики в MLflow — например,
train_loss залогирован, но val_metric нет), ResultAnalyzer:

- Помечает `mlflow_data_partial=True`
- Читает доступные метрики и включает в Failed Experiments таблицу
  с пометкой "(partial)"
- Не включает в `metrics_history` для BudgetController
- Включает в отчёт с пометкой — не скрывает (antigoal 3)

### 6.4 Runs без кода в артефактах

Если Claude Code не залогировал `.py` файл как артефакт (нарушение M-UAF-11):

- `code_artifact_present=False`
- Записывается в `session_analysis.json` как `compliance_issues`
- Метрика `mlflow_compliance` (M-UAF-11) уменьшается
- В отчёте: примечание "Code artifact missing for run {run_id}"
- DVC cross-reference для этого run невозможен -> `cross_ref_integrity` уменьшается

---

## 7. Системные ошибки UAF: анализ и отчёт

### 7.1 SystemErrorAnalyzer: что отслеживается

Это отдельный слой от ResultAnalyzer. SystemErrorAnalyzer анализирует ошибки
самого UAF как программной системы — не ML-экспериментов.

Запускается в конце pipeline в ResearchSessionController перед ReportGenerator.

**Категории системных ошибок:**

| ID | Категория | Источник данных |
|----|-----------|-----------------|
| SE-01 | UAF crash rate | `uaf_exit_status` из MLflow session summary |
| SE-02 | Ruff violations | RuffReport от RuffEnforcer |
| SE-03 | Budget overrun | `consumed_iterations` vs `max_iterations` из BudgetController |
| SE-04 | MLflow compliance | M-UAF-11 из ResultCollector |
| SE-05 | DVC commit failures | M-UAF-12 из DVCSetup |
| SE-06 | SIGTERM hard stop | M-UAF-08 из BudgetController |
| SE-07 | Report generation failure | M-ONLINE-05 от ReportGenerator |
| SE-08 | HumanOversightGate timeout | `approval_wait_time_seconds` > 86400 |
| SE-09 | Antigoal violation | M-ONLINE-08: antigoal_violations_count > 0 |

### 7.2 Алгоритм SystemErrorAnalyzer

```python
def analyze_system_errors(
    session_id: str,
    budget_status: BudgetStatus,
    ruff_report: RuffReport,
    mlflow_summary: SessionSummaryRun,
) -> SystemErrorReport:
    issues = []

    # SE-02: Ruff violations
    if ruff_report.ruff_clean_rate < 0.95:
        issues.append(SystemIssue(
            id="SE-02",
            severity="warning",
            description=f"Ruff clean rate {ruff_report.ruff_clean_rate:.1%} < 95%.",
            detail=f"Files with unfixable violations: {ruff_report.files_with_unfixable}",
            recommendation="Усилить ruff инструкцию в program.md шаблоне или проверить "
                           "что Claude Code применяет ruff --fix после каждого файла.",
        ))

    # SE-03: Budget overrun (только fixed mode)
    if budget_status.mode == "fixed" and budget_status.hard_stop:
        if budget_status.consumed_iterations > budget_status.max_iterations:
            overrun_pct = (budget_status.consumed_iterations - budget_status.max_iterations) \
                          / budget_status.max_iterations
            issues.append(SystemIssue(
                id="SE-03",
                severity="error",
                description=f"Budget overrun {overrun_pct:.1%}: "
                             f"{budget_status.consumed_iterations} итераций при лимите "
                             f"{budget_status.max_iterations}.",
                detail="Нарушение antigoal 6.",
                recommendation="Уменьшить grace_period_minutes или усилить "
                               "инструкцию по budget_status.json в program.md.",
            ))

    # SE-04: MLflow compliance
    if mlflow_summary.mlflow_compliance < 1.0:
        issues.append(SystemIssue(
            id="SE-04",
            severity="warning",
            description=f"MLflow compliance {mlflow_summary.mlflow_compliance:.1%} < 100%.",
            detail=f"Missing required fields in {int((1 - mlflow_summary.mlflow_compliance) "
                   f"* mlflow_summary.total_runs)} runs.",
            recommendation="Claude Code не следует MLflow инструкции в program.md. "
                           "Проверить что mlflow_context.json был доступен.",
        ))

    # SE-06: SIGTERM hard stop
    if mlflow_summary.terminated_by_sigterm:
        issues.append(SystemIssue(
            id="SE-06",
            severity="warning",
            description="Сессия завершена через SIGTERM (grace period истёк).",
            detail="Claude Code не остановился корректно после hard_stop=True.",
            recommendation="Увеличить grace_period_minutes или убедиться что "
                           "budget_status.json читается Claude Code регулярно.",
        ))

    # SE-09: Antigoal violations
    if mlflow_summary.antigoal_violations_count > 0:
        issues.append(SystemIssue(
            id="SE-09",
            severity="critical",
            description=f"Antigoal violations: {mlflow_summary.antigoal_violations_count}.",
            detail="Нарушение принципов проекта.",
            recommendation="Расследовать детали в MLflow. "
                           "Antigoal violations = архитектурная проблема UAF.",
        ))

    return SystemErrorReport(
        session_id=session_id,
        issues=issues,
        has_critical=any(i.severity == "critical" for i in issues),
        has_errors=any(i.severity == "error" for i in issues),
        has_warnings=any(i.severity == "warning" for i in issues),
    )
```

### 7.3 Что попадает в LaTeX/PDF отчёт из SystemErrorAnalyzer

SystemErrorReport включается в ReportGenerator как отдельная секция:

```latex
\section{UAF System Health}

\subsection{Code Quality (Ruff)}
% Данные из RuffReport:
%   - Total .py files: N
%   - Clean files: M (X%)
%   - Files with unfixable violations: K
%   - Таблица: файл / violations_before_fix / violations_after_fix / unfixable
%   - Если ruff_clean_rate >= 0.95: "\textcolor{green}{PASS}"
%   - Если < 0.95: "\textcolor{orange}{WARNING}"

\subsection{Budget Compliance}
% Данные из BudgetController:
%   - Budget mode: fixed / dynamic
%   - Consumed: N iterations (X% of limit) / $Y cost / Z hours
%   - Stop reason: metric_converged | safety_cap | llm_signal | hard_stop
%   - Если hard_stop через SIGTERM: предупреждение
%   - Если overrun: CRITICAL с деталями

\subsection{MLflow Logging Compliance}
% mlflow_compliance: X/N runs полностью соответствуют схеме
% Если < 100%: список runs с missing fields
% cross_ref_integrity: X/N runs имеют валидный DVC commit

\subsection{Antigoal Compliance}
% Для каждого antigoal (1-6): проверка выполнения
% Таблица: Antigoal / Статус (OK / VIOLATION) / Детали
% Если violations > 0: CRITICAL, список нарушений

\subsection{System Issues Log}
% Таблица всех SystemIssue из SystemErrorAnalyzer:
%   ID / Severity / Description / Recommendation
% Если issues пуст: "No system issues detected"
```

### 7.4 Crash rate и MLflow session summary

При unhandled exception в UAF (не в Claude Code):

```python
# В ResearchSessionController, except block:
mlflow.set_tag("uaf_exit_status", "error")
mlflow.set_tag("uaf_crash_reason", str(exception)[:200])
mlflow.log_text(traceback.format_exc(), "uaf_crash_traceback.txt")
```

Это гарантирует что даже при краше UAF в MLflow остаётся запись.
ReportGenerator при `uaf_exit_status=error` добавляет CRITICAL блок
в начало отчёта.

---

## 8. Что попадает в LaTeX/PDF отчёт по error analysis (сводно)

Ниже полная карта: компонент -> данные -> секция отчёта.

### 8.1 Секция Analysis and Findings (генерируется Claude Code в рамках сессии)

**Входные данные для Claude Code (финальный этап сессии):**

```
- session_analysis.json (весь файл):
    baseline_value, best_value, metric_delta_pct
    improvement_trajectory (список значений)
    top_params_by_correlation (top-3)
    improvement_hypotheses (список гипотез из правил H-01..H-09)
    segment_analysis (worst/best сегменты)
    systemic_failure (если есть)
- program_md_final: секция Final Conclusions (написанная Claude Code в процессе сессии)
- task_config: task_type, metric.name, metric.direction
```

**Что Claude Code генерирует и сохраняет в SESSION_DIR/report/sections/analysis_and_findings.md:**
- 2-3 абзаца: интерпретация паттернов в метрике
- Почему лучший run лучший (какие параметры)
- Почему failed experiments провалились (на основе failure_categories)
- Достигнута ли цель из task.yaml threshold (если задан)
- Не дублирует гипотезы — они выводятся отдельно в секции Recommendations

UAF ReportGenerator читает этот файл и вставляет содержимое в .tex шаблон.

### 8.2 Секция Failed Experiments (автоматическая)

Источник: failed_runs_analysis из ResultAnalyzer.

```latex
\subsection{Failed Experiments}

% Таблица: Step / Category / Error Summary / Recovery Hint
% Каждая строка — один failed run (серым шрифтом)
% Код ошибки: последние 3 строки traceback (моноширинный шрифт)

% Если systemic_failure detected:
\begin{tcolorbox}[colback=yellow!10, colframe=orange!50]
\textbf{Системная ошибка: {category}}
{description}
\end{tcolorbox}
```

### 8.3 Секция Feature Importance / SHAP

Источник: SHAP артефакты из MLflow (если shap_run успешен).

```latex
\subsection{Feature Importance (SHAP)}
% - Таблица top-10 features: feature / mean(|SHAP|) / rank
% - Figure: shap_bar_chart.pdf (вставляется как \includegraphics)
% - Если low_shap_features > 30%: блок рекомендации по удалению признаков
```

### 8.4 Секция Segment Analysis

Источник: segment_analysis из session_analysis.json.

```latex
\subsection{Performance by Segment}
% Таблица: сегмент / метрика / дельта от overall
% Worst сегмент выделен красным
% Best сегмент выделен зелёным
% Примечание если predictions.csv недоступен
```

### 8.5 Секция Improvement Hypotheses

Источник: improvement_hypotheses из session_analysis.json.

```latex
\subsection{Hypotheses for Next Iteration}
% Пронумерованный список гипотез
% Каждая гипотеза: краткое описание + обоснование (из правил H-*)
% Сортировка: по приоритету правила (H-01 первый)
```

### 8.6 Секция UAF System Health

Источник: SystemErrorReport от SystemErrorAnalyzer.

```latex
\section{UAF System Health}
% Подсекции: Code Quality, Budget Compliance,
%            MLflow Compliance, Antigoal Compliance, Issues Log
% (см. раздел 7.3 выше)
```

---

## 9. DataClass: SessionAnalysis

Полная структура `session_analysis.json`:

```python
@dataclass
class FailedRunAnalysis:
    run_id: str
    step_id: str
    failure_category: str
    error_summary: str
    traceback_tail: str
    recovery_hint: str
    systemic: bool
    mlflow_data_partial: bool
    code_artifact_present: bool


@dataclass
class SegmentResult:
    segment_name: str
    segment_metric: float
    segment_delta: float   # разница от overall_metric
    n_samples: int


@dataclass
class SegmentAnalysis:
    task_type: str
    overall_metric: float
    segments: dict[str, SegmentResult]
    worst_segment: str
    worst_segment_delta: float
    best_segment: str
    predictions_available: bool
    shap_available: bool


@dataclass
class ImprovementHypothesis:
    rule_id: str            # H-01 .. H-09
    hypothesis: str
    evidence: str           # что именно в данных послужило основанием
    priority: int           # 1 = высший


@dataclass
class SystematicFailure:
    detected: bool
    category: str | None
    occurrence_rate: float
    affected_steps: list[str]
    recommendation: str


@dataclass
class SessionAnalysis:
    session_id: str
    task_type: str
    metric_name: str
    metric_direction: str

    # Метрический профиль
    baseline_value: float | None
    best_value: float | None
    metric_delta_pct: float | None
    improvement_trajectory: list[float]
    trajectory_slope: float | None

    # Param-metric корреляции
    top_params_by_correlation: list[dict]   # [{param, correlation, range}]

    # Failed experiments
    failed_runs_analysis: list[FailedRunAnalysis]
    systemic_failure: SystematicFailure

    # Сегментация
    segment_analysis: SegmentAnalysis | None

    # SHAP
    shap_available: bool
    shap_top_features: list[str]
    shap_low_features: list[str]

    # Гипотезы
    improvement_hypotheses: list[ImprovementHypothesis]

    # Флаги
    no_successful_runs: bool
    predictions_available: bool
    analysis_complete: bool
```

Файл сохраняется как `.uaf/sessions/{id}/session_analysis.json`.
Логируется в MLflow: `mlflow.log_artifact("session_analysis.json")` в session summary run.

---

## 10. Ограничения ResultAnalyzer и что не анализируется

**Явные ограничения:**

1. **Сегментация требует prediction_csv.** Если Claude Code не залогировал
   предсказания (не выполнил инструкцию), сегментация недоступна.
   Это информационный deficit, не критическая ошибка. ResultAnalyzer
   отмечает `predictions_available=False` и пропускает сегментацию.

2. **SHAP только для tree-based моделей по умолчанию.** TreeExplainer
   не применим к нейросетям. Для PyTorch-моделей требуется GradientExplainer
   или KernelExplainer (значительно медленнее). Включение через
   `research_preferences.feature_importance=True` в task.yaml.

3. **param-metric корреляция требует >= 5 runs.** При малом числе runs
   (< 5 successful) корреляции не вычисляются — нестатистически.

4. **failure_category — keyword matching, не parsing.** Ложные срабатывания
   возможны при нестандартных exception сообщениях. Полный traceback
   всегда доступен в MLflow для ручного изучения.

5. **improvement_hypotheses — детерминированные правила без LLM.**
   Они не заменяют экспертный анализ. Назначение: структурированная отправная
   точка для следующей итерации, не окончательный вывод.

6. **Анализ не рекурсивный.** ResultAnalyzer не анализирует промежуточные
   чекпоинты внутри одного run (только финальные метрики run).
   Если нужен анализ динамики loss внутри обучения — Claude Code должен
   логировать step-метрики через `mlflow.log_metric(..., step=epoch)`.

---

## STAGE COMPLETE

Стадия 08-error-analysis завершена.

**Зафиксировано:**

- Стадия 07-baseline: SKIPPED/EMBEDDED. Baseline logic зафиксирована как
  обязательная часть шаблона program.md (Phase 1: Baseline, 4 уровня).
  Отдельный артефакт не создаётся.

- ResultAnalyzer (Слой A):
  - 8-шаговый алгоритм post-session анализа
  - Входные данные: all_runs из MLflow, program_md_final, task_config, ruff_report
  - Выход: session_analysis.json (DataClass SessionAnalysis)
  - Два момента вызова: BudgetController polling (минимальный) + post-session (полный)

- Feedback loop (3 канала):
  - Канал 1: program.md (внутренний Claude Code loop)
  - Канал 2: budget_status.json (metrics_history)
  - Канал 3: improvement_context.md при --resume

- Improvement hypotheses: 9 детерминированных правил (H-01..H-09),
  без LLM, максимум 5 гипотез по приоритету.

- SHAP: только tabular + tree-based, по умолчанию. Логируется как отдельный
  MLflow run (type=analysis). Артефакты: shap_values.npy, shap_importance.csv,
  shap_bar_chart.pdf.

- Сегментация ошибок:
  - Tabular: по cat features, quantile num feature, по классу
  - NLP: по длине текста, домену, языку
  - CV: по размеру изображения, яркости, классу / bbox size / per-class IoU
  - Требует prediction_csv / eval_results.csv в MLflow артефактах

- Failed experiments: классификация по 7 категориям (keyword matching),
  recovery hints (детерминированные), systemic failure detection (>= 50%),
  поддержка partial runs (antigoal 3 соблюдается).

- SystemErrorAnalyzer (Слой B): 9 категорий SE-01..SE-09,
  severity: critical/error/warning. Входные данные: BudgetStatus, RuffReport,
  SessionSummaryRun.

- Отчёт (6 секций от error analysis):
  - Analysis and Findings (Claude Code генерирует в рамках сессии, сохраняет в report/sections/)
  - Failed Experiments (автоматическая, antigoal 3)
  - Feature Importance/SHAP (автоматическая, conditional)
  - Segment Analysis (автоматическая, conditional на predictions_available)
  - Improvement Hypotheses (автоматическая, из правил H-*)
  - UAF System Health (автоматическая, из SystemErrorAnalyzer)

Переход к стадии 09-pipeline разрешён.
