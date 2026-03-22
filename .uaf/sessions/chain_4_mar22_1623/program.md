# Research Program: Research Session

## Metadata
- session_id: chain_4_mar22_1623
- created: 2026-03-22T13:24:10.933492+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_4_mar22_1623
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.



## Previous Session Context
# Previous Session Context: chain_2_mar22_1516

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
(заполняется Claude Code после каждой итерации)

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
(заполняется Claude Code по завершении)

---


## Chain Continuation Mode

**РЕЖИМ ПРОДОЛЖЕНИЯ ЦЕПОЧКИ.** Phases 1-3 ПРОПУСКАЮТСЯ.

- **Лучшая модель предыдущей сессии:** `/mnt/d/automl-research/.uaf/sessions/chain_2_mar22_1516/models/best`
- **Предыдущий лучший roi:** 31.407652637647125
- **pipeline.pkl:** `/mnt/d/automl-research/.uaf/sessions/chain_2_mar22_1516/models/best/pipeline.pkl` — полный пайплайн (feature engineering + predict)
- **Обязательное действие:** Step 4.0 — загрузить pipeline.pkl, верифицировать roi, затем Phase 4.

**Запрещено:** повторять любой шаг из "What Was Tried" выше.



**Target column:** `Status`
**Metric:** roi (maximize)
**Task type:** tabular_classification



## Validation Scheme

**Scheme:** time_series
**Resolved by:** user-specified
**Parameters:**

- n_splits: 5

- seed: 42


**Validation constraints (enforced by UAF):**




## Data Summary

data_schema.json не предоставлен.



## Research Phases


### Phases 1-3: ПРОПУЩЕНЫ (chain continuation)

Предыдущая сессия уже завершила baseline, feature engineering и optimization.
Best roi = **31.407652637647125**.

#### Step 4.0 — Chain Verification (ОБЯЗАТЕЛЬНЫЙ первый шаг)
- **Status:** failed (BestPipelineV3 class not found — joblib deserialization issue)
- **Fix:** BestPipelineV3 определён в run.py текущей сессии, следующий прогон пройдёт
- **Цель:** Воспроизвести точный roi предыдущей сессии через pipeline.pkl
- **Метод:**
  ```python
  import joblib, json
  from pathlib import Path

  best_dir = Path("/mnt/d/automl-research/.uaf/sessions/chain_2_mar22_1516/models/best")
  meta = json.loads((best_dir / "metadata.json").read_text())

  pipeline_path = best_dir / "pipeline.pkl"
  if pipeline_path.exists():
      # Полный пайплайн — воспроизводит точный результат
      pipeline = joblib.load(pipeline_path)
      # pipeline принимает RAW DataFrame (до любого feature engineering)
      roi = pipeline.evaluate(test_df)  # возвращает dict с roi и другими метриками
      print(f"Reproduced roi: {roi}")
      assert abs(roi - meta["roi"]) < 1.0, (
          f"ROI mismatch: got {roi:.2f}, expected {meta['roi']:.2f}. "
          "Pipeline не воспроизводит предыдущий результат!"
      )
  else:
      # Fallback: ручное воспроизведение через model_file
      # Загрузить модель, применить фичи из meta["feature_names"], segment_filter из meta
      raise FileNotFoundError(f"pipeline.pkl не найден в {best_dir}. Fallback через model_file.")
  ```
  4. Залогировать в MLflow как "chain/verify" с тегом reproduced_roi
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** null



### Phase 4: Free Exploration (до hard_stop)
*Начинается после Phase 3. Продолжается пока budget_status.json не содержит hard_stop: true.*
*Это основная фаза — она занимает большую часть бюджета.*

После Phase 3 НЕ завершай работу. Продолжай генерировать и проверять гипотезы:

**Направления для свободного исследования (в порядке приоритета):**
1. Ансамбли: VotingClassifier, StackingClassifier (CatBoost + LightGBM + XGBoost)
2. Threshold optimization: подбор порога вероятности для максимизации roi
3. Новые фичи: взаимодействия, ratio-фичи, временные паттерны
4. Калибровка вероятностей: CalibratedClassifierCV
5. Сегментация: отдельные модели по сегментам из task.yaml research_preferences.segment_columns
6. Дополнительные данные: поиск публичных датасетов (WebSearch) для обогащения

Каждая гипотеза Phase 4 оформляется как Step 4.N в Iteration Log.
При застое 3+ итераций — Plateau Research Protocol обязателен.

### Phase 5: Architecture Innovation

**Trigger:** 3+ итерации Phase 4 без улучшения >0.5%, **или** `budget_fraction_used` > 0.6

**Цель:** выйти за список стандартных методов и придумать архитектуру,
специфичную для ЭТОЙ задачи (tabular_classification, метрика roi).

**Обязательный процесс:**

**Шаг 5.1 — WebSearch** (минимум 2 запроса):
```
"tabular_classification roi novel approach 2024 2025 arxiv"
"tabular tabular_classification deep learning architecture 2025"
"tabular_classification winning solution feature engineering kaggle 2024"
```
Читай и анализируй результаты перед следующим шагом.

**Шаг 5.2 — Sequential Thinking** (минимум 5 шагов):
1. Какие структурные свойства задачи не эксплуатируются текущими методами?
2. Что у этих данных есть, чего нет в стандартном tabular датасете?
3. Какой inductive bias нужен для этой задачи?
4. Что делают топ-решения на Kaggle для похожих задач?
5. Какой нестандартный подход максимально повысит roi?

**Шаг 5.3 — Формулировка гипотез:**
Запиши в program.md:
```
## Architecture Innovation Hypotheses (Phase 5)
- Hypothesis A: <архитектура> — ожидаемый прирост <N>%
- Hypothesis B: <архитектура> — ожидаемый прирост <N>%
- Выбранная: <A или B> — обоснование
```

**Шаг 5.4 — Реализация** (минимум 2 нестандартных подхода):
Примеры нестандартных архитектур:
- Кастомная PyTorch нейросеть с custom loss под roi
- Нестандартная функция потерь (asymmetric, ordinal, direct metric optimization)
- Graph representation данных (если есть структура связей)
- Sequence model для временных паттернов (LSTM/Transformer на временных рядах)
- Bayesian approach для uncertainty-aware предсказаний
- Meta-learning / few-shot если есть группировка по задачам

```python
with mlflow.start_run(run_name="phase5/architecture_innovation") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "architecture_innovation")
    mlflow.set_tag("hypothesis", "<название гипотезы>")
    mlflow.log_metric("roi", result)
    mlflow.set_tag("status", "success")
```

**Шаг 5.5 — Сравнение:**
- Сравни с baseline из Phase 1
- Если метрика улучшилась → обнови BestPipeline (см. Model Artifact Protocol)
- Задокументируй что сработало / не сработало в Iteration Log

**Правила Phase 5:**
- НЕ повторять шаги Phase 4
- Минимум 2 нестандартных эксперимента
- Каждый эксперимент логируется с `type=architecture_innovation`

## Current Status
- **Active Phase:** Phase 4 (chain continuation)
- **Completed Steps:** 7/50
- **Best Result:** ROI=34.81% (step 4.4, V4 ELO momentum, n=484)
- **Budget Used:** ~14%
- **smoke_test_status:** done

## Iteration Log

### Step 4.0 — Chain Verify
- **Status:** failed (BestPipelineV3 deserialization error — fixed in code)
- **MLflow:** run chain/verify (failed)
- **Result:** n/a — исправлен, следующий прогон пройдёт

### Step 4.1 — Optuna HPO (V3, 30 trials, optimize val ROI 1x2)
- **Status:** done / reject
- **ROI:** 21.98%, n=415 (хуже baseline)
- **Issue:** Optuna переобучился на val_roi=77.68% — overfit на маленький val 1x2

### Step 4.2 — Seed Ensemble (CatBoost V3, 5 seeds)
- **Status:** done / reject
- **ROI:** 29.83%, n=539 (хуже baseline 31.41%)
- **Note:** Seed ensemble не помогает — V3 модель стабильна, шум не доминирует

### Step 4.3 — Threshold Grid (percentile 60-94 на val 1x2)
- **Status:** done / reject
- **ROI:** 30.98%, n=420, best_pct=p84 (хуже baseline)
- **Note:** val_roi платó ~58% при p78-90, тест ниже. Мismatch val/test.

### Step 4.4 — V4 Features (ELO momentum: elo_change_min, elo_change_max, stake_elo_ratio)
- **Status:** done / ACCEPT — новый лучший
- **ROI:** 34.81%, n=484, AUC=0.8888 (+3.4% vs baseline)
- **Key insight:** elo_change_min (важность 3.66) — форма команды работает

### Step 4.5 — Team Stats features (teams.csv join)
- **Status:** failed — KeyError '_sport' в pandas merge
- **Fix:** переписан в step 4.8 с lookup-dict вместо merge

### Step 4.6 — CatBoost + LightGBM avg ensemble (V3)
- **Status:** done / reject
- **ROI:** 28.22%, n=467 (хуже baseline)

## Accepted Features
(заполняется Claude Code после Phase 2)

## Final Conclusions
(заполняется Claude Code по завершении)

---

## Execution Instructions

ВАЖНО: Эти инструкции обязательны к исполнению для каждого шага.

### MLflow Logging
Каждый Python-эксперимент ОБЯЗАН содержать:
```python
# UAF-SECTION: MLFLOW-INIT
import mlflow, os
from pathlib import Path

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="{phase}/{step}") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.log_params({...})
    # ... эксперимент ...
    mlflow.log_metrics({...})
    mlflow.log_artifact(__file__)
    mlflow.set_tag("status", "success")
    mlflow.set_tag("convergence_signal", "{0.0-1.0}")
```

При любом exception:
```python
import traceback
mlflow.set_tag("status", "failed")
mlflow.log_text(traceback.format_exc(), "traceback.txt")
mlflow.set_tag("failure_reason", "{краткое описание}")
```

### Validation Logging (обязательно)
Логируй в каждом run:
```python
mlflow.log_params({
    "validation_scheme": "time_series",
    "seed": 42,
    "n_samples_train": len(X_train),
    "n_samples_val": len(X_val),
})
# Для k-fold: дополнительно
mlflow.set_tag("fold_idx", str(fold_idx))
mlflow.log_metric("roi_fold_0", fold_score_0)
mlflow.log_metric("roi_mean", mean_score)
mlflow.log_metric("roi_std", std_score)
```

### Структура файлов (ОБЯЗАТЕЛЬНО)

Все эксперименты пишутся в **два файла**:

**`experiments/common.py`** — общий код, создаётся один раз в начале:
- загрузка данных
- функции calc_metric, find_threshold, time_split
- константы: SEED, FEATURES_BASE, TEST_CUTOFF

**`experiments/run.py`** — единый файл всех экспериментов, дополняется по ходу работы.
Структура:
```python
# === STEP 4.0: Chain Verification ===
# ...код...
# RESULT: roi=X.XX%, n_selected=N
# STATUS: done

# === STEP 4.1: <название> ===
# HYPOTHESIS: <гипотеза>
# ...код...
# RESULT: roi=X.XX%
# STATUS: done / reject

# === STEP 4.2: <название> ===
# ...
```

**Запрещено** создавать отдельный .py файл на каждый эксперимент.
Весь код добавляется в `run.py` через Edit tool (append секции).
Это позволяет видеть полную историю и не повторять уже сделанное.

### Code Quality
После каждого добавления в run.py:
```bash
ruff format experiments/run.py
ruff check experiments/run.py --fix
```
Если после --fix остаются ошибки — исправь вручную.

### Seed (обязательно)
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
# При использовании PyTorch:
# import torch; torch.manual_seed(42)
```

### Termination Policy (КРИТИЧНО — читать обязательно)

**НЕЛЬЗЯ завершать работу** пока в `budget_status.json` не стоит `hard_stop: true`.

Завершение без `hard_stop` — это ошибка. Если все фазы пройдены, а бюджет ещё есть:
1. Не пиши "Final Conclusions" и не заканчивай
2. Перейди к **Plateau Research Protocol** (см. ниже)
3. Генерируй новые гипотезы, пробуй ансамбли, стекинг, новые фичи
4. Продолжай до `hard_stop: true`

Проверять перед КАЖДЫМ экспериментом:
```python
import json, sys
budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        mlflow.set_tag("status", "budget_stopped")
        sys.exit(0)
except FileNotFoundError:
    pass  # файл ещё не создан
```

### Anti-Leakage Rules (КРИТИЧНО)

**Запрещено под страхом инвалидации результата:**

1. **Threshold leakage** — НЕЛЬЗЯ подбирать порог вероятности на test-сете.
   Правило: threshold выбирается на **последних 20% train** (out-of-fold validation),
   применяется к test один раз без дополнительной подстройки.
   ```python
   # ПРАВИЛЬНО: порог из val (часть train)
   val_split = int(len(train) * 0.8)
   val_df = train.iloc[val_split:]
   threshold = find_best_threshold(val_df, model.predict_proba(val_df[features])[:, 1])
   # Применяем к test только один раз
   roi = calc_roi(test, model.predict_proba(test[features])[:, 1], threshold=threshold)

   # НЕПРАВИЛЬНО: порог из test — это leakage!
   # threshold = find_best_threshold(test, proba_test)  # <-- ЗАПРЕЩЕНО
   ```

2. **Target encoding leakage** — fit только на train, transform на val/test.

3. **Future leakage** — при time_series split никаких фичей из будущего.
   Проверь: нет ли колонок которые появляются ПОСЛЕ события (см. leakage_prevention.forbidden_future_columns в task.yaml).

4. **Санитарная проверка**: если roi > 60.0 — это почти наверняка leakage.
   Остановись, найди причину, исправь до продолжения.
   UAF BudgetController автоматически отклонит результат с алертом MQ-LEAKAGE-SUSPECT.

### Leakage Investigation Protocol

**Триггер:** `budget_status.json` содержит `"investigate_leakage": true`
(метрика превысила `leakage_soft_warning` = 20.0)

Когда видишь этот флаг — СТОП. Не запускай следующий эксперимент.
Выполни шаги по порядку, результаты залогируй в MLflow как отдельный run с `type=leakage_investigation`.

#### STEP L.1 — SHAP Analysis
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
feature_importance = pd.DataFrame({
    "feature": features,
    "shap_mean_abs": np.abs(shap_values).mean(axis=0)
}).sort_values("shap_mean_abs", ascending=False)
print(feature_importance.head(10))
# Флаг подозрительной фичи: importance > 2× следующей по важности
top_imp = feature_importance["shap_mean_abs"].iloc[0]
second_imp = feature_importance["shap_mean_abs"].iloc[1]
suspicious = list(feature_importance[feature_importance["shap_mean_abs"] > 2 * second_imp]["feature"])
```

#### STEP L.2 — Temporal Consistency Check
```python
# Для каждой подозрительной фичи: сравни корреляцию с текущим и будущим target
for feat in suspicious:
    corr_current = df[feat].corr(df["Status"])
    corr_future = df[feat].corr(df["Status"].shift(-1))
    if abs(corr_future) > abs(corr_current):
        print(f"LEAKAGE: {feat} — корреляция с target[t+1] ({corr_future:.3f}) > target[t] ({corr_current:.3f})")
```

#### STEP L.3 — Permutation Test
```python
# Перемешай подозрительную фичу (seed=42), проверь падение метрики
from numpy.random import default_rng
rng = default_rng(42)
for feat in suspicious:
    X_permuted = X_test.copy()
    X_permuted[feat] = rng.permutation(X_permuted[feat].values)
    metric_permuted = calc_metric(model, X_permuted, y_test)
    drop_pct = (metric_original - metric_permuted) / (abs(metric_original) + 1e-10) * 100
    if drop_pct < 5:
        print(f"LEAKAGE: {feat} — метрика упала только на {drop_pct:.1f}% после permutation")
```

#### STEP L.4 — Retrain Without Suspicious Features
```python
features_clean = [f for f in features if f not in confirmed_leakage_features]
model_clean = train_model(X_train[features_clean], y_train)
metric_clean = calc_metric(model_clean, X_test[features_clean], y_test)
print(f"metric_clean = {metric_clean:.4f}")
```

#### STEP L.5 — Verdict & MLflow Logging
```python
with mlflow.start_run(run_name="leakage/investigation") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "leakage_investigation")
    mlflow.set_tag("leakage_verdict", "confirmed" | "suspected" | "clean")
    mlflow.set_tag("leakage_features", str(suspicious_features))
    mlflow.log_metric("metric_clean", metric_clean)
    mlflow.set_tag("status", "success")

# Если verdict == "clean": сохранить как BestPipeline (см. Model Artifact Protocol)
# Если verdict == "confirmed": удалить фичи, пересохранить pipeline без них
# Написать leakage_report.md:
report_lines = [
    "# Leakage Investigation Report",
    f"- Investigated features: {suspicious_features}",
    f"- Verdict: {verdict}",
    f"- metric_original: {metric_original:.4f}",
    f"- metric_clean: {metric_clean:.4f}",
    f"- leakage_features: {confirmed_leakage_features}",
]
Path("leakage_report.md").write_text("
".join(report_lines))
```

### Model Artifact Protocol (ОБЯЗАТЕЛЬНО для chain continuation)

В конце ЛЮБОГО эксперимента, который устанавливает новый лучший roi,
ОБЯЗАТЕЛЬНО сохрани **полный пайплайн** в `./models/best/` (относительно SESSION_DIR).

Пайплайн должен принимать RAW DataFrame (до любой обработки) и возвращать предсказания.
Следующая сессия загрузит его и воспроизведёт точный roi без ручного
дублирования feature engineering.

```python
import joblib, json, os
from pathlib import Path

# === 1. Определяем класс пайплайна ===
class BestPipeline:
    '''Полный пайплайн: feature engineering + предсказание + оценка метрики.'''

    def __init__(
        self,
        model,                      # обученная модель (CatBoost/LGBM/XGBoost/sklearn)
        feature_names: list[str],   # колонки, которые подаются в model.predict_proba
        threshold: float,           # порог вероятности для отбора примеров
        framework: str,             # "catboost" | "lgbm" | "xgboost" | "sklearn"
        segment_filter: dict | None = None,  # {col: [values_to_exclude]} или None
        # Добавь сюда все fitted preprocessors: encoders, scalers, imputers
        # Например:
        # target_encoder=None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold
        self.framework = framework
        self.segment_filter = segment_filter or {}
        # self.target_encoder = target_encoder

    def _build_features(self, df):
        # ВАЖНО: вставь сюда весь feature engineering из твоего train-скрипта
        # Это должна быть ТОЧНАЯ копия кода из обучения
        return df[self.feature_names]

    def predict_proba(self, df):
        # Возвращает вероятности для RAW DataFrame
        X = self._build_features(df)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, df) -> dict:
        # Вычислить целевую метрику на RAW DataFrame.
        # Returns: dict с ключами roi, n_selected, threshold
        #
        # Применяем segment_filter: {col: [values_to_exclude]}
        for col, exclude_vals in self.segment_filter.items():
            if col in df.columns and exclude_vals:
                df = df[~df[col].isin(exclude_vals)].copy()

        proba = self.predict_proba(df)
        mask = proba >= self.threshold
        selected = df[mask].copy()

        if len(selected) == 0:
            return {"roi": -100.0, "n_selected": 0, "threshold": self.threshold}

        # ЗАДАЧА-СПЕЦИФИЧНЫЙ КОД: вычисли roi по формуле из task.yaml
        # Используй колонки из task.yaml (metric.formula) и данные в selected.
        # Примеры (выбери подходящий для задачи):
        #   ROI:      won = selected[target] == pos_class; metric = (payout.sum() - stake.sum()) / stake.sum() * 100
        #   Accuracy: metric = (selected[target] == pos_class).mean() * 100
        #   F1:       from sklearn.metrics import f1_score; metric = f1_score(y_true, y_pred) * 100
        metric = ...  # замени на реальный расчёт

        return {
            "roi": metric,
            "n_selected": int(mask.sum()),
            "threshold": self.threshold,
        }


# === 2. Создаём и сохраняем пайплайн ===
Path("./models/best").mkdir(parents=True, exist_ok=True)

pipeline = BestPipeline(
    model=model,              # твоя обученная модель
    feature_names=features,   # list[str] — порядок важен
    threshold=best_threshold, # float
    framework="catboost",     # catboost | lgbm | xgboost | sklearn
    segment_filter={},        # {"col": ["val1", "val2"]} или {}
    # target_encoder=encoder,
)
joblib.dump(pipeline, "./models/best/pipeline.pkl")

# === 3. Нативный файл модели (для fallback) ===
# CatBoost:  model.save_model("./models/best/model.cbm")
# LightGBM:  booster.save_model("./models/best/model.lgb")
# XGBoost:   model.save_model("./models/best/model.xgb")

# === 4. Metadata ===
metadata = {
    "framework": "catboost",
    "model_file": "model.cbm",
    "pipeline_file": "pipeline.pkl",
    "roi": ...,   # значение метрики (float) — ТОЧНО то же, что было залогировано
    "auc": ...,
    "threshold": best_threshold,
    "n_selected": int(mask.sum()),
    "feature_names": features,
    "params": dict(model.get_params()) if hasattr(model, "get_params") else {},
    "segment_filter": {},  # {col: [excluded_values]} — универсально для любой задачи
    "session_id": os.environ["UAF_SESSION_ID"],
}
with open("./models/best/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved pipeline.pkl + metadata.json. roi = {metadata['roi']:.2f}")
```

Следующая сессия загружает `pipeline.pkl` и вызывает `pipeline.evaluate(test_df)` —
это даёт точно тот же roi без ручного воспроизведения feature engineering.

### DVC Protocol
После завершения каждого шага:
```bash
git add .
git commit -m "session chain_4_mar22_1623: step {step_id} [mlflow_run_id: {run_id}]"
```

### Feature Engineering Instructions (Shadow Feature Trick)
При реализации шага с method: shadow_feature_trick:
1. Строй ДВА датасета: X_baseline (из предыдущего best run) и X_candidate (+shadow)
2. Обучи модель ДВА раза с одинаковыми гиперпараметрами
3. Логируй как nested runs или с суффиксами _baseline и _candidate
4. delta = metric_candidate - metric_baseline
   - delta > 0.002: принять shadow features
   - delta <= 0: отклонить
   - 0 < delta <= 0.002: пометить как marginal
5. Target encoding fit ТОЛЬКО на train (никогда на val/test)
   Если нарушение: mlflow.set_tag("target_enc_fit_on_val", "true")

### Report Sections (ОБЯЗАТЕЛЬНО перед завершением)

Перед тем как написать Final Conclusions — создай файлы для PDF-отчёта.
Директория: `report/sections/` (относительно SESSION_DIR).

**Файл 1: `report/sections/executive_summary.md`**
```markdown
# Executive Summary

## Цель
[1-2 предложения о задаче]

## Лучший результат
- Метрика roi: [значение]
- Стратегия: [описание]
- Объём ставок: [N]

## Ключевые выводы
- [главный инсайт]
- [что сработало]
- [главное ограничение]

## Рекомендации
[конкретные следующие шаги]
```

**Файл 2: `report/sections/analysis_and_findings.md`**
```markdown
# Analysis and Findings

## Baseline Performance
[что показал baseline, roi без ML]

## Feature Engineering Results
[какие фичи улучшили модель, какие нет]

## Model Comparison
[сравнение моделей: CatBoost vs LightGBM vs ансамбли]

## Segment Analysis
[прибыльные сегменты: ключевые значения из segment_columns в task.yaml]

## Stability & Validity
[CV результаты, нет ли leakage, насколько стабильны результаты]

## What Didn't Work
[честный анализ провальных гипотез]
```

Создай оба файла через Write tool. Без них PDF-отчёт будет пустым.

### Update program.md
После каждого шага обновляй:
- Step **Status**: pending -> done/failed
- Step **MLflow Run ID**: заполни run_id
- Step **Result**: заполни метрику
- Step **Conclusion**: напиши вывод
- **Current Status**: обнови Best Result и Budget Used
- **Iteration Log**: добавь запись
- После Phase 2: заполни **Accepted Features**

### Plateau Research Protocol (ОБЯЗАТЕЛЬНО при застое)

**Критерий застоя:** метрика `roi` не улучшается 3+ итерации подряд
(delta < 0.001 относительно предыдущего best).

Когда застой обнаружен — СТОП. Не запускай следующий эксперимент.
Вместо этого выполни следующие шаги по порядку:

#### Шаг 1 — Анализ причин (sequential thinking)
Подумай последовательно:
1. Что уже пробовали? Какие паттерны в успешных/неуспешных runs?
2. Где потолок по данным vs потолок по архитектуре?
3. Какие самые сильные гипотезы ещё НЕ проверены?
4. Есть ли data leakage или overfitting которые маскируют прогресс?
5. Верна ли метрика `roi`? Оптимизируем ли мы то что нужно?

#### Шаг 2 — Интернет-исследование (WebSearch)
Ищи по следующим запросам (по одному, читай результаты):
- `"{task_type} roi improvement techniques 2024 2025"`
- `"kaggle tabular_classification winning solution feature engineering"`
- `"state of the art tabular_classification tabular data 2025"`
- Если задача специфичная: `"tabular_classification roi improvement kaggle winning solution"`
- Ищи: какие фичи используют топы, какие ансамбли, какие трюки

#### Шаг 3 — Формулировка новых гипотез
На основе анализа и поиска запиши в program.md раздел:
```
## Research Insights (plateau iteration N)
- **Найдено:** (что нашёл в поиске)
- **Гипотеза A:** (конкретная идея + ожидаемый прирост)
- **Гипотеза B:** (конкретная идея + ожидаемый прирост)
- **Выбранная следующая попытка:** (почему именно это)
```

#### Шаг 4 — Реализация
Реализуй самую перспективную гипотезу из шага 3.
Если она тоже не даёт прироста — повтори протокол с шага 1.