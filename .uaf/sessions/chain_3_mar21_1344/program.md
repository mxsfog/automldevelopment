# Research Program: Research Session

## Metadata
- session_id: chain_3_mar21_1344
- created: 2026-03-21T10:44:23.218050+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_3_mar21_1344
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.



## Previous Session Context
# Previous Session Context: chain_2_mar21_1309

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | N_bets | Threshold | Run ID |
|------|--------|-----|-----|--------|-----------|--------|
| 1.1 | DummyClassifier | -3.07% | - | 14899 | - | 5f61a209 |
| 1.2 | Rule ML_P>=0.30 | +3.82% | - | 11310 | 0.30 | 457aec69 |
| 1.3 | LogisticRegression | +2.04% | 0.791 | 15 | 0.81 | 92445be7 |
| 1.4 | CatBoost basic | -0.02% | 0.793 | 2230 | 0.83 | 4c10cb92 |
| 2.1 | CatBoost+ELO+odds | +5.60% | 0.800 | 9551 | 0.46 | 6c92943e |
| 2.2 | +new features | +6.50% | 0.802 | 9675 | 0.46 | dbeddbac |
| 3.1 | Optuna HPO (AUC) | +6.87% | 0.803 | 9318 | 0.47 | e8d38342 |
| 4.1 | Stacking CB+LGB+XGB | +6.52% | 0.806 | 10096 | 0.44 | a34d6e62 |
| 4.2 | Edge+sport scan* | +15.73%* | 0.804 | 1167 | 0.45+e0.10 | 84f8c52f |
| 4.3 | Edge val-tuned | +14.82% | 0.803 | 1181 | p0.43+e0.10 | d987968b |
| 4.4 | Calibr+stability | +14.78% | 0.804 | 1185 | p0.42+e0.10 | db226171 |
| 4.5 | Kelly+retrain | +22.92%* | 0.798 | 1131 | p0.43+e0.10 | dcd0ba70 |
| 4.6 | Multi-edge 3models | +9.78% | 0.804 | 881 | stable-best | 9f93f294 |
| 4.7 | Q2 analysis+odds | +9.79% | 0.804 | 1806 | p0.43+e0.07 | 16d4cf1c |

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
(заполняется Claude Code по завершении)

---




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


### Phase 1: Baseline (MANDATORY)
**Goal:** Установить нижнюю границу и strong baseline
**Success Criterion:** Превысить random baseline по roi


#### Step 1.1 — Constant baseline
- **Hypothesis:** DummyClassifier (most_frequent) задаёт lower bound
- **Method:** dummy_classifier
- **Metric:** roi
- **Critical:** true
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** (заполняется Claude Code)
- **Conclusion:** (заполняется Claude Code)


#### Step 1.2 — Rule-based baseline
- **Hypothesis:** Простое пороговое правило по топ-1 признаку
- **Method:** threshold_rule
- **Metric:** roi
- **Critical:** false
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** (заполняется Claude Code)
- **Conclusion:** (заполняется Claude Code)


#### Step 1.3 — Linear baseline
- **Hypothesis:** LogisticRegression с базовыми фичами — linear baseline
- **Method:** logistic_regression
- **Metric:** roi
- **Critical:** true
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** (заполняется Claude Code)
- **Conclusion:** (заполняется Claude Code)


#### Step 1.4 — Non-linear baseline
- **Hypothesis:** CatBoost с дефолтами — strong non-linear baseline
- **Method:** catboost_default
- **Metric:** roi
- **Critical:** true
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** (заполняется Claude Code)
- **Conclusion:** (заполняется Claude Code)



### Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*



*Гипотезы о фичах будут сгенерированы Claude Code на основе data_schema.json*
*после завершения Phase 1 (max 5 шагов)*


### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Hyperparameter Optimization
- **Hypothesis:** Optuna TPE найдёт лучшие гиперпараметры
- **Method:** optuna_tpe
- **Metric:** roi
- **Critical:** false
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** null
- **Conclusion:** null



### Phase 4: Free Exploration (до hard_stop)
*Начинается после Phase 3. Продолжается пока budget_status.json не содержит hard_stop: true.*
*Это основная фаза — она занимает большую часть бюджета.*

После Phase 3 НЕ завершай работу. Продолжай генерировать и проверять гипотезы:

**Направления для свободного исследования (в порядке приоритета):**
1. Ансамбли: VotingClassifier, StackingClassifier (CatBoost + LightGBM + XGBoost)
2. Threshold optimization: подбор порога вероятности для максимизации roi
3. Новые фичи: взаимодействия, ratio-фичи, временные паттерны
4. Калибровка вероятностей: CalibratedClassifierCV
5. Сегментация: отдельные модели по Sport/Market/Is_Parlay
6. Дополнительные данные: поиск публичных датасетов (WebSearch) для обогащения

Каждая гипотеза Phase 4 оформляется как Step 4.N в Iteration Log.
При застое 3+ итераций — Plateau Research Protocol обязателен.

## Current Status
- **Active Phase:** Phase 1
- **Completed Steps:** 0/5
- **Best Result:** null
- **Budget Used:** 0%
- **smoke_test_status:** pending

## Iteration Log
(заполняется Claude Code после каждой итерации)

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

### Code Quality
После создания каждого Python-файла:
```bash
ruff format {filepath}
ruff check {filepath} --fix
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
   Проверь: нет ли колонок которые появляются ПОСЛЕ события (Payout_USD, финальный счёт).

4. **Санитарная проверка**: если roi > 35.0 — это почти наверняка leakage.
   Остановись, найди причину, исправь до продолжения.
   UAF BudgetController автоматически отклонит результат с алертом MQ-LEAKAGE-SUSPECT.

### Model Artifact Protocol (ОБЯЗАТЕЛЬНО для chain continuation)

В конце ЛЮБОГО эксперимента, который устанавливает новый лучший roi:

Сохраняй модель в `./models/best/` (относительно SESSION_DIR):

```python
import json
from pathlib import Path

Path("./models/best").mkdir(parents=True, exist_ok=True)

# Выбери формат по фреймворку:
# CatBoost:  model.save_model("./models/best/model.cbm")
# LightGBM:  booster.save_model("./models/best/model.lgb")
# XGBoost:   model.save_model("./models/best/model.xgb")
# sklearn:   import joblib; joblib.dump(pipeline, "./models/best/model.pkl")

import json, os
metadata = {
    "framework": "catboost",   # catboost | lgbm | xgboost | sklearn
    "model_file": "model.cbm",
    "roi": ...,         # best ROI value (float)
    "auc": ...,         # AUC (float)
    "threshold": ...,   # prediction threshold (float)
    "n_bets": ...,      # number of bets selected
    "feature_names": [...],  # list[str]
    "params": {...},          # hyperparameters dict
    "sport_filter": [...],    # list[str] — sports to EXCLUDE (if any), or []
    "session_id": os.environ["UAF_SESSION_ID"],
}
with open("./models/best/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

Это позволяет следующей сессии в цепочке загрузить модель и начать сразу с Phase 4.

### DVC Protocol
После завершения каждого шага:
```bash
git add .
git commit -m "session chain_3_mar21_1344: step {step_id} [mlflow_run_id: {run_id}]"
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
[прибыльные сегменты: спорт, рынки, odds диапазоны]

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