# Research Program: Research Session

## Metadata
- session_id: sports_10h
- created: 2026-03-20T09:11:51.652383+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/sports_10h
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.


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
- **Status:** done
- **MLflow Run ID:** 15193895325f4dc6aebb47d8eebe961b
- **Result:** roi_mean=-2.1120, roi_std=4.3186, accuracy=0.5459
- **Conclusion:** DummyClassifier (always predict "won") даёт ROI=-2.11%. Majority class = "won" (53.9-55.6%). Это lower bound — любая модель должна бить этот уровень.


#### Step 1.2 — Rule-based baseline
- **Hypothesis:** Простое пороговое правило по топ-1 признаку
- **Method:** threshold_rule
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** 2d9ec63bfdd041bf9d0bb35637eb54da
- **Result:** best_threshold=0, roi_mean=-2.12%, coverage=31.7%
- **Conclusion:** ML_Edge > 0 даёт ROI=-2.12%, не лучше dummy. Платформенный ML_Edge не содержит полезного сигнала для фильтрации ставок. Увеличение порога снижает coverage и ухудшает ROI.


#### Step 1.3 — Linear baseline
- **Hypothesis:** LogisticRegression с базовыми фичами — linear baseline
- **Method:** logistic_regression
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** f5bc9b706744477b8c0672c31e449101
- **Result:** roi_mean=1.88% (threshold=0.65), auc_mean=0.758
- **Conclusion:** LogisticRegression с 7 фичами (Odds, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, Is_Parlay, Outcomes_Count) даёт ROI=+1.88% при порогe 0.65. AUC=0.758 стабилен. Первый положительный ROI — модель начинает отбирать прибыльные ставки.


#### Step 1.4 — Non-linear baseline
- **Hypothesis:** CatBoost с дефолтами — strong non-linear baseline
- **Method:** catboost_default
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 89387127aa6c4166a7482cd18ee2b7d7
- **Result:** roi_mean=2.81% (threshold=0.55), auc_mean=0.766
- **Conclusion:** CatBoost с 10 фичами (7 num + 3 cat) даёт ROI=+2.81%. Feature importance: Odds доминирует (80.8%), ML_P_Implied (9.1%), Sport (3.9%). Market и is_parlay не используются (importance=0). Лучший результат Phase 1.



### Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*



#### Step 2.1-2.5 — Shadow Feature Tests
- **Groups tested:** implied_value, time_features, odds_transforms, sport_market_winrate, elo_features
- **Status:** done
- **MLflow Run ID:** 80d6e1a6319e4391b692b5ec800cd1ad
- **Result:** Все 5 групп отклонены (delta < 0 для всех)
- **Conclusion:** Базовый feature set CatBoost (Odds, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, Outcomes_Count, USD, Sport, Market, Is_Parlay) оптимален. Дополнительные фичи (implied_prob, value_gap, hour/dow, log_odds, sport_winrate, ELO) не улучшают ROI по shadow feature trick.


### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Hyperparameter Optimization
- **Hypothesis:** Optuna TPE найдёт лучшие гиперпараметры
- **Method:** optuna_tpe
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** a93d5fab03494b2ca4f020cf57ff1ab6
- **Result:** roi_mean=2.52% (threshold=0.57), auc_mean=0.764, 40 trials
- **Conclusion:** Optuna нашла конфигурацию с roi=2.52%, что не превышает дефолтный CatBoost (2.81%). Лучшие параметры: depth=5, lr=0.25, iterations=553, без class_weights. Высокая дисперсия по фолдам (от -2.3% до +8.5%) указывает на нестабильность ROI-метрики и ограниченный сигнал в данных.

## Current Status
- **Active Phase:** Phase 1
- **Completed Steps:** 6/6 (Phase 1 done, Phase 2 done, Phase 3 done)
- **Best Result:** roi_mean=+2.81% (CatBoost default, threshold=0.55)
- **Budget Used:** 20%
- **smoke_test_status:** pending

## Iteration Log
1. **Step 1.1** (DummyClassifier): roi_mean=-2.11%, roi_std=4.32%, acc=0.5459. Run: 15193895325f4dc6aebb47d8eebe961b
2. **Step 1.2** (Rule-based ML_Edge>0): roi_mean=-2.12%, coverage=31.7%. Run: 2d9ec63bfdd041bf9d0bb35637eb54da
3. **Step 1.3** (LogisticRegression): roi_mean=+1.88%, auc=0.758, threshold=0.65. Run: f5bc9b706744477b8c0672c31e449101
4. **Step 1.4** (CatBoost default): roi_mean=+2.81%, auc=0.766, threshold=0.55. Run: 89387127aa6c4166a7482cd18ee2b7d7
5. **Phase 2** (Shadow Feature Trick): 5 групп протестировано, 0 принято. Feature set не изменён. Run: 80d6e1a6319e4391b692b5ec800cd1ad
6. **Step 3.1** (Optuna TPE, 40 trials): roi_mean=+2.52%, auc=0.764. Не превзошёл дефолтный CatBoost. Run: a93d5fab03494b2ca4f020cf57ff1ab6

## Accepted Features
Базовый feature set (из Phase 1, без изменений после Phase 2):
- **Числовые:** Odds, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, Outcomes_Count, USD
- **Категориальные:** Sport, Market, Is_Parlay

## Final Conclusions

### Результаты
| Step | Модель | ROI mean | AUC | Threshold |
|------|--------|----------|-----|-----------|
| 1.1 | DummyClassifier | -2.11% | - | - |
| 1.2 | Rule-based (ML_Edge>0) | -2.12% | - | 0 |
| 1.3 | LogisticRegression | +1.88% | 0.758 | 0.65 |
| 1.4 | CatBoost (default) | **+2.81%** | 0.766 | 0.55 |
| 3.1 | CatBoost (Optuna) | +2.52% | 0.764 | 0.57 |

### Лучшая модель
CatBoost с дефолтными параметрами (iterations=500, depth=6, lr=0.1, auto_class_weights=Balanced) при threshold=0.55.

### Ключевые наблюдения
1. **Odds — доминирующий признак** (80.8% feature importance). Модель фактически учится фильтровать ставки по коэффициентам.
2. **ML_P_Implied — второй по важности** (9.1%). Вычисленная платформой implied probability содержит дополнительный сигнал.
3. **Feature engineering не помог.** Все 5 групп новых фичей (implied_value, time, odds_transforms, sport_winrate, ELO) ухудшили ROI по shadow feature trick.
4. **Optuna не улучшила дефолты.** 40 trials оптимизации не превзошли baseline CatBoost — сигнал в данных ограничен.
5. **ROI = +2.81% < цели 10%.** Текущий feature set не содержит достаточно информации для достижения целевого ROI.
6. **Парлаи убыточны** (ROI=-19.6%), синглы прибыльны (ROI=+2.3%). Фильтрация по is_parlay — простой, но эффективный шаг.
7. **Высокая дисперсия ROI по фолдам** (от -6.9% до +8.5%) указывает на нестабильность метрики.

### Рекомендации для дальнейшей работы
1. Сегментный анализ: отдельные модели для разных Sport (Soccer, Tennis, Basketball)
2. Фильтрация парлаев: ставить только на синглы (ROI +2.3% vs -19.6%)
3. Расширение данных: больше исторических данных для стабилизации ROI
4. Альтернативные ML-модели: LightGBM, XGBoost, ensemble
5. Калибровка вероятностей: Platt scaling или isotonic regression для лучшего threshold selection

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

### Budget Check (перед каждым экспериментом)
```python
import json
budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        # Завершить текущую работу, написать выводы, остановиться
        mlflow.set_tag("status", "budget_stopped")
        sys.exit(0)
except FileNotFoundError:
    pass  # файл ещё не создан
```

### DVC Protocol
После завершения каждого шага:
```bash
git add .
git commit -m "session sports_10h: step {step_id} [mlflow_run_id: {run_id}]"
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

### Update program.md
После каждого шага обновляй:
- Step **Status**: pending -> done/failed
- Step **MLflow Run ID**: заполни run_id
- Step **Result**: заполни метрику
- Step **Conclusion**: напиши вывод
- **Current Status**: обнови Best Result и Budget Used
- **Iteration Log**: добавь запись
- После Phase 2: заполни **Accepted Features**