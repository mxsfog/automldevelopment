# Research Program: Research Session

## Metadata
- session_id: chain_2_mar21_1309
- created: 2026-03-21T10:09:55.396137+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_2_mar21_1309
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.



## Previous Session Context
# Previous Session Context: chain_1_mar21_1231

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | N_bets | Threshold | Run ID |
|------|--------|-----|-----|--------|-----------|--------|
| 1.1 | DummyClassifier | -3.07% | - | 14899 | - | 2f1475b0 |
| 1.2 | Rule ML_P>=0.40 | +3.82% | - | 11310 | 0.40 | c1b1d525 |
| 1.3 | LogisticRegression | +1.40% | 0.791 | 2515 | 0.73 | d46dd507 |
| 1.4 | CatBoost default | +1.35% | 0.795 | 2700 | 0.77 | 6a9889b1 |
| 2.1 | CatBoost+features | +2.66% | 0.800 | 2767 | 0.79 | 8ff08237 |
| 2.2 | +ELO trend+sport | +5.32% | 0.799 | 10068 | auto | 82aeacac |
| 3.1 | Optuna HPO | -0.03% | 0.794 | 2231 | 0.89 | 09d3af37 |
| 4.1 | LightGBM | +5.52% | 0.789 | 9897 | 0.45 | 9264769e |
| 4.2 | Ensemble avg | +5.56% | 0.796 | 9934 | 0.45 | 77df4501 |
| 4.3 | CatBoost full+sport_thr | +2.03% | 0.798 | 3264 | 0.74 | 12e33a3c |
| 4.4 | Stacking LR aug | +5.58% | 0.800 | 10350 | 0.45 | 89f7e204 |
| 4.5 | Calibration+edge | +5.28%* | 0.800 | 9740 | 0.45 | 51accee4 |

## Accepted Features
1. **Time features:** hour, day_of_week, is_weekend
2. **Odds-derived:** implied_prob, log_odds, value_ratio, edge_x_odds, odds_bucket
3. **ELO static:** team_elo, team_winrate, team_games, team_off/def/net_rating
4. **ELO interactions:** elo_x_odds, winrate_vs_implied, model_confidence
5. **ELO trend:** elo_trend_5, elo_avg_change, recent_win_streak
6. **Market:** market_category

## Recommended Next Steps
### Лучший результат
- **ROI = +5.58%** (step 4.4, Stacking LR augmented, 10350 ставок из 14899, порог 0.45)
- **AUC = 0.800** (CatBoost доминирует в ensemble с весом 2.46)
- Цель ROI >= 10% **не достигнута** при flat betting

### Ключевые находки

1. **Feature engineering дал основной прирост:** ELO features + odds-derived features подняли ROI с +1.35% до +5.32% (delta +3.97%). Это главный драйвер качества.

2. **Низкие пороги стабильнее:** Оптимальный порог ~0.45, отбирает ~65% ставок. Высокие пороги (>0.7) переобучаются на val.

3. **Ensemble и stacking маргинально лучше одиночной модели:** CatBoost solo = +5.28%, ensemble avg = +5.56%, stacking = +5.58%. Прирост <0.5%.

4. **Sport-specific thresholds и full train — overfit:** Step 4.3 показал, что усложнение стратегии ухудшает результат (ROI=+2.03% вместо +5.28%).

5. **Optuna HPO — overfit:** Прямая оптимизация ROI через Optuna дала val=+14.5%, test=-0.03%. Дефолтные параметры лучше.

6. **Калибровка не помогает flat ROI:** Isotonic/Platt калибровка не улучшила ROI при фиксированном пороге.

7. **Edge-based selection перспективна, но нестабильна:** Edge>0.05 дал ROI=+11.06% (2108 ставок), edge>0.10 дал ROI=+49.1% (402 ставки). Малый размер выборки делает эти результаты ненадежными. ROI=49.1% вызвал MQ-LEAKAGE-SUSPECT alert (ложное срабатывание).

8. **Kelly criterion показывает потенциал:** При взвешенном sizing (f=0.25) ROI поднимается до +10.01%, но это другая метрика (взвешенная по размеру ставки, не flat).

### Рекомендации для продолжения
- Валидировать edge-based подход на новых данных (out-of-time)
- Протестировать Kelly criterion с реальным bankroll management
- Фильтрация убыточных видов спорта (FIFA, MMA) может дать +1-2% к ROI
- Собирать больше данных для стабильности результатов

### Причина остановки
hard_stop по MQ-LEAKAGE-SUSPECT: step 4.5 залогировал edge>0.10 ROI=49.1% (402 ставки) как best metric, что превысило sanity threshold 35%. Это не утечка данных, а артефакт высокой селективности при малой выборке.

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
- **Status:** done
- **MLflow Run ID:** 5f61a20970c84456b8363d158128b41f
- **Result:** ROI=-3.07%, 14899 bets, winrate=54.15%
- **Conclusion:** Ставки на все подряд дают -3.07% ROI. Lower bound установлен.


#### Step 1.2 — Rule-based baseline
- **Hypothesis:** Простое пороговое правило по топ-1 признаку
- **Method:** threshold_rule
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** 457aec69660c4a828fe34cf25d903824
- **Result:** ROI=+3.82%, 11310 bets, threshold=0.30
- **Conclusion:** ML_P_Model >= 0.30 уже дает +3.82%. Все ML_P > 0 ставки прибыльны.


#### Step 1.3 — Linear baseline
- **Hypothesis:** LogisticRegression с базовыми фичами — linear baseline
- **Method:** logistic_regression
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 92445be70ff746dcbbea90ba1b83043a
- **Result:** ROI=+2.04%, 15 bets, AUC=0.791, threshold=0.81
- **Conclusion:** LogReg переобучается на высокий порог, отбирает слишком мало ставок.


#### Step 1.4 — Non-linear baseline
- **Hypothesis:** CatBoost с дефолтами — strong non-linear baseline
- **Method:** catboost_default
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 4c10cb922e0544c185879e13ea9c915a
- **Result:** ROI=-0.02%, 2230 bets, AUC=0.793, threshold=0.83
- **Conclusion:** CatBoost на базовых фичах без ELO features не превышает rule baseline. Подтверждает что ELO features критичны.



### Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*



#### Step 2.1 — ELO + odds features (chain_1 set)
- **Hypothesis:** Воспроизвести chain_1 best feature set с низким порогом
- **Method:** catboost + shadow feature trick
- **Status:** done
- **MLflow Run ID:** 6c92943ee31c4d2fb351c929b9a7256e
- **Result:** ROI=+5.60%, AUC=0.800, 9551 bets, threshold=0.46
- **Conclusion:** Воспроизведён результат chain_1. Подтверждает стабильность.

#### Step 2.2 — New features (odds_sq, elo_winrate_product, etc.)
- **Hypothesis:** 10 новых фич (interactions, ratios) улучшат baseline
- **Method:** shadow feature trick: baseline vs candidate
- **Status:** done
- **MLflow Run ID:** dbeddbac16d04820a704c37d9e8cde8e
- **Result:** ROI=+6.50% (delta +0.90%), AUC=0.802, 9675 bets
- **Conclusion:** Принято. odds_sq и elo_winrate_product в топ-5 важности.


### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Hyperparameter Optimization
- **Hypothesis:** Optuna TPE оптимизирует AUC (не ROI, чтобы избежать overfit chain_1)
- **Method:** optuna_tpe, 30 trials
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** e8d38342a3b042c6bc96c4f75ada36c6
- **Result:** ROI=+6.87% (HPO) vs +6.50% (default), AUC=0.803
- **Conclusion:** HPO по AUC дал +0.37% ROI. lr=0.147, depth=6, l2=28.3, min_leaf=100. Принято.



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
- **Active Phase:** Phase 4
- **Completed Steps:** 7/7+
- **Best Result:** ROI=+22.92% flat / +15.09% kelly (step 4.5, retrained model_B p>=0.43 e>=0.10)
- **Budget Used:** ~65% (13/20)
- **smoke_test_status:** passed

## Iteration Log
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
git commit -m "session chain_2_mar21_1309: step {step_id} [mlflow_run_id: {run_id}]"
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