# Research Program: Research Session

## Metadata
- session_id: chain_2_mar21_1432
- created: 2026-03-21T11:33:19.148824+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_2_mar21_1432
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.



## Previous Session Context
# Previous Session Context: chain_1_mar21_1356

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | N_bets | MLflow Run | Status |
|------|--------|-----|--------|------------|--------|
| 1.1 | DummyClassifier | -1.96% | 14899 | 109416be6327 | done |
| 1.2 | Rule: ML_P_Model>=60 | +0.90% | 4353 | 7df7764ac6a4 | done |
| 1.3 | LogisticRegression thr=0.65 | +1.59% | 5524 | 577151579be4 | done |
| 1.4 | CatBoost default thr=0.60 | +0.80% | 5697 | 30a23de9af7d | done |
| 2.x | FE: Sport/Market enc accepted | +1.31% | 6902 | 42c6c09ee6c7 | done |
| 3.1 | Optuna CatBoost thr=0.55 | +1.47% | 6005 | 4ef3c43d8a85 | done |
| 4.1 | Threshold+Segments thr=0.70 | -0.99% | 1732 | 721d58c44c8e | done |
| 4.2 | EV Ensemble EV>=0.12 | +7.82% | 2535 | a4f8794503f4 | done |
| 4.3 | EV+Sport filter | -18.07% | 1495 | 111c2182dbe0 | done (sport filter hurts) |
| 4.4 | XGB+4model+extfeats | -3.07% | 3823 | de635e493a51 | done (extra feats=noise) |
| 4.5 | Full train ensemble EV>=0.12 | +16.02% | 2247 | 5914f58ebc3c | done (best result, CV mean=18.73% std=14.14%) |
| 4.6 | Stability: odds cap + EV grid | +30.84% | 898 | 16afcfc98d04 | done (EV>=0.25 no_cap volatile, odds cap kills ROI) |

## Accepted Features
Base (15): Odds, USD, Is_Parlay, Outcomes_Count, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, ML_Winrate_Diff, ML_Rating_Diff, Outcome_Odds, n_outcomes, mean_outcome_odds, max_outcome_odds, min_outcome_odds
+ Sport_target_enc, Sport_count_enc, Market_target_enc, Market_count_enc
Total: 19 features

## Recommended Next Steps
### Лучший результат
**ROI = +16.02%** (тест, n=2247) — цель 10% достигнута.

Стратегия: 3-model ensemble (CatBoost + LightGBM + LogReg) обученный на полном train без val split, отбор ставок по Expected Value >= 0.12 (EV = model_prob * odds - 1).

### Ключевые находки
1. **EV-based selection — главный прорыв.** Переход от probability threshold (ROI ~1.5%) к EV threshold (ROI ~16%) — самое значимое улучшение за всю сессию. Модель определяет, где букмекер недооценивает вероятность.

2. **Минимализм в фичах.** Из 5 групп feature engineering только Sport/Market encoding прошел проверку. Все попытки добавить фичи ухудшали результат. 19 фич — оптимум.

3. **ROI через high-odds value.** Прибыль приходит от ставок с высокими коэффициентами (avg_odds=33.9). При ограничении odds<=5 ROI падает до 2.6%. Стратегия рискованная, но прибыльная.

4. **Cross-validation подтверждает.** 5-fold CV: mean=18.73%, std=14.14%. 4/5 фолдов прибыльны. Последний фолд ~0% — возможный временной drift.

### Ограничения
- Высокая дисперсия (std=14.14%), один из фолдов на нуле
- Данные покрывают только 81 день — ограниченная временная глубина
- Средний коэффициент ~34 означает низкий win rate (~37%)
- Step 4.6 спровоцировал MQ-LEAKAGE-SUSPECT alert (CV fold ROI=81.63% > 35% threshold) — false positive, но указывает на экстремальную волатильность при EV>=0.25

### Рекомендации для следующей сессии
1. Исследовать калибровку вероятностей (Platt scaling, isotonic) для точного EV
2. Тестировать на более длинном периоде данных
3. Bankroll management (Kelly criterion) для контроля drawdown
4. Мониторинг drift — переобучение при деградации

---


## Chain Continuation Mode

**РЕЖИМ ПРОДОЛЖЕНИЯ ЦЕПОЧКИ.** Phases 1-3 ПРОПУСКАЮТСЯ.

- **Лучшая модель предыдущей сессии:** `/mnt/d/automl-research/.uaf/sessions/chain_1_mar21_1356/models/best`
- **Предыдущий лучший roi:** 16.02
- **Обязательное действие:** Step 4.0 — загрузить модель, верифицировать результат, затем Phase 4.

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
Best roi = **16.02**.

#### Step 4.0 — Chain Verification (ОБЯЗАТЕЛЬНЫЙ первый шаг)
- **Цель:** Убедиться что модель и среда работают корректно
- **Метод:**
  1. Загрузить metadata из `/mnt/d/automl-research/.uaf/sessions/chain_1_mar21_1356/models/best/metadata.json`
  2. Загрузить модель (формат указан в metadata.model_file)
  3. Загрузить данные, применить те же фичи и sport_filter что в metadata
  4. Вычислить roi на test — должен быть ≈ 16.02
  5. Залогировать в MLflow как "chain/verify"
- **Status:** done
- **MLflow Run ID:** 183c3ae0af8c46f98a6f0609ef89475f
- **Result:** ROI=16.02%, AUC=0.784, n_bets=2247, delta=0.00% — verified OK



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
- **Active Phase:** Phase 4 (complete — budget hard_stop)
- **Completed Steps:** 5 (4.0-4.4, 4.5 interrupted)
- **Best Result:** ROI=16.02% (unchanged from chain_1)
- **Budget Used:** 20% (hard_stop: MQ-LEAKAGE-SUSPECT false positive from step 4.4 regression val metric)
- **smoke_test_status:** passed

## Iteration Log

| Step | Method | ROI | N_bets | MLflow Run | Status |
|------|--------|-----|--------|------------|--------|
| 4.0 | Chain Verification | +16.02% | 2247 | 183c3ae0af8c | done (verified, delta=0.00%) |
| 4.1 | Calibrated EV + 4-model + EV grid | +14.04% | 2559 | 992f54b26653 | done (no improvement, calibration loosened threshold) |
| 4.2 | Stacking + Optuna CB + weighted | +9.00% | 3377 | cacbd6f5f1e8 | done (deep CB=worse EV calibration, simple model better) |
| 4.3 | Kelly + stratified EV | +16.02% | 2247 | 83c926074eaa | done (Kelly hurts, stratified=overfitted 52.94% on n=584) |
| 4.4 | Profit regression (CB+LGBM+Huber) | -5.36% | 6249 | 47db381fb18d | done (regression fails: skewed target, all approaches negative ROI on test) |

## Accepted Features
Base (15): Odds, USD, Is_Parlay, Outcomes_Count, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, ML_Winrate_Diff, ML_Rating_Diff, Outcome_Odds, n_outcomes, mean_outcome_odds, max_outcome_odds, min_outcome_odds
+ Sport_target_enc, Sport_count_enc, Market_target_enc, Market_count_enc
Total: 19 features

## Research Insights (plateau iteration 1)
- **Найдено:** Calibration > accuracy для ROI (arxiv 2303.06021: +34.69% vs -35.17%). Regression на profit используется в winning solutions. Cross-validated calibration робастнее prefit.
- **Гипотеза A:** Regression на profit (target=won*odds-1) — CatBoost regression вместо classification. Модель напрямую оптимизирует прибыльность. Ожидаемый прирост: 2-5% ROI.
- **Гипотеза B:** Cross-validated calibration (cv=5) вместо prefit — более robust калибровка вероятностей для EV. Ожидаемый прирост: 1-3% ROI.
- **Гипотеза C:** Two-stage model (classify won/lost + predict odds_error). Ожидаемый прирост: 1-2% ROI.
- **Выбранная следующая попытка:** A (regression на profit) — принципиально новый подход

## Final Conclusions

### Результат
ROI = **+16.02%** (не улучшен). Baseline из chain_1 оказался устойчив ко всем попыткам оптимизации.

### Что подтвердилось
1. **Simple ensemble is king.** 3-model average (CB+LGBM+LR) с простыми параметрами (depth=6, iter=200) — оптимальная архитектура. Усложнение ухудшает результат.
2. **EV-based selection работает.** Формула EV = p*odds - 1 >= 0.12 стабильно выбирает прибыльные ставки (CV mean=18.73%).
3. **Прибыль = high-odds value.** Стратегия находит ставки где букмекер недооценивает вероятность. Средний коэффициент отобранных ставок = 33.9.

### Что не работает (отрицательные результаты)
1. **Калибровка** — isotonic/Platt размывает вероятности, ухудшает EV selection
2. **Сложные модели** — Optuna CatBoost (depth=8) менее калиброван для EV
3. **Kelly criterion** — переносит вес с high-odds на low-odds, убивает ROI
4. **Profit regression** — skewed target (min=-1, max=+126) не поддаётся обучению
5. **Val-optimized thresholds** — не переносятся на test (temporal distribution shift)
6. **Weighted/stacking ensemble** — overfitting к val, нет improvement на test

### Рекомендации для следующей сессии
1. **Agreement-based selection** (step 4.5, прерван) — потенциально может снизить variance
2. **Больше данных** — 81 день мало для стабильной оценки, нужен 6+ месяцев
3. **Online learning** — rolling retrain window для адаптации к drift
4. **ELO features** — elo_history.csv не использован, может добавить signal

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
git commit -m "session chain_2_mar21_1432: step {step_id} [mlflow_run_id: {run_id}]"
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