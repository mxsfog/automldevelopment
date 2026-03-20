# Analysis and Findings

## Baseline Performance
- DummyClassifier: ROI=-3.07% — ставки без модели убыточны
- Rule-based (ML_Edge>=11.5): ROI=-4.89% — простые правила хуже случайного
- LogisticRegression: ROI=3.83%, AUC=0.7947 — первый положительный ROI, линейная модель уже полезна
- CatBoost default: ROI=0.16%, AUC=0.7981 — без оптимизации порогов CatBoost не лучше

Вывод: базовые модели без EV-фильтрации и feature engineering дают ROI около 0%.

## Feature Engineering Results

### Принятые фичи (33 total)
- **Base (8):** Odds, USD, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, Outcomes_Count, Is_Parlay_bool
- **Engineered (10):** log_odds, implied_prob, value_ratio, edge_x_ev, edge_abs, ev_positive, model_implied_diff, log_usd, log_usd_per_outcome, parlay_complexity
- **ELO (15):** team_elo_mean, elo_diff, team_winrate_diff, elo_spread и др.

### Отклонённые фичи
- **Temporal (5):** hour, day_of_week, is_weekend, cyclic encodings — AUC -0.005, ROI -8-12pp. Время события не несёт предиктивной информации.
- **Interactions (12):** wr_diff_x_odds, odds_x_elo_mean и др. — AUC +0.01, но ROI -1pp. CatBoost сам находит нужные interactions через splits.

### Ключевой инсайт
ELO-фичи дают основной прирост: team_winrate_diff — top-1 по feature importance. Фильтр has_elo==1.0 критичен: модель работает только на матчах с историей ELO.

## Model Comparison
- **CatBoost solo:** лучший результат. Optimized params из chain_5 (depth=7, lr=0.05, l2=5.0).
- **Stacking (CB+LGB+XGB):** ROI=49.14% — хуже solo CatBoost. Причина: данных мало (~5000 filtered samples), 3-way split уменьшает train слишком сильно.
- **LightGBM/XGBoost solo:** в stacking эксперименте показали AUC на уровне CatBoost, но ROI ниже из-за худшей калибровки.
- **Weighted average:** не улучшает vs CatBoost solo.

Вывод: на малых данных (5000 samples) одиночный CatBoost оптимален.

## Segment Analysis

### Прибыльные спорты (после фильтрации)
Модель обучается и предсказывает только на матчах с ELO, исключая Basketball, MMA, FIFA, Snooker. Оставшиеся спорты (Tennis, Football, Hockey, Baseball и др.) дают стабильно положительный ROI.

### Per-sport EV thresholds
Разные спорты требуют разных EV порогов. PS_floor15 стратегия подбирает оптимальный порог для каждого спорта на validation split с шагом 0.005, минимум floor=0.15. Это даёт +5-13pp ROI vs единый порог EV>=0.10.

### Odds диапазоны
Odds range фильтры (1.1-5.0, 1.2-3.5 и т.д.) не улучшают ROI поверх per-sport EV thresholds.

## Stability and Validity

### Multi-seed stability (10 seeds)
- ROI avg=61.30%, std=2.92%
- Все 10 seeds дают ROI > 55%
- AUC avg=0.8662, std=0.0026

### 5-fold expanding window CV
- PS_floor15: avg=36.79%, std=22.61%, 4/5 folds positive
- PS_floor10: avg=34.74%, std=18.82%, 5/5 folds positive
- CV ROI ниже test ROI — ожидаемо, т.к. ранние folds имеют меньше training data

### Bootstrap CI (1000 samples)
- 95% CI: [47.16%, 84.53%]
- Median: 66.62%
- Нижняя граница значительно выше 10% target

### Anti-leakage проверки
- Thresholds подбираются только на val (последние 20% train)
- Test используется один раз для финальной оценки
- Нет future leakage: все фичи доступны до события
- ELO-фичи рассчитаны на исторических данных (без lookahead)

## What Didn't Work

1. **Temporal features** (step 4.1): час, день недели, выходные — чистый шум. AUC и ROI падают.
2. **Stacking ensemble** (step 4.2): CB+LGB+XGB с LogReg meta-learner. Данных слишком мало для 3-way split. Solo CatBoost лучше.
3. **Interaction features** (step 4.5): 12 попарных комбинаций top фичей. AUC +0.01 но ROI не улучшился — CatBoost и так находит нужные splits.
4. **Odds-stratified EV** (step 4.3): разные EV пороги для favorites/balanced/underdogs — хуже per-sport подхода.
5. **EV+p sweep** (step 4.3): оптимизация единого EV и p порога — лучший единый порог EV=0.24+p=0.73 даёт ROI=62.80%, но PS_floor15 стабильнее на CV.
