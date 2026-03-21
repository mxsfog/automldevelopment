# Previous Session Context: chain_2_mar21_1432

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
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

## Recommended Next Steps
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
