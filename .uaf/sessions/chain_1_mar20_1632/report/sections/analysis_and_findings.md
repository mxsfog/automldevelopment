# Analysis and Findings

## Baseline Performance
Без ML все ставки дают ROI = -3.07% (bet_all baseline). Random selection: -5.97%. Правило ML_Edge >= 8 показало val ROI=15% vs test ROI=-3.53% -- классический overfitting. LogisticRegression: ROI=-1.40%, AUC=0.7913. CatBoost default: ROI=-2.86%, AUC=0.7938. Все baseline отрицательные -- задача нетривиальная.

## Feature Engineering Results

### Что помогло
- **Odds-derived features** (log_odds, implied_prob, value_ratio): приняты через shadow feature trick, delta ROI = +2.05%
- **Interaction features** (edge_x_ev, model_implied_diff): улучшают discrimination
- **Parlay complexity** (Outcomes_Count * Is_Parlay): полезна для разделения типов ставок

### Что не помогло
- **Target encoding** (sport_winrate, market_winrate): REJECTED, delta ROI = -1.09%. market_winrate доминировала в importance (50%) -- утечка
- **Temporal features** (hour, day_of_week, is_weekend): REJECTED, delta ROI = -3.44%. Шумные и не несут сигнала для ROI
- **Kelly fraction, odds_squared, model_confidence**: REJECTED, не улучшают модель
- **Sport как categorical feature**: маргинальный эффект +0.03%, по сути бесполезно

### Feature Importance
Из CatBoost feature importance: implied_prob (31%), Odds (28%), log_odds (26%), ML_P_Implied (11%). Остальные фичи -- менее 1% каждая. Модель преимущественно работает с odds-структурой.

## Model Comparison

| Модель | ROI (filtered) | AUC | Threshold | Notes |
|--------|---------------|-----|-----------|-------|
| CatBoost single | 2.66% | 0.7945 | 0.65 | После Optuna |
| Ensemble avg (CB+LGB+XGB) | 7.23% | 0.8095 | 0.60 | Лучший подход |
| Weighted ensemble (0.7/0.15/0.15) | 3.13% | 0.7939 | 0.60 | Overfitting весов |
| Stacking (LogReg meta) | 1.89% | - | - | Хуже среднего |
| CatBoost deeper (depth=5) | 2.23% | 0.8116 | 0.80 | Early stop iter 59 |
| CatBoost-cat (Sport) | 7.26% | 0.8097 | 0.60 | Маргинально лучше |

Equal average ensemble стабильно лучше любых весов и stacking. Причина: на шумных данных простое усреднение более робастно.

## Segment Analysis

### По спортам (test set, t=0.60)
| Спорт | ROI | N ставок |
|-------|-----|----------|
| Rugby | +42% | ~50 |
| Boxing | +24% | ~30 |
| Dota 2 | +19% | ~200 |
| Ice Hockey | +13% | ~500 |
| Soccer | +7.5% | ~3000 |
| CS:GO | -5% | ~300 |
| Basketball | -26% | ~800 |
| MMA | -41% | ~100 |
| FIFA | -19% | ~50 |

Исключение Basketball, MMA, FIFA, Snooker дало прирост с 3.20% до 7.23% (+4.03 п.п.).

### По диапазонам Odds
Лучший сегмент: Odds [1.5, 1.8) -- ROI=15.57% при n=1338. Но использование per-odds-range пороговOptimization переобучается на val и не переносится на test.

### По типу ставки
Singles (Is_Parlay='f') стабильнее parlays, но разница в рамках фильтрованного набора невелика.

## Stability & Validity

### Anti-Leakage
- Threshold выбирается на val (последние 20% train), применяется к test однократно
- Target encoding отвергнут из-за data leakage
- Time-series split исключает look-ahead bias
- Все ROI > 30% в segment analysis -- это малые выборки (n<100), не используются для принятия решений

### Стабильность
- Val-selected threshold (0.60) стабилен: coarse grid дает его в 8 из 8 экспериментов
- Fine val grid (0.01 step) переобучается: выбирает t=0.64 с ROI=6.81% (хуже)
- ROI на test устойчив: 7.23% +/- 0.03% через 8 итераций Phase 4
- Однако test set -- всего 3 дня. Долгосрочная стабильность не проверена

### Ограничения
- Test window: 2026-02-20 -- 2026-02-22 (3 дня) -- слишком короткий для надежной оценки
- ML_Team_Stats_Found всегда 'f' -- данные ELO/рейтингов не связаны со ставками
- ML_Winrate_Diff, ML_Rating_Diff: 100% NaN -- бесполезны

## What Didn't Work

1. **Target encoding** (sport/market winrate): утечка через target statistics, доминирует в importance но не генерализует
2. **Weighted ensemble**: оптимизация весов на val переобучается (CB=0.55/LGB=0.05/XGB=0.40 дал t=0.85 -> ROI=-0.36%)
3. **Stacking (LogReg meta-learner)**: 1.89% -- хуже простого среднего
4. **EV-based selection**: val ROI=60.82% vs test ROI=-27.65% -- катастрофическое переобучение
5. **Per-odds-range thresholds**: overfit на val, test ROI=1.77%
6. **Training only on good sports**: меньше данных = хуже обобщение (ROI=2.26%)
7. **Deeper CatBoost (depth=5)**: early stop iter 59, ROI=2.23% -- overfitting
8. **Temporal features**: hour/day_of_week -- шум, не помогают
9. **Isotonic regression calibration**: маргинальный эффект на AUC (+0.0001), не влияет на ROI при val threshold
