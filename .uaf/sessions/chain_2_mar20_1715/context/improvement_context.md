# Previous Session Context: chain_1_mar20_1632

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | Threshold | N_bets | Notes |
|------|--------|-----|-----|-----------|--------|-------|
| 1.1 | DummyClassifier | -3.07% | - | - | 14899 | Lower bound |
| 1.2 | ML_Edge rule | -3.53% | - | Edge>=8 | 3000 | Overfitting val vs test |
| 1.3 | LogisticRegression | -1.40% | 0.7913 | 0.75 | 1061 | Best Phase 1 |
| 1.4 | CatBoost default | -2.86% | 0.7938 | 0.85 | 1642 | Early stop iter 40 |
| 2.1 | Shadow FE (full) | 0.71% | 0.7820 | 0.50 | 8010 | Rejected: target enc leakage |
| 2.2 | Shadow FE (safe) | -0.81% | 0.7937 | 0.80 | 1862 | Accepted: +2.05% delta |
| 3.1 | Optuna CatBoost | 2.66% | 0.7945 | 0.65 | 2564 | depth=3, lr=0.059, no cw |
| 4.1 | Stacking CB+LGB+XGB | 3.20% | 0.7923 | 0.60 | 5769 | avg best, stack 1.89% |
| 4.2 | Weighted avg+thr | 3.13% | 0.7939 | 0.60 | 5498 | w=0.7/0.15/0.15 |
| 4.3 | Segment analysis | 3.20% | - | 0.60 | 5769 | Dota+IceH+Soccer best |
| 4.4 | Segment filter | 7.23% | - | 0.60 | 5066 | excl Basketball/MMA/FIFA |
| 4.5 | Optuna LGB+ens+seg | 7.26% | 0.8095 | 0.60 | 5066 | Optuna LGB marginal |
| 4.6 | Calibrated+fine thr | 7.95% | 0.8096 | 0.63 | 4333 | isotonic cal marginal |
| 4.7 | Odds range opt+wgt | 7.23% | - | 0.60 | 5066 | per-range/weight overfit |
| 4.8 | Fine val thr+new fch | 6.81% | 0.8095 | 0.64 | 4092 | fine val overfit, new fch rejected |
| 4.9 | Filtered training | 7.23% | 0.8095 | 0.60 | 5066 | train-filt worse, deep CB worse |
| 4.10 | EV-based selection | 7.23% | 0.8095 | 0.60 | 5066 | EV overfit on val |
| 4.11 | CatBoost cat Sport | 7.26% | 0.8097 | 0.60 | 5146 | marginal +0.03% |
| 4.12 | Optuna CB filt val | 7.32% | 0.8089 | 0.60 | 4975 | depth=2, marginal +0.09% |
| 4.13 | Market/Tourn filter | 6.81% | - | 0.60 | 5189 | market filter worse |

## Accepted Features
- log_odds: np.log1p(Odds)
- implied_prob: 1/Odds
- value_ratio: (ML_P_Model/100) / implied_prob
- edge_x_ev: ML_Edge * ML_EV
- edge_abs: abs(ML_Edge)
- ev_positive: ML_EV > 0
- model_implied_diff: ML_P_Model - ML_P_Implied
- log_usd: np.log1p(USD)
- log_usd_per_outcome: np.log1p(USD/Outcomes_Count)
- parlay_complexity: Outcomes_Count * Is_Parlay

## Recommended Next Steps
### Итоговый результат
- **Best ROI: 7.32%** (val-selected threshold, anti-leakage compliant)
- **Стратегия:** Ensemble (Optuna CatBoost depth=2 + LightGBM + XGBoost), equal average, threshold=0.60, исключение убыточных спортов (Basketball, MMA, FIFA, Snooker)
- **AUC:** 0.8089 на filtered test
- **N ставок:** 4 975 из 12 118 (41%)
- **Target 10% не достигнут.** Разрыв ~2.7 п.п.

### Что сработало
1. **Segment filtering** (+4.03 п.п.): исключение 4 убыточных спортов — единственное крупное улучшение в Phase 4
2. **Safe feature engineering** (+2.05 п.п.): log_odds, implied_prob, value_ratio без target encoding
3. **Optuna hyperparameter search** (+3.47 п.п.): переход от отрицательного ROI к положительному
4. **Equal average ensemble** (+0.54 п.п.): стабильнее любых взвешенных схем

### Что не сработало (9 подходов Phase 4)
- Weighted/stacking ensemble, EV-based selection, per-odds-range thresholds, temporal features, deeper CatBoost, filtered training, fine val threshold, categorical Sport — всё дало либо ухудшение, либо маргинальный эффект < 0.1 п.п.

### Почему потолок на ~7.3%
1. **Feature importance:** Odds/implied_prob доминируют (85%+). Модель по сути прогнозирует по структуре коэффициентов. Нет внешних данных (ELO, form, H2H)
2. **Val noise:** 3-дневный test window создает неустойчивость; val threshold overfit при fine grid
3. **Отсутствие team stats:** ML_Team_Stats_Found='f' всегда; ML_Winrate_Diff=100% NaN

### Рекомендации для следующей итерации
1. Обогатить данные внешними ELO/рейтингами — потенциал ~2-5 п.п.
2. Расширить test window до 2+ недель для надежной оценки
3. Автоматический мониторинг ROI по спортам с rolling window
4. Online learning: адаптация порога на свежих данных

---
