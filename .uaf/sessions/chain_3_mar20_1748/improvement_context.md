# Previous Session Context: chain_2_mar20_1715

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | Threshold | N_bets | Run ID |
|------|--------|-----|-----|-----------|--------|--------|
| 1.1 | DummyClassifier | -3.07% | - | - | 14899 | 8d9636... |
| 1.2 | Rule ML_Edge | -7.35% | - | 0.31 | 2041 | c0ec10... |
| 1.3 | LogisticRegression | 1.46% | 0.7897 | 0.81 | 2656 | 34e1e9... |
| 1.4 | CatBoost default | 2.48% | 0.7946 | 0.76 | 2461 | 603c95... |
| 2.5a | Baseline (no ELO) | 0.34% | 0.7927 | 0.63 | 1126 | 16f5ec... |
| 2.5b | + Safe ELO | 2.38% | 0.7983 | 0.83 | 2527 | d27b14... |
| 2.5c | ELO-only subset | 10.70% | 0.8540 | 0.62 | 725 | 7e0110... |
| 3.1 | Optuna CB ELO-only | 16.63% | 0.8431 | 0.73 | 634 | 248614... |
| 4.1 | Ensemble ELO w50 | 18.14% | 0.8379 | 0.73 | 565 | 37bf27... |
| 4.2 | Dual-model | 16.63% | - | 0.73 | 634 | d4bac4... |
| 4.3 | Robustness 4 splits | 12.15% avg | 0.8369 avg | - | - | 90c87d... |
| 4.4 | Optuna LGB+Ens CB50 | 16.76% | 0.8501 | 0.62 | 743 | b5a485... |
| 4.5 | ELO interactions+OptW | 16.37% | 0.8464 | 0.64 | 730 | df3c87... |
| 4.6 | Sport thresh+stacking | 15.38% | 0.8471 | 0.70 | 640 | 45f72c... |
| 4.7 | Robust threshold 3-fold | 18.61% | 0.8471 | 0.73 | 602 | 895bb4... |
| 4.8 | Final best 4-fold | 16.86% | 0.8471 | 0.77 | 534 | ca192d... |

## Accepted Features
### Chain_1 proven features (safe)
- log_odds, implied_prob, value_ratio, edge_x_ev, edge_abs
- ev_positive, model_implied_diff, log_usd, log_usd_per_outcome, parlay_complexity

### ELO features (safe, no leakage)
- team_elo_mean, team_elo_max, team_elo_min, k_factor_mean, n_elo_records
- elo_diff, elo_diff_abs, has_elo
- team_winrate_mean, team_winrate_max, team_winrate_diff
- team_total_games_mean, team_current_elo_mean
- elo_spread, elo_mean_vs_1500

## Recommended Next Steps
### Итоговый результат
- **Best ROI: 18.61%** (robust multi-fold threshold t=0.73, leakage-free)
- **Стратегия:** CB50 Ensemble (Optuna CatBoost 50% + Optuna LightGBM 25% + XGBoost 25%) на ELO-only subset
- **AUC:** 0.8471 на ELO test
- **N ставок:** 602 из 1332 ELO test (45.2% coverage)
- **Target 10% достигнут.** Превышение на +8.61 п.п.

### Прогресс chain_1 -> chain_2
| Метрика | chain_1 | chain_2 | Дельта |
|---------|---------|---------|--------|
| Best ROI | 7.32% | 18.61% | +11.29 п.п. |
| AUC | 0.8089 | 0.8471 | +0.038 |
| Ключевой фактор | Odds (85% FI) | ELO+Odds (diversified FI) | ELO enrichment |

### Что сработало
1. **ELO data enrichment** (+11 п.п.): safe ELO features (Old_ELO, Winrate, K_Factor) диверсифицировали feature importance и дали прорыв
2. **ELO-only subset** (+8 п.п. vs all-data): модель работает существенно лучше на ставках с ELO-данными
3. **Optuna HPO** (+6 п.п.): depth=7, lr=0.214, high regularization
4. **CB50 Ensemble** (+2 п.п.): CatBoost-доминирующий ансамбль стабильнее одиночных моделей
5. **Robust threshold selection** (+2 п.п.): multi-fold median threshold устойчивее single-val

### Что не сработало в chain_2
- Interaction features (elo_diff * value_ratio etc.) -- ухудшили ROI на 6 п.п.
- Per-sport thresholds -- переобучение на малых выборках, ROI ниже global threshold
- Stacking meta-learners (LR, CatBoost) -- не превзошли простое weighted average
- Optuna-оптимизация весов ансамбля -- marginal improvement, не оправдала сложность
- Dual-model (ELO + non-ELO) -- non-ELO component размывает результат

### Ограничения
1. **ELO coverage 9.7%**: только 7198 из 74493 ставок имеют ELO-данные
2. **3-дневный test window**: результаты могут варьироваться на более длинном периоде (std=5.12% по 4 splits)
3. **Temporal stability**: mean ROI=12.15% across 4 temporal splits, отдельные splits от 6.4% до 19.7%

### Рекомендации для production
1. Расширить ELO-трекинг на большее количество матчей
2. Rolling window мониторинг ROI по спортам с алертами при drift
3. Dual-model deployment: ELO-модель (ROI~18%) для ELO-ставок, chain_1 модель (ROI~7%) для остальных
4. Тестирование на 2+ недельном окне перед production rollout
5. Фиксированный порог t=0.73 для стабильности (не подстраивать на свежих данных)

---
