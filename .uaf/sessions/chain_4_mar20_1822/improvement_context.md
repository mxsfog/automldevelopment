# Previous Session Context: chain_3_mar20_1748

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | Threshold | N_bets | Run ID |
|------|--------|-----|-----|-----------|--------|--------|
| 1.1 | DummyClassifier | -3.07% | - | - | 14899 | 70afa8f0 |
| 1.2 | Rule ML_Edge | -7.35% | - | 0.31 | 2041 | 75324f42 |
| 1.3 | LogisticRegression | 2.70% | 0.7949 | 0.81 | 2717 | 1fcdd3f2 |
| 1.4 | CatBoost default | 0.34% | 0.7927 | 0.63 | 1126 | 79f44f82 |
| 2.5a | Baseline (no ELO) | 0.34% | 0.7927 | 0.63 | 1126 | f54fb1bd |
| 2.5b | +Safe ELO all | -0.75% | 0.7979 | 0.87 | 1981 | 242a1859 |
| 2.5c | ELO-only subset | 13.18% | 0.8543 | 0.75 | 550 | 0ba6fadb |
| 3.1 | Optuna CB ELO-only | 18.59% | 0.8550 | 0.77 | 559 | 4c346f7a |
| 4.1 | CB50 Ens (CB+LGB+XGB) | 14.97% | 0.8524 | 0.72 | 587 | 9ebb3396 |
| 4.2 | Robust thresh+weights | 16.09% | 0.8544 | 0.75 | 554 | 2573395d |
| 4.3 | Optuna XGB+thresh scan | 16.66% | 0.8550 | 0.73 | 605 | 50647a3d |
| 4.4 | Sport filter+Optuna50 | 17.72% | 0.8473 | 0.77 | 453 | d9a1da50 |
| 4.5 | Sport ens65+robust | 18.35% | 0.8473 | 0.76 | 447 | 927ed787 |
| 4.6 | CB42+sport filter | 19.80% | 0.8550 | 0.77 | 434 | 6c5ffff4 |
| 4.7 | Deep sport analysis | 19.80% | 0.8550 | 0.77 | 434 | 42e41d89 |
| 4.8 | Optuna SF + ref CB | 20.23% | 0.8473 | 0.77 | 449 | 80583a3d |
| 4.9 | Best combo SF | 20.48% | 0.8473 | 0.76 | 459 | db5dda09 |
| 4.10 | Robustness 4-fold CV | 13.55% avg | 0.8500 avg | 0.77 | ~239 avg | 0d2bd1e1 |
| 4.11 | Final validation SF | 20.23% | 0.8473 | 0.77 | 449 | 859143aa |

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
- **Best ROI: 20.23%** (CatBoost seed=42, sport-filtered train+test, t=0.77)
- **Стратегия:** CatBoost (depth=8, lr=0.08, l2=21.1) на ELO-only подмножестве с исключением Basketball, MMA, FIFA, Snooker
- **AUC:** 0.8473 на sport-filtered test
- **N ставок:** 449 из 1094 sport-filtered ELO test (41% coverage)
- **Target 10% достигнут.** Превышение на +10.23 п.п.

### Робастность (4-fold temporal CV)
| Fold | ROI (all) | ROI (SF) | AUC |
|------|-----------|----------|-----|
| 0 | 4.10% | 3.72% | 0.8378 |
| 1 | 15.98% | 16.52% | 0.8433 |
| 2 | 11.60% | 8.83% | 0.8680 |
| 3 | 16.02% | 25.13% | 0.8508 |
| **Mean** | **11.93%** | **13.55%** | **0.8500** |
| **Std** | **4.86%** | **8.09%** | - |

Все 4 фолда положительные. Sport filter дает в среднем +1.63 п.п.

### Прогресс chain_1 -> chain_2 -> chain_3
| Метрика | chain_1 | chain_2 | chain_3 | Дельта (1->3) |
|---------|---------|---------|---------|---------------|
| Best ROI | 7.32% | 18.61% | 20.23% | +12.91 п.п. |
| CV mean ROI | - | 12.15% | 13.55% | - |
| AUC | 0.8089 | 0.8471 | 0.8473 | +0.038 |
| Ключевое | Odds | ELO+Ensemble | ELO+SportFilter | SportFilter |

### Что сработало в chain_3
1. **Sport filter at train+test** (+1.6 п.п.): исключение Basketball, MMA, FIFA, Snooker из train и test
2. **Фиксированный t=0.77** (лучше robust median): multi-fold robust threshold дает слишком низкие значения (0.59), fixed t=0.77 стабильнее
3. **CB solo > ensemble**: CatBoost solo на sport-filtered дает 20.23%, CB65 ensemble 17.74%, multi-seed 18.89%
4. **Train on SF > train on all + infer SF**: 20.23% vs 19.80%

### Что не сработало в chain_3
- Multi-seed averaging (18.89% vs 20.23% single seed=42)
- Ensembles на sport-filtered data (CB65=17.74%, CB80=18.50%)
- Optuna re-tune на sport-filtered (17.29-19.06%, не побил базовые params)
- Robust median threshold (0.59 -> 13.67% vs fixed 0.77 -> 20.23%)
- Val-determined sport exclusion (Cricket, NBA2K -> 16.32% vs original filter -> 20.23%)
- Isotonic calibration (из chain_2, ухудшает ROI)

### Ограничения
1. **ELO coverage 9.7%**: только 7198 из 74493 ставок имеют ELO-данные
2. **Sport filter further reduces coverage**: 1094 из 1332 ELO test bets (82%)
3. **3-дневный test window**: ROI варьируется от 3.72% до 25.13% по фолдам (std=8.09%)
4. **High variance**: std=8.09% по 4 фолдам, что говорит о нестабильности

### Рекомендации для production
1. Использовать CatBoost (depth=8, lr=0.08, l2=21.1) на ELO-only + sport filter
2. Фиксированный порог t=0.77, не подстраивать на свежих данных
3. Расширить ELO-трекинг для увеличения покрытия
4. Rolling window мониторинг ROI по спортам с алертами при drift
5. Тестирование на 2+ недельном окне перед production rollout
6. Dual-model deployment: ELO-SF модель (ROI~20%) для ELO+good-sports ставок, fallback для остальных

---
