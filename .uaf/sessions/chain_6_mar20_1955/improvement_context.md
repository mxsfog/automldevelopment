# Previous Session Context: chain_5_mar20_1910

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | Threshold | N_bets | Run ID |
|------|--------|-----|-----|-----------|--------|--------|
| 1.1 | DummyClassifier | -3.07% | - | - | 14899 | a7e901dd |
| 1.2 | Rule ML_Edge>=11.5 | -4.89% | - | 11.5 | 2489 | 022f86ba |
| 1.3 | LogisticRegression | 2.62% | 0.7943 | 0.83 | 2593 | 668873f0 |
| 1.4 | CatBoost default | 1.72% | 0.7930 | 0.75 | 2822 | 1b0b9deb |
| 2.5 | ELO+SF CatBoost FT | 28.44% | 0.8623 | EV>=0+p77 | 328 | d749f6fe |
| 3.1 | Optuna TPE (50t) | 24.90% | 0.8575 | EV>=0+p77 | 300 | 2732faf7 |
| 4.1 | ML feats+XGB+ens | 28.44% | 0.8623 | EV>=0+p77 | 328 | afbff5fe |
| 4.2 | Calib+PerSport | 52.02%/29.62% | 0.8623 | PS_EV/Hybrid | 132/312 | 2244324b |
| 4.3 | PerSport EV 5-fold | 52.02% | 0.8623 | PS_EV | 132 | 9978a914 |
| 4.4 | Time+Market+OddsEV | 49.45% | 0.8623 | EV>=0.10+p77 | 157 | 0ff2a7d0 |
| 4.5 | Comprehensive 5-fold CV | 49.45% | 0.8623 | EV>=0.10+p77 | 157 | cd2bfbd5 |
| 4.6 | Combined best+model save | 57.42% | 0.8623 | PS_EV floor=0.10 | 110 | 76511638 |
| 4.7 | Kelly+robustness | 57.42% | 0.8623 | PS_EV floor=0.10 | 110 | 75402fce |
| 4.8 | LGB+ensemble strict EV | 49.45% | 0.8623 | CB EV>=0.10 | 157 | c7a41d06 |
| 4.9 | Blend+PS_EV combos | 57.42% | 0.8623 | CB+PS010 | 110 | 6d794a1d |
| 4.10 | Tournament features | 57.42% | 0.8623 | base (tourn=-17pp) | 110 | dbd65d82 |
| 4.11 | Multi-seed stability | 54.62% avg | 0.8672 | PS010 10 seeds | ~110 | 3297cc9e |
| 4.12 | Seed averaging (5) | 57.42% | 0.8702 | single PS010 wins | 110 | 7b3bd996 |
| 4.13 | Bootstrap CI | 57.42% | 0.8623 | PS010 [44.8%,72.4%] | 110 | a2b17f74 |

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

### Base features
- Odds, USD, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, Outcomes_Count, Is_Parlay_bool

## Recommended Next Steps
(заполняется Claude Code по завершении)

---
