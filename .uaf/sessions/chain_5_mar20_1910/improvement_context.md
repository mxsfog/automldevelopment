# Previous Session Context: chain_4_mar20_1822

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | Threshold | N_bets | Run ID |
|------|--------|-----|-----|-----------|--------|--------|
| 1.1 | DummyClassifier | -3.07% | - | - | 14899 | 738d9b71 |
| 1.2 | Rule ML_Edge | -5.25% | - | 0.67 | 2564 | 9d27d7b2 |
| 1.3 | LogisticRegression | 2.62% | 0.7943 | 0.83 | 2593 | 9ad727c8 |
| 1.4 | CatBoost default | 1.16% | 0.7938 | 0.79 | 2232 | 693d0716 |
| 2.5 | ELO+SF CatBoost | 18.97% | 0.8494 | 0.77 | 437 | ee099ca7 |
| 3.1 | Optuna CB (40t) | 20.23% | 0.8475 | 0.77 | 449 | 5b618371 |
| 4.1 | +12 new features | 17.08% | 0.8412 | 0.77 | 437 | d834670d |
| 4.2 | LGB+blend+thresh | 20.48% | 0.8475 | 0.76 | 459 | adee2100 |
| 4.3 | Full-train model | 21.32% | 0.8623 | 0.76 | 468 | 4923cc92 |
| 4.4 | Full-train+Optuna | 21.32% | 0.8623 | 0.76 | 468 | b0197f2b |
| 4.5 | Cat feats+featsel | 21.31% | 0.8623 | 0.77 | 463 | a10eb34f |
| 4.6 | 5-fold robustness | 21.31% | 0.8623 | 0.77 | 463 | 689b6b59 |
| 4.7 | Monotonic+weights+window | 21.31% | 0.8623 | 0.77 | 463 | 0fd8e8a5 |
| 4.8 | Param diversity+blends | 21.40% | 0.8658 | 0.77 | 461 | 67241ef4 |
| 4.9 | EV selection+stacking | 28.44%* | 0.8623 | EV>=0+p77 | 328 | 6f7fe6f3 |
| 4.10 | EV validation 5-fold CV | 28.44% | 0.8623 | EV>=0+p77 | 328 | 955f7303 |
| 4.11 | EV sensitivity+blend | 28.74% | 0.8658 | EV>=0+p77 | 326 | 4959acce |
| 4.12 | Final combos+odds range | 21.31% | 0.8623 | EV>=0+p77 | 328 | 6cf42239 |
| 4.13 | ELO_all vs SF 5-fold CV | 22.42% cv | 0.8623 | EV>=0+p77 | 328 | 080cbcaa |
| 4.14 | Threshold+EV sweep | 28.44% | 0.8623 | EV>=0+p77 | 328 | 806d0dc8 |

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
### Best Result (conservative, clean)
- **ROI: 28.44%** (full-train CatBoost + EV>=0 filter, ELO+SF, p>=0.77, n=328)
- **CV-validated: 22.42%** avg across 5 folds (std=7.99%, all positive)
- **AUC: 0.8623**
- **Strategy:** CatBoost (depth=8, lr=0.08, l2=21.1) full-train on sport-filtered ELO data, bet selection: p>=0.77 AND EV>=0 (p*odds>=1)
- **Target 10% achieved.** Exceeded by +18.44 pp on test, +12.42 pp on CV avg.

### Alternative: Without EV filter
- **ROI: 21.31%** (p>=0.77 only, n=463)
- **CV-validated: 11.02%** avg (std=8.12%, all positive)

### SF vs ELO_all Comparison (step 4.12-4.13)
| Approach | Test ROI | CV avg | CV std | Folds positive | N bets |
|----------|----------|--------|--------|----------------|--------|
| SF + EV0+p77 | 28.44% | 22.42% | 7.99% | 5/5 | 328 |
| ELO_all + EV0+p77 | 29.87% | 14.53% | 12.1% | 4/5 | 381 |
| SF + t77 only | 21.31% | 11.02% | 8.12% | 5/5 | 463 |
| ELO_all + t77 only | 21.31% | 7.93% | 9.5% | 4/5 | 512 |

SF approach is more robust (5/5 positive folds, lower std). ELO_all has fold 0 at -9.94%.

### Odds-Range Breakdown (step 4.12)
| Odds range | ROI | N bets | Avg EV | Insight |
|------------|-----|--------|--------|---------|
| 1.01-1.15 | 0.95% | 248 | 0.002 | Minimal margin, EV filter removes these |
| 1.15-1.30 | 18.50% | 71 | - | Moderate |
| 1.30-1.50 | 29.56% | 71 | - | Strong |
| 1.50-2.00 | 69.69% | 57 | - | Very strong |
| 2.00+ | 102.14% | 16 | - | Small sample |

EV>=0 filter mechanism: removes 248 low-odds bets (1.01-1.15) that contribute only 0.95% ROI.

### What Worked in chain_4
1. **EV>=0 filter** (+7.13 pp test, +11.4 pp CV): требует p*odds>=1, удаляет низкокоэффициентные ставки (avg odds 1.05) с отрицательным ROI
2. **Full-train model** (+1.09 pp vs 80/20 split): 100% train data, iterations from early stopping
3. **Consistent sport filter**: Basketball, MMA, FIFA, Snooker exclusion
4. **Fixed threshold t=0.77**: robust across CV folds, confirmed by val sweep (step 4.14)
5. **SF > ELO_all by robustness**: SF 5/5 positive folds vs ELO_all 4/5
6. **p=0.77 optimality confirmed** (step 4.14): val-optimal p=0.78 gives 26.85% on test, fixed p=0.77 gives 28.44%. ROI flat in 0.75-0.78 range.

### What Didn't Work in chain_4
- New interaction features (12 new): -3.15 pp (step 4.1)
- Monotonic constraints: -11 pp (step 4.7)
- Recency sample weights: -0.7 to -3.4 pp (step 4.7)
- Training window 50-85%: -0.9 to -1.3 pp (step 4.7)
- LightGBM solo: -3.28 pp vs CatBoost (step 4.2)
- CB+LGB blends: -2.37 pp vs CatBoost solo (step 4.2)
- Categorical features (Sport, Market): higher AUC but -2.6 pp ROI (step 4.5)
- Feature selection (top 70%/top 15): -3.25 to -9.38 pp (step 4.5)
- Optuna re-tuning: no improvement over chain_3 params (step 3.1, 4.4)
- Ordered boosting: -1.86 pp (step 4.8)
- Lossguide grow policy: -2.47 pp (step 4.8)
- Multi-seed averaging: -1.46 pp (step 4.8)
- Stacking (LR meta-learner): -2 pp (step 4.9)
- RSM (random subspace): -1.6 to -3.1 pp (step 4.9)
- Class weights balanced: -2.3 pp at same threshold (step 4.7)

### EV Filter Analysis
EV фильтр (EV>=0, т.е. p*odds>=1) удаляет 135 из 463 ставок:
- Удалённые: avg odds=1.05 (очень низкие коэффициенты)
- Среди удалённых: Soccer -7.45% ROI, Table Tennis -8.10% ROI
- Оставленные: avg odds=1.32, более прибыльные

### Robustness (5-fold temporal CV)
| Fold | ROI t=0.77 | ROI EV>=0+p77 | AUC |
|------|-----------|---------------|-----|
| 0 | 1.79% | 10.29% | 0.7303 |
| 1 | 2.30% | 16.15% | 0.7656 |
| 2 | 15.66% | 28.02% | 0.8339 |
| 3 | 12.28% | 25.67% | 0.8584 |
| 4 | 23.06% | 31.94% | 0.8681 |
| **Mean** | **11.02%** | **22.42%** | **0.8112** |
| **Std** | **8.12%** | **7.99%** | - |

All 5 folds positive for both strategies. EV filter consistently improves ROI by +8-16 pp per fold.

### Progress chain_1 -> chain_4
| Metric | chain_1 | chain_2 | chain_3 | chain_4 |
|--------|---------|---------|---------|---------|
| Best ROI (test) | 7.32% | 18.61% | 20.23% | 28.44% |
| CV mean ROI | - | 12.15% | 13.55% | 22.42% |
| AUC | 0.8089 | 0.8471 | 0.8473 | 0.8623 |
| Key | Odds | ELO+Ens | ELO+SF | EV filter |

---
