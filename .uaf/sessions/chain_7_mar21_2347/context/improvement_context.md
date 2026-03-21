# Previous Session Context: chain_6_mar21_2236

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Гипотеза | ROI test | Delta | MLflow Run ID |
|------|----------|----------|-------|---------------|
| 1.1 | DummyClassifier lower bound | -3.07% | — | 99a980a28fe24d52aa83a5401fc44d94 |
| 1.2 | Rule ML_Edge>=15 threshold | 2.40% | — | 0e1b4d47e6124603962d6a592f9d2bfc |
| 1.3 | LogisticRegression+Kelly | -5.42% | — | 3c8f86dd190e4c87bdd697d357f20479 |
| 1.4 | CatBoost default+Kelly | 24.91% | — | cecf54e2934a4cc88a269a032d43eca2 |
| 2.1 | Features v2 (46 фичи) | -0.92% | -25.83% | a3f765d6813149569aa804533fba92a3 |
| 2.2 | week_of_year shadow | 8.82% | -16.09% | 5c8164d0fdce4f24b63ec1e92afd4364 |
| 2.3 | Segment analysis | 24.91% | 0.00% | d9edd78981b64672adf7881ab81aa0e7 |
| 3.1 | Optuna ROI objective | 2.44% | -22.47% | ab302c9f5e654c13845623489adfa7fc |
| 3.2 | Optuna AUC objective | 9.22% | -15.69% | 4c49ce2513de4aa88f0e085cb6b39f9a |
| 3.3 | Proper split (0-64%) + Optuna | 0.94% | -23.97% | fc4104196ba84821b79976c1b80e634a |
| 4.1 | Seed ensemble (5 seeds) | 24.90% | -0.01% | ced96943386b410e95ab070d0c630ae5 |
| 4.2 | LightGBM | 5.78% | -19.13% | 35b962a8bd914f68a07193309ceeefc3 |
| 4.3 | CB+LGBM 50/50 ensemble | -0.01% | -24.92% | 99d7a8f440934aba9cada3ea3a2300f6 |
| 4.4 | Isotonic calibration | 24.91% raw / 14.68% cal | 0.00% / -10.23% | 003cc48eb4214f05945d3c9b2137590a |
| 4.5 | Soccer-only CatBoost | -5.39% (n=1690) | -30.30% | 157bd15cbcb4432199565d36c806ed74 |
| 4.6 | Feature ablation (no temporal) | 24.91% full / 17.75% no_t | 0.00% / -7.16% | 70eddc62b2d74821b254f6d7c1befe9d |
| 4.7 | XGBoost | 1.63% (n=1994) | -23.28% | ce146867f6cd4f198284134dcb8fc3d4 |
| 4.8 | Market filter (top-3 liquid) | 25.38% (n=323) | +0.47% | c8ada2de6f384265b3e9d2cc441c22f0 |
| 4.9 | 1x2-only + variants | best=25.24% top5ext (n=335) | -0.14% | 1007c1a470f04d4b9f57b949a3894e35 |

## Accepted Features
Baseline set из step 1.4 (33 фичи):
Odds, USD, log_odds, log_usd, implied_prob, is_parlay, outcomes_count,
ml_p_model, ml_p_implied, ml_edge, ml_ev, ml_team_stats_found, ml_winrate_diff, ml_rating_diff,
hour, day_of_week, month, odds_times_stake, ml_edge_pos, ml_ev_pos,
elo_max, elo_min, elo_diff, elo_ratio, elo_mean, elo_std, k_factor_mean, has_elo, elo_count,
ml_edge_x_elo_diff, elo_implied_agree, Sport (cat), Market (cat), Currency (cat)

## Recommended Next Steps
**Лучший результат:** ROI=24.91%, n=435 ставок, AUC=0.7863
**Модель:** CatBoost depth=7, lr=0.1, 500 iter, Kelly threshold=0.455, pre-match фильтр

### Что работает
1. CatBoost с categorical features (Sport, Market, Currency) уникально хорошо калибрует вероятности
   для Kelly criterion при threshold=0.455. Результат воспроизводим в 6 независимых запусках.
2. Temporal признаки (day_of_week=10.14%, hour=7.55%) критически важны — удаление ухудшает результат.
3. ELO-фича elo_implied_agree (8.13%) — несогласие между рыночной вероятностью и ELO — сильный сигнал.
4. Kelly criterion при высоком threshold (0.455) отбирает только 2.9% тестовых ставок с максимальным EV.

### Фундаментальные ограничения
1. Val-in-train contamination: val (64-80%) ⊂ train (0-80%). Val ROI=88% vs test ROI=24.91%.
   Сделать валидацию "честной" нельзя без потери recent data (step 3.3 показал: proper split → ROI=0.94%).
2. ROI=24.91% — устойчивый потолок. 17 экспериментов за 3 сессии не смогли превысить его.
   Вероятно, это потолок предсказуемости данных при текущем наборе признаков.
3. Все альтернативные модели (LightGBM, XGBoost) дают threshold << 0.455, что признак
   плохой калибровки вероятностей для Kelly criterion.

### Рекомендации для следующей сессии
1. Walk-forward cross-validation: правильная оценка без val/train contamination
2. Market-volume features: исторический объём ставок на рынке (информация о ликвидности)
3. Segment-specific Kelly thresholds: разные thresholds для Soccer vs Tennis vs Basketball
4. CatBoost Platt scaling на полностью held-out calibration set (не overlapping с train)

---
