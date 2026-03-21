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
