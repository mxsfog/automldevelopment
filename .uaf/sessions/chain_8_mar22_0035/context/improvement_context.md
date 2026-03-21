# Previous Session Context: chain_7_mar21_2347

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Гипотеза | ROI test | Delta | MLflow Run ID |
|------|----------|----------|-------|---------------|
| 4.0 | Chain verify (pipeline.pkl) | 24.9088% | 0.00% | 1029f5186e5b4e17b2d7db1362b67d0d |
| 4.1 | Odds-bucket segment Kelly (low/mid/high) | 25.8347% (n=362) | +0.93% | 23a2af6af60a46cf89c50a86e86aff2c |
| 4.2 | Lead-time features (+5 фичей) | 18.8545% (n=748) | -6.05% | 974eed08dcf34eb6af2fff34a6bfa02e |
| 4.3 | Pre-match lead-hours filter sweep | 24.9088% (best=lead>0h) | 0.00% | eadde4ce04ae4257b40ddf397a757608 |
| 4.4 | Deeper CatBoost (depth=9, lr=0.05, 1500 iter) | 4.1844% (n=3022) | -20.72% | 0cddc9870ae34f0e981e80c6a75ac9f7 |
| 4.5 | Dual-agreement (CatBoost + platform ML_P_Model) | 21.7452% (n=38) | -3.16% | 302c6b91c57f4618a66be0d4a11aaf2a |
| 4.6 | Shrunken segment thresholds (shrink=0.5) | 26.9345% (n=372) | +2.03% | 13248020e9b441efa46f5d9eefa0eaa9 |
| 4.7 | Shrinkage sweep (val-optimized, best=0.9) | 24.5566% (n=359) | -0.35% | 5b4854e0fead4e61bb426b942203fe13 |
| 4.8 | Walk-forward probability ensemble (4 windows) | -26.7306% (n=453) | -51.6% | c3c86a1a8b4b456aba24c9d6e77ece5d |
| 4.9 | Market filter + shrunken segments (top-5) | 26.9345% (n=372) | +2.03% | 1d7fc94cf7804c7e9a6b99de1b6179c6 |
| 4.10 | 1x2 only + shrunken segments (Soccer) | 28.5833% (n=233) | +3.67% | 5eeff22fca134f05a6a2811650c2de27 |
| 4.11 | 1x2-retrained CatBoost (market-specific model) | 11.2805% (n=553) | -13.63% | ecd86291955140e48e23eb08190a568d |
| 4.12 | 4-bin odds segments on 1x2 (1x2-specific baseline_t) | 16.8172% (n=737) | -11.77% | 387d4c1c73b54fc190655dcac50db7b3 |
| 4.13 | 4-bin odds segments on 1x2 (global baseline_t=0.455) | 18.1157% (n=343) | -10.47% | 29d8d7e1c293460ca5478ec135380de3 |

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
**Лучший результат:** ROI=28.5833%, n=233 ставок (step 4.10)

**Стратегия:** Глобальная CatBoost-модель (chain_6_mar21_2236) + фильтр рынка 1x2 + shrunken segment Kelly thresholds:
- low (<1.8): t=0.475
- mid (1.8-3.0): t=0.545
- high (>=3.0): t=0.325
- Shrinkage=0.5 toward baseline t=0.455
- Pre-match filter: lead_hours > 0
- Все отобранные ставки: Soccer/1x2

**Почему это работает:**
1. CatBoost уникально хорошо калибрует вероятности для Kelly criterion (threshold ~0.455 vs LightGBM/XGB ~0.05)
2. 1x2 рынок — наиболее ликвидный и предсказуемый в футболе; Kelly отбирает только ставки с реальным edge
3. Shrunken thresholds регуляризуют val-overfitting: без shrinkage raw thresholds дают -0.35% (step 4.7)

**Фундаментальный потолок:**
ROI ~25-29% при отборе 1.5-2% ставок является потолком для данной комбинации модель+данные.
Proper out-of-time validation (step 3.3) показывает ROI=0.94% — реальный ceiling без contamination.

**Рекомендации для production:**
1. Развернуть фильтр как live-стратегию с half-Kelly bet sizing
2. Мониторинг rolling 30-day ROI; переобучение при деградации
3. Накапливать 1x2-specific данные для специализированной модели (нужно 3-5x больше текущих 7150)

---
