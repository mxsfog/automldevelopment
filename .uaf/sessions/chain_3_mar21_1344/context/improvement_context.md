# Previous Session Context: chain_2_mar21_1309

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | AUC | N_bets | Threshold | Run ID |
|------|--------|-----|-----|--------|-----------|--------|
| 1.1 | DummyClassifier | -3.07% | - | 14899 | - | 5f61a209 |
| 1.2 | Rule ML_P>=0.30 | +3.82% | - | 11310 | 0.30 | 457aec69 |
| 1.3 | LogisticRegression | +2.04% | 0.791 | 15 | 0.81 | 92445be7 |
| 1.4 | CatBoost basic | -0.02% | 0.793 | 2230 | 0.83 | 4c10cb92 |
| 2.1 | CatBoost+ELO+odds | +5.60% | 0.800 | 9551 | 0.46 | 6c92943e |
| 2.2 | +new features | +6.50% | 0.802 | 9675 | 0.46 | dbeddbac |
| 3.1 | Optuna HPO (AUC) | +6.87% | 0.803 | 9318 | 0.47 | e8d38342 |
| 4.1 | Stacking CB+LGB+XGB | +6.52% | 0.806 | 10096 | 0.44 | a34d6e62 |
| 4.2 | Edge+sport scan* | +15.73%* | 0.804 | 1167 | 0.45+e0.10 | 84f8c52f |
| 4.3 | Edge val-tuned | +14.82% | 0.803 | 1181 | p0.43+e0.10 | d987968b |
| 4.4 | Calibr+stability | +14.78% | 0.804 | 1185 | p0.42+e0.10 | db226171 |
| 4.5 | Kelly+retrain | +22.92%* | 0.798 | 1131 | p0.43+e0.10 | dcd0ba70 |
| 4.6 | Multi-edge 3models | +9.78% | 0.804 | 881 | stable-best | 9f93f294 |
| 4.7 | Q2 analysis+odds | +9.79% | 0.804 | 1806 | p0.43+e0.07 | 16d4cf1c |

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
(заполняется Claude Code по завершении)

---
