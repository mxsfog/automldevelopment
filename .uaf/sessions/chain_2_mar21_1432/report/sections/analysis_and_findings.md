# Analysis and Findings

## Baseline Performance
Верифицированный baseline из chain_1: ROI=+16.02%, AUC=0.784. Воспроизведён с delta=0.00% (step 4.0). Стратегия: 3-model ensemble (CatBoost + LightGBM + LogReg), EV threshold >= 0.12, full train.

## Feature Engineering Results
19 принятых фичей из предыдущей сессии (15 базовых + 4 sport/market encoding). В этой сессии новые фичи не тестировались — фокус был на архитектуре и selection strategy.

## Model Comparison

| Подход | ROI test | N bets | Вывод |
|--------|----------|--------|-------|
| Baseline 3-model avg, EV>=0.12 | **+16.02%** | 2247 | Best, стабилен |
| Calibrated 3-model (isotonic prefit) | +14.04% | 2559 | Калибровка ослабила порог |
| 4-model + Optuna CB (depth=8) | +9.00% | 3377 | Глубокая модель хуже калибрована |
| Weighted ensemble (opt на val) | +8.37% | 3738 | Веса не переносятся |
| CatBoost Optuna solo | +6.35% | 4642 | Одна модель хуже ансамбля |
| Stratified EV (per-odds) | +52.94% | 584 | Overfitting (val ROI=304%) |
| Kelly criterion (f=0.25) | +7.03% | 2247 | Kelly перевешивает с high-odds |
| Profit regression (CatBoost) | -5.36% | 6249 | Regression не работает (skewed target) |
| Profit regression (Huber) | -2.28% | 6242 | Huber лучше, но всё равно отрицательный |

## Segment Analysis
- Прибыль концентрирована в high-odds сегменте (avg_odds=33.9)
- Low-odds ставки (odds<=5) дают ROI=+2.6% — мало value
- Sport/market фильтрация ухудшает результат (chain_1 step 4.3)
- Kelly criterion перекладывает вес с high-odds на low-odds, уничтожая ROI

## Stability & Validity
- 5-fold time series CV: mean=18.73%, std=14.14%
- 4/5 фолдов прибыльны, fold 4 ~ 0%
- Нет leakage: threshold выбран на val, применён к test один раз
- Stratified EV (ROI=52.94%) — false positive (overfitting к val noise), не принят
- Profit regression val ROI=157% вызвал MQ-LEAKAGE-SUSPECT alert (false positive)

## What Didn't Work

### 1. Калибровка вероятностей (step 4.1)
Isotonic prefit калибровка размывала вероятности, EV порог падал с 0.12 до 0.11, захватывая больше шумных ставок.

### 2. Более сложные модели (step 4.2)
Optuna-тюнированный CatBoost (depth=8, iter=873) даёт менее калиброванные вероятности для EV-расчёта. Простая модель (depth=6, iter=200) оптимальна для этой задачи.

### 3. Kelly criterion (step 4.3)
Kelly stake = EV/(odds-1) пропорционально уменьшает ставку на high-odds. Но именно high-odds — источник прибыли. Kelly оптимизирует log-wealth, а не ROI.

### 4. Profit regression (step 4.4)
Target profit = won*odds - 1 имеет extreme skew (min=-1, max=+126). Модели не могут обучиться на таком распределении — даже Huber loss не спасает. Classification + post-hoc EV принципиально лучше, т.к. разделяет оценку вероятности и расчёт profit.

### 5. Val-optimized thresholds
Любая оптимизация порогов на val (stratified EV, weight search) не переносится на test из-за distribution shift между val и test периодами.
