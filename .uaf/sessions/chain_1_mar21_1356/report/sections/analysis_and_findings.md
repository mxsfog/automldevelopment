# Analysis and Findings

## Baseline Performance
- DummyClassifier (все ставки): ROI = -1.96% (n=14899)
- Случайный отбор 50%: ROI = -5.92% (n=7363)
- Правило ML_P_Model>=60: ROI = +0.90% (n=4353) — первый положительный ROI
- LogisticRegression (thr=0.65): ROI = +1.59% (n=5524), AUC=0.7918
- CatBoost default (thr=0.60): ROI = +0.80% (n=5697), AUC=0.7934

Все baseline-модели дают положительный, но слабый ROI (< 2%). Модели фильтруют в основном low-odds/high-probability ставки (win_rate > 90%, avg_odds ~ 1.07).

## Feature Engineering Results
Метод Shadow Feature Trick: обучение двух моделей (baseline vs candidate) с одинаковыми гиперпараметрами.

| Группа фич | Delta ROI | Статус |
|------------|-----------|--------|
| Odds-based (implied_prob, margin и др.) | -0.10 | rejected |
| Sport/Market target encoding | +6.92 | accepted |
| Temporal (hour, day_of_week) | -6.25 | rejected |
| ML interactions (edge_x_odds и др.) | -5.68 | rejected |
| Complexity (odds_spread, odds_cv) | -6.89 | rejected |

Только 4 фичи Sport/Market encoding прошли проверку. Добавление любых других групп ухудшало результат. Вероятная причина: избыточность (Odds уже несет большую часть информации), шум на малых выборках.

## Model Comparison
- CatBoost: основная модель, AUC~0.79, Odds доминирует (51.6% importance)
- LightGBM: сравнимый AUC, добавляет diversity в ансамбль
- LogisticRegression: слабее по AUC, но стабилизирует ансамбль
- XGBoost: добавление 4-й модели + дополнительных фич ухудшило ROI (-3.07% vs +7.82%)

3-model ensemble (CB+LGBM+LR) — оптимальный вариант. Добавление моделей или фич = переобучение.

## Segment Analysis
- Dota2: +4.58% ROI (segment-level)
- Cricket: +4.13% ROI
- CS2: +2.54% ROI
- Ice Hockey, Basketball: отрицательный ROI на val

Попытка исключить "плохие" виды спорта на val привела к ROI -18.07% на test. Сегментный анализ на val не переносится — слишком мало данных для надежной сегментации.

## Stability & Validity
5-fold time series CV для лучшей конфигурации (EV>=0.12, full train):

| Fold | ROI | N bets |
|------|-----|--------|
| 0 | +31.90% | 1921 |
| 1 | +12.45% | 2463 |
| 2 | +38.03% | 1952 |
| 3 | +11.59% | 1924 |
| 4 | -0.32% | 1893 |
| **Mean** | **+18.73%** | - |
| **Std** | **14.14%** | - |

4 из 5 фолдов прибыльны. Fold 4 (последний по времени) почти на нуле — возможный drift.

Anti-leakage проверки:
- Threshold (EV>=0.12) выбран на val, применен к test один раз
- Target encoding fit только на train
- Нет future-looking фичей (Payout_USD, Score удалены)
- Test ROI 16.02% < 35% sanity threshold

Замечание: ROI обусловлен высокими коэффициентами. При odds cap <= 5 ROI падает до ~2.6%. Стратегия прибыльна за счет выявления value в высоких коэффициентах, что несет высокий риск.

## What Didn't Work
1. **Probability threshold optimization** (Step 4.1): модель выбирает только safe bets с очень низкими коэффициентами (avg_odds=1.06). ROI = -0.99%
2. **Sport segmentation** (Step 4.3): фильтрация по видам спорта на val дала ROI -18.07% на test. Overfitting на малых сегментах
3. **4-model ensemble + extra features** (Step 4.4): XGBoost + odds_features + interaction_features = ROI -3.07%. Дополнительные фичи = шум, ухудшение EV-калибровки
4. **Odds capping** (Step 4.6): ограничение max_odds убивает ROI. При odds<=5 лучший ROI = 6.48% (EV>=0.25, n=361). Alpha находится именно в high-odds ставках
5. **Optuna hyperparameter tuning** (Step 3.1): минимальный прирост (+1.47% vs +1.31%). Гиперпараметры не критичны для данной задачи
