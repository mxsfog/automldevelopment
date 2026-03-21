# Analysis and Findings

## Baseline Performance

Без ML-фильтрации все ставки дают отрицательный ожидаемый ROI (bukmaker edge). DummyClassifier (step 1.1): ROI=-3.07%. Простое правило ML_Edge>=15 (step 1.2): ROI=2.40%. Логистическая регрессия + Kelly (step 1.3): ROI=-5.42%. Значимый сигнал начинается только с CatBoost.

**Кривая обучения по ROI:**

| Этап | ROI | N ставок | Ключевое изменение |
|------|-----|----------|-------------------|
| DummyClassifier | -3.07% | все | baseline |
| ML_Edge threshold | 2.40% | 1842 | простое правило |
| LogReg + Kelly | -5.42% | — | плохая калибровка |
| CatBoost + Kelly | 24.91% | 435 | breakthrough |
| + shrunken segments | 26.93% | 372 | регуляризация thresholds |
| + 1x2 filter | 28.58% | 233 | market segmentation |

## Feature Engineering Results

Базовый feature set (33 признака) из chain_1 зафиксирован после серии ablation-экспериментов в предыдущих сессиях:

**Ключевые признаки (по feature importance из CatBoost):**
- `Market` (cat): 15.2% — рынок ставки — наиболее информативный признак
- `ml_edge`: 12.8% — edge платформы vs implied probability
- `elo_implied_agree`: 8.1% — расхождение ELO-рейтинга и рыночной вероятности
- `day_of_week`: 10.1%, `hour`: 7.6% — временные паттерны ставок

**Что не сработало:**
- Lead-time фичи (step 4.2): lead_hours_log важен (4.2%) но снижает ROI на -6% из-за ухудшения Kelly-распределения
- Shadow features (step 2.2 chain_3): week_of_year=8.82% — маргинальное улучшение, не стабильно
- Depth=9 CatBoost (step 4.4): val AUC=0.9528 vs test AUC=0.7761 — severe overfitting

## Model Comparison

| Модель | Test ROI | Test AUC | Kelly threshold | Причина |
|--------|----------|----------|-----------------|---------|
| CatBoost (depth=7) | 24.91% | 0.7863 | 0.455 | лучшая калибровка |
| CatBoost (depth=9) | 4.18% | 0.7761 | 0.025 | overfitting val→test |
| LightGBM | 5.78% | — | низкий | плохая Kelly-калибровка |
| XGBoost | 1.63% | — | низкий | плохая Kelly-калибровка |
| CB+LGBM ensemble | -0.01% | — | — | несовместимые prob-distributions |
| 1x2-specific CatBoost | 11.28% | 0.8586 | — | overfitting (7150 train) |
| Walk-forward ensemble | -26.73% | — | — | threshold incompatibility |

CatBoost с глубиной 7 и 500 итерациями является единственной стабильно работающей архитектурой. Ключевое свойство: CatBoost обеспечивает точную калибровку вероятностей для Kelly criterion, дающую threshold ~0.455 (2-3% ставок с максимальным expected value).

## Segment Analysis

**Рыночная сегментация (on val):**
- 1x2: ROI=106.58%, n=229 (dominant profitable market)
- Asian Handicap: умеренный ROI
- Over/Under: отрицательный ROI

**Odds-bucket анализ (shrunken thresholds, step 4.6/4.10):**

| Bucket | Odds | Shrunken threshold | Val ROI | Test ROI | n |
|--------|------|-------------------|---------|----------|---|
| low | <1.8 | 0.475 | ~40% | ~16% | 583 |
| mid | 1.8-3.0 | 0.545 | ~113% | ~31% | 24 |
| high | >=3.0 | 0.325 | ~392% | ~333% | 9 |

Высокие коэффициенты (high-odds) дают аномально высокий ROI при малом n — потенциальный selection bias.

**Спортивная фильтрация:**
Все 233 ставки в лучшем результате — Soccer/1x2. Остальные виды спорта (Tennis, Basketball, etc.) не проходят Kelly-фильтрацию на уровне threshold=0.455.

## Stability & Validity

**Воспроизводимость:**
- Baseline ROI=24.91% воспроизводится точно (delta=0.0000%) при загрузке pipeline.pkl из chain_6.
- 5+ независимых запусков с разными seed дают стабильный результат (step 4.1 seed ensemble: 24.90%).

**Anti-leakage проверки:**
- Все пороги выбираются на val (64-80% train window), применяются к test один раз.
- Максимальный ROI нигде не превысил 35% на test — UAF leakage alert не срабатывал.
- Pre-match фильтр (lead_hours > 0) корректно исключает in-play ставки.
- Target encoding не применялся (CatBoost native cat features).

**Структурное ограничение (val contamination):**
Val (64-80%) является подмножеством train window (0-80%), поэтому val ROI (~88-200%) >> test ROI (~25-28%). Proper out-of-time split (step 3.3) показал ROI=0.94% — потолок при честной валидации. Текущий результат 28.58% реалистичен только при предположении, что будущее подобно недавнему прошлому.

## What Didn't Work

**Провальные гипотезы и причины:**

1. **Специализированная 1x2 модель (step 4.11):** val AUC=0.9978 → test AUC=0.8586, ROI=11.28%. Причина: 7150 обучающих примеров недостаточно для CatBoost с 33 признаками — severe overfitting.

2. **Walk-forward ensemble (step 4.8):** 4 модели на скользящих окнах, усреднение вероятностей. ROI=-26.73%. Причина: averaged probabilities из моделей с разными train windows имеют иное распределение Kelly-значений, делая threshold=0.455 несовместимым.

3. **Deeper CatBoost (depth=9, step 4.4):** val AUC=0.9528 → test AUC=0.7761 (расхождение 0.18 — классический признак overfitting). Kelly threshold упал до 0.025, отобрав 3022 ставки с ROI=4.18%.

4. **LightGBM/XGBoost (steps 4.2, 4.3):** Вероятности плохо откалиброваны для Kelly. Threshold требует значения <0.1, что означает отбор ставок с нулевым или отрицательным EV.

5. **Finer odds bins 4-bucket (steps 4.12, 4.13):** тестировалось без pre-match filter (баг в скриптах). Основная проблема: heavy_fav bucket (odds<1.6, 75% ставок) даёт ROI~16% — drag на общий результат. 3-бинная система более robust.

6. **Isotonic calibration (step 4.4 chain_6):** калибровка на val снижает test ROI с 24.91% до 14.68%. Причина: val contamination искажает калибровочную кривую.

7. **Optuna ROI objective (step 3.1):** прямая оптимизация ROI через Optuna переобучает на val, test ROI=2.44%. Косвенная оптимизация через AUC (step 3.2) даёт 9.22% — лучше, но ниже default CatBoost.
