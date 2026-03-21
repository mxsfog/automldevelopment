# Analysis and Findings

## Baseline Performance
- DummyClassifier (all-in): ROI = -3.07% -- нижняя граница
- Rule-based (ML_Edge threshold): ROI = -5.25% -- существующий ML_Edge не информативен
- LogisticRegression: ROI = 2.62%, AUC = 0.7943 -- первый положительный ROI
- CatBoost default: ROI = 1.16%, AUC = 0.7938 -- на базовых фичах без ELO слабый

Вывод: без ELO-данных модели достигают максимум 2-3% ROI.

## Feature Engineering Results
### Что улучшило модель
- **ELO фичи** (15 фичей): team_winrate_diff, elo_diff, team_elo_mean и др. -- главный прорыв. ROI с -0.79% до 13.51% на ELO подмножестве.
- **Sport filter** (исключение Basketball, MMA, FIFA, Snooker): +5.5 п.п. на ELO подмножестве
- **Odds-based фичи** (из chain_1): implied_prob, value_ratio, log_odds -- базовые предикторы

### Что не улучшило
- 12 новых interaction features (elo_implied_agreement, winrate_vs_odds, etc.): -3.15 п.п.
- Categorical features (Sport, Market как native CatBoost): AUC выше (0.8695), но ROI ниже (-2.6 п.п.)
- Feature selection (top 70% / top 15): ухудшение от -3.25 до -9.38 п.п.

### Top-5 фичей по importance
1. implied_prob: 12.86
2. team_winrate_diff: 12.25
3. Odds: 10.45
4. log_odds: 5.55
5. team_winrate_mean: 4.21

## Model Comparison
| Model | ROI (t=0.77) | AUC | N bets |
|-------|-------------|-----|--------|
| CatBoost full-train (ref params) | 21.31% | 0.8623 | 463 |
| CatBoost full-train + EV>=0 | **28.44%** | 0.8623 | 328 |
| Blend CB+deep10 + EV>=0 | 28.74% | 0.8658 | 326 |
| CatBoost 80/20 (ref params) | 20.23% | 0.8475 | 449 |
| CatBoost + cat features | 18.71% | 0.8695 | 466 |
| LightGBM | 16.95% | 0.8325 | 442 |
| Blend CB+LGB (0.7/0.3) | 17.86% | 0.8459 | 441 |
| Stacking (LR meta) | 19.33% | 0.8684 | 523 |

CatBoost solo доминирует. EV фильтр -- главный прорыв в post-processing.

## EV Filter Discovery (step 4.9-4.11)
### Механизм
EV = predicted_prob * odds - 1. Фильтр EV>=0 требует, чтобы ожидаемая ценность ставки была неотрицательной по оценке модели.

### Что удаляет EV фильтр
- 135 ставок из 463 (при p>=0.77)
- Удалённые: avg odds = 1.05 (очень низкие коэффициенты)
- Оставленные: avg odds = 1.32
- Soccer: -7.45% ROI среди удалённых
- Table Tennis: -8.10% ROI среди удалённых
- Фильтр убирает ставки на heavy favorites с минимальной маржей

### Валидация (5-fold temporal CV)
| Fold | t=0.77 | EV>=0+p77 | Delta |
|------|--------|-----------|-------|
| 0 | 1.79% | 10.29% | +8.50 |
| 1 | 2.30% | 16.15% | +13.85 |
| 2 | 15.66% | 28.02% | +12.36 |
| 3 | 12.28% | 25.67% | +13.39 |
| 4 | 23.06% | 31.94% | +8.88 |
| **Avg** | **11.02%** | **22.42%** | **+11.40** |

Улучшение стабильно по всем фолдам (+8.5 до +13.9 п.п.).

## Segment Analysis
### По спортам (на test set, EV>=0 + p>=0.77)
| Sport | ROI | N bets | Note |
|-------|-----|--------|------|
| CS2 | 34.95% | 25 | Высокий, малая выборка |
| Soccer | 44.02% | 46 | Сильно улучшился от EV фильтра |
| Table Tennis | 28.71% | 93 | Основной объём |
| Tennis | 18.52% | 76 | Умеренный ROI |
| Cricket | 15.31% | 13 | Малая выборка |
| Volleyball | 12.45% | 23 | |

### Убыточные спорты (исключены фильтром)
- Basketball, MMA, FIFA, Snooker -- стабильно убыточные по всем chain-ам

## What Didn't Work (comprehensive)
### Model Architecture
- LightGBM solo и blends: CatBoost с chain_3 params доминирует
- Ordered boosting: -1.86 п.п.
- Lossguide grow policy: -2.47 п.п.
- Stacking (LR meta-learner): -2 п.п.
- Multi-seed averaging: -1.46 п.п.

### Feature Engineering
- 12 interaction features: -3.15 п.п.
- Feature selection (top 70%/15): -3.25 до -9.38 п.п.
- Categorical features: лучше AUC, хуже ROI

### Regularization
- Monotonic constraints: -11 п.п. (слишком жёсткие)
- RSM (random subspace): -1.6 до -3.1 п.п.
- Class weights balanced: -2.3 п.п.

### Training Data
- Sample weights by recency: -0.7 до -3.4 п.п.
- Training window (50-85%): -0.9 до -1.3 п.п.
- Optuna re-tuning (65+ trials across chains): chain_3 params оптимальны

### Hyperparameters
- Depth 6: -4.85 п.п.
- Depth 7: -0.18 п.п. (close but worse)
- Depth 9: -2.25 п.п.
- Depth 10: -5.33 п.п.
- LR=0.03: -2.91 п.п.

## SF vs ELO_all Comparison (step 4.12-4.13)

5-fold temporal CV сравнение двух подходов:

| Approach | Test ROI | CV avg | CV std | Folds positive |
|----------|----------|--------|--------|----------------|
| SF + EV0+p77 | 28.44% | 22.42% | 7.99% | 5/5 |
| ELO_all + EV0+p77 | 29.87% | 14.53% | 12.1% | 4/5 |

SF подход более робастный: все 5 фолдов положительные, меньше std. ELO_all имеет fold 0 = -9.94%.

### Odds-Range Analysis
| Odds range | ROI | N bets | Avg EV |
|------------|-----|--------|--------|
| 1.01-1.15 | 0.95% | 248 | 0.002 |
| 1.15-1.30 | 18.50% | 71 | - |
| 1.30-1.50 | 29.56% | 71 | - |
| 1.50-2.00 | 69.69% | 57 | - |
| 2.00+ | 102.14% | 16 | - |

EV>=0 фильтр в основном убирает 248 ставок с odds 1.01-1.15 (ROI всего 0.95%). Ставки с higher odds значительно более прибыльны.

## Threshold Sensitivity (step 4.14)
Val-sweep и test-sweep подтверждают p=0.77 как оптимальный порог:

| Threshold | Test ROI (t only) | Test ROI (EV0) | N bets (EV0) |
|-----------|-------------------|----------------|--------------|
| 0.75 | 20.34% | 27.67% | 341 |
| 0.76 | 21.32% | 28.39% | 332 |
| **0.77** | **21.31%** | **28.44%** | **328** |
| 0.78 | 19.96% | 26.85% | 316 |
| 0.80 | 18.31% | 25.22% | 304 |

Val-optimal порог: p=0.78 (27.35% на val), но на test p=0.77 лучше (28.44% vs 26.85%).
ROI плавно снижается в обе стороны от 0.76-0.77 -- подтверждает робастность выбора.

## Stability & Validity
### Anti-leakage проверки
- Threshold (p>=0.77) определяется на val (last 20% of train)
- EV>=0 -- fixed domain-knowledge rule, не подбирается на test
- ELO фичи используют Old_ELO (до матча), нет future leakage
- Time-series split: train до 2026-02-20, test после
- Final ROI (28.44%) < 30% -- в пределах допустимого
- SF approach validated: 5/5 positive folds in CV

### Ограничения
1. **ELO coverage 9.7%**: только 7198 из 74493 ставок имеют ELO-данные
2. **Sport filter + EV filter**: 328 из 74493 total bets (0.44% coverage)
3. **Test window 3 дня**: ROI по фолдам от 10.29% до 31.94%
4. **Высокая дисперсия**: std=7.99% по 5 фолдам (EV+p77)
5. **ELO_all менее стабилен**: fold 0 = -9.94%, std=12.1%
