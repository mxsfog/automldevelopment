# Analysis and Findings

## Phase 1: Baselines (steps 1.1-1.4)

| Step | Method | ROI | Вывод |
|------|--------|-----|-------|
| 1.1 | DummyClassifier | -3.07% | Lower bound, все ставки убыточны |
| 1.2 | Rule ML_Edge | -7.35% | ML_Edge как единственный фильтр не работает |
| 1.3 | LogisticRegression | 2.70% | Линейная модель уже положительна |
| 1.4 | CatBoost default | 0.34% | Early stopping на 4 итерации, Odds = 96% FI |

## Phase 2: Feature Engineering (steps 2.5a-c)

Ключевое открытие: **ELO-only подмножество** дает прорыв.

- Без ELO: ROI=0.34% (Odds-доминирование, 96% FI)
- +ELO на всех данных: ROI=-0.75% (низкое покрытие 9.8% размывает)
- **ELO-only subset: ROI=13.18%** (прорыв, diversified FI)

ELO-фичи (team_winrate_diff, elo_diff, k_factor_mean) диверсифицировали feature importance и дали модели информацию о силе команд, которую odds не полностью capture.

## Phase 3: HPO (step 3.1)

Optuna TPE (40 trials) на ELO-only CatBoost:
- **ROI=18.59%** (depth=8, lr=0.08, l2=21.1, t=0.77)
- +5.41 п.п. vs default CatBoost ELO-only

## Phase 4: Free Exploration (steps 4.1-4.11)

### Что протестировано

**Ensembles (4.1-4.3, 4.5, 4.9):**
- CB50 (50% CB + 25% LGB + 25% XGB): ROI=14.97%
- CB65 (65% CB + 20% LGB + 15% XGB): ROI=18.35%
- CB80 (80% CB + 10% LGB + 10% XGB): ROI=18.50%
- Вывод: ensembles не превосходят CB solo (20.23%) на sport-filtered data

**Robust threshold (4.2, 4.5):**
- Multi-fold median threshold: 0.59-0.73 (нестабилен)
- Fixed t=0.77 стабильно лучше на test
- Вывод: для малых выборок adaptive threshold переобучается, фиксированный надежнее

**Optuna XGBoost (4.3):**
- XGB solo ROI=16.66%
- Не превосходит CatBoost

**Sport filter (4.4, 4.6-4.9, 4.11):**
- Train on all + infer SF: ROI=19.80%
- **Train on SF + infer SF: ROI=20.23%** (best)
- Per-sport analysis: Basketball -4.62% (main drag), Soccer +35.37%, CS2 +35.18%
- Val-determined optimal combo (Cricket, NBA2K) хуже chain_2 original filter

**Calibration (4.4 from chain_2):**
- Isotonic calibration: ROI=13.02% (ухудшает)

**Multi-seed averaging (4.6, 4.9):**
- 5 seeds (42-46): ROI=18.89% (хуже single seed=42 = 20.23%)
- Averaging сглаживает полезные паттерны

**Fresh Optuna (4.4, 4.8):**
- Wider search (depth 5-10): overfitting (val=20.71%, test=8.55%)
- Conservative search on SF: не побил базовые params

### Per-sport ROI (test set, t=0.77)

| Sport | ROI | N bets | Вывод |
|-------|-----|--------|-------|
| Soccer | 35.37% | 60 | Top performer |
| CS2 | 35.18% | 35 | Top performer |
| Table Tennis | 21.80% | 107 | Стабильный, высокий volume |
| MMA | 22.23% | 4 | Мало данных, исключен |
| Ice Hockey | 20.31% | 6 | Мало данных |
| FIFA | 15.94% | 8 | Мало данных, исключен |
| Tennis | 12.25% | 118 | Стабильный, высокий volume |
| Dota 2 | 11.95% | 9 | Мало данных |
| Cricket | 10.90% | 33 | Стабильный |
| League of Legends | 10.58% | 8 | Мало данных |
| Volleyball | 9.59% | 27 | Стабильный |
| Baseball | 7.53% | 7 | Мало данных |
| Basketball | -4.62% | 90 | **Убыточен, исключен** |

## Robustness Analysis

4-fold temporal CV (expanding window):
- **Mean ROI: 13.55%** (all folds positive)
- **Std: 8.09%** (high variance inherent to betting data)
- Worst fold: 3.72% (still positive)
- Best fold: 25.13%

Sport filter improvement over no filter: +1.63 п.п. в среднем по фолдам.
