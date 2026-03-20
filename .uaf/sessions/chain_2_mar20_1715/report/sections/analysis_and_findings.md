# Analysis and Findings

## Baseline Performance

| Step | Method | ROI | AUC | N_bets |
|------|--------|-----|-----|--------|
| 1.1 | DummyClassifier (most_frequent) | -3.07% | - | 14899 |
| 1.2 | Rule-based (ML_Edge threshold) | -7.35% | - | 2041 |
| 1.3 | LogisticRegression | 1.46% | 0.7897 | 2656 |
| 1.4 | CatBoost default | 2.48% | 0.7946 | 2461 |

Без ML ставки дают -3.07% (все ставки проигрывают маржу букмекера). Rule-based подход ещё хуже. Первый положительный ROI -- LogisticRegression с порогом 0.81. CatBoost default даёт 2.48%, но останавливается на 19 итерациях из-за early stopping -- данные слишком зашумлены для глубокой модели без информативных фичей.

Ключевое наблюдение из Phase 1: feature importance CatBoost показал 85%+ на Odds/implied_prob. Модель по сути прогнозирует по структуре коэффициентов, а не по содержательным спортивным данным.

## Feature Engineering Results

### Chain_1 features (proven safe, Shadow Feature Trick)
10 фичей, принятых через shadow feature trick в chain_1 (delta > 0.002):
- `log_odds`, `implied_prob`, `value_ratio` -- трансформации коэффициентов
- `edge_x_ev`, `edge_abs`, `ev_positive`, `model_implied_diff` -- взаимодействия Edge/EV
- `log_usd`, `log_usd_per_outcome`, `parlay_complexity` -- размер ставки и парлай

### ELO features (новое в chain_2)
15 safe ELO-фичей из `elo_history.csv` и `teams.csv`:
- Рейтинговые: `team_elo_mean`, `elo_diff`, `elo_spread`, `elo_mean_vs_1500`
- Виннрейт: `team_winrate_mean`, `team_winrate_diff`
- Объем: `team_total_games_mean`, `k_factor_mean`, `has_elo`

**Leakage detection**: первоначальные ELO-фичи включали `ELO_Change` (изменение рейтинга после матча) -- это target leakage, т.к. ELO_Change зависит от результата матча. ROI с leakage был 25.90%, после удаления -- 10.70%.

**Эффект ELO-фичей:**
- Все данные + safe ELO: ROI=2.38% (vs 0.34% baseline), +2.04 п.п.
- ELO-only subset: ROI=10.70%, AUC=0.8540 -- прорывной результат

**Ограничение**: ELO-данные доступны только для 9.7% ставок (7198 из 74493).

## Model Comparison

Все модели обучены на ELO-only subset (ставки с данными ELO рейтингов):

| Модель | ROI | AUC | Threshold | N_bets |
|--------|-----|-----|-----------|--------|
| CatBoost default | 10.70% | 0.8540 | 0.62 | 725 |
| Optuna CatBoost | 16.63% | 0.8431 | 0.73 | 634 |
| Optuna LightGBM | ~14% | ~0.84 | - | - |
| XGBoost default | ~11% | ~0.83 | - | - |
| Ensemble CB50 (CB 50%/LGB 25%/XGB 25%) | **16.76%** | **0.8501** | 0.62 | 743 |
| Rank average (CB+LGB+XGB) | 15.09% | 0.84 | - | - |

Optuna дал значительное улучшение CatBoost (+5.93 п.п.). Основные гиперпараметры: depth=7, lr=0.214, l2_leaf_reg=1.15.

Ensemble CB50 -- лучший результат. Взвешенное среднее с доминированием CatBoost (50%) стабильнее, чем equal average или rank average.

## Segment Analysis

### По спортам (ELO-only subset, best threshold)
| Спорт | ROI | Комментарий |
|-------|-----|-------------|
| Soccer | 26% | Лучший сегмент, больше всего ELO-данных |
| Tennis | 17% | Стабильный |
| Table Tennis | 16% | Стабильный |
| Basketball | 11% | Прибыльный на ELO-subset (в отличие от non-ELO) |

**Все спорты прибыльны** на ELO-subset. Это контрастирует с chain_1, где Basketball, MMA, FIFA, Snooker были убыточны и требовали фильтрации.

### Dual-model стратегия
- ELO-модель: ROI=16.63%, 634 ставки
- Non-ELO модель: ROI=4.72%, фильтрованный ~5%
- Комбинированная: ROI=5.73% -- non-ELO component размывает результат

Рекомендация: использовать ELO-модель для ELO-ставок, chain_1 модель с segment filtering для остальных.

## Stability & Validity

### Temporal stability (4 temporal splits)
| Test % | ROI | AUC | Период |
|--------|-----|-----|--------|
| 15% | 19.69% | 0.85 | Последние дни |
| 20% | 6.40% | 0.83 | |
| 25% | 8.67% | 0.83 | |
| 30% | 13.83% | 0.84 | |

- **Mean ROI: 12.15%, std: 5.12%**
- При фиксированном пороге t=0.60: все 4 сплита >7%, 3 из 4 >12%
- Разброс ожидаем для 3-дневного test window на спортивных данных

### Anti-leakage compliance
- Threshold выбирается на val (последние 20% train), применяется к test однократно
- ELO_Change/New_ELO исключены (post-match data)
- Target encoding не используется
- Все фичи доступны до начала матча (pre-match)
- Максимальный ROI=16.76% -- в пределах разумного (< 30% sanity check)

## What Didn't Work

### Chain_1 (потолок 7.32%)
9 подходов Phase 4 chain_1 дали маргинальный эффект (<0.1 п.п. каждый):
- Weighted/stacking ensemble
- EV-based selection
- Per-odds-range thresholds
- Temporal features
- Deeper CatBoost
- Filtered training
- Fine val threshold
- Categorical Sport encoding
- Market/Tournament filtering

**Причина**: Odds/implied_prob доминировали 85% feature importance. Без внешних данных модель не могла выйти за пределы информации в коэффициентах.

### Chain_2
- **Dual-model стратегия**: non-ELO component (~5% ROI) размывает результат ELO-модели (16.76%)
- **ELO coverage**: только 9.7% ставок имеют ELO-данные. Это основное ограничение для масштабирования
- **Rank average ensemble**: чуть хуже weighted (15.09% vs 16.76%)
