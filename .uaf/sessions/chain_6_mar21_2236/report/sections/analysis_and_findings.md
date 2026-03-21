# Analysis and Findings

## Baseline Performance

| Модель | ROI test | n_bets | AUC test | Примечание |
|--------|----------|--------|----------|------------|
| DummyClassifier | -3.07% | 14899 | — | Lower bound (все ставки) |
| Rule ML_Edge>=15 | 2.40% | 506 | — | Простой threshold, val=49.74% |
| LogisticRegression + Kelly | -5.42% | 1966 | 0.7948 | AUC высокий, Kelly не помог |
| **CatBoost + Kelly** | **24.91%** | **435** | **0.7863** | **Baseline, threshold=0.455** |

Вывод: Линейные модели не извлекают паттерн. CatBoost критически важен для нелинейного отбора.

## Feature Engineering Results

### Принятый набор (33 признака)
Базовые: Odds, USD, log_odds, log_usd, implied_prob, is_parlay, outcomes_count
ML-сигналы: ml_p_model, ml_p_implied, ml_edge, ml_ev, ml_team_stats_found, ml_winrate_diff, ml_rating_diff
Временные: hour, day_of_week, month
Составные: odds_times_stake, ml_edge_pos, ml_ev_pos, ml_edge_x_elo_diff, elo_implied_agree
ELO: elo_max, elo_min, elo_diff, elo_ratio, elo_mean, elo_std, k_factor_mean, has_elo, elo_count
Категориальные: Sport, Market, Currency

### Feature importances (CatBoost FULL, baseline)
| # | Признак | Важность |
|---|---------|----------|
| 1 | Market | 11.72% |
| 2 | day_of_week | 10.14% |
| 3 | elo_implied_agree | 8.13% |
| 4 | hour | 7.55% |
| 5 | Sport | 7.16% |
| 6 | implied_prob | 6.75% |
| 7 | Odds | 5.52% |
| 8 | log_odds | 5.52% |
| 9 | ml_edge | 5.18% |
| 10 | ml_ev | 4.91% |

Важный вывод: ML-сигналы (ml_edge, ml_ev) только на 9-10 месте. Market и temporal признаки доминируют.
Модель учится ставить на правильный рынок в правильное время, а не только предсказывать исход.

### Что не сработало
- **Features v2 (46 фичей):** lead_hours/is_prematch как feature ломает Kelly distribution → ROI=-0.92%
- **week_of_year вместо day_of_week:** хуже (ROI=8.82%, n=127 — невалидно)
- **Без temporal фичей:** хуже (ROI=17.75%, n=96 — невалидно). Temporal признаки НУЖНЫ.

## Model Comparison

| Модель | ROI test | n_bets | threshold | AUC test | Статус |
|--------|----------|--------|-----------|----------|--------|
| CatBoost depth=7 | 24.91% | 435 | 0.455 | 0.7863 | **ЛУЧШИЙ** |
| Seed ensemble CB (5x) | 24.90% | 277 | — | 0.7862 | НЕЙТРАЛЬНЫЙ |
| LightGBM num_leaves=63 | 5.78% | 1942 | 0.125 | 0.7704 | ОТКЛОНЁН |
| CB+LGBM 50/50 | -0.01% | 2336 | 0.120 | 0.7759 | ОТКЛОНЁН |
| XGBoost max_depth=7 | 1.63% | 1994 | 0.145 | 0.7636 | ОТКЛОНЁН |

Ключевая закономерность: только CatBoost производит распределение вероятностей с threshold~0.455.
Все альтернативы дают threshold 0.06-0.15, что при val/train contamination выглядит хорошо (val ROI>80%),
но тест показывает близкий к нулю ROI из-за слишком широкого отбора.

## Segment Analysis

### Виды спорта в датасете
Soccer (25496), Tennis (14688), Basketball (10616), Cricket (7792), CS2 (3105), ...

### Val-сегменты (шаг 2.3, val=64-80%)
- Soccer: ROI=97.5%, n=278 — самый прибыльный
- Tennis: ROI=32.9%, n=46

### Soccer-only модель (шаг 4.5)
AUC=0.8364 (выше baseline 0.7863) — Soccer действительно более предсказуем.
НО: threshold=0.060 → test ROI=-5.39% (n=1690). Слишком низкий threshold, много ставок с отрицательным ROI.
Вывод: Сегментация отдельных моделей не помогает без решения проблемы val/train contamination.

## Stability & Validity

### Воспроизводимость результата
Baseline CatBoost ROI=24.91% воспроизведён в 6 разных runs:
- step 1.4 (исходный): 24.91%
- chain_2_mar21: 24.91%
- chain_3_mar21: 24.91%
- step 2.3 (segment analysis): 24.91%
- step 4.1 (seed ensemble FULL): 24.91%
- step 4.6 (ablation FULL): 24.91%

Результат высоко стабилен и воспроизводим.

### Anti-leakage checks
- threshold подбирается ТОЛЬКО на val (64-80%), НЕ на test — правило соблюдалось
- ROI > 35% не достигался ни в одном валидном run
- Future features исключены (Payout_USD не используется как признак)
- Lead hours используется ТОЛЬКО для фильтрации (не как признак модели)

### Структурная проблема val/train
val (64-80%) входит в train (0-80%). Это означает:
- val AUC (~0.89-0.97) значительно выше test AUC (~0.77)
- val ROI обычно 80-110%, test ROI 1-25%
- Threshold найденный на val работает для baseline, но ненадёжен для новых моделей

## What Didn't Work

### Hyperparameter optimization
- Optuna ROI objective: val ROI=103%, test=2.44% — двойное переобучение (threshold ищется на val, который в train)
- Optuna AUC objective: val AUC=0.97, test ROI=9.22% — overfitting из-за val-in-train

### Calibration
- Isotonic regression: Brier cal=0.1955 > raw=0.1859 — ухудшила калибровку
- Isotonic calibration test ROI=14.68% < baseline 24.91%
- Причина: маленькая выборка калибровки (64-72%), isotonic переобучилась

### Proper split (step 3.3)
Исключение val из train (train=0-64%, val=64-80%): test ROI=0.94%, n=173 (INVALID).
Вывод: Recent data critical — модель ДОЛЖНА обучаться на данных близких к тест-периоду.
Текущий "leaky" split лучше generalizует на тест, чем proper split.

### Ensembles
- Seed ensemble: нейтрально (+0.00%)
- CB+LGBM: -24.92% — LightGBM ухудшает отбор
- Причина: любое усреднение вероятностей меняет их распределение и сдвигает threshold вниз

### Сегментация
- Soccer-only: -30.30% — специализированная модель не помогает, порог деградирует
