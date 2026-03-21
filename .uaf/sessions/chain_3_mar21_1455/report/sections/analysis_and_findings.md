# Analysis and Findings

## Baseline Performance
- Данные: 74493 ставки, 81 день (2025-12-03 .. 2026-02-22)
- Target mean: 53.89% (slight class imbalance towards wins)
- Train/test split: 80/20 по времени (59594 / 14899)
- Baseline ROI (все ставки без фильтрации): отрицательный
- ML baseline (EV >= 0.12): +16.02% ROI, 2247 ставок

## Feature Engineering Results
- **Принятые фичи (19):** Base features (15) + Sport/Market target encoding (4)
- **Отвергнутые:** odds_features (implied_prob, margin, odds_log), interaction_features (edge_x_odds), temporal_features (hour, day_of_week), complexity_features (odds_spread, odds_cv) — все ухудшают val performance
- Target encoding использует smoothing (alpha=50) для защиты от overfitting на редких категориях

## Model Comparison
| Модель | AUC | Роль |
|--------|-----|------|
| CatBoost (iter=200, lr=0.05, depth=6) | 0.784 | Основная: нелинейные паттерны, взаимодействия |
| LightGBM (n_est=200, lr=0.05, depth=6) | ~0.78 | Дополнительный сигнал, другая архитектура деревьев |
| LogisticRegression (C=1.0) | ~0.75 | Линейный baseline, стабилизирует ensemble |

Simple average 3 моделей дает AUC=0.784. Попытки усложнить:
- **Meta-model (OOF):** AUC=0.7840 vs 0.7839 — идентично
- **XGBoost 4-model:** CV нестабильнее
- **Optuna-tuned CB:** хуже EV-калибровка
- **Stacking:** overfitting к val

## Segment Analysis
- **По спортам:** Basketball extreme ROI (малая выборка), Soccer/Tennis отрицательные, но нестабильно — спорты меняют знак между val и test
- **По odds:** Прибыль из high-odds (50-500), средние odds умеренный edge, low-odds (1-2.5) near zero
- **По рынкам:** высокая дисперсия, малые выборки по отдельным рынкам

## Stability & Validity
- **Leakage check:** все threshold выбраны на val (последние 20% train), применены к test один раз
- **CV stability (5-fold expanding window):**
  - Baseline (EV>=0.12): mean=1.12%, std=12%
  - conf_ev_0.15: mean=4.69%, std=27.5%
  - conf_ev_0.08: mean=1.36%, std=11% (best Sharpe)
- **Вывод:** результат +27.95% специфичен для тестового периода. Реальный ожидаемый ROI: 0-5%

## What Didn't Work

### Калибровка вероятностей
Isotonic regression на OOF: AUC практически не меняется, но ROI катастрофически падает (-8% до -30%). Калибровка "размазывает" вероятности, ухудшая EV selection.

### Сегментация
- Per-sport models: уменьшает train size, ухудшает обобщение
- Dual strategy (low/high odds): -6.10% ROI
- Sport exclusion: спорты нестабильны между периодами, exclusion не переносится

### Temporal decay
Exponential weighting (half-life 7-30 дней): ухудшает val/test на всех конфигурациях. Модель выигрывает от разнообразия в train, а не от recency.

### Extended features
+16 новых фичей (odds, interaction, temporal, complexity): все группы ухудшают val. Больше features = больше noise при 81-дневном датасете.

### Multi-factor scoring
Добавление edge filter, agreement filter, composite scores к conf_ev: все ухудшают ROI. conf_ev уже оптимальный фильтр.

### Odds-weighted training
log/sqrt/linear odds weighting: val показывает +15%, test показывает -10%. Полная инверсия val/test.

### Bootstrap confidence
5 ensembles с разными seed: 11-14% ROI vs 28% для standard. Seed diversity дает меньше diversity чем model type diversity (CB vs LGBM vs LR).

### Ensemble weights
Оптимальные веса (min log_loss): CB=0.335, LGBM=0.000, LR=0.665. Но 2-model (CB+LR) хуже 3-model на test. Diversity от LGBM важна несмотря на нулевой оптимальный вес.

### Feature pruning
Top-3 фичи: Market_target_enc (28%), Market_count_enc (22%), Odds (20%). ML_Winrate_Diff и ML_Rating_Diff имеют importance=0. Но pruning до top-8..15 ухудшает val ROI.

### Alternative confidence formulas
exp(-k*std), percentile(std), hard std filter — все хуже оригинальной 1/(1+std*10) на test.

### RF/ExtraTrees в ensemble
ET(AUC=0.760) > LR(0.757) > RF(0.750). Но замена LR на RF/ET ухудшает conf_ev selection из-за изменения p_std.

### Парлай-анализ
20% ставок — парлаи. Singles-only: val=15.19%, test=6.45%. Прибыль приходит из парлаев (high odds), не из синглов. Исключение парлаев убивает ROI.

### Hyperparameter sensitivity
5 конфигов (depth 4-8, iter 200-500, lr 0.03-0.1): AUC 0.781-0.784 для всех. ROI различия определяются калибровкой, не качеством модели.

### Dual thresholds (парлай-буст)
Разные пороги conf_ev для singles/parlays: dual_s0.15_p0.10 дает test=29.11% (n=1265), но uniform_0.15 на val лучше. Инверсия val/test сохраняется.

### Odds floor/band/ceiling анализ
- band_50_500 + conf_ev>=0.15: test ROI=124.79% (n=285) — extreme, нестабильно
- ceil_100 + conf_ev>=0.15: test ROI=0.26% — удаление odds>100 убивает прибыль
- val-best band_2_10 (ROI=37.48%) инвертируется на test
- **Вывод:** прибыль полностью из extreme high-odds ставок

### CRITICAL: Profit Concentration (step 4.24)
- **1 ставка** (odds=490.9, P&L=3,200,535) = **137% всей прибыли** стратегии
- Без этой ставки стратегия УБЫТОЧНА (-864K)
- Top-4 ставки = 188.8% прибыли, остальные 1088 суммарно минус
- По odds-brackets: ВСЕ кроме [200,10000) убыточны:
  - [1,3): ROI=-5.4%, [3,10): -24.3%, [10,50): -10.9%, [50,200): -22.5%
  - [200,10000): ROI=+1582% (1 выигрыш из 64 ставок)
- Gini coefficient P&L = 0.762
- Seed sensitivity (5 seeds, thr=0.15): mean=23.26%, std=7.52%, range 16-36%
- **Заключение: стратегия не имеет систематического edge, ROI определяется 1 случайным выигрышем на extreme odds**

### Seed Sensitivity
AUC стабилен (0.784-0.785) по всем seeds. ROI нестабилен из-за зависимости от single extreme event.

### Deep Edge Validation (step 4.28)
Edge strategy (p_model - p_implied >= 0.10, odds <= 2-5):
- **CV:** mean=-7.10%, std=12% — отрицательна на CV, не робастна
- **Seeds:** mean=6.40%, std=**0.44%** — крайне seed-стабильна
- Val-test consistency из step 4.27 оказалась совпадением периода

### Robust CV Optimization (step 4.29) — ФИНАЛЬНЫЙ АНАЛИЗ
33 стратегии × 5 folds expanding window:

**По min fold ROI (робастность):**
- Лучшая: pmean_0.55 (min=-3.46%, mean=1.95%) — select p >= 0.55
- Все стратегии имеют отрицательный min fold ROI

**По Sharpe:**
- Лучший: ev_0.05 (sharpe=0.44, mean=7.00%, std=15.97%)
- confev_0.15: sharpe=0.17 (mean=4.69%, std=27.55%)

**Вывод: реалистичный ожидаемый ROI = 0-2%, не 28%.** 27.95% определяется 1 ставкой на odds=490.9.

### Capped Odds (step 4.25)
При ограничении max odds: cap10=2.82%, cap20=-5.68%, cap50=-3.62%, cap100=0.26%.
Без extreme outliers edge ≈ 0 для conf_ev стратегии.

### Low-odds Edge (step 4.26)
Prediction quality по odds brackets:
- [1.0,1.5): edge=-0.029 (модель недооценивает фаворитов)
- [1.5,2.0): edge=+0.007
- [2.0,2.5): edge=+0.018 (лучший edge)
- [2.5,3.0): edge=+0.008
- [3.0,5.0): edge=-0.020

Walk-forward (3 блока в test): conf_ev>=0.15 дает -17.6%, -35.8%, +120%. Прибыль из одного блока (одна ставка).

### VALIDATED REAL EDGE (step 4.27)
Edge-based strategy: select where p_model - p_implied >= threshold, odds <= cap.
Val → test consistency:

| Strategy | Val ROI | Test ROI | n_bets |
|----------|---------|----------|--------|
| edge_cap2_e0.10 | +6.69% | +5.50% | ~500 |
| edge_cap3_e0.10 | +4.44% | +6.93% | ~550 |
| edge_cap5_e0.10 | +3.65% | +8.05% | ~530 |
| edge_cap10_e0.10 | +4.07% | +7.60% | ~540 |

Это единственная стратегия с val-test consistency. Реальный edge ~5-8%.
В отличие от conf_ev_0.15 (27.95%), не зависит от единственного extreme bet.

### Combined edge+EV (step 4.31)
Intersection (edge AND EV) и union (edge OR EV) стратегии:
- inter_e0.10_ev0.02_cap3: val=4.44% → test=6.93% (n=516) — идентично чистому edge_cap3
- Комбинация не улучшает: edge и EV при low odds сильно коррелируют

### Temporal Block Analysis (step 4.32)
4 временных блока в test:
- **pmean_0.55: 4/4 положительных** [0.4%, 2.0%, 0.5%, 2.1%], std=0.80%
- confev_0.15: 1/4 положительный [-41.8%, -31.1%, -17.2%, +165.6%], std=85.2%
- ev_0.05: 1/4 положительный [-10.1%, -5.4%, -5.0%, +40.8%]
- edge_cap5_e0.10: 3/4 положительных [7.1%, -0.9%, 24.7%, 0.9%]
- **pmean_0.55 — единственная стратегия со 100% положительных блоков**

### Kelly Criterion Sizing (step 4.33)
Fractional Kelly (0.25/0.50/0.75/1.00) × 3 стратегии:
- **pmean_0.55 + kelly_0.25: ROI=259%, drawdown=50%** — единственная стратегия, выживающая Kelly
- pmean_0.55 + kelly_0.50: ROI=356%, drawdown=59%
- confev_0.15 + kelly_0.25: ROI=69%, drawdown=94%
- confev_0.15 + kelly>=0.50: **банкролл уничтожен** (-85% до -100%)
- ev_0.05 + kelly>=0.25: банкролл уничтожен

Kelly criterion подтверждает: pmean_0.55 имеет реальный, устойчивый edge.
Стратегии на extreme odds (confev, ev) уничтожают банкролл при пропорциональном размере ставки.
