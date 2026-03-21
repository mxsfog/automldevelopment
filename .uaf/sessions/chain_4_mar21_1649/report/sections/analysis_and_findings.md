# Analysis and Findings

## 1. Walk-Forward Validation (Steps 4.1-4.2)

Expanding window, недельный ретрейн, 6 тестовых блоков.

| Стратегия | Overall ROI | Pos blocks | N bets |
|-----------|-------------|------------|--------|
| conf_ev_0.15 | +11.44% | 4/6 | 6116 |
| ev_0.05 | +1.77% | 3/6 | 14582 |
| pmean_0.55 | -1.37% | 3/6 | 36104 |
| Platt + conf_ev | +2.24% | 3/6 | 8714 |

Platt scaling ухудшает результат, размывая conf_ev фильтр.

## 2. Decomposed Filtering (Step 4.8) -- лучшая стратегия

Разложение conf_ev на отдельные фильтры:

| Стратегия | ROI | Pos blocks | Формула |
|-----------|-----|------------|---------|
| ev005_agree_p02 | **+13.80%** | **5/6** | EV>=0.05 AND p_std<=0.02 |
| conf_ev_0.15 | +11.44% | 4/6 | EV*conf >= 0.15 |
| agree_only_p02 | +1.79% | 3/6 | p_std<=0.02 |

## 3. Odds Cap Analysis (Steps 4.6, 4.9)

| Ограничение | ev005_agree ROI | N bets |
|-------------|-----------------|--------|
| Без ограничений | +13.80% | 6718 |
| odds <= 50 | -1.02% | ~5800 |
| odds <= 10 | -0.14% | 5517 |
| odds 2-5 | +3.79% | 1557 |

**Весь ROI от extreme odds.** При разумных ограничениях edge исчезает.

## 4. Seed Stability (Step 4.12)

| Стратегия | Mean ROI | Std ROI | Range |
|-----------|----------|---------|-------|
| ev005_agree_p02 | 12.11% | 2.10% | 8.9-14.8% |
| conf_ev_015 | 12.11% | 2.19% | 8.3-14.9% |

Обе EV-стратегии seed-стабильны (все seeds положительные).

## 5. Permutation Test (Steps 4.13, 4.15)

**Uncapped (all odds):** ROI невалиден для permutation (perm_mean=1136% из-за extreme odds).

**Capped (odds<=10):**
- ROI: real=-0.14% vs perm_mean=21.50%, p=1.0 (NOT significant)
- **AUC: real=0.7278 vs perm_mean=0.5002, p=0.020 (SIGNIFICANT)**

Модель ПРЕДСКАЗЫВАЕТ (AUC значимо лучше random), но НЕ ГЕНЕРИРУЕТ ROI при moderate odds.

## 6. Random Baseline (Step 4.19)

| Метрика | Model | Random | p-value |
|---------|-------|--------|---------|
| ROI (all odds) | +13.80% | -1.36% (std=3.08%) | **0.010** |
| ROI (odds<=10) | -0.14% | -2.15% (std=1.21%) | n/a |

Модель значимо лучше random selection (p=0.01).

## 7. Calibration Analysis (Step 4.17) -- ROOT CAUSE

**EV-selected ставки (ev>=0.05, p_std<=0.02):**
- Predicted win rate: 50.4%
- **Actual win rate: 45.0%**
- Calibration gap: **+5.3%** (overconfidence)

При odds<=10: gap = +6.4% (predicted=60.5%, actual=54.1%).

**Break-even по odds brackets:**
| Bracket | Predicted WR | Actual WR | Gap | ROI |
|---------|-------------|-----------|-----|-----|
| [1-2] | 0.686 | 0.678 | +0.007 | **+3.7%** |
| [2-3] | 0.450 | 0.424 | +0.025 | -0.4% |
| [3-5] | 0.250 | 0.251 | -0.002 | -6.5% |
| [5-10] | 0.146 | 0.143 | +0.003 | -2.9% |
| [50-1000] | 0.025 | 0.017 | +0.008 | +132.8% |

EV selection **усиливает overconfidence**: выбирает ставки где модель максимально переоценивает P(win).

## 8. Adversarial Validation (Step 4.21) -- DISTRIBUTION SHIFT

Mean adversarial AUC = **0.878** (train vs test distinguishable).

| Block | Adv AUC | Odds Drift | Train WR | Test WR |
|-------|---------|------------|----------|---------|
| 0 | 0.695 | +9.6% | 0.487 | 0.503 |
| 1 | 0.888 | **+78.9%** | 0.490 | 0.527 |
| 2 | 0.941 | -9.4% | 0.496 | 0.543 |
| 3 | **0.983** | +6.7% | 0.502 | 0.553 |
| 4 | 0.901 | +4.3% | 0.522 | 0.549 |
| 5 | 0.864 | +29.5% | 0.538 | 0.540 |

Данные радикально меняются между неделями. Block 1 имеет +79% drift в odds.

## 9. Implied Probability Blend (Steps 4.30-4.32) -- BREAKTHROUGH

Blend модельных вероятностей с implied (рыночными):
p_final = alpha * p_model + (1 - alpha) * p_implied

| Стратегия | Mean ROI | Std | Range | N bets |
|-----------|----------|-----|-------|--------|
| 3m_raw (baseline) | 12.10% | 2.10% | 8.9-14.8% | ~6700 |
| 3m_blend50 | 17.57% | 2.05% | 14.6-20.7% | ~4400 |
| 4m_raw | 13.82% | 0.90% | 13.2-15.5% | ~5800 |
| **4m_blend50** | **21.48%** | **1.78%** | **18.8-24.3%** | **~3800** |

Blend работает потому что уменьшает overconfidence: EV-фильтр выбирает бет только если модель СИЛЬНО уверена (edge сохраняется после shrinkage к implied).

## 10. Drift Mitigation (Steps 4.22-4.24)

- **Time-weighted training:** ВСЕ хуже uniform (linear=7.54%, exp_fast=10.68%)
- **Drop drift features:** drop7=14.88% adv_auc 0.878→0.721, marginal
- **ML-only features:** 6.28% (encoding фичи несут сигнал несмотря на drift)
- **Isotonic calibration:** 20.65% per-model, но 3/6 блоков (shift ломает калибровку)

## 11. Bet Sizing and Thresholds (Steps 4.25, 4.28)

- **Kelly criterion:** ROI=0.46-0.65% (Kelly HURTS — занижает ставки на extreme odds)
- **Bracket-specific EV:** high_ev_only(EV>=0.15)=36.91% 4/6 (концентрация на extreme odds)
- **Moderate-only focus:** -2.52% (odds 1-10 не работают при ЛЮБЫХ пороговых)

## 12. Approaches That Failed (Steps 4.3-4.4, 4.7, 4.10-4.11, 4.16, 4.18, 4.20)

- **P&L regression:** -5.68% (P&L слишком шумный)
- **Adaptive threshold:** Overfits, пороги 0.04-0.38
- **pmean + agreement combo:** -0.65% (pmean отсекает profitable extreme odds)
- **Rolling window:** 4.55-10.78% (хуже expanding, мало данных)
- **Calibration fix (shrinkage):** 4.71-9.15% (хуже baseline)
- **Low-odds only (1-2):** -0.89% to -1.63%
- **Edge-predicting model:** -0.79% to -1.50%
- **Monotonic constraints:** +15.22% conf_ev (marginal), но основной фильтр без изменений

## Выводы

1. **Модель имеет доказанную предсказательную силу** (AUC=0.73, p=0.02; vs random p=0.01)
2. **Лучшая стратегия: 4m_blend50** — ROI=21.48% (std=1.78%), 5/6 блоков, seed-stable [18.8-24.3%]
3. **Blend с implied probability** (+65% vs raw) — ключевое улучшение: уменьшает overconfidence
4. **Рынок эффективен при moderate odds (2-10)**: ROI отрицательный при любой стратегии
5. **Positive ROI = extreme odds**: ставки с odds 50+ создают весь profit
6. **Root cause негативного ROI при low odds**: overconfidence на 5-6% у EV-selected ставок
7. **Значительный covariate shift** (adv AUC=0.878) между train и test периодами
8. **Данных недостаточно** (81 день) для robust стратегии при любых odds
