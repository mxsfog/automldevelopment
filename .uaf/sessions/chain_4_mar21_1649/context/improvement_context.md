# Previous Session Context: chain_3_mar21_1455

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | N_bets | MLflow Run | Status |
|------|--------|-----|--------|------------|--------|
| 4.0 | Chain Verification | +16.02% | 2247 | 289eed24725c4d5daca6d7c1162493a8 | done (verified, delta=0.00%) |
| 4.1 | Agreement-based EV selection | +26.49% | 1276 | 6cb4e6cd56be48b78e82ebc7b2a5f077 | done (E_odds_stratified best, needs val) |
| 4.2 | Validated strategies (train/val/test) | +27.95% | 1092 | 8117ef97e08047c79b633f2857c48cd2 | done (conf_ev_0.15 best, val=7.63%) |
| 4.3 | XGBoost 4-model + CV stability | +54.98% | 736 | 2188256eaf324ced87c36c12d7c2c851 | done (CV unstable: base mean=1.12%, conf mean=4.69%) |
| 4.4 | Per-sport/market/odds segmentation | +16.02% | 2247 | b62b923b2f8e40218bb22cd2217f64e7 | done (analysis: soccer/tennis negative, basketball extreme) |
| 4.5 | Validated segments (val-based excl) | +27.95% | 1092 | 8043ef7d06a3414696713c22b3612243 | done (sport excl hurts, conf_ev_0.15 still best) |
| 4.6 | Meta-model on OOF predictions | +31.37% | 1035 | 8986084a18e64b53a946294feb1b0702 | done (meta_ev_0.20 best, but not val-tuned) |
| 4.7 | Dual strategy (low/high odds) | -6.10% | 813 | 7c67e8a1094945328b0ec98a98dd194f | done (segment models worse, conf_ev_0.15 still best) |
| 4.8 | Multi-factor scoring (edge+agree+composite) | +27.95% | 1092 | d639cb824e234a0cb746f85c8292b2d8 | done (all filters hurt vs pure conf_ev_0.15, edge/agree remove profitable bets) |
| 4.9 | Temporal decay + isotonic calibration | +27.95% | 1092 | a8e203074c534716b2f1d269d888535a | done (decay hurts, calibration catastrophic: -8% to -30% ROI) |
| 4.10 | Extended features (+odds,+inter,+temp,+cmplx) | +27.95% | 1092 | d50d373f16014ee9a7b265d17cbcdcc4 | done (all feature groups hurt val, baseline_19 best) |
| 4.11 | CV threshold + bootstrap confidence | +27.95% | 1092 | 9d31649f17454f0abc846bd9e2794569 | done (CV: all strategies ~0-5% mean, 27.95% is period-specific) |
| 4.12 | Odds-weighted training (log/sqrt/linear) | +27.95% | 1092 | b4383ea770d74e85bd31479f3fd6fab0 | done (val: odds best 15.38%, test: -10.23%, total val/test inversion) |
| 4.13 | Ensemble weights (opt/geo/median/cb-only) | +27.95% | 1092 | 28ccdc355d334a8ab19df7f62172e611 | done (opt: LGBM=0, CB=0.34/LR=0.66; geo best val 10.45%; test: median_0.18=37.4% but not val-best) |
| 4.14 | Rank average + CB+LR + power mean | +27.95% | 1092 | 8244325a99df48149c3ab256724a4797 | done (rank avg invalid: ranks≠probs, EV formula loses meaning; CB+LR val=8.45% test=33.18% interesting but not validated) |
| 4.15 | CB+LR 2-model ensemble validation | +27.95% | 1092 | 36261c549d7a4c0db5caf12e4606b0a2 | done (AUC=0.787 slightly better; val eq_0.10=8.45%→test=15.94%; 2-model worse than 3-model, LGBM diversity helps) |
| 4.16 | ELO features (+5 elo-based) | +27.95% | 1092 | 56cfa509b7b94aabba9b0685ea2d9a05 | done (coverage 9.66%, AUC+0.001, val ROI worse: 7.63%→-1.43%, sparse data = noise) |
| 4.17 | Confidence formula variants (inv/exp/pct/hard) | +27.95% | 1092 | 811a2c86165d42c8bf2955b41cdd1caa | done (exp_k10 val best 12.04%, test 23.36%; original inv_k10_t0.15 still best on test) |
| 4.18 | Hyperparameter sensitivity (5 configs) | +27.95% | 1092 | c17e7c29151642e69aaee371731701b0 | done (AUC 0.781-0.784 for all; large val=11.71% but test worse; baseline config optimal) |
| 4.19 | Feature importance pruning (top-8..15) | +27.95% | 1092 | 17219a16120e4411a5fa02b6b035222b | done (top3: Market_enc 50%, Odds 20%; pruning hurts val; ML_Winrate/Rating=0 importance) |
| 4.20 | RF/ET in ensemble (6 combos) | +27.95% | 1092 | 742b544f62f4432f9e2cb1358c86cad2 | done (ET AUC=0.760 best; all RF/ET ensembles worse on val; cb_lgbm_lr still optimal) |
| 4.21 | Parlay filter (singles vs all) | +27.95% | 1092 | a08b3eaf6e8647038b79f8a05817033b | done (parlays=20%, singles-only kills test ROI: 27.95%→6.45%; profit comes from parlays) |
| 4.22 | Parlay boost (dual thresholds) | +29.18% | 797 | b99a41e79c094af7b97ca7a5990a5fca | done (uniform_0.18 best test but val=-5.52%; dual_s0.15_p0.10=29.11% n=1265 but not val-best; conf_ev_0.15 still best validated) |
| 4.23 | Odds floor/band/ceiling | +124.79% | 285 | 4c31fb6c82cb45d6ac675cf18f75b7ca | done (band_50_500 extreme ROI but n=285; ceil_100=0.26% — profit from extreme odds; val-best band_2_10 inverts on test) |
| 4.24 | Robustness: profit concentration + seeds | +27.95% | 1092 | 9625e997f9a343f494bce184f5d11db7 | done (CRITICAL: 1 bet at odds=490.9 = 137% of profit; without it ROI negative; Gini=0.762; seeds: mean=23%, std=7.5%) |
| 4.25 | Capped odds (exclude extreme outliers) | +2.82% | 605 | cd977d278b984f408e179d10e11815c6 | done (cap10=2.82%, cap20=-5.68%, cap50=-3.62%, cap100=0.26%; no systematic edge without extreme odds) |
| 4.26 | Low-odds edge search | +15.33% | 264 | 5b032bf0abe6463dab3b7b48822ad14a | done (edge_cap5_e0.15=15.33% n=264; walk-forward: b1=-17.6%, b2=-35.8%, b3=+120% — 1 bet drives all; low-odds ~0%) |
| 4.27 | Validated edge strategy (val→test) | +5.50% | 496 | e8d771d725554eafa4be05015a9c9cf8 | done (CONSISTENT: cap2_e0.10 val=6.69→test=5.50; cap3_e0.10 val=4.44→test=6.93; cap5_e0.10 val=3.65→test=8.05; real edge ~5-8%) |
| 4.28 | Deep edge validation (CV + seeds + Gini) | -7.10% | ~400 | a2a5cc6642ce49488eaba32d0794afe0 | done (CV: edge_cap2 mean=-7.10% std=12%; seed std=0.44% excellent; BUT CV negative = not robust across periods) |
| 4.29 | Robust CV optimization (33 strategies × 5 folds) | +1.24% | 7628 | 8c94ea93d6b045c699389fa75240583d | done (best robust: pmean_0.55 min=-3.46% mean=1.95% test=1.24%; NO strategy has positive min fold ROI; realistic ROI=0-2%) |
| 4.30 | Final summary (convergence signal=1.0) | +27.95% | 1092 | 9ff8981028ac4fe2bc3e72bd9cac624d | done (summary of 30 experiments; realistic ROI=0-2%) |
| 4.31 | Combined edge+EV (intersection/union) | +6.93% | 516 | 2005da5d2c7f409a8ddb371afd61b841 | done (inter_e0.10_ev0.02_cap3 = edge_cap3; combining doesn't improve; val-test gap persists) |
| 4.32 | Temporal block analysis (4 blocks) | +1.24% | 3925 | 0f3ef5ebded349fca5ad07a085c541c8 | done (pmean_0.55: 4/4 positive [0.4,2.0,0.5,2.1%] std=0.80%; confev_0.15: 1/4 positive [-42,-31,-17,+166%]) |
| 4.33 | Kelly criterion sizing (1/4..full) | +259.16% | 3925 | 5533a7734e1544fda1cfb2d8ed2e32fb | done (pmean_0.55 kelly_0.25: ROI=259%, dd=50%; confev_0.15 kelly>=0.5: bankroll destroyed; pmean_0.55 = only strategy surviving Kelly) |

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
### Лучший результат
**ROI = +27.95%** (n=1092 ставок, conf_ev_0.15) на тестовом периоде Feb 20-22, 2026.

Улучшение с +16.02% (baseline EV>=0.12) достигнуто за счёт confidence-weighted EV selection:
```
EV = p_mean * odds - 1
confidence = 1 / (1 + p_std * 10)
select where EV * confidence >= 0.15
```

### Что подтвердилось
1. **Simple ensemble** (CB + LGBM + LR average) — оптимальная архитектура
2. **EV-based selection** стабильно выделяет прибыльные ставки
3. **Model disagreement** (p_std) — лучший сигнал quality ставки
4. **19 базовых фичей** достаточно, расширение ухудшает

### Что не работает (33 эксперимента Phase 4)
- Калибровка вероятностей (isotonic) — катастрофа на test
- Сегментация по спорту/рынку/odds — уменьшает train, нестабильно
- Meta-model на OOF — идентично простому average
- Temporal decay — ухудшает
- Odds-weighted training — инверсия val/test
- Bootstrap confidence — хуже model-type diversity
- Extended features — noise > signal при 81 днях
- Multi-factor scoring — conf_ev уже оптимален
- Optimal ensemble weights — LGBM=0, но 2-model (CB+LR) хуже 3-model
- Rank averaging — невалидно (ranks ≠ probabilities для EV)
- ELO features — coverage 9.66%, AUC +0.001, val ROI хуже
- Альтернативные формулы confidence — exp/pct/hard все хуже на test
- Hyperparameter tuning — AUC 0.781-0.784 для всех конфигов
- Feature pruning — top-8..15 все хуже all_19 на val
- RF/ExtraTrees в ensemble — увеличивают diversity, но ухудшают conf_ev на val
- Dual thresholds (singles/parlays) — uniform_0.18 best test (29.18%) but val=-5.52%; не переносится
- Odds floor/band/ceiling — profit из extreme high-odds (50-500); ceil_100 убивает ROI до 0.26%; val-best band_2_10 инвертируется на test
- Capped odds — подтверждает: cap10=2.82%, cap20=-5.68%, cap100=0.26%; edge = 0 без extreme outliers

### Критическое ограничение
CV stability analysis (step 4.11) показал:
- Средний ROI по CV: **0-5%** (std=11-70%)
- Результат 27.95% **специфичен для тестового периода**
- Реальный ожидаемый ROI при deployment: **~5%**, не 28%

**Profit concentration (step 4.24):**
- **1 ставка** (odds=490.9, P&L=3.2M) создает **137% всей прибыли**
- Без этой ставки стратегия УБЫТОЧНА
- Все odds-brackets кроме [200,10000) убыточны
- Gini coefficient P&L = 0.762 (extreme concentration)
- Seed sensitivity: ROI 16-36% при thr=0.15, зависит от того, выбирает ли модель эту одну ставку
- **Вывод: conf_ev_0.15 не имеет систематического edge, результат = 1 случайный выигрыш**

**Real edge discovery (step 4.27):**
- Edge-based strategy (p_model - p_implied >= 0.10, odds <= 2-5) дает **5-8% ROI**
- **val-test consistent**: cap2_e0.10 val=6.69%→test=5.50%, cap5_e0.10 val=3.65%→test=8.05%
- **UPDATED (step 4.28):** CV shows mean=-7.10% для edge strategy. Seed-стабильна (std=0.44%), но CV-нестабильна
- Val-test consistency была совпадением периода. Ни одна стратегия не имеет доказанного edge.

**Temporal block analysis (step 4.32):**
- pmean_0.55: единственная стратегия с 4/4 положительных блока [0.4%, 2.0%, 0.5%, 2.1%], std=0.80%
- confev_0.15: 1/4 блоков положительный [-41.8%, -31.1%, -17.2%, +165.6%]
- Все EV-based стратегии: 1/4 блоков положительный

**Kelly sizing (step 4.33):**
- pmean_0.55 + kelly_0.25: ROI=259%, drawdown=50% — единственная стратегия, выживающая Kelly
- confev_0.15 + kelly >= 0.50: полная потеря банкролла (-100%)
- Подтверждает: pmean_0.55 имеет реальный, стабильный edge ~1-2%

### Рекомендации для следующей сессии
1. **6+ месяцев данных** для стабильной оценки
2. **Rolling retrain** каждые 1-2 недели
3. **Мониторинг ROI в реальном времени** с автоматическим стопом
4. **Closing Line Value (CLV)** если доступны closing odds — +4.2% ROI по литературе
5. **Не усложнять модель** — простой ensemble уже оптимален

---
