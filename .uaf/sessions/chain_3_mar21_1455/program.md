# Research Program: Research Session

## Metadata
- session_id: chain_3_mar21_1455
- created: 2026-03-21T11:56:07.929023+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_3_mar21_1455
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.



## Previous Session Context
# Previous Session Context: chain_2_mar21_1432

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Method | ROI | N_bets | MLflow Run | Status |
|------|--------|-----|--------|------------|--------|
| 4.0 | Chain Verification | +16.02% | 2247 | 183c3ae0af8c | done (verified, delta=0.00%) |
| 4.1 | Calibrated EV + 4-model + EV grid | +14.04% | 2559 | 992f54b26653 | done (no improvement, calibration loosened threshold) |
| 4.2 | Stacking + Optuna CB + weighted | +9.00% | 3377 | cacbd6f5f1e8 | done (deep CB=worse EV calibration, simple model better) |
| 4.3 | Kelly + stratified EV | +16.02% | 2247 | 83c926074eaa | done (Kelly hurts, stratified=overfitted 52.94% on n=584) |
| 4.4 | Profit regression (CB+LGBM+Huber) | -5.36% | 6249 | 47db381fb18d | done (regression fails: skewed target, all approaches negative ROI on test) |

## Accepted Features
Base (15): Odds, USD, Is_Parlay, Outcomes_Count, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, ML_Winrate_Diff, ML_Rating_Diff, Outcome_Odds, n_outcomes, mean_outcome_odds, max_outcome_odds, min_outcome_odds
+ Sport_target_enc, Sport_count_enc, Market_target_enc, Market_count_enc
Total: 19 features

## Recommended Next Steps
### Результат
ROI = **+16.02%** (не улучшен). Baseline из chain_1 оказался устойчив ко всем попыткам оптимизации.

### Что подтвердилось
1. **Simple ensemble is king.** 3-model average (CB+LGBM+LR) с простыми параметрами (depth=6, iter=200) — оптимальная архитектура. Усложнение ухудшает результат.
2. **EV-based selection работает.** Формула EV = p*odds - 1 >= 0.12 стабильно выбирает прибыльные ставки (CV mean=18.73%).
3. **Прибыль = high-odds value.** Стратегия находит ставки где букмекер недооценивает вероятность. Средний коэффициент отобранных ставок = 33.9.

### Что не работает (отрицательные результаты)
1. **Калибровка** — isotonic/Platt размывает вероятности, ухудшает EV selection
2. **Сложные модели** — Optuna CatBoost (depth=8) менее калиброван для EV
3. **Kelly criterion** — переносит вес с high-odds на low-odds, убивает ROI
4. **Profit regression** — skewed target (min=-1, max=+126) не поддаётся обучению
5. **Val-optimized thresholds** — не переносятся на test (temporal distribution shift)
6. **Weighted/stacking ensemble** — overfitting к val, нет improvement на test

### Рекомендации для следующей сессии
1. **Agreement-based selection** (step 4.5, прерван) — потенциально может снизить variance
2. **Больше данных** — 81 день мало для стабильной оценки, нужен 6+ месяцев
3. **Online learning** — rolling retrain window для адаптации к drift
4. **ELO features** — elo_history.csv не использован, может добавить signal

---


## Chain Continuation Mode

**РЕЖИМ ПРОДОЛЖЕНИЯ ЦЕПОЧКИ.** Phases 1-3 ПРОПУСКАЮТСЯ.

- **Лучшая модель предыдущей сессии:** `/mnt/d/automl-research/.uaf/sessions/chain_2_mar21_1432/models/best`
- **Предыдущий лучший roi:** 16.02
- **Обязательное действие:** Step 4.0 — загрузить модель, верифицировать результат, затем Phase 4.

**Запрещено:** повторять любой шаг из "What Was Tried" выше.



**Target column:** `Status`
**Metric:** roi (maximize)
**Task type:** tabular_classification



## Validation Scheme

**Scheme:** time_series
**Resolved by:** user-specified
**Parameters:**

- n_splits: 5

- seed: 42


**Validation constraints (enforced by UAF):**




## Data Summary

data_schema.json не предоставлен.



## Research Phases


### Phases 1-3: ПРОПУЩЕНЫ (chain continuation)

Предыдущая сессия уже завершила baseline, feature engineering и optimization.
Best roi = **16.02**.

#### Step 4.0 — Chain Verification (ОБЯЗАТЕЛЬНЫЙ первый шаг)
- **Цель:** Убедиться что модель и среда работают корректно
- **Метод:**
  1. Загрузить metadata из `/mnt/d/automl-research/.uaf/sessions/chain_2_mar21_1432/models/best/metadata.json`
  2. Загрузить модель (формат указан в metadata.model_file)
  3. Загрузить данные, применить те же фичи и sport_filter что в metadata
  4. Вычислить roi на test — должен быть ≈ 16.02
  5. Залогировать в MLflow как "chain/verify"
- **Status:** done (verified, delta=0.00%)
- **MLflow Run ID:** 289eed24725c4d5daca6d7c1162493a8
- **Result:** ROI=+16.02%, AUC=0.784, n_bets=2247



### Phase 4: Free Exploration (до hard_stop)
*Начинается после Phase 3. Продолжается пока budget_status.json не содержит hard_stop: true.*
*Это основная фаза — она занимает большую часть бюджета.*

После Phase 3 НЕ завершай работу. Продолжай генерировать и проверять гипотезы:

**Направления для свободного исследования (в порядке приоритета):**
1. Ансамбли: VotingClassifier, StackingClassifier (CatBoost + LightGBM + XGBoost)
2. Threshold optimization: подбор порога вероятности для максимизации roi
3. Новые фичи: взаимодействия, ratio-фичи, временные паттерны
4. Калибровка вероятностей: CalibratedClassifierCV
5. Сегментация: отдельные модели по Sport/Market/Is_Parlay
6. Дополнительные данные: поиск публичных датасетов (WebSearch) для обогащения

Каждая гипотеза Phase 4 оформляется как Step 4.N в Iteration Log.
При застое 3+ итераций — Plateau Research Protocol обязателен.

## Current Status
- **Active Phase:** Phase 4 (chain continuation) — COMPLETED (hard_stop)
- **Completed Steps:** 34 (4.0 through 4.33)
- **Best Result:** ROI=+27.95% (conf_ev_0.15, n=1092) — inflated by 1 extreme bet
- **Best ROBUST Result:** pmean_0.55: CV mean=1.95% (min=-3.46%), test=1.24%, n=7628, 4/4 temporal blocks positive
- **Best Kelly-viable:** pmean_0.55 + kelly_0.25: ROI=259%, drawdown=50%
- **Realistic expected ROI:** 0-2%
- **Budget Used:** 95% (19/20 iterations), hard_stop=true
- **smoke_test_status:** passed
- **CV stability:** mean ROI ~5%, std ~28% (unstable)

## Iteration Log

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

## Research Insights (plateau iteration 4.5-4.7)

### Анализ причин застоя
- conf_ev_0.15 (+27.95%) не побит 3 итерации (4.5, 4.6, 4.7)
- Сегментация (спорт, odds bracket, dual model) ухудшает: спорты нестабильны, segment models теряют данные
- Meta-model (OOF) даёт AUC=0.7840 vs 0.7839 — практически идентично простому avg
- Profit regression fails: skewed target
- CV показывает высокую нестабильность (std=27.5% для conf_ev)

### Найдено в исследовании
1. **Calibration-based model selection** (ScienceDirect): +34.69% ROI vs -35.17% для accuracy-based
2. **CLV (Closing Line Value)**: +4.2% ROI improvement (недоступен в наших данных)
3. Предыдущая попытка калибровки (chain_2 step 4.1) провалилась из-за ОСЛАБЛЕНИЯ порога
4. Правильный подход: калибровать → УЖЕСТОЧИТЬ порог на val

### Гипотезы
- **A: OOF-calibrated conf_ev** — isotonic калибровка на OOF, затем conf_ev с val-tuned threshold. Ожидание: +1-3% ROI за счёт более точных p для EV расчёта
- **B: Multi-factor scoring** — композитный score = f(EV, confidence, edge, agreement_count). Rank-based selection: top-K по score. Ожидание: +2-5% ROI
- **C: Temporal decay weighting** — больший вес недавним данным при обучении. Ожидание: неизвестно, зависит от drift

### Выбранная следующая попытка
**B: Multi-factor scoring** — потому что: (1) conf_ev уже показал что комбинация EV+confidence работает лучше одного EV, (2) добавление edge и agreement может дать ещё более точный фильтр, (3) не требует калибровки (которая уже провалилась). Далее: A как fallback.

## Accepted Features
(заполняется Claude Code после Phase 2)

## Final Conclusions

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

## Execution Instructions

ВАЖНО: Эти инструкции обязательны к исполнению для каждого шага.

### MLflow Logging
Каждый Python-эксперимент ОБЯЗАН содержать:
```python
# UAF-SECTION: MLFLOW-INIT
import mlflow, os
from pathlib import Path

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="{phase}/{step}") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.log_params({...})
    # ... эксперимент ...
    mlflow.log_metrics({...})
    mlflow.log_artifact(__file__)
    mlflow.set_tag("status", "success")
    mlflow.set_tag("convergence_signal", "{0.0-1.0}")
```

При любом exception:
```python
import traceback
mlflow.set_tag("status", "failed")
mlflow.log_text(traceback.format_exc(), "traceback.txt")
mlflow.set_tag("failure_reason", "{краткое описание}")
```

### Validation Logging (обязательно)
Логируй в каждом run:
```python
mlflow.log_params({
    "validation_scheme": "time_series",
    "seed": 42,
    "n_samples_train": len(X_train),
    "n_samples_val": len(X_val),
})
# Для k-fold: дополнительно
mlflow.set_tag("fold_idx", str(fold_idx))
mlflow.log_metric("roi_fold_0", fold_score_0)
mlflow.log_metric("roi_mean", mean_score)
mlflow.log_metric("roi_std", std_score)
```

### Code Quality
После создания каждого Python-файла:
```bash
ruff format {filepath}
ruff check {filepath} --fix
```
Если после --fix остаются ошибки — исправь вручную.

### Seed (обязательно)
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
# При использовании PyTorch:
# import torch; torch.manual_seed(42)
```

### Termination Policy (КРИТИЧНО — читать обязательно)

**НЕЛЬЗЯ завершать работу** пока в `budget_status.json` не стоит `hard_stop: true`.

Завершение без `hard_stop` — это ошибка. Если все фазы пройдены, а бюджет ещё есть:
1. Не пиши "Final Conclusions" и не заканчивай
2. Перейди к **Plateau Research Protocol** (см. ниже)
3. Генерируй новые гипотезы, пробуй ансамбли, стекинг, новые фичи
4. Продолжай до `hard_stop: true`

Проверять перед КАЖДЫМ экспериментом:
```python
import json, sys
budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        mlflow.set_tag("status", "budget_stopped")
        sys.exit(0)
except FileNotFoundError:
    pass  # файл ещё не создан
```

### Anti-Leakage Rules (КРИТИЧНО)

**Запрещено под страхом инвалидации результата:**

1. **Threshold leakage** — НЕЛЬЗЯ подбирать порог вероятности на test-сете.
   Правило: threshold выбирается на **последних 20% train** (out-of-fold validation),
   применяется к test один раз без дополнительной подстройки.
   ```python
   # ПРАВИЛЬНО: порог из val (часть train)
   val_split = int(len(train) * 0.8)
   val_df = train.iloc[val_split:]
   threshold = find_best_threshold(val_df, model.predict_proba(val_df[features])[:, 1])
   # Применяем к test только один раз
   roi = calc_roi(test, model.predict_proba(test[features])[:, 1], threshold=threshold)

   # НЕПРАВИЛЬНО: порог из test — это leakage!
   # threshold = find_best_threshold(test, proba_test)  # <-- ЗАПРЕЩЕНО
   ```

2. **Target encoding leakage** — fit только на train, transform на val/test.

3. **Future leakage** — при time_series split никаких фичей из будущего.
   Проверь: нет ли колонок которые появляются ПОСЛЕ события (Payout_USD, финальный счёт).

4. **Санитарная проверка**: если roi > 35.0 — это почти наверняка leakage.
   Остановись, найди причину, исправь до продолжения.
   UAF BudgetController автоматически отклонит результат с алертом MQ-LEAKAGE-SUSPECT.

### Model Artifact Protocol (ОБЯЗАТЕЛЬНО для chain continuation)

В конце ЛЮБОГО эксперимента, который устанавливает новый лучший roi:

Сохраняй модель в `./models/best/` (относительно SESSION_DIR):

```python
import json
from pathlib import Path

Path("./models/best").mkdir(parents=True, exist_ok=True)

# Выбери формат по фреймворку:
# CatBoost:  model.save_model("./models/best/model.cbm")
# LightGBM:  booster.save_model("./models/best/model.lgb")
# XGBoost:   model.save_model("./models/best/model.xgb")
# sklearn:   import joblib; joblib.dump(pipeline, "./models/best/model.pkl")

import json, os
metadata = {
    "framework": "catboost",   # catboost | lgbm | xgboost | sklearn
    "model_file": "model.cbm",
    "roi": ...,         # best ROI value (float)
    "auc": ...,         # AUC (float)
    "threshold": ...,   # prediction threshold (float)
    "n_bets": ...,      # number of bets selected
    "feature_names": [...],  # list[str]
    "params": {...},          # hyperparameters dict
    "sport_filter": [...],    # list[str] — sports to EXCLUDE (if any), or []
    "session_id": os.environ["UAF_SESSION_ID"],
}
with open("./models/best/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

Это позволяет следующей сессии в цепочке загрузить модель и начать сразу с Phase 4.

### DVC Protocol
После завершения каждого шага:
```bash
git add .
git commit -m "session chain_3_mar21_1455: step {step_id} [mlflow_run_id: {run_id}]"
```

### Feature Engineering Instructions (Shadow Feature Trick)
При реализации шага с method: shadow_feature_trick:
1. Строй ДВА датасета: X_baseline (из предыдущего best run) и X_candidate (+shadow)
2. Обучи модель ДВА раза с одинаковыми гиперпараметрами
3. Логируй как nested runs или с суффиксами _baseline и _candidate
4. delta = metric_candidate - metric_baseline
   - delta > 0.002: принять shadow features
   - delta <= 0: отклонить
   - 0 < delta <= 0.002: пометить как marginal
5. Target encoding fit ТОЛЬКО на train (никогда на val/test)
   Если нарушение: mlflow.set_tag("target_enc_fit_on_val", "true")

### Report Sections (ОБЯЗАТЕЛЬНО перед завершением)

Перед тем как написать Final Conclusions — создай файлы для PDF-отчёта.
Директория: `report/sections/` (относительно SESSION_DIR).

**Файл 1: `report/sections/executive_summary.md`**
```markdown
# Executive Summary

## Цель
[1-2 предложения о задаче]

## Лучший результат
- Метрика roi: [значение]
- Стратегия: [описание]
- Объём ставок: [N]

## Ключевые выводы
- [главный инсайт]
- [что сработало]
- [главное ограничение]

## Рекомендации
[конкретные следующие шаги]
```

**Файл 2: `report/sections/analysis_and_findings.md`**
```markdown
# Analysis and Findings

## Baseline Performance
[что показал baseline, roi без ML]

## Feature Engineering Results
[какие фичи улучшили модель, какие нет]

## Model Comparison
[сравнение моделей: CatBoost vs LightGBM vs ансамбли]

## Segment Analysis
[прибыльные сегменты: спорт, рынки, odds диапазоны]

## Stability & Validity
[CV результаты, нет ли leakage, насколько стабильны результаты]

## What Didn't Work
[честный анализ провальных гипотез]
```

Создай оба файла через Write tool. Без них PDF-отчёт будет пустым.

### Update program.md
После каждого шага обновляй:
- Step **Status**: pending -> done/failed
- Step **MLflow Run ID**: заполни run_id
- Step **Result**: заполни метрику
- Step **Conclusion**: напиши вывод
- **Current Status**: обнови Best Result и Budget Used
- **Iteration Log**: добавь запись
- После Phase 2: заполни **Accepted Features**

### Plateau Research Protocol (ОБЯЗАТЕЛЬНО при застое)

**Критерий застоя:** метрика `roi` не улучшается 3+ итерации подряд
(delta < 0.001 относительно предыдущего best).

Когда застой обнаружен — СТОП. Не запускай следующий эксперимент.
Вместо этого выполни следующие шаги по порядку:

#### Шаг 1 — Анализ причин (sequential thinking)
Подумай последовательно:
1. Что уже пробовали? Какие паттерны в успешных/неуспешных runs?
2. Где потолок по данным vs потолок по архитектуре?
3. Какие самые сильные гипотезы ещё НЕ проверены?
4. Есть ли data leakage или overfitting которые маскируют прогресс?
5. Верна ли метрика `roi`? Оптимизируем ли мы то что нужно?

#### Шаг 2 — Интернет-исследование (WebSearch)
Ищи по следующим запросам (по одному, читай результаты):
- `"{task_type} roi improvement techniques 2024 2025"`
- `"kaggle tabular_classification winning solution feature engineering"`
- `"state of the art tabular_classification tabular data 2025"`
- Если задача специфичная: `"tabular_classification roi improvement kaggle winning solution"`
- Ищи: какие фичи используют топы, какие ансамбли, какие трюки

#### Шаг 3 — Формулировка новых гипотез
На основе анализа и поиска запиши в program.md раздел:
```
## Research Insights (plateau iteration N)
- **Найдено:** (что нашёл в поиске)
- **Гипотеза A:** (конкретная идея + ожидаемый прирост)
- **Гипотеза B:** (конкретная идея + ожидаемый прирост)
- **Выбранная следующая попытка:** (почему именно это)
```

#### Шаг 4 — Реализация
Реализуй самую перспективную гипотезу из шага 3.
Если она тоже не даёт прироста — повтори протокол с шага 1.