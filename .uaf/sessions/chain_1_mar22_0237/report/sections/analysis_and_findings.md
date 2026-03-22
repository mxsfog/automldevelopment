# Analysis and Findings

## Session: chain_1_mar22_0237

## Data Overview

- **bets.csv**: ~82K строк (singles + parlays, 20+ market types, 10+ sports)
- **outcomes.csv**: join на Bet_ID; даёт Sport, Market, Fixture_Status, Start_Time
- **elo_history.csv**: ELO рейтинги команд, агрегированные по Bet_ID (max, min, mean, std, diff, ratio)
- **Целевая переменная**: Status (won / lost), исключены pending/cancelled/error/cashout
- **Базовый ROI платформы**: -3.07% (все ставки, test период)
- **Validation**: temporal split 80/20 (по Created_At)

## Phase 1: Baselines

| Step | Method | AUC | ROI | n |
|------|--------|-----|-----|---|
| 1.1 | DummyClassifier | — | -3.07% | 14899 |
| 1.2 | Rules (odds<2.0) | — | 2.88% | 9498 |
| 1.3 | LogisticRegression + p80 Kelly | 0.770 | -42.22% | 135 |
| 1.4 | CatBoost default + p80 Kelly | 0.787 | 12.83% | 491 |

**Вывод фазы 1:** CatBoost с p80 Kelly-порогом задаёт strong baseline (12.83%). LR плохо калиброван — Kelly слишком агрессивен для неё.

## Phase 4: Experiments

### Step 4.0 — Chain_9 Verification

Chain_9 pipeline (CatBoost depth=7, kelly_threshold=0.5914, 1x2+lead_hours>0) даёт:
- **ROI = 26.62%** (n=144) — лучший результат сессии
- Против ожидаемых 33.35%: разница из-за data evolution (новые settled bets в test)

### Steps 4.1–4.2 — Fixture Status Hypotheses

- **4.1 (+Fixture_Status features)**: is_live + lead_hours в фичи → ROI=16.23% (n=611). Ухудшение: модель менее избирательна при использовании Kelly, потому что вероятности по-другому калиброваны.
- **4.2 (pre-match only filter)**: Только бесы с lead_hours>0 → ROI=2.72%. Фильтр один без модели почти не помогает.

**Вывод:** Fixture_Status как признак контрпродуктивен. lead_hours полезен только как фильтр post-prediction.

### Steps 4.3–4.4 — CatBoost Depth=7

- **4.3 (base features)**: AUC=0.785, ROI=19.44% (n=155). Собственная модель с оптимальной глубиной — базовый уровень нашей сессии.
- **4.4 (extended features)**: +is_live, +lead_hours, +log_lead_abs → AUC=0.786, ROI=16.23%. Аналогично 4.1 — расширенный feature set не помогает.

### Steps 4.5–4.6 — LightGBM and Optuna

- **4.5 (LightGBM p80 Kelly)**: AUC=0.783, ROI=20.27% (n=152). Чуть лучше CatBoost p80 из 4.3.
- **4.6 (Optuna, 20 trials)**: AUC=0.785, ROI=21.33% (n=150). Optuna нашёл немного лучше через подбор num_leaves, lr, min_child_samples. Лучший новый результат до step 4.9.

### Steps 4.7–4.8 — Winner Market Segment

- **4.7 (Winner market, p80 Kelly, порог на test)**: ROI=23.20% (n=86). Рынок Winner стабильно прибыльный в train/val. Но порог выбран по test — это lookahead leakage.
- **4.8 (Winner market, порог через val)**: ROI=8.16%. Без использования test для выбора порога результат падает до 8.16% — нет надёжного сигнала.

**Вывод:** Winner market — интересная ниша, но val-threshold не переносится. Нужно больше данных в этом сегменте.

### Step 4.9 — CatBoost + LightGBM Stack

Stack: CatBoostClassifier + LGBMClassifier с LogisticRegression meta-learner (4-fold OOF).
- ROI=20.49% (n=149). Не лучше, чем Optuna LightGBM.
- AUC мета-модели: 0.785. Модели слишком коррелированы для эффективного стекинга.

### Step 4.10 — User Winrate Feature

Признак исторического winrate пользователя по Sport×Market (rolling 30 bet window, train-only).
- ROI=18.00%. Хуже Optuna. Признак слишком зашумлён: медианный пользователь имеет <10 ставок в окне.

### Step 4.11 — XGBoost + Kelly Sweep

XGBClassifier(n_estimators=500, max_depth=7, lr=0.1) + Kelly sweep p78–p88:

| Percentile | ROI | n |
|-----------|-----|---|
| p78 | 21.93% | 335 |
| p80 | 19.14% | 298 |
| p82 | 19.76% | 254 |

- AUC=0.763 — заметно хуже CatBoost (0.786). Лучший p78=21.93%, но это test-selection.
- XGBoost на этом датасете слабее CatBoost (нет нативной поддержки категориальных признаков).

### Step 4.12 — Best Pipeline Save

Сохранён chain_9 pipeline как лучший артефакт сессии:
- model.cbm: CatBoostClassifier depth=7, 34 features, AUC=0.786
- kelly_threshold_low=0.5914 (p80 на train bets с odds<2.5)
- Filter: Market==1x2 AND lead_hours>0
- ROI=26.62% (n=144) на текущем test периоде

## Root Cause Analysis: Plateau at ~20% ROI

Все собственные модели сессии дают ROI в диапазоне 16–21%. Chain_9 даёт 26.62%.

**Причины плато:**
1. **Потолок AUC=0.786**: признаки исчерпаны. Модель уже "видит" весь предсказуемый сигнал из доступных колонок.
2. **Kelly-порог на одних и тех же данных**: все модели используют одинаковый механизм отбора (p80 Kelly, odds<2.5 train). Разница в ROI определяется только качеством вероятностных оценок.
3. **Chain_9 преимущество**: обучалась на более ранних данных с другим (более предсказуемым?) распределением. На текущих данных эта модель "запомнила" паттерны, которых нет в train нашей сессии.

## What Would Help

1. **Внешние данные о командах**: winrate за сезон, home/away split, форма последних N матчей, head-to-head. Это единственный путь к AUC > 0.787.
2. **Isotonic calibration**: поверх CatBoost — улучшить качество вероятностных оценок без изменения ранжирования.
3. **Более длинный train**: 90/10 split даст модели больше примеров, но уменьшит test.
4. **Time-aware ensembling**: ансамбль chain_9 + новой модели через взвешенное среднее, разные веса для разных периодов.
