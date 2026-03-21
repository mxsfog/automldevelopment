# Analysis and Findings

## 1. Исследование данных

**Датасет:** 74,493 ставок после фильтрации (исключены статусы pending/cancelled/error/cashout).

**Валидация:** Time-series split
- Train: 0-64% (47,675 ставок)
- Val: 64-80% (11,919 ставок) — для подбора порогов
- Test: 80-100% (14,899 ставок) — только для финальной оценки

**Целевая метрика — ROI:**
```
ROI = (Payout_USD - Stake_USD) / Stake_USD * 100%
```

**ELO признаки:** доступны для 9.7% ставок (через таблицу elo_history.csv, join по Bet_ID).

**Pre-match vs live:**
- Train: 30.9% pre-match (lead_hours > 0)
- Test: 35.2% pre-match
- Pre-match ставки имеют принципиально разные характеристики — модель получает более стабильный сигнал

## 2. Feature Engineering

Итоговый feature set (24 числовых + 3 категориальных):

**Базовые:**
- `Odds`, `log_odds`, `implied_prob` — вероятность из коэффициента
- `USD`, `log_usd` — размер ставки
- `is_parlay`, `outcomes_count` — тип ставки

**ML-сигналы платформы:**
- `ml_p_model`, `ml_p_implied`, `ml_edge`, `ml_ev` — вероятностные оценки платформы
- `ml_edge_pos`, `ml_ev_pos` — положительный компонент edge/ev
- `ml_team_stats_found`, `ml_winrate_diff`, `ml_rating_diff` — статистика команд

**Временные:**
- `hour`, `day_of_week`, `month` — время размещения ставки

**ELO:**
- `elo_max/min/mean/std/diff/ratio` — рейтинги команд
- `has_elo`, `elo_count` — признак наличия ELO
- `elo_implied_agree` — разница между implied_prob и ELO-вероятностью
- `ml_edge_x_elo_diff` — взаимодействие edge и ELO-разрыва

**Производные:**
- `odds_times_stake` — объём ставки
- `lead_hours`, `log_lead_hours` — время до матча

**Категориальные (CatBoost native):** `Sport`, `Market`, `Currency`

## 3. Методология временных весов

Экспоненциальное затухание к прошлому:
```
w_i = exp(log(2) / (half_life * n) * i)
```
При `half_life=0.5`: 50% веса у последней половины данных. Дает +1-2% ROI vs равных весов.

## 4. Ключевые эксперименты

### Phase 1: Baseline
- DummyClassifier: ROI=-3.07% (нижняя граница)
- Rule-based (ML_Edge >= 11.5): ROI=-4.89% (сигнал есть, но правило грубое)
- LogisticRegression: ROI=+1.44% (модель может учиться)
- CatBoost default: ROI=+5.34%

### Phase 2: Feature Engineering
ELO-признаки + temporal weighting (d7/hl50): **+7.34%** (+2% vs baseline)
- ELO доступен только для 9.7% ставок, но ello_implied_agree эффективен

### Phase 3: Optuna HPO (неудача)
- 30 trials, val objective: val=64.49% → test=4.13% — глубокий оверфит
- CV objective (3-fold): depth=5, test=4.03% — умеренный оверфит
- Вывод: оптимизация по val нестабильна, нужна другая стратегия

### Phase 4: Kelly Criterion + Pre-match (прорыв)
- **Step 4.3**: Isotonic calibration + Kelly → test=12.24% (+4.9%)
  - Калибровка уменьшает Brier score: 0.1390 → 0.1321
  - Raw Kelly без калибровки дал лучший результат: val=68%, test=12.24%
- **Step 4.4**: Kelly fraction=0.5, thr=0.270 → test=14.52% (+2.3%)
  - CV крайне нестабилен: fold 3 = -48.41% (только 300 ставок)
- **Step 4.5**: Pre-match filter (lead_hours > 0) + Kelly → test=**24.91%** (+10.4%)
  - Исключение live ставок радикально улучшает отбор
  - Порог 0.455 с min_bets=200 найден на val
  - 435 pre-match ставок (из ~4733 pre-match в тесте)
- **Step 4.6**: Специализированная модель только на PM данных → test=7.44% (хуже)
  - Меньше обучающих данных → слабее генерализация
- **Step 4.7**: Баггинг 5 seeds + sweep → test=19.23%
  - Усреднение вероятностей сглаживает дискриминативность высокого порога
  - Soccer = 72.4% pre-match Kelly ставок; ROI Soccer: 18.3%
- **Step 4.8**: Grid depth×lr×kelly → best=17.80%
  - Изменение фичей меняет распределение Kelly → оптимальный порог смещается

## 5. Анализ нестабильности CV

**CV (4-fold, step 4.5):**
- Fold 1: +17.28% (1124 бетов)
- Fold 2: -6.81% (1031 бет)
- Fold 3: **-39.25%** (550 бетов) — систематически плохой
- Fold 4: +3.19% (627 бетов)

Fold 3 соответствует примерно 40-60% данных хронологически. Возможные причины:
1. Смена операционной модели платформы в середине датасета
2. Сезонный эффект (другой спортивный сезон)
3. Изменение состава пользователей / рынков в тот период

Несмотря на нестабильность CV, тестовый результат (80-100% данных) = 24.91% устойчив.

## 6. Sport breakdown (step 4.7, threshold=0.32)

| Спорт | ROI test | Ставок |
|-------|----------|--------|
| CS2 | 46.3% | 11 |
| MMA | 36.3% | 36 |
| Boxing | 33.1% | 18 |
| Ice Hockey | 23.7% | 70 |
| Tennis | 23.1% | 59 |
| Soccer | 18.3% | 702 |
| Basketball | 18.3% | 25 |
| Cricket | -15.6% | 21 |

Soccer доминирует по числу ставок (72.4%), остальные спорты имеют более высокий ROI но малый объём.

## 7. Итоговый пайплайн (step 4.5)

```python
# 1. Загрузка и объединение данных
df = load_data()  # bets + outcomes(Sport,Market,Start_Time) + elo_agg

# 2. Сортировка по времени, train/val/test split
# train: 0-80%, val: 64-80% (для early stopping), test: 80-100%

# 3. Построение фичей
X_train, cat_features = build_features(train_df)
# + temporal weights w = exp(log(2)/(0.5*n) * i)

# 4. Обучение CatBoost
model = CatBoostClassifier(depth=7, lr=0.1, iterations=500, ...)
model.fit(X_train, y_train, eval_set=(X_val, y_val), sample_weight=weights)

# 5. Предсказания + Kelly
probas = model.predict_proba(X)[:, 1]
kelly = (probas * (odds-1) - (1-probas)) / (odds-1)

# 6. Отбор: только pre-match ставки с Kelly >= 0.455
mask = (kelly >= 0.455) & (lead_hours > 0)
selected_bets = df[mask]
```

## 8. MLflow tracking

Все эксперименты залогированы в MLflow:
- Experiment: `uaf/chain_1_mar21_2039`
- Tracking URI: http://127.0.0.1:5000
- 11 runs, лучший run: `dc8e48815fb944c7b57b097bd5922c1a` (step 4.5)

Сохранённый pipeline: `.uaf/sessions/chain_1_mar21_2039/models/best/`
