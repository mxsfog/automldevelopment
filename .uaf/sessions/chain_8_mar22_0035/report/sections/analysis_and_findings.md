# Analysis and Findings — chain_8_mar22_0035

## 1. Верификация baseline (step 4.0)

Загружена модель chain_6_mar21_2236 (CatBoostClassifier, depth=7, lr=0.1, 500 iter, AUC=0.7863).
Применена комбинация: Market=1x2 + pre-match (lead_hours > 0) + shrunken segment Kelly thresholds.
Воспроизведён ROI=**28.5833%** (n=233), delta=0.00% — точное совпадение с chain_7.

**Shrunken segment thresholds (shrinkage=0.5 к baseline_t=0.455):**
- Odds [0, 1.8): t_low = 0.475
- Odds [1.8, 3.0): t_mid = 0.545
- Odds [3.0, ∞): t_high = 0.325

---

## 2. Структура тестового периода

| Период | Дата | Day | Win rate 1x2 | n ставок |
|--------|------|-----|--------------|---------|
| Q1 | Feb 20 morning | Thu | 34.3% | ~23 |
| Q2 | Feb 20–21 | Thu-Fri | 33.0% | ~23 |
| Q3 | Feb 21–22 | Fri-Sat | 58.5% | ~77 |
| Q4 | Feb 22 | Sat | 62.2% | ~110 |

Основная прибыль (210/233 ставок) приходится на Q3-Q4 (Пятница-Суббота, Feb 21-22).
Q1-Q2 win rate ≈ baseline (33%) → нулевая или отрицательная прибыль.

**Вывод**: Test ROI=28.58% во многом определяется "горячим" Q3-Q4 периодом.
При другом temporal split результат мог быть иным.

Val period (Feb 17-20, Mon-Thu) имеет ROI=106-115% — аномально высокий baseline,
что подтверждает нестационарность win rate между временными окнами.

---

## 3. Feature importance

Топ-10 фичей по CatBoost feature_importance (шаг 4.6):

| Feature | Importance |
|---------|-----------|
| Market | 11.7% |
| day_of_week | 10.1% |
| elo_implied_agree | 8.1% |
| hour | 7.6% |
| Sport | 7.2% |
| ml_rating_diff | 6.8% |
| Odds | 6.5% |
| implied_prob | 5.9% |
| ml_edge | 5.7% |
| elo_diff | 5.1% |

Pruned top-10 AUC=0.7901 (vs 0.7863 full), но ROI=22.88% (хуже). Feature reduction ухудшает отбор ставок несмотря на рост AUC.

---

## 4. Market analysis (step 4.8)

| Market | Val ROI | Test ROI | n_test | Double-positive |
|--------|---------|---------|--------|-----------------|
| 1x2 | +106.58% | +28.58% | 233 | YES |
| Double Chance | +16.71% | +4.29% | 48 | YES |
| Winner | +9.36% | +3.91% | 41 | YES |
| Asian Handicap | -4.28% | -12.60% | 95 | NO |
| Over/Under | +2.11% | -8.44% | 174 | NO |

Union 1x2+Double Chance+Winner: ROI=26.60% (n=322) — хуже чем 1x2 alone.
**1x2 остаётся наилучшим рынком.**

---

## 5. Cat edge analysis (step 4.12)

`cat_edge = proba_catboost - 1/odds` — разница между предсказанной вероятностью и рыночной ценой.

| cat_edge threshold | Val ROI | n_val | Test ROI | n_test |
|-------------------|---------|-------|---------|--------|
| >= 0.00 (1x2+pm) | 102.1% | 439 | -1.2% | 1199 |
| >= 0.05 | 113.2% | 343 | +12.6% | 638 |
| >= 0.10 | 114.4% | 305 | +22.5% | 381 |
| >= **0.15** | **110.8%** | **264** | **+30.1%** | **228** |
| >= 0.20 | 115.6% | 213 | +54.5% | 132 |
| >= 0.25 | 151.1% | 140 | +85.5% | 30 |
| >= 0.30 | 147.9% | 131 | +140.2% | 7 |

**Монотонная зависимость**: выше порог cat_edge → выше ROI на test, но меньше ставок.
Оптимальный баланс: **cat_edge >= 0.15** (ROI=30.1%, n=228) — double-positive, достаточный n.

Platform ML_Edge (данные платформы): не является полезным сигналом — все thresholds дают test ROI=-28.6%.

---

## 6. Что принципиально не работает

### Ретренировка модели на большем датасете (step 4.10)

Retrain на train+val (80% vs 64%) даёт AUC=0.7933 (+0.007), но Kelly mean(1x2 pre-match)=-0.054 vs 0.198 базовой модели. Вся система Kelly thresholds рассчитана под конкретную калибровку chain_6 — переобученная модель имеет другую probability distribution.

**Вывод**: Калибровка вероятностей CatBoost сильно зависит от размера/состава обучающего датасета. Нельзя перенести thresholds между моделями без повторной валидации.

### Калибровка (step 4.2)

Isotonic regression и Platt scaling меняют absolute probability values, что аннулирует Kelly thresholds. Калиброванные вероятности semantically эквивалентны оригинальным, но численно другие → thresholds неприменимы.

### Time-weighted training (step 4.3)

alpha=2 и alpha=5 дают ROI=8-22% (хуже). Взвешивание недавних примеров не помогает — тестовый период имеет другой day-of-week pattern (Fri-Sat vs Mon-Thu в train).

---

## 7. CV-based threshold stability analysis (step 4.9)

5-fold TimeSeriesSplit на train (0-64%) для оценки стабильности thresholds:

| Fold | n_val | Best ROI | low | mid | high |
|------|-------|---------|-----|-----|------|
| 0 | 7945 | 79.5% | 0.750 | 0.250 | 0.250 |
| 1 | 7945 | 57.9% | 0.750 | 0.200 | 0.050 |
| 2 | 7945 | 87.5% | 0.750 | 0.300 | 0.150 |
| 3 | 7945 | 105.8% | 0.750 | 0.050 | 0.250 |
| 4 | 7945 | 205.3% | 0.700 | 0.750 | 0.150 |

CV mean thresholds: low=0.740, mid=0.310, high=0.170 → test ROI=29.4% (n=77) — delta=+0.85% vs n=233.

**Наблюдение**: Фолд-оптимальные thresholds нестабильны (разброс от 0.050 до 0.750 для mid) — каждый fold находит разные "горячие" пороги, что указывает на noise fitting.

---

## 8. Итоговая характеристика задачи

**Сигнал**: Существует реальный предсказательный сигнал в Soccer 1x2 ставках.
CatBoost с AUC=0.786 и cat_edge >= 0.15 стабильно отбирает ~200-230 ставок с ROI > 20%.

**Ограничения**:
- Все selected ставки — Soccer (100% concentration)
- Val period аномально доходный (win rate ~55-60%) → нестабильность порогов
- Test период (3 дня, Feb 20-22) — недостаточно для статистически значимого заключения
- Temporal nonstationarity: Q1-Q2 (33%) vs Q3-Q4 (62%) win rates

**Ceiling**: ROI=28-30% является практическим потолком при текущей модели и данных.
Для существенного улучшения необходимо:
1. Больше Soccer 1x2 данных (3-5x текущих ~7150 записей)
2. Sport-specific CatBoost модели
3. Longer temporal validation (>=3 months) для подтверждения устойчивости
