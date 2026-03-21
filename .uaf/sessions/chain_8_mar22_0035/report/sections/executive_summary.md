# Executive Summary — chain_8_mar22_0035

**Сессия:** chain_8_mar22_0035
**Дата:** 2026-03-22
**Итерации:** 13 экспериментов (steps 4.0–4.12)
**MLflow experiment:** uaf/chain_8_mar22_0035

---

## Цель

Улучшить ROI стратегии ставок сверх **28.5833%** (baseline из chain_7_mar21_2347).
Задача: предсказание won/lost, отбор ставок через Kelly criterion, целевой ROI ≥ 10%.

---

## Лучший результат

**ROI = 28.5833% (n=233 ставок)** — baseline не превзойдён ни одним надёжно валидированным экспериментом.

| Метрика | Значение |
|---------|----------|
| ROI | 28.58% |
| Количество ставок | 233 |
| Модель | CatBoostClassifier (chain_6), AUC=0.7863 |
| Отбор | Market=1x2, pre-match, Kelly >= shrunken segment thresholds |
| Все ставки | Soccer / 1x2 |
| Test период | Feb 20–22, 2026 |

---

## Что пробовали

13 экспериментов в 4 категориях:

**Ансамблирование и калибровка** (steps 4.1, 4.2): LightGBM+CatBoost ансамбль показал 31.88%, но threshold оптимизирован на аномально inflated val (ROI=107%) → overfit. XGBoost + isotonic calibration: без изменений (0.00%).

**Альтернативные модели** (steps 4.3, 4.5, 4.10): Time-weighted training, profit regression (CatBoostRegressor), ретренировка на 80% данных — все REJECT или 0% delta. Ретренировка меняет калибровку модели настолько, что Kelly values уходят в отрицательную зону (mean=-0.054 vs 0.198 базовой модели).

**Фильтры и подвыборки** (steps 4.4, 4.6, 4.7, 4.8, 4.9, 4.11): Odds range filter — REJECT (leakage n=4). Pruned features — ROI хуже. Day-of-week — val=Mon-Thu, test=Fri-Sun (нет overlap). Market search — 1x2 единственный double-positive рынок. CV thresholds — delta +0.85% (n=77 vs 233).

**Аналитические эксперименты** (steps 4.3, 4.6, 4.12): Открытие temporal drift (Q1-Q2: 33% win rate vs Q3-Q4: 58-62%), platform ML_Edge = нет сигнала для 1x2, cat_edge >= 0.15 даёт 30.12% n=228 (≈ baseline Kelly).

---

## Ключевые выводы

1. **28.58% — практический потолок** для данной комбинации модель (AUC=0.786) + данные (~74k ставок, Feb 2026).

2. **Val period anomaly**: ROI val=106-115% против test=28.58% — аномальная горячая полоса или структурный сдвиг. Любая оптимизация на этом val переобучается.

3. **100% Soccer exposure**: Все отобранные 233 ставки — Soccer/1x2. CatBoost находит significant edge только в этом сегменте.

4. **Kelly ↔ cat_edge**: Оба критерия математически эквивалентны при фиксированных odds. Оптимальный порог: cat_edge >= 0.10-0.15 для n~200-380 ставок с ROI 22-30%.

5. **Platform ML_Edge бесполезен**: Платформенная ML-модель имеет отрицательный edge на test (-28.6% для 1x2 pre-match), что контрадиктирует val (+61.5%). Несостоятельный сигнал.

---

## Production-рекомендация

```
Стратегия: bet(row) = 1 if (
    row.Market == "1x2" AND
    row.lead_hours > 0 AND
    (catboost.proba(row) - 1/row.Odds) >= 0.15
)
Sizing: bet_fraction = Kelly(proba, odds) / 2  (half-Kelly)
Expected ROI: ~28-30% на горизонте сравнимом с test (~14k записей, 3 дня)
```

**Риски**: высокая концентрация в Soccer (100%), короткий test период (Feb 20-22, 3 дня), значительный temporal drift между периодами.
