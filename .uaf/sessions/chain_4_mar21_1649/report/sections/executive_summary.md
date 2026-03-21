# Executive Summary

## Задача
Предсказание исхода спортивных ставок (won/lost) для максимизации ROI на отобранных ставках. Данные: 74,493 ставки за период 2025-12-03 -- 2026-02-22 (81 день), 20+ рынков, 10+ видов спорта.

## Методология
- **Модель:** 4-model ensemble (CatBoost + LightGBM + LogisticRegression + XGBoost)
- **Blend:** p_final = 0.5 * p_model + 0.5 * p_implied (implied = 1/odds)
- **Фичи:** 19 признаков (odds, ML-метрики, target/count encoding для Sport/Market)
- **Валидация:** Walk-forward с недельным ретрейном (expanding window, 6 тестовых блоков)
- **Отбор ставок:** EV = p_final * odds - 1 >= 0.05 AND p_std <= 0.02

## Ключевые результаты

### Лучшая стратегия: 4m_blend50 (ev005_agree_p02)
- **Walk-forward ROI:** 21.80% (5/6 положительных блоков, n=3783)
- **Seed stability:** mean=21.48%, std=1.78% (все 5 seeds положительные: 18.8-24.3%)
- **Улучшение:** +65% vs предыдущий best (3m_raw: 13.80%, mean=12.10%, std=2.10%)

### Эволюция стратегий (36 экспериментов)
| Стратегия | Mean ROI | Std | Range | Sharpe | Cap10 |
|-----------|----------|-----|-------|--------|-------|
| 3m_raw (baseline) | 12.10% | 2.10% | 8.9-14.8% | 5.76 | -0.14% |
| 3m_blend50 | 17.57% | 2.05% | 14.6-20.7% | 8.57 | -1.81% |
| 4m_raw | 13.82% | 0.90% | 13.2-15.5% | 15.40 | -2.11% |
| **4m_blend50** | **21.48%** | **1.78%** | **18.8-24.3%** | **12.06** | **-3.41%** |
| 5m_blend50 | 28.74% | 5.59% | 19.2-36.6% | 5.15 | -5.84% |

### Критические ограничения

1. **Зависимость от extreme odds:** С ограничением odds <= 10: ROI = -3.41%. Весь ROI от ставок с odds 50+.
2. **Overconfidence:** EV-selected ставки имеют calibration gap +5.3% (predicted=50.4%, actual=45.0%).
3. **Covariate shift:** Adversarial AUC = 0.878 между train и test блоками. Данные меняются радикально.
4. **Profit concentration:** Extreme odds ставки = основной driver P&L.

### Доказательства предсказательной силы
- **AUC = 0.73** (permutation p=0.020, значимо)
- **Model vs Random selection:** p=0.010 (значимо)
- Модель РЕАЛЬНО предсказывает, но edge недостаточен для ROI при moderate odds

### Реалистичная оценка
- **Moderate odds (2-10):** ROI = -3.41% to -0.14%. Модель уменьшает убытки vs random (-2.15%), но не генерирует прибыль.
- **Все коэффициенты:** ROI = 18.8-24.3% (seed-stable), но зависит от extreme odds.

## Рекомендации
1. Модель имеет реальную предсказательную силу, но недостаточную для profitable deployment при normal odds
2. Для deployment с extreme odds: мониторинг profit concentration, max drawdown стоп, bankroll management
3. Blend с implied probability (50/50) значительно улучшает selection quality
4. Для улучшения: 6+ месяцев данных, closing line value (CLV), модели с учетом covariate shift
5. Рынок эффективен при odds 2-10: модель не может обойти implied probability
