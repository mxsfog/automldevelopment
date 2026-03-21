# Executive Summary

## Цель
Предсказание победы ставки (won/lost) на спортивных событиях. Задача — достичь ROI >= 10% на отобранных моделью ставках при использовании flat betting.

## Лучший результат
- Метрика ROI: **+16.02%** (тест)
- Стратегия: 3-model ensemble (CatBoost + LightGBM + LogReg) обученный на полном train, отбор ставок по Expected Value >= 0.12
- Объем ставок: 2247 из 14899 тестовых (15.1%)
- AUC: 0.784
- Win rate: 36.6%, средний коэффициент: 33.9
- Cross-validation: mean ROI = 18.73%, std = 14.14% (5 фолдов)

## Ключевые выводы
- EV-based selection (EV = model_prob * odds - 1) — ключевой прорыв. Переход от probability threshold к EV threshold увеличил ROI с +1.5% до +16%
- Из 5 групп feature engineering только Sport/Market target encoding дал значимый прирост (+6.92 delta ROI). Остальные группы (odds-based, temporal, interactions, complexity) добавляли шум
- ROI обусловлен высокими коэффициентами (avg_odds=33.9). При ограничении odds<=5 ROI падает до +2.6%. Стратегия работает за счет выявления value в high-odds ставках

## Рекомендации
1. Для продакшена использовать EV>=0.12 без ограничения odds, но с мониторингом drawdown
2. Высокая дисперсия (CV std=14.14%) требует bankroll management (Kelly criterion или фиксированный процент)
3. Периодическая переобучение модели на свежих данных (данные покрывают 81 день)
4. Исследовать калибровку вероятностей для более точного EV-расчета
