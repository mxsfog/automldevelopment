# Executive Summary

## Цель
Предсказание победы ставок на спортивных событиях для достижения ROI >= 10% на отобранных ставках. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта.

## Лучший результат
- Метрика ROI: **28.44%** (test), **22.42%** (5-fold CV avg)
- Стратегия: CatBoost (depth=8, lr=0.08, l2=21.1), full-train на ELO-подмножестве с sport filter (без Basketball, MMA, FIFA, Snooker), выбор ставок: p>=0.77 AND EV>=0 (p*odds>=1)
- AUC: 0.8623
- Объём ставок: 328 из 1094 sport-filtered ELO test bets (30% coverage)

## Альтернатива (без EV фильтра)
- ROI: 21.31% (test), 11.02% (CV avg)
- Объём: 463 ставок (42% coverage)

## Ключевые выводы
- EV фильтр (p*odds>=1) -- главное открытие chain_4, даёт +7.1 п.п. на test, +11.4 п.п. на CV
- ELO-фичи (winrate, elo_diff, team stats) -- главный предиктор качества ставки
- Full-train модель (100% train data) даёт +1 п.п. vs 80/20 split
- CatBoost solo доминирует все альтернативы (LGB, blends, stacking, ensembles)
- Sport filter стабильно добавляет 2-5 п.п., SF подход 5/5 positive folds vs ELO_all 4/5
- Порог p=0.77 подтверждён val-sweep (step 4.14): ROI стабилен в диапазоне 0.75-0.78
- Odds-range analysis: ставки с odds 1.01-1.15 дают только 0.95% ROI, EV фильтр их удаляет

## Рекомендации
1. Использовать CatBoost full-train + EV>=0 + p>=0.77 как основную стратегию
2. Fallback: p>=0.77 без EV фильтра для максимального покрытия ставок (21.31% ROI)
3. Расширить ELO-трекинг: текущее покрытие 9.7% ограничивает объём
4. Rolling window мониторинг ROI по спортам с алертами при деградации
5. Тестирование на 2+ недельном out-of-time окне перед production
6. Dual-model: ELO-SF модель для ELO ставок, fallback модель для остальных
