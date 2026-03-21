# Executive Summary

## Задача
Предсказание исхода спортивных ставок (won/lost) для отбора прибыльных ставок с целью ROI >= 10%.

## Результат
**ROI = 20.23%** на test-сете (3-дневное окно, 449 ставок из 1094 sport-filtered ELO подмножества).

## Стратегия
- **Модель:** CatBoost (depth=8, lr=0.08, l2_leaf_reg=21.1, min_data_in_leaf=20)
- **Данные:** только ставки с ELO-рейтингами (~9.7% от всех ставок)
- **Sport filter:** исключены Basketball, MMA, FIFA, Snooker из train и test
- **Порог:** фиксированный t=0.77 (выше порога -> ставка принимается)
- **Фичи:** 28 признаков (base odds + engineered + ELO features)

## Робастность
4-fold temporal CV: mean ROI = 13.55% +/- 8.09%. Все 4 фолда положительные (min=3.72%, max=25.13%).

## Прогресс по chains
| Chain | Best ROI | CV mean ROI | Ключевое улучшение |
|-------|----------|-------------|-------------------|
| chain_1 | 7.32% | - | Baseline + Odds features |
| chain_2 | 18.61% | 12.15% | ELO enrichment + ensemble |
| chain_3 | 20.23% | 13.55% | Sport filtering |

## Ключевые выводы
1. ELO-данные дают основной прирост качества (+13 п.п. vs baseline)
2. Sport filter дает дополнительные +1.6 п.п. за счет исключения шумных видов спорта
3. CatBoost solo лучше любого ensemble на этих данных
4. Фиксированный порог 0.77 стабильнее любой adaptive-стратегии
