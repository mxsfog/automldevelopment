# Analysis and Findings

## Phase 1: Baselines
Установлены 4 базовые модели:
- DummyClassifier: ROI=-3.07% (lower bound)
- Rule ML_Edge>=11.5: ROI=-4.89% (простое правило хуже random)
- LogisticRegression: ROI=2.62%, AUC=0.7943 (linear baseline, положительный ROI)
- CatBoost default: ROI=1.72%, AUC=0.7930 (non-linear baseline)

## Phase 2: Feature Engineering
Воспроизведен chain_4 baseline: ROI=28.44% (EV>=0+p77), AUC=0.8623.
ELO фичи + sport filter + full-train + EV фильтр дают основной скачок.

## Phase 3: Optuna
Optuna overfitted (val=38% -> test=25%, delta=-3.54pp). Chain_4 params подтверждены как оптимальные.

## Phase 4: Free Exploration

### Ключевое открытие: строгость EV фильтра
EV>=0.10 почти удваивает CV ROI по сравнению с EV>=0:
- EV>=0: CV=11.10%, test=28.44%, n=134/fold
- EV>=0.10: CV=21.73%, test=49.45%, n=47/fold

Механизм: EV>=0.10 удаляет ставки с положительным, но маленьким EV (0-10%).
Эти ставки имеют низкий ROI из-за маржи букмекера.

### Per-sport EV thresholds
Per-sport оптимизация EV порога на val дает ещё +5.34pp по CV (27.07% vs 21.73%).
Разные спорта имеют разную оптимальную строгость:
- Table Tennis, Tennis, Soccer: EV>=0.22-0.24 (нужен высокий edge)
- CS2, Dota 2: EV>=0.17
- Остальные: EV>=0.10 (floor)

### Что не сработало
- Time features (hour, day_of_week): -2.25pp
- Kelly criterion staking: хуже flat (44% vs 49%)
- LightGBM: CV=15.03% vs CB=21.73%
- CB+LGB blend: CV=24.23% (лучше CB solo), но хуже CB+PS_EV
- Blend+PS_EV: CV=22.63% (хуже CB+PS_EV=27.07%)
- Platt/Isotonic calibration: -2.9/-13.9pp
- Optuna re-tuning: overfitting
- Odds filter поверх EV>=0.10: не добавляет ценности

### Robustness
- Leave-one-sport-out: ROI=43-54% при удалении любого спорта
- min_bets sensitivity: стабильный (57.17-57.56%)
- Все 5 фолдов CV положительные для всех ключевых стратегий
- Per-sport breakdown: Table Tennis 47.89%, Tennis 38.86%, Soccer 73.22%, CS2 90.58%

## Bootstrap Confidence Intervals (Step 4.13)
| Strategy | ROI | 95% CI | N | P(ROI>0) |
|----------|-----|--------|---|----------|
| t77 | 21.31% | [14.84%, 27.79%] | 463 | 100% |
| ev0_p77 | 28.44% | [21.13%, 36.62%] | 328 | 100% |
| ev010_p77 | 49.45% | [40.01%, 60.49%] | 157 | 100% |
| ps_ev010 | 57.42% | [44.81%, 72.39%] | 110 | 100% |

## Multi-seed Stability (Step 4.11)
10 seeds, PS_EV floor=0.10: avg=54.62%, std=4.19%, min=46.03%, max=59.98%.
Результат не зависит от конкретного random seed.

## Рекомендации для chain_6
1. Per-sport EV floor=0.10 подтвержден. Можно пробовать per-sport модели для Top-3 спортов.
2. Tournament-level features (агрегаты по турнирам) не исследованы.
3. Temporal decay weights: свежие данные важнее.
4. Online learning: обновление модели по мере поступления данных.
