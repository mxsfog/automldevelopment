# Executive Summary

## Session: chain_1_mar22_0237

**Date:** 2026-03-22
**Model:** claude-sonnet-4-6
**MLflow Experiment:** uaf/chain_1_mar22_0237 (experiment_id=34)

## Goal

Превысить ROI=33.35% (chain_9 best) на задаче отбора спортивных ставок.
Минимальный целевой ROI: 10% при n >= 50 ставок на тестовом периоде.

## Best Result

**ROI = 26.62%** (n=144 ставки, test период ~20% по времени)
Достигнут с pipeline: chain_9 CatBoostClassifier (depth=7, AUC=0.786) + p80 Kelly + фильтр 1x2 + lead_hours>0.

## Key Findings

1. Целевой ROI=33.35% не воспроизведён из-за data evolution: данные, бывшие "pending" в chain_8/9, завершились и изменили состав датасета.

2. Все 7 новых обученных моделей (CatBoost d7, LightGBM, XGBoost, Optuna, стек, сегментация) дают ROI в диапазоне 16–21% — существенно хуже chain_9.

3. Потолок AUC зафиксирован на уровне ~0.786. Без внешних данных о командах/матчах улучшение невозможно.

4. Признаки Fixture_Status (live/prematch) и lead_hours, добавленные в feature set, снижают ROI с 19.44% до 16.23% — ухудшают калибровку модели, делая Kelly менее избирательным.

5. Рынок Winner показывает положительный ROI в train, но порог не переносится на test.

## Artifacts

- **Best model**: `models/best/model.cbm` (chain_9 CatBoost, 34 features)
- **Kelly threshold**: 0.5914 (p80 на train LOW odds<2.5)
- **Filter**: Market==1x2 AND lead_hours>0 (pre-match only)
- **MLflow runs**: 16 completed runs
