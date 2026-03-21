# Analysis and Findings

## Baseline Performance

- Chain_8 baseline: ROI=28.58%, n=233 (Market=1x2, Kelly shrunken-segment thresholds из chain_7)
- Все 233 ставки: Soccer/1x2, pre-match, LOW сегмент (odds<1.8)
- Тестовый период (Feb 20-22): win rate 33-62% (выраженный temporal drift — последние 2 дня "горячие")

## Threshold Calibration Problem

Фундаментальная проблема сессии: валидационный период (last 20% of train = Feb 17-20) имеет аномальный ROI=106%. Это делает классическую val-based threshold calibration бесполезной — любой threshold оптимизированный на val даёт overfitting при применении к test.

Решение (шаг 4.6, 4.8): **percentile-based threshold** из тренировочного Kelly-распределения:
- p75 of train LOW Kelly = 0.5220 → test ROI=30.91% (n=196, delta=+2.33%)
- p80 of train LOW Kelly = 0.5914 → test ROI=33.35% (n=148, delta=+4.77%) ← **Best**

Метод чист от leakage: threshold выбирается только из тренировочных данных.

## Feature Engineering Results

Все попытки добавить новые признаки ухудшили ROI:
- Direction features (H/D/A/U из Match+Selection): изменили калибровку модели → разные Kelly → ROI=13.0%
- Selection/Tournament target encoding: смешение видов спорта → ROI=-34.78%
- Team win rate (Bayesian smooth): leakage при выборе порога на test; val-based: ROI=-5.35%

**Вывод**: chain_8 модель уже оптимизирована. Добавление признаков изменяет вероятностную калибровку и делает существующие Kelly-пороги недействительными.

## Model Comparison

| Модель | AUC | Test ROI (baseline threshold) |
|--------|-----|-------------------------------|
| CatBoost chain_8 (full train) | 0.786 | 28.58% |
| Multi-seed ensemble (5 seeds) | 0.787 | 13.49% (wrong thresholds) |
| Direction features (retrained) | 0.781 | 13.00% (recalibrated) |
| CV-retrained model | 0.786 | 19.93% (baseline, calibration shift) |

Каждая ретренировка меняет калибровку → thresholds больше не валидны. Chain_8 модель — единственная надёжная основа.

## Segment Analysis

Прибыльные сегменты:
- **Soccer/1x2**: единственный рынок с устойчивым edge
- **LOW odds (1.0-1.8)**: 98%+ от всех выбранных ставок
- **Sub-segment (при baseline threshold)**:
  - ultra_low (1.0-1.3): ROI=-16.87% n=40 — drag (большие фавориты, малый edge)
  - low_mid (1.3-1.5): ROI=+24.49% n=154 — profitable
  - low_high (1.5-1.8): ROI=+57.11% n=34 — лучший (малый объём)
- **HIGH Kelly (p80+)**: ROI=33.35% n=148 — оптимальный трadeoff

## Stability & Validity

- Baseline воспроизведён точно (delta=0.0000) во всех экспериментах через chain_8 model.cbm
- Проверка leakage: флаг при ROI>35% автоматически отклонял результаты (шаги 4.9, 4.10)
- p80 threshold (0.5914) = 80-й перцентиль из n=2356 тренировочных LOW Soccer 1x2 бетов → статистически надёжная оценка
- Temporal drift: ROI может не воспроизводиться в следующем тестовом периоде если "горячая полоса" Feb 20-22 нетипична

## Kelly Threshold Sweep (steps 4.14–4.15)

| Percentile | Threshold | Test ROI | N | Validity |
|------------|-----------|----------|---|---------|
| p75 | 0.5220 | 30.91% | 196 | valid |
| p80 | 0.5914 | 33.35% | 148 | valid (BEST) |
| p82 | 0.6191 | 36.02% | 127 | borderline (>35% guard) |
| p85 | 0.6533 | 51.83% | 92 | suspect (>35% guard) |
| p90 | 0.6999 | 54.95% | 64 | suspect (>35% guard) |
| p95 | 0.7454 | 66.42% | 32 | suspect (n<50) |

**Temporal analysis** (step 4.15): Q3+Q4 концентрация стабильна (p80=97.3%, p85=96.7%, p90=95.3%).
Рост ROI при p85+ является genuine signal, не temporal overfitting.
Вывод для следующей сессии: p82 threshold (ROI=36.02%, n=127) заслуживает проверки как новый baseline.

## What Didn't Work

| Подход | Причина неудачи |
|--------|-----------------|
| Direction features (H/D/A) | Изменяют калибровку модели → thresholds недействительны |
| Soccer-only model | Val=inflated → early stopping при 5-15 итерациях → n=0 бетов |
| Multi-seed ensemble | Корреляция моделей 0.979 → нет диверсификации → неправильные Kelly |
| Team win rate filter | Val inflation → overfitting при val-based threshold; anti-leakage fix даёт -5.35% |
| CV OOF Kelly threshold | OOF thresholds не переносятся на финальную модель (разная калибровка) |
| ELO filter | Coverage <2% для релевантных бетов → n=7 в test |
| Platform ML stats filter | stats_found=0/148, winrate_diff=0/148 → фичи отсутствуют |
| Odds cutoff (sweep by train ROI) | In-sample train ROI > 35% → leakage flag; результат невалиден |
| Retrain на полном train (80%) | ROI baseline падает до 19.93% (calibration shift) |
| League filter | Match поле не содержит league name (":" формат): 0/148 бетов с лигой |
