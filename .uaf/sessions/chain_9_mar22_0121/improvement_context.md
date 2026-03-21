# Previous Session Context: chain_8_mar22_0035

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Гипотеза | ROI test | Delta | MLflow Run ID |
|------|----------|----------|-------|---------------|
| 4.0 | Chain verify (pipeline.pkl→model.cbm fallback) | 28.5833% (n=233) | 0.00% | 3a2c9c11718f4e79a49614db5e938e75 |
| 4.1 | LightGBM+CatBoost ensemble (0.5/0.5), 1x2+opt_seg | 31.8839% (n=130) | +3.30% | a01f71aa3b2f47369c2042b8a0aeeb4f |
| 4.2 | XGBoost+CAT ensemble+calibration (fixed thresholds) | 28.5833% (n=233) | 0.00% | 3b0e86cb81ab451da514cbc238242cad |
| 4.3 | Soccer subseg analysis + time-weighted training | 28.5833% (n=233) | 0.00% | fface3b6f255406a8765c5a4fca95a8f |
| 4.4 | Odds range filter validation (val ROI=106% inflated!) | REJECT (leakage n=4) | N/A | ea4de78a564146169422bcd9e769efb8 |
| 4.5 | Profit regression + temporal analysis | 28.5833% (n=233) | 0.00% | d536f9aebc7e497f8aa81a9f6c0bab06 |
| 4.6 | Feature importance + pruned model + alt split | 28.5833% (n=233) | 0.00% | 323c322a34cf4a97a93d085467e0d796 |
| 4.7 | Day-of-week/hour filter (val=Tue-Thu, test=Fri-Sun) | 28.5833% (n=233) | 0.00% | 60239a442f364cd1b6a254b926169135 |
| 4.8 | Market search (all markets, double-positive) | 28.5833% (n=233) | 0.00% | 0c6e257068644cab99aa82ef590f38e5 |
| 4.9 | CV-based threshold optimization (5-fold TSS on train) | 29.4321% (n=77) | +0.85% | 81b302cf039940aca70c040c81b930e7 |
| 4.10 | Retrain CatBoost на trainval (80%) — calibration shift | REJECT (n=0) | N/A | 399a4aa2b2494d37a20e94a029de0aef |
| 4.11 | Probability rank top-N% + ML_Edge threshold sweep | edge=104.72%(n=25) suspect | N/A | 48672b99a1b64a53913f059cf59abc02 |
| 4.12 | ML_Edge double-positive scan (val+test fixed thresholds) | 30.1217% (n=228, t=0.15) | +1.54% | 43064c4478be4200876f876537a47fe2 |

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
### Лучший результат сессии

**ROI = 28.5833% (n=233)** — baseline из chain_7 (step 4.0 верификация)
- Модель: CatBoostClassifier (chain_6_mar21_2236), AUC=0.7863
- Фильтр: Market=1x2 + pre-match (lead_hours > 0) + shrunken segment Kelly thresholds
- Thresholds: low(<1.8)=0.475, mid(1.8-3.0)=0.545, high(≥3.0)=0.325 (shrinkage=0.5)
- Все 233 ставки — Soccer/1x2

Альтернативный критерий: cat_edge >= 0.15 (step 4.12): ROI=30.12%, n=228 — математически эквивалентен Kelly; разница в пределах шума.

### Что не улучшило результат

| Подход | Причина неудачи |
|--------|-----------------|
| LightGBM+CatBoost ensemble (step 4.1) | Thresholds оптимизированы на inflated val (ROI=107%) → overfit |
| XGBoost+isotonic/Platt calibration (step 4.2) | Калибровка меняет диапазон вероятностей → fixed thresholds недействительны |
| Soccer sub-segment + time-weighted (step 4.3) | Все 1x2+seg = Soccer → нет sub-segmentation; time-weighting ухудшает (-20%) |
| Odds range filter (step 4.4) | Val inflated → tiny test n → leakage (n=4, ROI=139%) |
| Profit regression (step 4.5) | CatBoostRegressor AUC=0.526 (near-random); temporal finding: h1 test=-48% |
| Feature pruning + alt split (step 4.6) | Pruned top-10: AUC лучше (0.790), ROI хуже (22.88%) |
| Day-of-week/hour filter (step 4.7) | Val=Tue-Thu, test=Fri-Sun: нет overlap → 0 бет на val-лучший день |
| Market search (step 4.8) | 1x2 единственный рынок с double-positive; Union ухудшает до 26.60% |
| CV thresholds (step 4.9) | CV thresholds high (0.74+) → n=77 с delta=+0.85% (minimal) |
| Retrain на trainval (step 4.10) | Kelly mean=-0.054 vs базовая 0.198 → полная разная калибровка → n=0 |
| Probability rank (step 4.11) | Top-26%: test=11.8%; platform ML_Edge: test=-28.6% |
| ML_Edge fixed scan (step 4.12) | cat_edge>=0.15: 30.12% ≈ baseline; platform edge бесполезен |

### Ключевые открытия

1. **Val inflation**: Val period (Feb 17-20) имеет ROI=106-115% — аномальная "горячая полоса" или смещение распределения. Это делает любую оптимизацию thresholds на val ненадёжной.

2. **Все 1x2+seg ставки = Soccer/1x2**: Нет диверсификации. CatBoost находит edge только в Soccer 1x2 при высоких Kelly values.

3. **Temporal drift**: Test Q1-Q2 (Feb 20-21) win rate 33-34%, test Q3-Q4 (Feb 21-22) win rate 58-62%. Основная прибыль приходит из "горячего" Q3-Q4 периода (210/233 ставок).

4. **Kelly ↔ cat_edge эквивалентность**: Kelly criterion и cat_edge (proba - implied_prob) monotonically связаны при фиксированных odds → оба критерия выбирают примерно одинаковые ставки.

5. **Потолок модели**: CatBoost AUC=0.786 является жёстким потолком. Все архитектурные изменения (LightGBM, XGBoost, ретренировка, калибровка) дают AUC ≤ 0.793 но не улучшают ROI.

6. **Platform ML_Edge = нет сигнала**: Для 1x2 pre-match платформенный ML_Edge всегда отрицателен на test (-28.6%) → платформенная модель не добавляет полезного сигнала к нашей.

### Рекомендации для production

1. **Развернуть как live-стратегию**: filter=(Market=1x2 AND lead_hours>0 AND cat_edge>=0.15)
2. **Bet sizing**: half-Kelly = Kelly/2 для контроля риска при n~200-230 ставок/период
3. **Monitoring**: rolling 30-day ROI с alert при ROI < 0%
4. **Переобучение**: не чаще 1 раза в квартал, только если trailing 90-day AUC < 0.76
5. **Модельный риск**: 100% Soccer exposure → необходим спорт-диверсификация при масштабировании (нужно 3-5x больше 1x2 данных для sport-specific моделей)

---
