# Research Program: Research Session

## Metadata
- session_id: chain_1_mar21_2039
- created: 2026-03-21T17:39:23.643824+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_1_mar21_2039
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.




**Target column:** `Status`
**Metric:** roi (maximize)
**Task type:** tabular_classification



## Validation Scheme

**Scheme:** time_series
**Resolved by:** user-specified
**Parameters:**

- n_splits: 5

- seed: 42


**Validation constraints (enforced by UAF):**




## Data Summary

data_schema.json не предоставлен.



## Research Phases


### Phase 1: Baseline (MANDATORY)
**Goal:** Установить нижнюю границу и strong baseline
**Success Criterion:** Превысить random baseline по roi


#### Step 1.1 — Constant baseline
- **Hypothesis:** DummyClassifier (most_frequent) задаёт lower bound
- **Method:** dummy_classifier
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** b96b63ac4d97485596bdb7b7d38a2d78
- **Result:** ROI=-3.07% (14899 ставок), CV=-2.15%±2.51%
- **Conclusion:** Lower bound установлен. Все ставки подряд дают -3.07%.


#### Step 1.2 — Rule-based baseline
- **Hypothesis:** Простое пороговое правило по топ-1 признаку
- **Method:** threshold_rule
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** 7bc62f77a1e04aa38d3e2236bd5301ae
- **Result:** ROI=-4.89% test (val=17.37%, threshold ML_Edge>=11.5)
- **Conclusion:** Сильный val/test разрыв — правило переподогнано на val. Не улучшает baseline.


#### Step 1.3 — Linear baseline
- **Hypothesis:** LogisticRegression с базовыми фичами — linear baseline
- **Method:** logistic_regression
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 98d36dde36fb4495ad8b8206063ce627
- **Result:** ROI=1.44% test (AUC=0.795, threshold=0.81), CV=0.09%±1.16%
- **Conclusion:** Linear baseline превышает random. AUC высокий (0.79). Нелинейность нужна для максимизации ROI.


#### Step 1.4 — Non-linear baseline
- **Hypothesis:** CatBoost с дефолтами — strong non-linear baseline
- **Method:** catboost_default
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 145644060fd240d0bbd7149cbabd00c7
- **Result:** ROI=5.34% test (AUC=0.789, threshold=0.71), CV=2.18%±7.45%
- **Conclusion:** Strong baseline установлен. Топ фичи: implied_prob, Market, day_of_week, Odds. ELO фичи ещё не добавлены — Phase 2.



### Phase 2: Feature Engineering (MANDATORY)

#### Step 2.1 — ELO features + temporal weighting
- **Hypothesis:** ELO рейтинги + exponential sample weights улучшат CatBoost
- **Method:** elo_join + make_sample_weights(half_life=0.5) + depth grid
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 1c5e6bda3f104dfe9ab04cf4a0fa60a1
- **Result:** ROI=7.34% (d7_hl50pct, 5009 ставок), CV=3.47%±4.62%
- **Conclusion:** ELO match rate 9.7%. Лучший конфиг: depth=7, half_life=0.5. Delta +2.00% vs baseline. Топ фичи: Market, day_of_week, elo_implied_agree, hour, Sport.


### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Hyperparameter Optimization
- **Hypothesis:** Optuna TPE найдёт лучшие гиперпараметры
- **Method:** optuna_tpe
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** 7208d680668a42788487a8edfbff20f2
- **Result:** ROI=4.13% test (val=64.49%, depth=9, lr=0.3) — хуже step 2.1
- **Conclusion:** Optuna переобуcился на val: depth=9+high_lr → AUC_val=0.979 vs AUC_test=0.754. Оптимизация val ROI нестабильна. Лучший результат остаётся step 2.1: d7_hl50pct ROI=7.34%.



### Phase 4: Free Exploration (до hard_stop)
*Начинается после Phase 3. Продолжается пока budget_status.json не содержит hard_stop: true.*
*Это основная фаза — она занимает большую часть бюджета.*

После Phase 3 НЕ завершай работу. Продолжай генерировать и проверять гипотезы:

**Направления для свободного исследования (в порядке приоритета):**
1. Ансамбли: VotingClassifier, StackingClassifier (CatBoost + LightGBM + XGBoost)
2. Threshold optimization: подбор порога вероятности для максимизации roi
3. Новые фичи: взаимодействия, ratio-фичи, временные паттерны
4. Калибровка вероятностей: CalibratedClassifierCV
5. Сегментация: отдельные модели по Sport/Market/Is_Parlay
6. Дополнительные данные: поиск публичных датасетов (WebSearch) для обогащения

Каждая гипотеза Phase 4 оформляется как Step 4.N в Iteration Log.
При застое 3+ итераций — Plateau Research Protocol обязателен.

## Current Status
- **Active Phase:** Phase 4
- **Completed Steps:** 7/7
- **Best Result:** ROI=24.91% (pre-match Kelly, step 4.5)
- **Budget Used:** 55%
- **smoke_test_status:** done

## Iteration Log

| Step | Гипотеза | ROI test | Delta | MLflow Run ID |
|------|----------|----------|-------|---------------|
| 1.1 | DummyClassifier (lower bound) | -3.07% | — | b96b63ac4d97485596bdb7b7d38a2d78 |
| 1.2 | Rule ML_Edge>=11.5 | -4.89% | — | 7bc62f77a1e04aa38d3e2236bd5301ae |
| 1.3 | LogisticRegression | 1.44% | — | 98d36dde36fb4495ad8b8206063ce627 |
| 1.4 | CatBoost default | 5.34% | — | 145644060fd240d0bbd7149cbabd00c7 |
| 2.1 | ELO features + temporal weighting (d7_hl50pct) | **7.34%** | +2.00% | 1c5e6bda3f104dfe9ab04cf4a0fa60a1 |
| 3.1 | Optuna HPO (30 trials, val objective) | 4.13% | -3.21% | 7208d680668a42788487a8edfbff20f2 |
| 4.1 | Optuna CV-objective (25 trials, 3-fold) | 4.03% | -3.31% | 8ace7d37a28a4b3c90796db036195d4d |
| 4.2 | Text features + Tournament + bet_lead_hours | 1.28% | -6.06% | 0fe60f4b395347ee88224fb35ff83645 |
| 4.3 | Isotonic calibration + **Kelly criterion selection** | **12.24%** | **+4.90%** | dd5bd761384f4b8393033fa5e9465e90 |
| 4.4 | Kelly fraction=0.5, thr=0.270 | **14.52%** | **+2.28%** | 9264a14912144aed8160c8cb21bc73da |
| 4.5 | Pre-match filter + stable Kelly (min_bets=500) | **24.91%** | **+10.39%** | dc8e48815fb944c7b57b097bd5922c1a |

## Accepted Features

Финальный feature set после Phase 2 (step 2.1):
- **Базовые:** Odds, USD, log_odds, log_usd, implied_prob, is_parlay, outcomes_count
- **ML сигналы:** ml_p_model, ml_p_implied, ml_edge, ml_ev, ml_team_stats_found, ml_winrate_diff, ml_rating_diff, ml_edge_pos, ml_ev_pos
- **Временные:** hour, day_of_week, month
- **Производные:** odds_times_stake, ml_edge_x_elo_diff
- **ELO:** elo_max, elo_min, elo_diff, elo_ratio, elo_mean, elo_std, k_factor_mean, has_elo, elo_count, elo_implied_agree
- **Категориальные:** Sport, Market, Currency

## Research Insights (plateau iteration 1)
- **Найдено:** Ключевая находка из веб-поиска (arxiv 2303.06021): калибровка вероятностей даёт +34.69% ROI vs -35.17% без калибровки. Isotonic regression для нелинейной калибровки. 2026 исследование: xG + isotonic → ~15% ROI на football.
- **Гипотеза A:** Isotonic regression calibration на val + Kelly criterion selection вместо probability threshold. Kelly=(p*b-(1-p))/b, где b=Odds-1. Ожидаемый прирост: +2-5%.
- **Гипотеза B:** Calibrated probability + fractional Kelly (0.5x Kelly) для выбора ставок. Фильтрация только Kelly > 0.03.
- **Гипотеза C:** LightGBM с нативной поддержкой категориальных фич + isotonic calibration.
- **Выбранная следующая попытка:** Гипотеза A — isotonic calibration + Kelly criterion. Calibration — наименее исследованный аспект в предыдущих сессиях, с потенциально большим влиянием.

## Final Conclusions
(заполняется Claude Code по завершении)

---

## Execution Instructions

ВАЖНО: Эти инструкции обязательны к исполнению для каждого шага.

### MLflow Logging
Каждый Python-эксперимент ОБЯЗАН содержать:
```python
# UAF-SECTION: MLFLOW-INIT
import mlflow, os
from pathlib import Path

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="{phase}/{step}") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.log_params({...})
    # ... эксперимент ...
    mlflow.log_metrics({...})
    mlflow.log_artifact(__file__)
    mlflow.set_tag("status", "success")
    mlflow.set_tag("convergence_signal", "{0.0-1.0}")
```

При любом exception:
```python
import traceback
mlflow.set_tag("status", "failed")
mlflow.log_text(traceback.format_exc(), "traceback.txt")
mlflow.set_tag("failure_reason", "{краткое описание}")
```

### Validation Logging (обязательно)
Логируй в каждом run:
```python
mlflow.log_params({
    "validation_scheme": "time_series",
    "seed": 42,
    "n_samples_train": len(X_train),
    "n_samples_val": len(X_val),
})
# Для k-fold: дополнительно
mlflow.set_tag("fold_idx", str(fold_idx))
mlflow.log_metric("roi_fold_0", fold_score_0)
mlflow.log_metric("roi_mean", mean_score)
mlflow.log_metric("roi_std", std_score)
```

### Code Quality
После создания каждого Python-файла:
```bash
ruff format {filepath}
ruff check {filepath} --fix
```
Если после --fix остаются ошибки — исправь вручную.

### Seed (обязательно)
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
# При использовании PyTorch:
# import torch; torch.manual_seed(42)
```

### Termination Policy (КРИТИЧНО — читать обязательно)

**НЕЛЬЗЯ завершать работу** пока в `budget_status.json` не стоит `hard_stop: true`.

Завершение без `hard_stop` — это ошибка. Если все фазы пройдены, а бюджет ещё есть:
1. Не пиши "Final Conclusions" и не заканчивай
2. Перейди к **Plateau Research Protocol** (см. ниже)
3. Генерируй новые гипотезы, пробуй ансамбли, стекинг, новые фичи
4. Продолжай до `hard_stop: true`

Проверять перед КАЖДЫМ экспериментом:
```python
import json, sys
budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        mlflow.set_tag("status", "budget_stopped")
        sys.exit(0)
except FileNotFoundError:
    pass  # файл ещё не создан
```

### Anti-Leakage Rules (КРИТИЧНО)

**Запрещено под страхом инвалидации результата:**

1. **Threshold leakage** — НЕЛЬЗЯ подбирать порог вероятности на test-сете.
   Правило: threshold выбирается на **последних 20% train** (out-of-fold validation),
   применяется к test один раз без дополнительной подстройки.
   ```python
   # ПРАВИЛЬНО: порог из val (часть train)
   val_split = int(len(train) * 0.8)
   val_df = train.iloc[val_split:]
   threshold = find_best_threshold(val_df, model.predict_proba(val_df[features])[:, 1])
   # Применяем к test только один раз
   roi = calc_roi(test, model.predict_proba(test[features])[:, 1], threshold=threshold)

   # НЕПРАВИЛЬНО: порог из test — это leakage!
   # threshold = find_best_threshold(test, proba_test)  # <-- ЗАПРЕЩЕНО
   ```

2. **Target encoding leakage** — fit только на train, transform на val/test.

3. **Future leakage** — при time_series split никаких фичей из будущего.
   Проверь: нет ли колонок которые появляются ПОСЛЕ события (Payout_USD, финальный счёт).

4. **Санитарная проверка**: если roi > 35.0 — это почти наверняка leakage.
   Остановись, найди причину, исправь до продолжения.
   UAF BudgetController автоматически отклонит результат с алертом MQ-LEAKAGE-SUSPECT.

### Model Artifact Protocol (ОБЯЗАТЕЛЬНО для chain continuation)

В конце ЛЮБОГО эксперимента, который устанавливает новый лучший roi,
ОБЯЗАТЕЛЬНО сохрани **полный пайплайн** в `./models/best/` (относительно SESSION_DIR).

Пайплайн должен принимать RAW DataFrame (до любой обработки) и возвращать предсказания.
Следующая сессия загрузит его и воспроизведёт точный roi без ручного
дублирования feature engineering.

```python
import joblib, json, os
from pathlib import Path

# === 1. Определяем класс пайплайна ===
class BestPipeline:
    '''Полный пайплайн: feature engineering + предсказание + оценка метрики.'''

    def __init__(
        self,
        model,                      # обученная модель (CatBoost/LGBM/XGBoost/sklearn)
        feature_names: list[str],   # колонки, которые подаются в model.predict_proba
        threshold: float,           # порог вероятности для фильтрации ставок
        sport_filter: list[str],    # виды спорта для ИСКЛЮЧЕНИЯ (пустой список = не фильтровать)
        framework: str,             # "catboost" | "lgbm" | "xgboost" | "sklearn"
        # Добавь сюда все fitted preprocessors: encoders, scalers, imputers
        # Например:
        # target_encoder=None,
        # elo_scaler=None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold
        self.sport_filter = sport_filter
        self.framework = framework
        # self.target_encoder = target_encoder
        # self.elo_scaler = elo_scaler

    def _build_features(self, df):
        # ВАЖНО: вставь сюда весь feature engineering из твоего train-скрипта
        # Это должна быть ТОЧНАЯ копия кода из обучения
        # Например:
        # df = df.copy()
        # df["odds_bucket"] = pd.cut(df["Odds"], bins=[1, 1.5, 2.0, 3.0, 10], labels=False)
        # if self.target_encoder:
        #     df["sport_enc"] = self.target_encoder.transform(df[["Sport"]])
        # ...
        return df[self.feature_names]

    def predict_proba(self, df):
        # Возвращает вероятности для RAW DataFrame
        X = self._build_features(df)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, df) -> dict:
        # Вычислить ROI и другие метрики на RAW DataFrame.
        # Returns: dict с ключами roi, n_selected, threshold
        # Фильтрация по sport_filter (ИСКЛЮЧАЕМ указанные виды)
        if self.sport_filter:
            df = df[~df["Sport"].isin(self.sport_filter)].copy()

        proba = self.predict_proba(df)
        mask = proba >= self.threshold
        selected = df[mask].copy()

        if len(selected) == 0:
            return {"roi": -100.0, "n_selected": 0, "threshold": self.threshold}

        # ROI = (выигрыши - общие ставки) / общие ставки * 100
        won_mask = selected["Status"] == "won"
        total_stake = selected["USD"].sum()
        total_payout = selected.loc[won_mask, "Payout_USD"].sum()
        roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0

        return {
            "roi": roi,
            "n_selected": int(mask.sum()),
            "threshold": self.threshold,
        }


# === 2. Создаём и сохраняем пайплайн ===
Path("./models/best").mkdir(parents=True, exist_ok=True)

pipeline = BestPipeline(
    model=model,              # твоя обученная модель
    feature_names=features,   # list[str] — порядок важен
    threshold=best_threshold, # float
    sport_filter=[],          # list[str] если есть фильтрация
    framework="catboost",     # catboost | lgbm | xgboost | sklearn
    # target_encoder=encoder, # если использовался
)
joblib.dump(pipeline, "./models/best/pipeline.pkl")

# === 3. Нативный файл модели (для fallback) ===
# CatBoost:  model.save_model("./models/best/model.cbm")
# LightGBM:  booster.save_model("./models/best/model.lgb")
# XGBoost:   model.save_model("./models/best/model.xgb")

# === 4. Metadata ===
metadata = {
    "framework": "catboost",
    "model_file": "model.cbm",
    "pipeline_file": "pipeline.pkl",
    "roi": ...,   # значение метрики (float) — ТОЧНО то же, что было залогировано
    "auc": ...,
    "threshold": best_threshold,
    "n_bets": int(mask.sum()),
    "feature_names": features,
    "params": dict(model.get_params()) if hasattr(model, "get_params") else {},
    "sport_filter": [],
    "session_id": os.environ["UAF_SESSION_ID"],
}
with open("./models/best/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved pipeline.pkl + metadata.json. roi = {metadata['roi']:.2f}")
```

Следующая сессия загружает `pipeline.pkl` и вызывает `pipeline.evaluate(test_df)` —
это даёт точно тот же roi без ручного воспроизведения feature engineering.

### DVC Protocol
После завершения каждого шага:
```bash
git add .
git commit -m "session chain_1_mar21_2039: step {step_id} [mlflow_run_id: {run_id}]"
```

### Feature Engineering Instructions (Shadow Feature Trick)
При реализации шага с method: shadow_feature_trick:
1. Строй ДВА датасета: X_baseline (из предыдущего best run) и X_candidate (+shadow)
2. Обучи модель ДВА раза с одинаковыми гиперпараметрами
3. Логируй как nested runs или с суффиксами _baseline и _candidate
4. delta = metric_candidate - metric_baseline
   - delta > 0.002: принять shadow features
   - delta <= 0: отклонить
   - 0 < delta <= 0.002: пометить как marginal
5. Target encoding fit ТОЛЬКО на train (никогда на val/test)
   Если нарушение: mlflow.set_tag("target_enc_fit_on_val", "true")

### Report Sections (ОБЯЗАТЕЛЬНО перед завершением)

Перед тем как написать Final Conclusions — создай файлы для PDF-отчёта.
Директория: `report/sections/` (относительно SESSION_DIR).

**Файл 1: `report/sections/executive_summary.md`**
```markdown
# Executive Summary

## Цель
[1-2 предложения о задаче]

## Лучший результат
- Метрика roi: [значение]
- Стратегия: [описание]
- Объём ставок: [N]

## Ключевые выводы
- [главный инсайт]
- [что сработало]
- [главное ограничение]

## Рекомендации
[конкретные следующие шаги]
```

**Файл 2: `report/sections/analysis_and_findings.md`**
```markdown
# Analysis and Findings

## Baseline Performance
[что показал baseline, roi без ML]

## Feature Engineering Results
[какие фичи улучшили модель, какие нет]

## Model Comparison
[сравнение моделей: CatBoost vs LightGBM vs ансамбли]

## Segment Analysis
[прибыльные сегменты: спорт, рынки, odds диапазоны]

## Stability & Validity
[CV результаты, нет ли leakage, насколько стабильны результаты]

## What Didn't Work
[честный анализ провальных гипотез]
```

Создай оба файла через Write tool. Без них PDF-отчёт будет пустым.

### Update program.md
После каждого шага обновляй:
- Step **Status**: pending -> done/failed
- Step **MLflow Run ID**: заполни run_id
- Step **Result**: заполни метрику
- Step **Conclusion**: напиши вывод
- **Current Status**: обнови Best Result и Budget Used
- **Iteration Log**: добавь запись
- После Phase 2: заполни **Accepted Features**

### Plateau Research Protocol (ОБЯЗАТЕЛЬНО при застое)

**Критерий застоя:** метрика `roi` не улучшается 3+ итерации подряд
(delta < 0.001 относительно предыдущего best).

Когда застой обнаружен — СТОП. Не запускай следующий эксперимент.
Вместо этого выполни следующие шаги по порядку:

#### Шаг 1 — Анализ причин (sequential thinking)
Подумай последовательно:
1. Что уже пробовали? Какие паттерны в успешных/неуспешных runs?
2. Где потолок по данным vs потолок по архитектуре?
3. Какие самые сильные гипотезы ещё НЕ проверены?
4. Есть ли data leakage или overfitting которые маскируют прогресс?
5. Верна ли метрика `roi`? Оптимизируем ли мы то что нужно?

#### Шаг 2 — Интернет-исследование (WebSearch)
Ищи по следующим запросам (по одному, читай результаты):
- `"{task_type} roi improvement techniques 2024 2025"`
- `"kaggle tabular_classification winning solution feature engineering"`
- `"state of the art tabular_classification tabular data 2025"`
- Если задача специфичная: `"tabular_classification roi improvement kaggle winning solution"`
- Ищи: какие фичи используют топы, какие ансамбли, какие трюки

#### Шаг 3 — Формулировка новых гипотез
На основе анализа и поиска запиши в program.md раздел:
```
## Research Insights (plateau iteration N)
- **Найдено:** (что нашёл в поиске)
- **Гипотеза A:** (конкретная идея + ожидаемый прирост)
- **Гипотеза B:** (конкретная идея + ожидаемый прирост)
- **Выбранная следующая попытка:** (почему именно это)
```

#### Шаг 4 — Реализация
Реализуй самую перспективную гипотезу из шага 3.
Если она тоже не даёт прироста — повтори протокол с шага 1.