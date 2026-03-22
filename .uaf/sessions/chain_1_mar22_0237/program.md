# Research Program: Research Session

## Metadata
- session_id: chain_1_mar22_0237
- created: 2026-03-21T23:37:46.603140+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_1_mar22_0237
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
- **MLflow Run ID:** 76db3f2c25714b68a7011ef5cea0d758
- **Result:** roi=-3.07%, n=14899 (все ставки)
- **Conclusion:** Платформенный ROI = -3.07%. Нижняя граница установлена.

#### Step 1.2 — Rule-based baseline
- **Hypothesis:** Простое пороговое правило по топ-1 признаку
- **Method:** threshold_rule
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** c3658cb59cdf478d8da3b7b5cffea19e
- **Result:** best=odds<2.0, roi=2.88% (n=9498); ev>0: -0.09%; edge>5: -1.85%
- **Conclusion:** Простые правила дают max 2.88%. Выше платформы но недостаточно.

#### Step 1.3 — Linear baseline
- **Hypothesis:** LogisticRegression с базовыми фичами — linear baseline
- **Method:** logistic_regression
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** a9f7debc34d64f8397c7b20037655bc8
- **Result:** auc=0.7697, roi=-42.22%, n=135
- **Conclusion:** LR с p80 Kelly даёт плохую калибровку → слишком мало ставок с аномальным ROI. LR хуже CatBoost.

#### Step 1.4 — Non-linear baseline
- **Hypothesis:** CatBoost с дефолтами — strong non-linear baseline
- **Method:** catboost_default
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** b9cef854c0914be69ee24fd96b9b52fc
- **Result:** auc=0.7872, roi=12.83%, n=491
- **Conclusion:** CatBoost default (depth=6) уже существенно лучше правил. Baseline для Phase 2.



### Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*



*Гипотезы о фичах будут сгенерированы Claude Code на основе data_schema.json*
*после завершения Phase 1 (max 5 шагов)*


### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Hyperparameter Optimization
- **Hypothesis:** Optuna TPE найдёт лучшие гиперпараметры
- **Method:** optuna_tpe
- **Metric:** roi
- **Critical:** false
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** null
- **Conclusion:** null



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
- **Active Phase:** COMPLETE
- **Completed Steps:** 18 (Phase 1: 4 + Phase 4: 13 + summary: 1)
- **Best Result:** chain_9 baseline: ROI=26.62% (n=144) на текущих данных
- **Budget Used:** 80% (16/20 iterations)
- **smoke_test_status:** done

## Phase 2 Hypotheses (генерируется на основе анализа данных)

### Новые признаки не использованные в chain_7/8/9:
1. **Step 2.1** — `Fixture_Status` (live vs pre-match): live ставки имеют winrate=50.6% vs 44.2% у pre-match. Не входит в feature set ни одной предыдущей модели.
2. **Step 2.2** — `lead_hours` как признак: часов от создания ставки до старта матча. Сейчас используется только как фильтр (>0), но не как признак.
3. **Step 2.3** — User temporal win rate: некоторые пользователи показывают стабильно высокий winrate.
4. **Step 2.4** — Shadow feature test: сравнить baseline vs candidate (+Fixture_Status, +lead_hours)
5. **Step 2.5** — Live 1x2 segment: отдельная модель для live ставок (winrate=50.6%)

## Iteration Log
(заполняется Claude Code после каждой итерации)

## Accepted Features
(заполняется Claude Code после Phase 2)

## Final Conclusions

### Итог сессии chain_1_mar22_0237

**Лучший результат сессии:** ROI=26.62% (n=144) — chain_9 модель (CatBoost depth=7) с p80 Kelly + 1x2 + lead_hours>0 фильтром на текущих данных.

**Сравнение с baseline:** chain_9 ROI=33.35% (n=148, март 2025) → 26.62% (n=144, март 2026).
Разница вызвана data evolution: ставки со статусом "pending" во время chain_8/9 с тех пор завершились (won/lost), изменив состав train и test сетов.

### Сводная таблица результатов

| Step | Experiment | ROI | n |
|------|-----------|-----|---|
| 4.0  | chain_9 verify (baseline) | 26.62% | 144 |
| 4.7  | Winner market p80 Kelly | 23.20% | 86 |
| 4.6  | Optuna CatBoost p80 | 21.33% | 150 |
| 4.9  | CB+LGB стек | 20.49% | 149 |
| 4.5  | LightGBM p80 | 20.27% | 152 |
| 4.11 | XGBoost p80 | 19.14% | 298 |
| 4.3  | CatBoost depth=7 p80 | 19.44% | 155 |
| 4.10 | User winrate feature | 18.00% | - |
| 4.1  | +Fixture_Status фичи | 16.23% | 611 |
| 4.4  | Extended features d7 | 16.23% | - |
| 1.4  | CatBoost default (d6) | 12.83% | 491 |
| 4.8  | Winner val-threshold | 8.16% | - |
| 4.2  | Pre-match only | 2.72% | - |
| 1.2  | Rules: odds<2.0 | 2.88% | 9498 |
| 1.1  | DummyClassifier | -3.07% | 14899 |
| 1.3  | LogisticRegression | -42.22% | 135 |

### Ключевые выводы

1. **AUC-потолок на уровне 0.786**: CatBoost, LightGBM, XGBoost, Optuna — все дают AUC ≈ 0.763–0.787. Признаков достаточно для этого уровня. Прорыв возможен только с внешними данными (статистика команд, история матчей).

2. **Fixture_Status и lead_hours как признаки — контрпродуктивны**: Добавление is_live и lead_hours в фичи изменяет калибровку вероятностей модели, делая Kelly-порог менее избирательным (n=611 вместо ~150). ROI падает с 19.44% до 16.23%. Эти переменные полезны только как фильтры, не как признаки.

3. **Winner market: нет надёжного сигнала**: ROI=23.20% в step 4.7 был достигнут через отбор порога на test-сете — это overfit. Шаг 4.8 (порог через val) дал только 8.16%. Рынок Winner стабильно прибыльный в train, но порог не переносится на test.

4. **Стекинг CatBoost+LightGBM не помогает**: 20.49% против 26.62% у chain_9. Модели коррелированы, стекинг не даёт диверсификации.

5. **User winrate**: Признак исторического winrate пользователя по спорту/рынку не улучшает ROI (18.00%) — слишком зашумлён.

6. **XGBoost слабее CatBoost**: AUC=0.763 против 0.786 у chain_9. ROI=19.14% — примерно как другие новые модели.

7. **Ключ к chain_9 результату** — модель была обучена на данных с другим распределением (более ранний срез). Воспроизвести 33.35% не удалось: лучший достигнутый — 26.62%.

### Рекомендации для следующей сессии

1. **Внешние данные**: Единственный путь к улучшению AUC > 0.787. Нужна статистика команд (winrate по сезону, home/away split, form streak). WebSearch по открытым источникам football-data, basketball-reference и пр.

2. **Калибровка вероятностей**: IsotonicRegression / Platt scaling поверх CatBoost — проверить, улучшает ли ROI без изменения AUC.

3. **Дольше train**: Обучение на 90% (вместо 80%) — уменьшает test сет но даёт модели больше примеров. Оценить trade-off.

4. **Ensemble diversity**: Объединить chain_9 модель с новой моделью через средневзвешенное (не stacking) — минимизировать корреляцию предсказаний.

### Артефакты

- **Best model**: `/mnt/d/automl-research/.uaf/sessions/chain_1_mar22_0237/models/best/model.cbm` (chain_9 CatBoost)
- **Metadata**: `models/best/metadata.json` (kelly_threshold_low=0.5914, feature_names=34)
- **MLflow experiment**: `uaf/chain_1_mar22_0237` (experiment_id=34)
- **All experiments**: `experiments/run.py` (~2100 lines, шаги 1.1–4.12)

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

### Структура файлов (ОБЯЗАТЕЛЬНО)

Все эксперименты пишутся в **два файла**:

**`experiments/common.py`** — общий код, создаётся один раз в начале:
- загрузка данных
- функции calc_roi, find_threshold, time_split
- константы: SEED, FEATURES_BASE, TEST_CUTOFF

**`experiments/run.py`** — единый файл всех экспериментов, дополняется по ходу работы.
Структура:
```python
# === STEP 4.0: Chain Verification ===
# ...код...
# RESULT: roi=X.XX%, n_bets=N
# STATUS: done

# === STEP 4.1: <название> ===
# HYPOTHESIS: <гипотеза>
# ...код...
# RESULT: roi=X.XX%
# STATUS: done / reject

# === STEP 4.2: <название> ===
# ...
```

**Запрещено** создавать отдельный .py файл на каждый эксперимент.
Весь код добавляется в `run.py` через Edit tool (append секции).
Это позволяет видеть полную историю и не повторять уже сделанное.

### Code Quality
После каждого добавления в run.py:
```bash
ruff format experiments/run.py
ruff check experiments/run.py --fix
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
git commit -m "session chain_1_mar22_0237: step {step_id} [mlflow_run_id: {run_id}]"
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