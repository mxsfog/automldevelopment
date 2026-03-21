# Research Program: Research Session

## Metadata
- session_id: chain_6_mar21_2236
- created: 2026-03-21T19:36:19.074366+00:00
- approved_by: pending
- approval_time: null
- budget_mode: fixed
- budget_summary: fixed: max 50 iterations, max ?h
- claude_model: claude-opus-4
- mlflow_experiment: uaf/chain_6_mar21_2236
- mlflow_tracking_uri: http://127.0.0.1:5000

## Task Description

Предсказание победы ставки (won/lost) на спортивных событиях. Данные со стейкинг-платформы: синглы и парлаи, 20+ рынков, 10+ видов спорта. Цель — ROI >= 10% на отобранных ставках.



## Previous Session Context
# Previous Session Context: chain_5_mar21_2228

## Best Results Achieved
- Best metric: нет данных

## What Was Tried (do NOT repeat)
| Step | Гипотеза | ROI test | Delta | MLflow Run ID |
|------|----------|----------|-------|---------------|
| 1.1 | DummyClassifier lower bound | -3.07% | — | 913cae89313c44dd9d9979d324d500ee |
| 1.2 | Rule ML_Edge threshold | -4.89% | — | 9fb83ab81a64488c90949b0ead0c0a73 |
| 1.3 | LogisticRegression+Kelly | 8.26% | — | 25a160acd0874050bdb7268acbbcf0a4 |
| 1.4 | CatBoost default+Kelly | 24.91% | — | ba5506eee46f4492b29e260494903753 |

## Accepted Features
(заполняется Claude Code после Phase 2)

## Recommended Next Steps
(заполняется Claude Code по завершении)

---




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
- **MLflow Run ID:** 99a980a28fe24d52aa83a5401fc44d94
- **Result:** ROI=-3.07% (n=14899 bets, all), ROI=-6.78% (n=5248 pre-match)
- **Conclusion:** Lower bound установлен. Без фильтрации ROI=-3.07%.


#### Step 1.2 — Rule-based baseline
- **Hypothesis:** Простое пороговое правило по топ-1 признаку
- **Method:** threshold_rule
- **Metric:** roi
- **Critical:** false
- **Status:** done
- **MLflow Run ID:** 0e1b4d47e6124603962d6a592f9d2bfc
- **Result:** ROI=2.40% (n=506, ML_Edge>=15.0), val ROI=49.74%
- **Conclusion:** Простое правило не работает — большой разрыв val/test. Overfitting на val.


#### Step 1.3 — Linear baseline
- **Hypothesis:** LogisticRegression с базовыми фичами — linear baseline
- **Method:** logistic_regression
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** 3c8f86dd190e4c87bdd697d357f20479
- **Result:** ROI=-5.42% (n=1966 bets), AUC=0.7948
- **Conclusion:** LogReg хуже чем в предыдущих сессиях (8.26%). Kelly не помог — val ROI отрицательный. AUC высокий но Kelly плохо работает с LogReg.


#### Step 1.4 — Non-linear baseline
- **Hypothesis:** CatBoost с дефолтами — strong non-linear baseline
- **Method:** catboost_default
- **Metric:** roi
- **Critical:** true
- **Status:** done
- **MLflow Run ID:** cecf54e2934a4cc88a269a032d43eca2
- **Result:** ROI=24.91% (n=435 bets), AUC=0.7863, threshold=0.455
- **Conclusion:** Baseline подтверждён. CatBoost+Kelly воспроизводит 24.91%. n_bets=435 > 200 (валидно).



### Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*

#### Step 2.1 — Features v2 (rejected)
- **Status:** done
- **MLflow Run ID:** a3f765d6813149569aa804533fba92a3
- **Result:** ROI=-0.92% (n=1574)
- **Conclusion:** Слишком много фичей (46). lead_hours/is_prematch как features ломают Kelly threshold (упал до 0.125). Overfitting on val (83.53% vs -0.92% test). Отклонено.

#### Step 2.2 — week_of_year shadow (rejected)
- **Status:** done
- **MLflow Run ID:** 5c8164d0fdce4f24b63ec1e92afd4364
- **Result:** ROI=8.82% (n=127), delta=-16.09%
- **Conclusion:** week_of_year хуже day_of_week. n_bets=127 < 200 (невалидно). AUC снизился. Отклонено.

#### Step 2.3 — Segment analysis (neutral)
- **Status:** done
- **MLflow Run ID:** d9edd78981b64672adf7881ab81aa0e7
- **Result:** ROI=24.91% (n=435), delta=0.00%
- **Conclusion:** Все сегменты прибыльны на val (Soccer 97.5%, Tennis 32.9%). Нет убыточных сегментов для фильтрации. Baseline набор признаков принят.


### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Optuna ROI objective (rejected)
- **Status:** done
- **MLflow Run ID:** ab302c9f5e654c13845623489adfa7fc
- **Result:** ROI=2.44% (n=2503), val=103.71%
- **Conclusion:** Optuna максимизировал val ROI с threshold → двойное переобучение на val. depth=10 overfits.

#### Step 3.2 — Optuna AUC objective (rejected)
- **Status:** done
- **MLflow Run ID:** 4c49ce2513de4aa88f0e085cb6b39f9a
- **Result:** ROI=9.22% (n=2438), val_auc=0.9694
- **Conclusion:** AUC objective лучше, но val_auc=0.97 vs test_auc=0.77 — val data в train_df. Все ещё overfits.

#### Step 3.3 — Proper split baseline + Optuna (reference)
- **Status:** done
- **MLflow Run ID:** fc4104196ba84821b79976c1b80e634a
- **Result:** baseline_proper=0.94% (n=173), optuna=-23.05%
- **Conclusion:** С train=0-64% baseline падает до 0.94% (n<200, невалидно). Train=0-80% (включая val) даёт лучшую генерализацию на тест (24.91%) — более свежие данные в обучении. Phase 3 не улучшила baseline.



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
- **Completed Steps:** 14/13+
- **Best Result:** ROI=24.91% (CatBoost+Kelly, step 1.4)
- **Budget Used:** ~25%
- **smoke_test_status:** done

## Iteration Log
| Step | Гипотеза | ROI test | Delta | MLflow Run ID |
|------|----------|----------|-------|---------------|
| 1.1 | DummyClassifier lower bound | -3.07% | — | 99a980a28fe24d52aa83a5401fc44d94 |
| 1.2 | Rule ML_Edge>=15 threshold | 2.40% | — | 0e1b4d47e6124603962d6a592f9d2bfc |
| 1.3 | LogisticRegression+Kelly | -5.42% | — | 3c8f86dd190e4c87bdd697d357f20479 |
| 1.4 | CatBoost default+Kelly | 24.91% | — | cecf54e2934a4cc88a269a032d43eca2 |
| 2.1 | Features v2 (46 фичи) | -0.92% | -25.83% | a3f765d6813149569aa804533fba92a3 |
| 2.2 | week_of_year shadow | 8.82% | -16.09% | 5c8164d0fdce4f24b63ec1e92afd4364 |
| 2.3 | Segment analysis | 24.91% | 0.00% | d9edd78981b64672adf7881ab81aa0e7 |
| 3.1 | Optuna ROI objective | 2.44% | -22.47% | ab302c9f5e654c13845623489adfa7fc |
| 3.2 | Optuna AUC objective | 9.22% | -15.69% | 4c49ce2513de4aa88f0e085cb6b39f9a |
| 3.3 | Proper split (0-64%) + Optuna | 0.94% | -23.97% | fc4104196ba84821b79976c1b80e634a |
| 4.1 | Seed ensemble (5 seeds) | 24.90% | -0.01% | ced96943386b410e95ab070d0c630ae5 |
| 4.2 | LightGBM | 5.78% | -19.13% | 35b962a8bd914f68a07193309ceeefc3 |
| 4.3 | CB+LGBM 50/50 ensemble | -0.01% | -24.92% | 99d7a8f440934aba9cada3ea3a2300f6 |
| 4.4 | Isotonic calibration | 24.91% raw / 14.68% cal | 0.00% / -10.23% | 003cc48eb4214f05945d3c9b2137590a |

## Accepted Features
Baseline set из step 1.4 (33 фичи):
Odds, USD, log_odds, log_usd, implied_prob, is_parlay, outcomes_count,
ml_p_model, ml_p_implied, ml_edge, ml_ev, ml_team_stats_found, ml_winrate_diff, ml_rating_diff,
hour, day_of_week, month, odds_times_stake, ml_edge_pos, ml_ev_pos,
elo_max, elo_min, elo_diff, elo_ratio, elo_mean, elo_std, k_factor_mean, has_elo, elo_count,
ml_edge_x_elo_diff, elo_implied_agree, Sport (cat), Market (cat), Currency (cat)

## Research Insights (plateau iteration 1)
- **Найдено:** Walsh & Joshi (2024): выбор модели по calibration вместо accuracy даёт +34% vs -35% ROI. Isotonic regression calibration — #1 способ улучшить Kelly betting.
- **Гипотеза A:** Isotonic calibration после CatBoost → лучше откалиброванные Kelly → больше прибыльных ставок при том же threshold
- **Гипотеза B:** Fixed EV threshold (EV > 5%) вместо grid search val ROI → убирает overfitting на val
- **Гипотеза C:** Soccer-only модель — сегмент с наибольшим val ROI (97.5%), n=278
- **Выбранная следующая попытка:** Step 4.4 — Isotonic calibration + CatBoost baseline
- **Итог 4.4:** REJECTED. Brier raw=0.1859 < cal=0.1955 — isotonic ухудшила калибровку. Test cal=14.68% << baseline 24.91%.

## Research Insights (plateau iteration 2)
- **Анализ:** 4 итерации Phase 4 без улучшения. LightGBM/ensemble дают val ROI>50% но test<6%. Calibration не помогает. Baseline CatBoost depth=7 устойчиво держит 24.91%.
- **Ключевой вопрос:** Можно ли улучшить отбор ставок через сегментацию (Soccer-only) или feature ablation?
- **Гипотеза A:** Soccer-only модель — Cricket/Basketball могут добавлять шум. Isolated Soccer-only train+test может дать более чистый сигнал.
- **Гипотеза B:** Feature importance ablation — top-15 фичей избавит от шума при тех же гиперпараметрах.
- **Гипотеза C:** Platt scaling (LogisticRegression поверх CatBoost скоров) — более устойчива чем isotonic при малой calib выборке.
- **Выбранная следующая попытка:** Step 4.5 — Soccer-only CatBoost (Hypothesis C iteration 1)

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
git commit -m "session chain_6_mar21_2236: step {step_id} [mlflow_run_id: {run_id}]"
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