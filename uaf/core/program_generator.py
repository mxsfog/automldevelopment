"""ProgramMdGenerator — подготовка context/ пакета для Claude Code."""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)

# Базовые типы задач для определения baseline логики
_BASELINE_STEPS: dict[str, list[dict[str, str]]] = {
    "tabular_classification": [
        {
            "id": "1.1",
            "name": "Constant baseline",
            "method": "dummy_classifier",
            "hypothesis": "DummyClassifier (most_frequent) задаёт lower bound",
            "critical": "true",
        },
        {
            "id": "1.2",
            "name": "Rule-based baseline",
            "method": "threshold_rule",
            "hypothesis": "Простое пороговое правило по топ-1 признаку",
            "critical": "false",
        },
        {
            "id": "1.3",
            "name": "Linear baseline",
            "method": "logistic_regression",
            "hypothesis": "LogisticRegression с базовыми фичами — linear baseline",
            "critical": "true",
        },
        {
            "id": "1.4",
            "name": "Non-linear baseline",
            "method": "catboost_default",
            "hypothesis": "CatBoost с дефолтами — strong non-linear baseline",
            "critical": "true",
        },
    ],
    "tabular_regression": [
        {
            "id": "1.1",
            "name": "Constant baseline",
            "method": "dummy_regressor",
            "hypothesis": "DummyRegressor (mean) задаёт lower bound",
            "critical": "true",
        },
        {
            "id": "1.2",
            "name": "Linear baseline",
            "method": "linear_regression",
            "hypothesis": "LinearRegression с базовыми фичами",
            "critical": "true",
        },
        {
            "id": "1.3",
            "name": "Non-linear baseline",
            "method": "catboost_default",
            "hypothesis": "CatBoost с дефолтами — non-linear baseline",
            "critical": "true",
        },
    ],
    "nlp_classification": [
        {
            "id": "1.1",
            "name": "Constant baseline",
            "method": "dummy_classifier",
            "hypothesis": "DummyClassifier задаёт lower bound",
            "critical": "true",
        },
        {
            "id": "1.2",
            "name": "TF-IDF + LR baseline",
            "method": "tfidf_logreg",
            "hypothesis": "TF-IDF + LogisticRegression — классический NLP baseline",
            "critical": "true",
        },
        {
            "id": "1.3",
            "name": "Pretrained embeddings baseline",
            "method": "sentence_transformer_logreg",
            "hypothesis": "Pretrained embeddings + classifier",
            "critical": "false",
        },
    ],
    "time_series": [
        {
            "id": "1.1",
            "name": "Naive baseline",
            "method": "naive_last_value",
            "hypothesis": "Naive (last value) — нижняя граница",
            "critical": "true",
        },
        {
            "id": "1.2",
            "name": "Statistical baseline",
            "method": "exponential_smoothing",
            "hypothesis": "Exponential Smoothing — классический метод",
            "critical": "true",
        },
        {
            "id": "1.3",
            "name": "ML baseline",
            "method": "lightgbm_lag_features",
            "hypothesis": "LightGBM с lag-фичами",
            "critical": "false",
        },
    ],
}

# Дефолтные шаги для неизвестных типов задач
_DEFAULT_BASELINE_STEPS = [
    {
        "id": "1.1",
        "name": "Constant baseline",
        "method": "dummy",
        "hypothesis": "Простейший baseline для lower bound",
        "critical": "true",
    },
    {
        "id": "1.2",
        "name": "Linear baseline",
        "method": "linear_model",
        "hypothesis": "Линейная модель как первый нетривиальный baseline",
        "critical": "true",
    },
    {
        "id": "1.3",
        "name": "Non-linear baseline",
        "method": "gradient_boosting",
        "hypothesis": "Gradient boosting с дефолтами",
        "critical": "true",
    },
]

# Максимум 5 shadow feature шагов в Phase 2
_MAX_FEATURE_STEPS = 5

# Шаблон program.md
_PROGRAM_MD_TEMPLATE = """\
# Research Program: {{ task_title }}

## Metadata
- session_id: {{ session_id }}
- created: {{ created_at }}
- approved_by: pending
- approval_time: null
- budget_mode: {{ budget_mode }}
- budget_summary: {{ budget_summary }}
- claude_model: claude-opus-4
- mlflow_experiment: uaf/{{ session_id }}
- mlflow_tracking_uri: {{ mlflow_tracking_uri }}

## Task Description

{{ task_description }}

**Target column:** `{{ target_column }}`
**Metric:** {{ metric_name }} ({{ metric_direction }})
**Task type:** {{ task_type }}

{% if constraints %}
**Constraints:**
{% for k, v in constraints.items() %}
- {{ k }}: {{ v }}
{% endfor %}
{% endif %}

## Validation Scheme

**Scheme:** {{ validation_scheme }}
**Resolved by:** {{ validation_resolved_by }}
**Parameters:**
{% for k, v in validation_params.items() %}
- {{ k }}: {{ v }}
{% endfor %}

**Validation constraints (enforced by UAF):**
{% for constraint in validation_constraints %}
- {{ constraint }}
{% endfor %}

{% if validation_warnings %}
**Critical warnings:**
{% for w in validation_warnings %}
- {{ w }}
{% endfor %}
{% endif %}

## Data Summary

{{ data_summary }}

{% if feature_hints %}
**Feature hints (из data_schema.json):**
{% for hint in feature_hints %}
- {{ hint }}
{% endfor %}
{% endif %}

## Research Phases

### Phase 1: Baseline (MANDATORY)
**Goal:** Установить нижнюю границу и strong baseline
**Success Criterion:** Превысить random baseline по {{ metric_name }}

{% for step in baseline_steps %}
#### Step {{ step.id }} — {{ step.name }}
- **Hypothesis:** {{ step.hypothesis }}
- **Method:** {{ step.method }}
- **Metric:** {{ metric_name }}
- **Critical:** {{ step.critical }}
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** (заполняется Claude Code)
- **Conclusion:** (заполняется Claude Code)

{% endfor %}

### Phase 2: Feature Engineering (MANDATORY)
*Выполняется после Phase 1 всегда*
*Пропускается только если skip_feature_engineering: true в task.yaml*

{% for step in feature_steps %}
#### Step {{ step.id }} — Shadow: {{ step.name }}
- **Hypothesis:** {{ step.hypothesis }}
- **Method:** shadow_feature_trick
- **Shadow features:** {{ step.shadow_features }}
- **Baseline run ID:** (заполняется после Phase 1)
- **Metric:** {{ metric_name }}
- **Critical:** false
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** null
- **Conclusion:** null

{% endfor %}
{% if not feature_steps %}
*Гипотезы о фичах будут сгенерированы Claude Code на основе data_schema.json*
*после завершения Phase 1 (max {{ max_feature_steps }} шагов)*
{% endif %}

### Phase 3: Model Optimization (MANDATORY)
*Включается после фиксации feature set из Phase 2*
*Optuna Hyperparameter Search на лучшей конфигурации*

#### Step 3.1 — Hyperparameter Optimization
- **Hypothesis:** Optuna TPE найдёт лучшие гиперпараметры
- **Method:** optuna_tpe
- **Metric:** {{ metric_name }}
- **Critical:** false
- **Status:** pending
- **MLflow Run ID:** null
- **Result:** null
- **Conclusion:** null

## Current Status
- **Active Phase:** Phase 1
- **Completed Steps:** 0/{{ total_steps }}
- **Best Result:** null
- **Budget Used:** 0%
- **smoke_test_status:** pending

## Iteration Log
(заполняется Claude Code после каждой итерации)

## Accepted Features
(заполняется Claude Code после Phase 2)

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
    "validation_scheme": "{{ validation_scheme }}",
    "seed": {{ seed }},
    "n_samples_train": len(X_train),
    "n_samples_val": len(X_val),
})
# Для k-fold: дополнительно
mlflow.set_tag("fold_idx", str(fold_idx))
mlflow.log_metric("{{ metric_name }}_fold_0", fold_score_0)
mlflow.log_metric("{{ metric_name }}_mean", mean_score)
mlflow.log_metric("{{ metric_name }}_std", std_score)
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
random.seed({{ seed }})
np.random.seed({{ seed }})
# При использовании PyTorch:
# import torch; torch.manual_seed({{ seed }})
```

### Budget Check (перед каждым экспериментом)
```python
import json
budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        # Завершить текущую работу, написать выводы, остановиться
        mlflow.set_tag("status", "budget_stopped")
        sys.exit(0)
except FileNotFoundError:
    pass  # файл ещё не создан
```

### DVC Protocol
После завершения каждого шага:
```bash
git add .
git commit -m "session {{ session_id }}: step {step_id} [mlflow_run_id: {run_id}]"
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

### Update program.md
После каждого шага обновляй:
- Step **Status**: pending -> done/failed
- Step **MLflow Run ID**: заполни run_id
- Step **Result**: заполни метрику
- Step **Conclusion**: напиши вывод
- **Current Status**: обнови Best Result и Budget Used
- **Iteration Log**: добавь запись
- После Phase 2: заполни **Accepted Features**
"""


@dataclass
class FeatureHypothesis:
    """Гипотеза о новом признаке для Phase 2.

    Атрибуты:
        step_id: идентификатор шага (2.1, 2.2 ...).
        name: название признака или группы.
        hypothesis: текст гипотезы.
        shadow_features: список имён shadow-признаков.
    """

    step_id: str
    name: str
    hypothesis: str
    shadow_features: list[str]


def _load_task_yaml(task_path: Path) -> dict[str, Any]:
    """Читает task.yaml.

    Args:
        task_path: путь к task.yaml.

    Returns:
        Словарь с конфигурацией задачи.

    Raises:
        ValueError: файл не найден или некорректный формат.
    """
    if not task_path.exists():
        raise ValueError(f"task.yaml не найден: {task_path}")
    with task_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"task.yaml должен быть YAML словарём: {task_path}")
    return data


def _load_data_schema(schema_path: Path | None) -> dict[str, Any]:
    """Читает data_schema.json.

    Args:
        schema_path: путь к schema файлу.

    Returns:
        Словарь схемы данных или пустой словарь.
    """
    if not schema_path or not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text())


def _load_feature_registry(registry_path: Path | None) -> dict[str, Any]:
    """Читает feature_registry.json если есть (для --resume сессий).

    Args:
        registry_path: путь к feature_registry.json.

    Returns:
        Словарь реестра фич или пустой словарь.
    """
    if not registry_path or not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text())


def _build_data_summary(schema: dict[str, Any]) -> str:
    """Формирует краткое описание данных из schema.

    Args:
        schema: data_schema.json содержимое.

    Returns:
        Текстовое summary.
    """
    if not schema:
        return "data_schema.json не предоставлен."

    splits = schema.get("splits", {})
    lines = []
    for name, info in splits.items():
        n_rows = info.get("n_rows", "?")
        n_cols = info.get("n_cols", "?")
        lines.append(f"- {name}: {n_rows} строк, {n_cols} колонок")

    features = schema.get("features", {})
    feature_types: dict[str, int] = {}
    for _col, meta in features.items():
        t = meta.get("type", "unknown")
        feature_types[t] = feature_types.get(t, 0) + 1

    if feature_types:
        type_str = ", ".join(f"{v} {k}" for k, v in feature_types.items())
        lines.append(f"- Типы признаков: {type_str}")

    missing_info = schema.get("missing_values", {})
    if missing_info:
        total_missing = sum(missing_info.values())
        if total_missing > 0:
            cols_with_missing = sum(1 for v in missing_info.values() if v > 0)
            lines.append(f"- Пропуски: {cols_with_missing} колонок с пропусками")

    hints = schema.get("task_hints", {})
    if hints.get("class_imbalance"):
        lines.append(f"- Дисбаланс классов: {hints['class_imbalance']}")

    return "\n".join(lines) if lines else "Статистика недоступна."


def _build_feature_hints(schema: dict[str, Any], already_tried: set[str]) -> list[str]:
    """Извлекает feature engineering hints из task_hints.

    Args:
        schema: data_schema.json.
        already_tried: имена фич из предыдущих сессий.

    Returns:
        Список строк-подсказок.
    """
    hints_raw: list[str] = []
    task_hints = schema.get("task_hints", {})
    for key in ("potential_feature_engineering", "leakage_warnings", "recommended_encoding"):
        val = task_hints.get(key)
        if isinstance(val, list):
            hints_raw.extend(val)
        elif isinstance(val, str) and val:
            hints_raw.append(val)
    return [h for h in hints_raw if h not in already_tried]


def _generate_feature_hypotheses(
    schema: dict[str, Any],
    task_type: str,
    already_tried: set[str],
    max_steps: int = _MAX_FEATURE_STEPS,
) -> list[FeatureHypothesis]:
    """Детерминированная генерация гипотез о признаках по data_schema.json.

    Применяет правила FG-T-*, FG-N-*, FG-I-*, FG-C-* без LLM вызовов.

    Args:
        schema: data_schema.json.
        task_type: тип задачи.
        already_tried: имена уже опробованных признаков.
        max_steps: максимальное число гипотез.

    Returns:
        Список FeatureHypothesis (до max_steps).
    """
    hypotheses: list[FeatureHypothesis] = []
    features = schema.get("features", {})
    step_counter = 1

    # FG-T-*: временные признаки
    datetime_cols = [col for col, meta in features.items() if meta.get("type") in ("datetime", "timestamp")]
    for col in datetime_cols:
        if len(hypotheses) >= max_steps:
            break
        shadows = []
        for feat_name in (f"hour_of_day_{col}", f"day_of_week_{col}", f"is_weekend_{col}"):
            if feat_name not in already_tried:
                shadows.append(feat_name)
        if shadows:
            hypotheses.append(
                FeatureHypothesis(
                    step_id=f"2.{step_counter}",
                    name=f"datetime features from {col}",
                    hypothesis=f"Временные признаки из {col} (час, день недели, выходной)",
                    shadow_features=shadows[:3],
                )
            )
            step_counter += 1

    # FG-N-*: логарифм для скошенных распределений
    for col, meta in features.items():
        if len(hypotheses) >= max_steps:
            break
        stats = meta.get("stats", {})
        skewness = stats.get("skewness", 0)
        feat_name = f"log1p_{col}"
        if (
            meta.get("type") in ("numeric", "float", "int")
            and abs(float(skewness)) > 2.0
            and feat_name not in already_tried
        ):
            hypotheses.append(
                FeatureHypothesis(
                    step_id=f"2.{step_counter}",
                    name=f"log1p transform {col}",
                    hypothesis=f"log1p({col}) для скошенного распределения (skewness={skewness:.1f})",
                    shadow_features=[feat_name],
                )
            )
            step_counter += 1

    # FG-C-*: target encoding для высококардинальных категориальных
    for col, meta in features.items():
        if len(hypotheses) >= max_steps:
            break
        cardinality = meta.get("unique_count", 0)
        if meta.get("type") in ("categorical", "string", "object") and int(cardinality) > 10:
            shadows = []
            for feat_name in (f"{col}_target_enc", f"{col}_count_enc"):
                if feat_name not in already_tried:
                    shadows.append(feat_name)
            if shadows:
                hypotheses.append(
                    FeatureHypothesis(
                        step_id=f"2.{step_counter}",
                        name=f"encoding for {col}",
                        hypothesis=f"Target/count encoding для {col} (cardinality={cardinality})",
                        shadow_features=shadows,
                    )
                )
                step_counter += 1

    # FG-I-*: взаимодействия для пар numeric с высокой корреляцией с target
    numeric_cols = [
        col for col, meta in features.items()
        if meta.get("type") in ("numeric", "float", "int")
        and meta.get("correlation_with_target") is not None
    ]
    numeric_cols_sorted = sorted(
        numeric_cols,
        key=lambda c: abs(float(features[c].get("correlation_with_target", 0))),
        reverse=True,
    )
    top_cols = numeric_cols_sorted[:3]
    for i, col_a in enumerate(top_cols):
        if len(hypotheses) >= max_steps:
            break
        for col_b in top_cols[i + 1 :]:
            if len(hypotheses) >= max_steps:
                break
            ratio_feat = f"{col_a}_div_{col_b}"
            if ratio_feat not in already_tried:
                b_stats = features.get(col_b, {}).get("stats", {})
                if float(b_stats.get("min", 0)) != 0:
                    hypotheses.append(
                        FeatureHypothesis(
                            step_id=f"2.{step_counter}",
                            name=f"ratio {col_a}/{col_b}",
                            hypothesis=f"Ратио {col_a}/{col_b} — взаимодействие топ признаков",
                            shadow_features=[ratio_feat],
                        )
                    )
                    step_counter += 1

    return hypotheses[:max_steps]


class ProgramMdGenerator:
    """Подготовка context/ пакета для Claude Code.

    Не делает LLM вызовов. Детерминированно читает task.yaml + data_schema.json
    и записывает структурированный контекст в SESSION_DIR/context/.

    Также генерирует шаблон program.md который Claude Code заполняет при старте.

    Args:
        session_dir: директория сессии (.uaf/sessions/{session_id}/).
        mlflow_tracking_uri: URI MLflow сервера.
    """

    def __init__(self, session_dir: Path, mlflow_tracking_uri: str = "http://127.0.0.1:5000") -> None:
        self.session_dir = session_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._context_dir = session_dir / "context"

    def prepare_context(
        self,
        task_path: Path,
        session_id: str,
        data_schema_path: Path | None = None,
        validation_report: Any | None = None,
        feature_registry_path: Path | None = None,
        improvement_context_path: Path | None = None,
    ) -> Path:
        """Подготавливает context/ пакет в SESSION_DIR.

        Args:
            task_path: путь к task.yaml.
            session_id: идентификатор сессии.
            data_schema_path: путь к data_schema.json.
            validation_report: ValidationReport (из ValidationChecker.run_pre_session).
            feature_registry_path: путь к feature_registry.json (для --resume).
            improvement_context_path: путь к improvement_context.md (для --resume).

        Returns:
            Путь к context/ директории.
        """
        self._context_dir.mkdir(parents=True, exist_ok=True)

        task_data = _load_task_yaml(task_path)
        schema = _load_data_schema(data_schema_path)
        registry = _load_feature_registry(feature_registry_path)

        # Копируем исходные файлы в context/
        shutil.copy2(task_path, self._context_dir / "task.yaml")
        if data_schema_path and data_schema_path.exists():
            shutil.copy2(data_schema_path, self._context_dir / "data_schema.json")

        # Генерируем data_context.json
        data_ctx = self._build_data_context(schema, task_data)
        (self._context_dir / "data_context.json").write_text(
            json.dumps(data_ctx, ensure_ascii=False, indent=2)
        )

        # Генерируем validation_context.json
        val_ctx = self._build_validation_context(validation_report)
        (self._context_dir / "validation_context.json").write_text(
            json.dumps(val_ctx, ensure_ascii=False, indent=2)
        )

        # improvement_context.md (для --resume)
        if improvement_context_path and improvement_context_path.exists():
            shutil.copy2(
                improvement_context_path,
                self._context_dir / "improvement_context.md",
            )
            logger.info("improvement_context.md скопирован из предыдущей сессии")

        # Генерируем шаблон program.md
        program_md = self._render_program_md(
            task_data=task_data,
            schema=schema,
            registry=registry,
            session_id=session_id,
            validation_report=validation_report,
        )
        program_md_path = self.session_dir / "program.md"
        program_md_path.write_text(program_md, encoding="utf-8")

        logger.info(
            "Context/ пакет подготовлен: %s (program.md: %d байт)",
            self._context_dir,
            len(program_md),
        )
        return self._context_dir

    def _build_data_context(
        self,
        schema: dict[str, Any],
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Строит data_context.json — краткое summary данных.

        Args:
            schema: data_schema.json.
            task_data: task.yaml.

        Returns:
            Словарь context.
        """
        task = task_data.get("task", {})
        data = task_data.get("data", {})
        metric = task_data.get("metric", {})
        ctx: dict[str, Any] = {
            "task_type": task.get("type", "unknown"),
            "target_column": (
                data.get("target_column")
                or task.get("dataset", {}).get("target_column", "")
            ),
            "metric_name": metric.get("name") or task.get("metric", {}).get("name", "metric"),
            "metric_direction": (
                metric.get("direction") or task.get("metric", {}).get("direction", "maximize")
            ),
        }

        if schema:
            ctx["n_rows_train"] = schema.get("splits", {}).get("train", {}).get("n_rows", None)
            ctx["n_cols"] = schema.get("splits", {}).get("train", {}).get("n_cols", None)
            ctx["feature_types"] = {}
            for _col, meta in schema.get("features", {}).items():
                t = meta.get("type", "unknown")
                ctx["feature_types"][t] = ctx["feature_types"].get(t, 0) + 1
            ctx["task_hints"] = schema.get("task_hints", {})

        return ctx

    def _build_validation_context(self, validation_report: Any | None) -> dict[str, Any]:
        """Строит validation_context.json.

        Args:
            validation_report: ValidationReport объект или None.

        Returns:
            Словарь context.
        """
        if validation_report is None:
            return {"scheme": "unknown", "resolved_by": "unknown", "checks": []}
        return validation_report.to_dict()

    def _render_program_md(
        self,
        task_data: dict[str, Any],
        schema: dict[str, Any],
        registry: dict[str, Any],
        session_id: str,
        validation_report: Any | None,
    ) -> str:
        """Рендерит program.md через Jinja2 шаблон.

        Args:
            task_data: task.yaml содержимое.
            schema: data_schema.json.
            registry: feature_registry.json.
            session_id: ID сессии.
            validation_report: ValidationReport.

        Returns:
            Готовый текст program.md.
        """
        task = task_data.get("task", {})
        data = task_data.get("data", {})
        metric_cfg = task_data.get("metric", {})
        budget = task_data.get("budget", {})
        prefs = task_data.get("research_preferences", {})
        validation_cfg = task_data.get("validation", {})

        task_type = task.get("type", "tabular_classification")
        metric_name = metric_cfg.get("name") or task.get("metric", {}).get("name", "metric")
        metric_direction = metric_cfg.get("direction") or task.get("metric", {}).get("direction", "maximize")
        budget_mode = budget.get("mode", "fixed")
        seed = validation_cfg.get("seed", 42)

        # Формируем budget summary
        if budget_mode == "fixed":
            max_iter = budget.get("max_iterations", "?")
            max_time = budget.get("max_time_hours", "?")
            budget_summary = f"fixed: max {max_iter} iterations, max {max_time}h"
        else:
            cap_iter = budget.get("safety_cap", {}).get("max_iterations", "50")
            budget_summary = f"dynamic (convergence-based), cap {cap_iter} iterations"

        # Baseline шаги
        baseline_steps = _BASELINE_STEPS.get(task_type, _DEFAULT_BASELINE_STEPS)

        # Feature engineering гипотезы
        already_tried: set[str] = set()
        if registry:
            for feat in registry.get("engineered_features", []):
                already_tried.update(feat.get("shadow_features", [feat.get("name", "")]))

        skip_fe = prefs.get("skip_feature_engineering", False)
        feature_steps: list[dict[str, Any]] = []
        if not skip_fe:
            hypotheses = _generate_feature_hypotheses(schema, task_type, already_tried)
            feature_steps = [
                {
                    "id": h.step_id,
                    "name": h.name,
                    "hypothesis": h.hypothesis,
                    "shadow_features": ", ".join(h.shadow_features),
                }
                for h in hypotheses
            ]

        # Validation параметры
        scheme = "auto"
        val_params: dict[str, Any] = {}
        val_constraints: list[str] = []
        val_warnings: list[str] = []
        val_resolved_by = "user-specified"

        if validation_report is not None:
            scheme = getattr(validation_report, "scheme", "auto")
            val_resolved_by = getattr(validation_report, "resolved_by", "user-specified")
            val_params = {
                "n_splits": validation_cfg.get("n_splits", 5),
                "shuffle": validation_cfg.get("shuffle", True),
                "seed": seed,
                "group_col": validation_cfg.get("group_col", "null"),
                "stratify_col": validation_cfg.get("stratify_col", "null"),
                "test_holdout": validation_cfg.get("test_holdout", True),
                "test_ratio": validation_cfg.get("test_ratio", 0.1),
            }
            for check in getattr(validation_report, "checks", []):
                if check.status == "WARN" and check.hint:
                    val_warnings.append(check.hint)
                if check.code.startswith("VS-S") or check.code.startswith("VS-G"):
                    if check.status == "PASS":
                        val_constraints.append(check.message)
        else:
            scheme = validation_cfg.get("scheme", "auto")
            val_params = {
                "n_splits": validation_cfg.get("n_splits", 5),
                "seed": seed,
            }

        # Data summary
        data_summary = _build_data_summary(schema)
        feature_hints = _build_feature_hints(schema, already_tried)

        # Подсчёт шагов
        total_steps = len(baseline_steps) + len(feature_steps) + 1  # +1 для Phase 3

        # Constraints
        constraints = task.get("constraints", {})

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(_PROGRAM_MD_TEMPLATE)
        return template.render(
            task_title=task.get("title", "Research Session"),
            session_id=session_id,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            budget_mode=budget_mode,
            budget_summary=budget_summary,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            task_description=task.get("description", task.get("problem_statement", "")),
            target_column=(
                data.get("target_column")
                or task.get("dataset", {}).get("target_column", "target")
            ),
            metric_name=metric_name,
            metric_direction=metric_direction,
            task_type=task_type,
            constraints=constraints,
            validation_scheme=scheme,
            validation_resolved_by=val_resolved_by,
            validation_params=val_params,
            validation_constraints=val_constraints,
            validation_warnings=val_warnings,
            seed=seed,
            data_summary=data_summary,
            feature_hints=feature_hints,
            baseline_steps=baseline_steps,
            feature_steps=feature_steps,
            max_feature_steps=_MAX_FEATURE_STEPS,
            total_steps=total_steps,
        )
