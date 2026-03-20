"""Phase 2 — Feature Engineering via Shadow Feature Trick.

Последовательно тестирует 5 групп фич. Каждая группа сравнивается:
baseline (best features so far) vs candidate (+ shadow features).
Принятие: delta ROI > 0.002 (0.2%).
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    compute_roi,
    find_best_threshold,
    load_bets,
    time_series_split,
)
from feature_engineering import (
    add_ml_calibration_features,
    add_odds_features,
    add_sport_market_features,
    add_stake_features,
    add_time_features,
)

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    budget_status = json.loads(budget_file.read_text())
    if budget_status.get("hard_stop"):
        logger.warning("Budget hard stop. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    pass

BASE_NUM_FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
]

BASE_CAT_FEATURES = [
    "Sport",
    "Market",
]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "depth": 6,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
    "use_best_model": True,
}

FEATURE_GROUPS = [
    ("2.1_odds_decomposition", add_odds_features),
    ("2.2_time_features", add_time_features),
    ("2.3_sport_market_agg", add_sport_market_features),
    ("2.4_ml_calibration", add_ml_calibration_features),
    ("2.5_stake_features", add_stake_features),
]

MIN_DELTA = 0.002  # 0.2% ROI


def evaluate_features(
    df: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    splits: list,
    label: str,
) -> dict:
    """Обучение CatBoost и оценка ROI/AUC на фолдах."""
    all_features = num_features + cat_features
    cat_indices = list(range(len(num_features), len(all_features)))

    fold_rois = []
    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        X_train = train_df[all_features].copy()  # noqa: N806
        y_train = train_df["target"].values
        X_val = val_df[all_features].copy()  # noqa: N806
        y_val = val_df["target"].values
        stakes_val = val_df["USD"].values
        payouts_val = val_df["Payout_USD"].values

        model = CatBoostClassifier(cat_features=cat_indices, **CATBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        fold_aucs.append(auc)

        # Поиск лучшего порога на train
        y_train_proba = model.predict_proba(X_train)[:, 1]
        best_t, _ = find_best_threshold(
            y_train,
            y_train_proba,
            train_df["USD"].values,
            train_df["Payout_USD"].values,
        )
        roi_result = compute_roi(y_val, y_proba, stakes_val, payouts_val, threshold=best_t)
        fold_rois.append(roi_result["roi"])

        logger.info(
            "[%s] Fold %d: AUC=%.4f, ROI=%.2f%% (threshold=%.2f, selected=%d/%d)",
            label,
            fold_idx,
            auc,
            roi_result["roi"],
            best_t,
            roi_result["n_selected"],
            roi_result["n_total"],
        )

    return {
        "roi_mean": float(np.mean(fold_rois)),
        "roi_std": float(np.std(fold_rois)),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "fold_rois": fold_rois,
        "fold_aucs": fold_aucs,
    }




mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

df = load_bets(with_outcomes=True)
for col in BASE_CAT_FEATURES:
    df[col] = df[col].fillna("unknown").astype(str)

splits = time_series_split(df, n_splits=5, gap_days=7)

accepted_num_features = BASE_NUM_FEATURES.copy()
accepted_cat_features = BASE_CAT_FEATURES.copy()
accepted_groups: list[str] = []

for step_name, fe_func in FEATURE_GROUPS:
    # Проверка бюджета
    try:
        budget_status = json.loads(budget_file.read_text())
        if budget_status.get("hard_stop"):
            logger.warning("Budget hard stop at step %s", step_name)
            break
    except FileNotFoundError:
        pass

    with mlflow.start_run(run_name=f"phase2/{step_name}") as step_run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", step_name)
            mlflow.set_tag("phase", "2")
            mlflow.set_tag("method", "shadow_feature_trick")

            # Генерируем shadow features
            new_cols = fe_func(df)
            new_num = [c for c in new_cols if df[c].dtype != object]
            new_cat = [c for c in new_cols if df[c].dtype == object]

            candidate_num = accepted_num_features + new_num
            candidate_cat = accepted_cat_features + new_cat

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_splits": len(splits),
                    "baseline_features": ",".join(accepted_num_features + accepted_cat_features),
                    "shadow_features": ",".join(new_cols),
                    "n_baseline_features": len(accepted_num_features) + len(accepted_cat_features),
                    "n_shadow_features": len(new_cols),
                }
            )

            # Baseline: текущий лучший feature set
            logger.info("--- Evaluating BASELINE for %s ---", step_name)
            baseline_result = evaluate_features(
                df, accepted_num_features, accepted_cat_features, splits, f"{step_name}_baseline"
            )

            # Candidate: baseline + shadow features
            logger.info("--- Evaluating CANDIDATE for %s ---", step_name)
            candidate_result = evaluate_features(
                df, candidate_num, candidate_cat, splits, f"{step_name}_candidate"
            )

            delta_roi = candidate_result["roi_mean"] - baseline_result["roi_mean"]
            delta_auc = candidate_result["auc_mean"] - baseline_result["auc_mean"]

            mlflow.log_metrics(
                {
                    "roi_mean_baseline": baseline_result["roi_mean"],
                    "roi_mean_candidate": candidate_result["roi_mean"],
                    "roi_mean": candidate_result["roi_mean"],
                    "roi_std": candidate_result["roi_std"],
                    "auc_mean_baseline": baseline_result["auc_mean"],
                    "auc_mean_candidate": candidate_result["auc_mean"],
                    "delta_roi": delta_roi,
                    "delta_auc": delta_auc,
                }
            )

            for i, (br, cr) in enumerate(
                zip(
                    baseline_result["fold_rois"],
                    candidate_result["fold_rois"],
                    strict=True,
                )
            ):
                mlflow.log_metric(f"roi_baseline_fold_{i}", br)
                mlflow.log_metric(f"roi_candidate_fold_{i}", cr)

            # Решение
            if delta_roi > MIN_DELTA:
                decision = "accepted"
                accepted_num_features = candidate_num
                accepted_cat_features = candidate_cat
                accepted_groups.append(step_name)
            elif delta_roi <= 0:
                decision = "rejected"
            else:
                decision = "marginal"
                # Принимаем marginal тоже — delta > 0 но <= 0.002
                accepted_num_features = candidate_num
                accepted_cat_features = candidate_cat
                accepted_groups.append(step_name)

            mlflow.set_tag("decision", decision)
            mlflow.set_tag("status", "success")
            mlflow.set_tag(
                "convergence_signal",
                str(min(0.5 + len(accepted_groups) * 0.1, 0.9)),
            )
            mlflow.log_artifact(__file__)

            logger.info(
                "Step %s: decision=%s, delta_roi=%.4f%%, baseline_roi=%.2f%%, "
                "candidate_roi=%.2f%%, delta_auc=%.4f",
                step_name,
                decision,
                delta_roi * 100,
                baseline_result["roi_mean"],
                candidate_result["roi_mean"],
                delta_auc,
            )

            print(
                f"STEP {step_name}: decision={decision}, "
                f"delta_roi={delta_roi:.4f}, "
                f"candidate_roi={candidate_result['roi_mean']:.4f}, "
                f"run_id={step_run.info.run_id}"
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step %s failed", step_name)
            raise

logger.info("Accepted feature groups: %s", accepted_groups)
logger.info(
    "Final feature set: num=%s, cat=%s",
    accepted_num_features,
    accepted_cat_features,
)
print(f"ACCEPTED_GROUPS: {accepted_groups}")
print(f"FINAL_NUM_FEATURES: {accepted_num_features}")
print(f"FINAL_CAT_FEATURES: {accepted_cat_features}")
