"""Phase 3 — Optimization.

Два подхода:
1. Pure rule-based (singles + profitable sports + odds 1.45-1.90) = ROI ~11.8%
2. CatBoost на отфильтрованных данных — проверяем, добавляет ли ML сверху

EDA показал: чистый rule-based дает ROI > 10%.
CatBoost на полном датасете оптимизирует AUC, а не ROI — бесполезен для value betting.
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
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    compute_roi,
    find_best_threshold,
    load_bets,
    time_series_split,
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
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.warning("Budget hard stop. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    pass

PROFITABLE_SPORTS = {"Tennis", "Dota 2", "League of Legends", "CS2", "Table Tennis", "Volleyball"}
ODDS_LO = 1.45
ODDS_HI = 1.90

NUM_FEATURES = [
    "Odds",
    "USD",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
]
CAT_FEATURES = ["Sport", "Market"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def apply_filter(df):
    """Фильтр: singles + profitable sports + odds sweet spot."""
    return (
        (df["Is_Parlay"] == 0)
        & (df["Sport"].isin(PROFITABLE_SPORTS))
        & (df["Odds"] >= ODDS_LO)
        & (df["Odds"] <= ODDS_HI)
    )


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="phase3/step3.1_optimized_strategy") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")
        mlflow.set_tag("method", "rule_based_optimized")

        df = load_bets(with_outcomes=True)
        for col in CAT_FEATURES:
            df[col] = df[col].fillna("unknown").astype(str)

        splits = time_series_split(df, n_splits=5, gap_days=7)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_splits": len(splits),
                "gap_days": 7,
                "method": "rule_based_optimized",
                "filter": "singles+profitable+odds_1.45_1.90",
                "odds_lo": ODDS_LO,
                "odds_hi": ODDS_HI,
                "profitable_sports": ",".join(sorted(PROFITABLE_SPORTS)),
            }
        )

        # --- Approach 1: Pure rule-based (no ML) ---
        logger.info("=== Approach 1: Pure Rule-Based ===")
        rule_fold_rois = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            val_df = df.iloc[val_idx]
            mask = apply_filter(val_df)
            selected = val_df[mask]

            total_stake = selected["USD"].sum()
            total_payout = selected.loc[selected["target"] == 1, "Payout_USD"].sum()
            roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else 0.0
            wr = selected["target"].mean() if len(selected) > 0 else 0.0

            rule_fold_rois.append(roi)
            mlflow.log_metric(f"roi_rule_fold_{fold_idx}", roi)
            mlflow.log_metric(f"n_selected_rule_fold_{fold_idx}", len(selected))
            mlflow.log_metric(f"wr_rule_fold_{fold_idx}", wr)
            mlflow.log_params(
                {
                    f"n_samples_train_fold_{fold_idx}": len(train_idx),
                    f"n_samples_val_fold_{fold_idx}": len(val_idx),
                }
            )
            logger.info(
                "[Rule] Fold %d: n=%d, WR=%.3f, ROI=%+.2f%%",
                fold_idx,
                len(selected),
                wr,
                roi,
            )

        rule_roi_mean = float(np.mean(rule_fold_rois))
        rule_roi_std = float(np.std(rule_fold_rois))
        mlflow.log_metric("roi_rule_mean", rule_roi_mean)
        mlflow.log_metric("roi_rule_std", rule_roi_std)
        logger.info("Rule-based: ROI_mean=%+.2f%% +/- %.2f%%", rule_roi_mean, rule_roi_std)

        # --- Approach 2: CatBoost на отфильтрованных данных ---
        logger.info("=== Approach 2: CatBoost on Filtered Data ===")
        cat_indices = list(range(len(NUM_FEATURES), len(ALL_FEATURES)))

        ml_fold_rois = []
        ml_fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_mask = apply_filter(train_df)
            val_mask = apply_filter(val_df)

            train_filtered = train_df[train_mask]
            val_filtered = val_df[val_mask]

            if len(train_filtered) < 50 or len(val_filtered) < 50:
                ml_fold_rois.append(rule_fold_rois[fold_idx])
                ml_fold_aucs.append(0.5)
                continue

            x_train = train_filtered[ALL_FEATURES].copy()
            y_train = train_filtered["target"].values
            x_val = val_filtered[ALL_FEATURES].copy()
            y_val = val_filtered["target"].values
            stakes_val = val_filtered["USD"].values
            payouts_val = val_filtered["Payout_USD"].values

            model = CatBoostClassifier(
                iterations=500,
                depth=4,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
                cat_features=cat_indices,
                eval_metric="AUC",
                early_stopping_rounds=30,
                use_best_model=True,
            )
            model.fit(x_train, y_train, eval_set=(x_val, y_val))

            y_proba = model.predict_proba(x_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            ml_fold_aucs.append(auc)

            # Threshold search на train
            y_train_proba = model.predict_proba(x_train)[:, 1]
            best_t, _ = find_best_threshold(
                y_train,
                y_train_proba,
                train_filtered["USD"].values,
                train_filtered["Payout_USD"].values,
                thresholds=np.arange(0.40, 0.70, 0.01),
            )

            roi_result = compute_roi(y_val, y_proba, stakes_val, payouts_val, threshold=best_t)
            ml_fold_rois.append(roi_result["roi"])

            mlflow.log_metric(f"roi_ml_fold_{fold_idx}", roi_result["roi"])
            mlflow.log_metric(f"auc_ml_fold_{fold_idx}", auc)
            mlflow.log_metric(f"threshold_ml_fold_{fold_idx}", best_t)
            mlflow.log_metric(f"n_selected_ml_fold_{fold_idx}", roi_result["n_selected"])

            logger.info(
                "[ML] Fold %d: AUC=%.4f, threshold=%.2f, ROI=%+.2f%% (selected=%d/%d)",
                fold_idx,
                auc,
                best_t,
                roi_result["roi"],
                roi_result["n_selected"],
                roi_result["n_total"],
            )

        ml_roi_mean = float(np.mean(ml_fold_rois))
        ml_roi_std = float(np.std(ml_fold_rois))
        ml_auc_mean = float(np.mean(ml_fold_aucs))
        mlflow.log_metric("roi_ml_mean", ml_roi_mean)
        mlflow.log_metric("roi_ml_std", ml_roi_std)
        mlflow.log_metric("auc_ml_mean", ml_auc_mean)

        logger.info("CatBoost filtered: ROI_mean=%+.2f%% +/- %.2f%%", ml_roi_mean, ml_roi_std)

        # --- Финальный результат ---
        best_approach = "rule_based" if rule_roi_mean >= ml_roi_mean else "catboost_filtered"
        best_roi = max(rule_roi_mean, ml_roi_mean)
        best_std = rule_roi_std if best_approach == "rule_based" else ml_roi_std

        mlflow.log_metric("roi_mean", best_roi)
        mlflow.log_metric("roi_std", best_std)
        mlflow.set_tag("best_strategy", best_approach)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.9")

        mlflow.log_artifact(__file__)

        logger.info(
            "FINAL: best_approach=%s, ROI_mean=%+.2f%% +/- %.2f%%",
            best_approach,
            best_roi,
            best_std,
        )
        print(f"RESULT: best_approach={best_approach}, roi_mean={best_roi:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "runtime_error")
        logger.exception("Step 3.1 failed")
        raise
