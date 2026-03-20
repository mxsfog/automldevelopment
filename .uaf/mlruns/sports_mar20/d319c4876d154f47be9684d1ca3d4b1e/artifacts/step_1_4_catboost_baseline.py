"""Step 1.4 — Non-linear baseline (CatBoost default)."""

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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

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

NUM_FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
]

CAT_FEATURES = [
    "Sport",
    "Market",
]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="phase1/step1.4_catboost_default") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("method", "catboost_default")

        df = load_bets(with_outcomes=True)

        # Заполняем NaN в категориальных фичах
        for col in CAT_FEATURES:
            df[col] = df[col].fillna("unknown").astype(str)

        splits = time_series_split(df, n_splits=5, gap_days=7)

        cat_indices = [ALL_FEATURES.index(c) for c in CAT_FEATURES]

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_splits": len(splits),
                "gap_days": 7,
                "method": "catboost_default",
                "features": ",".join(ALL_FEATURES),
                "n_features": len(ALL_FEATURES),
                "cat_features": ",".join(CAT_FEATURES),
                "iterations": 1000,
                "learning_rate": "auto",
                "depth": 6,
            }
        )

        fold_rois = []
        fold_aucs = []
        fold_details = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[ALL_FEATURES].copy()
            y_train = train_df["target"].values
            X_val = val_df[ALL_FEATURES].copy()
            y_val = val_df["target"].values
            stakes_val = val_df["USD"].values
            payouts_val = val_df["Payout_USD"].values

            model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                random_seed=42,
                verbose=100,
                cat_features=cat_indices,
                eval_metric="AUC",
                early_stopping_rounds=50,
                use_best_model=True,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=100,
            )

            y_pred_proba = model.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_pred_proba)
            fold_aucs.append(auc)

            # Поиск лучшего порога на train
            y_train_proba = model.predict_proba(X_train)[:, 1]
            stakes_train = train_df["USD"].values
            payouts_train = train_df["Payout_USD"].values

            best_t, _ = find_best_threshold(
                y_train,
                y_train_proba,
                stakes_train,
                payouts_train,
            )

            roi_result = compute_roi(
                y_val, y_pred_proba, stakes_val, payouts_val, threshold=best_t
            )
            fold_rois.append(roi_result["roi"])

            # Дополнительные метрики
            y_pred_binary = (y_pred_proba >= best_t).astype(int)
            prec = precision_score(y_val, y_pred_binary, zero_division=0)
            rec = recall_score(y_val, y_pred_binary, zero_division=0)
            f1 = f1_score(y_val, y_pred_binary, zero_division=0)

            roi_05 = compute_roi(y_val, y_pred_proba, stakes_val, payouts_val, threshold=0.5)

            mlflow.log_metrics(
                {
                    f"roi_fold_{fold_idx}": roi_result["roi"],
                    f"roi_default_fold_{fold_idx}": roi_05["roi"],
                    f"auc_fold_{fold_idx}": auc,
                    f"threshold_fold_{fold_idx}": best_t,
                    f"n_selected_fold_{fold_idx}": roi_result["n_selected"],
                    f"precision_fold_{fold_idx}": prec,
                    f"recall_fold_{fold_idx}": rec,
                    f"f1_fold_{fold_idx}": f1,
                }
            )
            mlflow.log_params(
                {
                    f"n_samples_train_fold_{fold_idx}": len(train_idx),
                    f"n_samples_val_fold_{fold_idx}": len(val_idx),
                    f"best_iteration_fold_{fold_idx}": model.best_iteration_,
                }
            )

            logger.info(
                "Fold %d: AUC=%.4f, threshold=%.2f, ROI=%.2f%% (selected=%d/%d), "
                "P=%.3f, R=%.3f, F1=%.3f, best_iter=%d",
                fold_idx,
                auc,
                best_t,
                roi_result["roi"],
                roi_result["n_selected"],
                roi_result["n_total"],
                prec,
                rec,
                f1,
                model.best_iteration_,
            )
            fold_details.append(roi_result)

        mean_roi = float(np.mean(fold_rois))
        std_roi = float(np.std(fold_rois))
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        mlflow.log_metrics(
            {
                "roi_mean": mean_roi,
                "roi_std": std_roi,
                "roc_auc_mean": mean_auc,
                "roc_auc_std": std_auc,
            }
        )

        # Feature importance (последний фолд)
        importances = model.get_feature_importance()
        for feat, imp in zip(ALL_FEATURES, importances, strict=True):
            mlflow.log_metric(f"importance_{feat}", imp)
            logger.info("Feature %s: importance=%.2f", feat, imp)

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.3")

        logger.info(
            "CatBoost default: ROI_mean=%.2f%% +/- %.2f%%, AUC_mean=%.4f +/- %.4f",
            mean_roi,
            std_roi,
            mean_auc,
            std_auc,
        )
        print(f"RESULT: roi_mean={mean_roi:.4f}, roi_std={std_roi:.4f}, auc={mean_auc:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "runtime_error")
        logger.exception("Step 1.4 failed")
        raise
