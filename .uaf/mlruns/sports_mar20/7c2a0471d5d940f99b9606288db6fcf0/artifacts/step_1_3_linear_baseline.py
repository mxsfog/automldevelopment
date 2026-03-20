"""Step 1.3 — Linear baseline (LogisticRegression)."""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

FEATURES = [
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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="phase1/step1.3_logistic_regression") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("method", "logistic_regression")

        df = load_bets(with_outcomes=True)
        splits = time_series_split(df, n_splits=5, gap_days=7)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_splits": len(splits),
                "gap_days": 7,
                "method": "logistic_regression",
                "features": ",".join(FEATURES),
                "n_features": len(FEATURES),
                "C": 1.0,
                "max_iter": 1000,
            }
        )

        fold_rois = []
        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[FEATURES].values
            y_train = train_df["target"].values
            X_val = val_df[FEATURES].values
            y_val = val_df["target"].values
            stakes_val = val_df["USD"].values
            payouts_val = val_df["Payout_USD"].values

            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
                ]
            )
            pipe.fit(X_train, y_train)

            y_pred_proba = pipe.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_pred_proba)
            fold_aucs.append(auc)

            # Поиск лучшего порога на train
            y_train_proba = pipe.predict_proba(X_train)[:, 1]
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

            # Также ROI при threshold=0.5 (default)
            roi_05 = compute_roi(y_val, y_pred_proba, stakes_val, payouts_val, threshold=0.5)

            mlflow.log_metrics(
                {
                    f"roi_fold_{fold_idx}": roi_result["roi"],
                    f"roi_default_fold_{fold_idx}": roi_05["roi"],
                    f"auc_fold_{fold_idx}": auc,
                    f"threshold_fold_{fold_idx}": best_t,
                    f"n_selected_fold_{fold_idx}": roi_result["n_selected"],
                }
            )
            mlflow.log_params(
                {
                    f"n_samples_train_fold_{fold_idx}": len(train_idx),
                    f"n_samples_val_fold_{fold_idx}": len(val_idx),
                }
            )

            logger.info(
                "Fold %d: AUC=%.4f, threshold=%.2f, ROI=%.2f%% (selected=%d/%d), ROI@0.5=%.2f%%",
                fold_idx,
                auc,
                best_t,
                roi_result["roi"],
                roi_result["n_selected"],
                roi_result["n_total"],
                roi_05["roi"],
            )

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

        # Логируем feature importance (coefficients)
        # Фильтруем фичи, где imputer не дропнул колонки
        pipe_full = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )
        X_full = df[FEATURES].values
        pipe_full.fit(X_full, df["target"].values)
        kept_mask = ~np.all(np.isnan(X_full), axis=0)
        kept_features = [f for f, m in zip(FEATURES, kept_mask, strict=True) if m]
        coefs = pipe_full.named_steps["clf"].coef_[0]
        for feat, coef in zip(kept_features, coefs, strict=True):
            mlflow.log_metric(f"coef_{feat}", coef)
            logger.info("Feature %s: coef=%.4f", feat, coef)

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.2")

        logger.info(
            "LogReg: ROI_mean=%.2f%% +/- %.2f%%, AUC_mean=%.4f +/- %.4f",
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
        logger.exception("Step 1.3 failed")
        raise
