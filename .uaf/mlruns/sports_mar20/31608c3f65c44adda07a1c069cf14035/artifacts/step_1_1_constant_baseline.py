"""Step 1.1 — Constant baseline (DummyClassifier). Нижняя граница ROI."""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import compute_roi, load_bets, time_series_split  # noqa: E402

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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="phase1/step1.1_constant_baseline") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("method", "dummy_classifier")

        df = load_bets(with_outcomes=True)
        splits = time_series_split(df, n_splits=5, gap_days=7)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_splits": len(splits),
                "gap_days": 7,
                "method": "dummy_classifier_most_frequent",
            }
        )

        fold_rois = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            y_train = df.loc[train_idx, "target"].values
            y_val = df.loc[val_idx, "target"].values
            stakes_val = df.loc[val_idx, "USD"].values
            payouts_val = df.loc[val_idx, "Payout_USD"].values

            # DummyClassifier (most_frequent): предсказывает majority class
            # majority = won (53.9%), значит предсказываем всем p=1 (ставим на все)
            majority_class = int(y_train.mean() >= 0.5)
            y_pred_proba = np.full(len(y_val), float(majority_class))

            roi_result = compute_roi(y_val, y_pred_proba, stakes_val, payouts_val, threshold=0.5)
            fold_rois.append(roi_result["roi"])

            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_result["roi"])
            mlflow.log_metric(f"n_selected_fold_{fold_idx}", roi_result["n_selected"])
            mlflow.log_metric(f"win_rate_fold_{fold_idx}", roi_result["win_rate_selected"])

            mlflow.log_params(
                {
                    f"n_samples_train_fold_{fold_idx}": len(train_idx),
                    f"n_samples_val_fold_{fold_idx}": len(val_idx),
                }
            )

            logger.info(
                "Fold %d: ROI=%.2f%%, selected=%d/%d, win_rate=%.4f",
                fold_idx,
                roi_result["roi"],
                roi_result["n_selected"],
                roi_result["n_total"],
                roi_result["win_rate_selected"],
            )

        mean_roi = float(np.mean(fold_rois))
        std_roi = float(np.std(fold_rois))

        mlflow.log_metric("roi_mean", mean_roi)
        mlflow.log_metric("roi_std", std_roi)
        mlflow.log_artifact(__file__)

        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.0")

        logger.info("Constant baseline: ROI_mean=%.2f%% +/- %.2f%%", mean_roi, std_roi)
        logger.info("Run ID: %s", run.info.run_id)

        print(f"RESULT: roi_mean={mean_roi:.4f}, roi_std={std_roi:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "runtime_error")
        logger.exception("Step 1.1 failed")
        raise
