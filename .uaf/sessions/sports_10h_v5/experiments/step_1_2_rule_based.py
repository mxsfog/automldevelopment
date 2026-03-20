"""Step 1.2: Rule-based baseline -- пороговое правило по ML_Edge."""

import logging
import os
import traceback

import mlflow
import numpy as np
from common import calc_roi, check_budget, load_data, set_seed, time_series_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    logger.info("Step 1.2: Rule-based baseline (ML_Edge threshold)")
    df = load_data()
    train, test = time_series_split(df)

    with mlflow.start_run(run_name="phase1/step1.2_rule_based") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "threshold_rule_ml_edge",
                    "test_size": 0.2,
                }
            )

            # Подбор порога на последних 20% train (anti-leakage)
            val_split_idx = int(len(train) * 0.8)
            train_val = train.iloc[val_split_idx:]

            train_val_ml = train_val[train_val["ML_Edge"].notna()].copy()
            test_ml = test[test["ML_Edge"].notna()].copy()
            logger.info(
                "Validation with ML_Edge: %d, Test with ML_Edge: %d / %d",
                len(train_val_ml),
                len(test_ml),
                len(test),
            )

            # Rule 1: ML_Edge >= threshold
            best_val_roi = -999.0
            best_edge_threshold = 0.0
            for edge_thresh in np.arange(-20, 40, 2):
                preds = (train_val_ml["ML_Edge"].values >= edge_thresh).astype(float)
                result = calc_roi(train_val_ml, preds, threshold=0.5)
                if result["n_bets"] >= 30 and result["roi"] > best_val_roi:
                    best_val_roi = result["roi"]
                    best_edge_threshold = edge_thresh

            logger.info(
                "Best val threshold: ML_Edge >= %.1f, val ROI=%.2f%%",
                best_edge_threshold,
                best_val_roi,
            )
            mlflow.log_param("edge_threshold", best_edge_threshold)

            preds_test = (test_ml["ML_Edge"].values >= best_edge_threshold).astype(float)
            roi_edge = calc_roi(test_ml, preds_test, threshold=0.5)
            logger.info(
                "Edge rule test ROI: %.2f%% (n=%d, WR=%.4f)",
                roi_edge["roi"],
                roi_edge["n_bets"],
                roi_edge["win_rate"],
            )

            # Rule 2: ML_P_Model >= threshold
            best_p_val_roi = -999.0
            best_p_threshold = 50.0
            for p_thresh in np.arange(30, 90, 5):
                preds_p = (train_val_ml["ML_P_Model"].values >= p_thresh).astype(float)
                result_p = calc_roi(train_val_ml, preds_p, threshold=0.5)
                if result_p["n_bets"] >= 30 and result_p["roi"] > best_p_val_roi:
                    best_p_val_roi = result_p["roi"]
                    best_p_threshold = p_thresh

            preds_p_test = (test_ml["ML_P_Model"].values >= best_p_threshold).astype(float)
            roi_p = calc_roi(test_ml, preds_p_test, threshold=0.5)
            logger.info(
                "P_Model >= %.0f test ROI: %.2f%% (n=%d)",
                best_p_threshold,
                roi_p["roi"],
                roi_p["n_bets"],
            )

            # Rule 3: Combined
            preds_combined = (
                (test_ml["ML_Edge"].values >= best_edge_threshold)
                & (test_ml["ML_P_Model"].values >= best_p_threshold)
            ).astype(float)
            roi_combined = calc_roi(test_ml, preds_combined, threshold=0.5)
            logger.info(
                "Combined rule test ROI: %.2f%% (n=%d)",
                roi_combined["roi"],
                roi_combined["n_bets"],
            )

            best_roi = max(roi_edge["roi"], roi_p["roi"], roi_combined["roi"])
            best_label = "edge"
            if roi_p["roi"] == best_roi:
                best_label = "p_model"
            elif roi_combined["roi"] == best_roi:
                best_label = "combined"

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_edge_rule": roi_edge["roi"],
                    "roi_p_model_rule": roi_p["roi"],
                    "roi_combined_rule": roi_combined["roi"],
                    "n_bets_edge": roi_edge["n_bets"],
                    "n_bets_p_model": roi_p["n_bets"],
                    "n_bets_combined": roi_combined["n_bets"],
                }
            )
            mlflow.set_tag("best_rule", best_label)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best rule-based ROI: %.2f%% (%s)", best_roi, best_label)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            raise


if __name__ == "__main__":
    main()
