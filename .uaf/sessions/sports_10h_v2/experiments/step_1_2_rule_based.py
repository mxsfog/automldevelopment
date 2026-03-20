"""Step 1.2: Rule-based baseline -- пороговое правило по ML_Edge."""

import logging
import os
import traceback

import mlflow
import numpy as np
from common import (
    calc_roi,
    check_budget,
    load_data,
    set_seed,
    time_series_split,
)

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

            # Используем ML_Edge как сигнал: bet when edge > threshold
            test_ml = test[test["ML_Edge"].notna()].copy()
            logger.info("Test with ML_Edge: %d / %d", len(test_ml), len(test))

            if len(test_ml) == 0:
                logger.warning("No ML_Edge data in test set")
                mlflow.log_metrics({"roi": 0.0})
                mlflow.set_tag("status", "success")
                mlflow.set_tag("convergence_signal", "0.0")
                return

            # Оптимизация порога на train
            train_ml = train[train["ML_Edge"].notna()].copy()
            best_train_roi = -999.0
            best_threshold = 0.0

            for edge_thresh in np.arange(-20, 40, 2):
                preds = (train_ml["ML_Edge"].values >= edge_thresh).astype(float)
                result = calc_roi(train_ml, preds, threshold=0.5)
                if result["n_bets"] >= 50 and result["roi"] > best_train_roi:
                    best_train_roi = result["roi"]
                    best_threshold = edge_thresh

            logger.info(
                "Best train threshold: ML_Edge >= %.1f, train ROI=%.2f%%",
                best_threshold,
                best_train_roi,
            )
            mlflow.log_param("edge_threshold", best_threshold)

            # Применяем на test
            preds_test = (test_ml["ML_Edge"].values >= best_threshold).astype(float)
            roi_result = calc_roi(test_ml, preds_test, threshold=0.5)

            logger.info(
                "Test ROI: %.2f%% (n=%d, WR=%.4f, selected=%.1f%%)",
                roi_result["roi"],
                roi_result["n_bets"],
                roi_result["win_rate"],
                roi_result["pct_selected"],
            )

            # Также проверяем ML_P_Model > threshold
            best_p_roi = -999.0
            best_p_threshold = 0.5
            for p_thresh in np.arange(0.3, 0.85, 0.05):
                preds_p = (train_ml["ML_P_Model"].values >= p_thresh * 100).astype(float)
                result_p = calc_roi(train_ml, preds_p, threshold=0.5)
                if result_p["n_bets"] >= 50 and result_p["roi"] > best_p_roi:
                    best_p_roi = result_p["roi"]
                    best_p_threshold = p_thresh

            preds_p_test = (test_ml["ML_P_Model"].values >= best_p_threshold * 100).astype(float)
            roi_p_result = calc_roi(test_ml, preds_p_test, threshold=0.5)
            logger.info(
                "ML_P_Model >= %.0f%% test ROI: %.2f%% (n=%d)",
                best_p_threshold * 100,
                roi_p_result["roi"],
                roi_p_result["n_bets"],
            )

            # Combined rule: ML_Edge >= threshold AND ML_P_Model >= p_threshold
            preds_combined = (
                (test_ml["ML_Edge"].values >= best_threshold)
                & (test_ml["ML_P_Model"].values >= best_p_threshold * 100)
            ).astype(float)
            roi_combined = calc_roi(test_ml, preds_combined, threshold=0.5)
            logger.info(
                "Combined rule test ROI: %.2f%% (n=%d)",
                roi_combined["roi"],
                roi_combined["n_bets"],
            )

            # Выбираем лучший результат
            best_roi = max(roi_result["roi"], roi_p_result["roi"], roi_combined["roi"])
            best_label = "edge"
            if roi_p_result["roi"] == best_roi:
                best_label = "p_model"
            elif roi_combined["roi"] == best_roi:
                best_label = "combined"

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_edge_rule": roi_result["roi"],
                    "roi_p_model_rule": roi_p_result["roi"],
                    "roi_combined_rule": roi_combined["roi"],
                    "n_bets_edge": roi_result["n_bets"],
                    "n_bets_p_model": roi_p_result["n_bets"],
                    "n_bets_combined": roi_combined["n_bets"],
                    "best_rule": 0,  # logged as tag
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
