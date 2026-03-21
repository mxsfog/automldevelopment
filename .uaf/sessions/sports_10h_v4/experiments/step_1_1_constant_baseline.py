"""Step 1.1: Constant baseline (DummyClassifier most_frequent)."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import calc_roi, check_budget, load_data, set_seed, time_series_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uaf/sports_10h_v4")
SESSION_ID = os.environ.get("UAF_SESSION_ID", "sports_10h_v4")


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step_1_1_constant_baseline") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.1")
            mlflow.set_tag("phase", "1")

            df = load_data()
            train, test = time_series_split(df)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "dummy_classifier_most_frequent",
                    "gap_days": 7,
                }
            )

            # DummyClassifier: всегда предсказывает most_frequent (won, target=1)
            # Это значит: ставим на все -> ROI как если бы ставили на все
            majority_class = train["target"].mode()[0]
            logger.info(
                "Majority class: %d (%.2f%%)", majority_class, train["target"].mean() * 100
            )

            # Strategy 1: predict all as majority (always bet)
            preds_all = np.ones(len(test))  # always predict won
            result_all = calc_roi(test, preds_all, threshold=0.5)
            logger.info(
                "Always bet - ROI: %.2f%%, bets: %d", result_all["roi"], result_all["n_bets"]
            )

            # Strategy 2: predict nothing (never bet)
            preds_none = np.zeros(len(test))
            result_none = calc_roi(test, preds_none, threshold=0.5)
            logger.info(
                "Never bet - ROI: %.2f%%, bets: %d", result_none["roi"], result_none["n_bets"]
            )

            # Strategy 3: random baseline (predict target_mean probability)
            target_mean = train["target"].mean()
            preds_random = np.full(len(test), target_mean)
            result_random_50 = calc_roi(test, preds_random, threshold=0.5)
            logger.info(
                "Random (p=%.3f, thr=0.5) - ROI: %.2f%%, bets: %d",
                target_mean,
                result_random_50["roi"],
                result_random_50["n_bets"],
            )

            # Log all metrics
            mlflow.log_metrics(
                {
                    "roi_always_bet": result_all["roi"],
                    "n_bets_always_bet": result_all["n_bets"],
                    "precision_always_bet": result_all["precision"],
                    "roi_never_bet": result_none["roi"],
                    "roi_random": result_random_50["roi"],
                    "target_mean_train": target_mean,
                    "target_mean_test": float(test["target"].mean()),
                    "roi": result_all["roi"],  # primary metric
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Primary ROI (always bet): %.2f%%", result_all["roi"])

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.1")
            logger.exception("Step 1.1 failed")
            raise


if __name__ == "__main__":
    main()
