"""Step 1.1: Constant baseline (DummyClassifier) -- lower bound ROI."""

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
    logger.info("Step 1.1: Constant baseline")
    df = load_data()
    train, test = time_series_split(df)

    with mlflow.start_run(run_name="phase1/step1.1_constant_baseline") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("phase", "1")

        try:
            train_win_rate = train["target"].mean()
            test_win_rate = (test["Status"] == "won").mean()

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "dummy_classifier_most_frequent",
                    "test_size": 0.2,
                }
            )

            # Dummy: predict all as won (most_frequent = won, 53.8%)
            roi_bet_all = calc_roi(test, np.ones(len(test)), threshold=0.5)
            logger.info("ROI (bet on all): %.2f%%, WR=%.4f", roi_bet_all["roi"], test_win_rate)

            # Random baseline
            np.random.seed(42)
            roi_random = calc_roi(test, np.random.rand(len(test)), threshold=0.5)
            logger.info("ROI (random 50%%): %.2f%%", roi_random["roi"])

            # Primary metric: ROI when betting on all (dummy predicts all = won)
            primary_roi = roi_bet_all["roi"]

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "roi_bet_all": roi_bet_all["roi"],
                    "roi_random_50": roi_random["roi"],
                    "n_bets_selected": roi_bet_all["n_bets"],
                    "win_rate_test": test_win_rate,
                    "train_win_rate": train_win_rate,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("ROI constant baseline: %.2f%%", primary_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.1")
            raise


if __name__ == "__main__":
    main()
