"""Step 1.1 — Constant baseline (DummyClassifier).

Гипотеза: DummyClassifier (most_frequent) задает lower bound по ROI.
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
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    get_base_features,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    with mlflow.start_run(run_name="phase1/step_1_1_constant_baseline") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            features = get_base_features()

            train, test = time_series_split(df, test_size=0.2)
            logger.info("Train: %d, Test: %d", len(train), len(test))

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                    "method": "dummy_classifier",
                    "strategy": "most_frequent",
                }
            )

            x_train = train[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train["target"]
            y_test = test["target"]

            # DummyClassifier — most_frequent
            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy.fit(x_train, y_train)

            # Probas — будут одинаковые для всех
            probas_test = dummy.predict_proba(x_test)[:, 1]
            logger.info("Dummy probas unique: %s", np.unique(probas_test))

            # ROI если ставить на всё (порог = 0)
            roi_all = calc_roi(test, np.ones(len(test)), threshold=0.5)
            logger.info("ROI all bets (test): %s", roi_all)

            # ROI random baseline
            roi_random = calc_roi(test, np.random.rand(len(test)), threshold=0.5)
            logger.info("ROI random 50%%: %s", roi_random)

            # Dummy classifier — ставит на всё (most_frequent = won)
            roi_dummy = calc_roi(test, probas_test, threshold=0.5)
            logger.info("ROI dummy most_frequent: %s", roi_dummy)

            mlflow.log_metrics(
                {
                    "roi_all_bets": roi_all["roi"],
                    "roi_random_50": roi_random["roi"],
                    "roi_dummy": roi_dummy["roi"],
                    "n_bets_all": roi_all["n_bets"],
                    "n_bets_random": roi_random["n_bets"],
                    "n_bets_dummy": roi_dummy["n_bets"],
                    "win_rate_test": float(y_test.mean()),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("=== RESULTS ===")
            logger.info("ROI all bets: %.2f%% (n=%d)", roi_all["roi"], roi_all["n_bets"])
            logger.info("ROI random: %.2f%% (n=%d)", roi_random["roi"], roi_random["n_bets"])
            logger.info("ROI dummy: %.2f%% (n=%d)", roi_dummy["roi"], roi_dummy["n_bets"])

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 1.1 failed")
            raise


if __name__ == "__main__":
    main()
