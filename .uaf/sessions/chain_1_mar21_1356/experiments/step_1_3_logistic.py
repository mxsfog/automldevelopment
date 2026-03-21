"""Step 1.3 — Linear baseline (LogisticRegression).

Гипотеза: LogisticRegression с базовыми фичами — linear baseline.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    find_best_threshold,
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
    with mlflow.start_run(run_name="phase1/step_1_3_logistic") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            features = get_base_features()

            train, test = time_series_split(df, test_size=0.2)
            logger.info("Train: %d, Test: %d", len(train), len(test))

            # Val split из train (последние 20%)
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val = train.iloc[val_split:]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "logistic_regression",
                    "features": ",".join(features),
                    "n_features": len(features),
                }
            )

            x_train = train_fit[features].fillna(0)
            x_val = val[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train_fit["target"]
            y_val = val["target"]
            y_test = test["target"]

            # Скалируем
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_val_s = scaler.transform(x_val)
            x_test_s = scaler.transform(x_test)

            # LogisticRegression
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            )
            model.fit(x_train_s, y_train)

            # Predict
            probas_val = model.predict_proba(x_val_s)[:, 1]
            probas_test = model.predict_proba(x_test_s)[:, 1]

            # AUC
            auc_val = roc_auc_score(y_val, probas_val)
            auc_test = roc_auc_score(y_test, probas_test)
            logger.info("AUC val=%.4f, test=%.4f", auc_val, auc_test)

            # Подбор порога на val (anti-leakage)
            best_threshold, val_result = find_best_threshold(val, probas_val, min_bets=50)
            logger.info(
                "Best threshold=%.2f, val ROI=%.2f%% (n=%d)",
                best_threshold,
                val_result.get("roi", 0),
                val_result.get("n_bets", 0),
            )

            # Применяем к test один раз
            test_result = calc_roi(test, probas_test, threshold=best_threshold)
            logger.info("Test result: %s", test_result)

            # Также ROI по нескольким порогам (для анализа)
            for thr in [0.5, 0.55, 0.6, 0.65, 0.7]:
                r = calc_roi(test, probas_test, threshold=thr)
                logger.info("Test thr=%.2f: ROI=%.2f%%, n=%d", thr, r["roi"], r["n_bets"])

            mlflow.log_metrics(
                {
                    "auc_val": auc_val,
                    "auc_test": auc_test,
                    "roi_val": val_result.get("roi", 0),
                    "roi_test": test_result["roi"],
                    "n_bets_test": test_result["n_bets"],
                    "threshold": best_threshold,
                    "win_rate_test": test_result.get("win_rate", 0),
                }
            )

            # Feature importance
            for i, feat in enumerate(features):
                mlflow.log_metric(f"coef_{feat}", float(model.coef_[0][i]))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 1.3 failed")
            raise


if __name__ == "__main__":
    main()
