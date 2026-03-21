"""Step 1.3: Linear baseline -- LogisticRegression."""

import logging
import os
import traceback

import mlflow
import numpy as np
from common import (
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
    logger.info("Step 1.3: LogisticRegression baseline")
    df = load_data()
    train, test = time_series_split(df)

    feature_cols = get_feature_columns()

    with mlflow.start_run(run_name="phase1/step1.3_logistic") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")

        try:
            val_split_idx = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split_idx]
            train_val = train.iloc[val_split_idx:]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "logistic_regression",
                    "test_size": 0.2,
                    "features": ",".join(feature_cols),
                    "n_features": len(feature_cols),
                    "C": 1.0,
                    "max_iter": 1000,
                }
            )

            x_fit = train_fit[feature_cols].values.astype(float)
            y_fit = train_fit["target"].values
            x_val = train_val[feature_cols].values.astype(float)
            x_test = test[feature_cols].values.astype(float)
            y_test = test["target"].values

            # Replace NaN with 0
            x_fit = np.nan_to_num(x_fit, nan=0.0)
            x_val = np.nan_to_num(x_val, nan=0.0)
            x_test = np.nan_to_num(x_test, nan=0.0)

            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x_fit)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(x_fit, y_fit)

            # Threshold on val
            proba_val = model.predict_proba(x_val)[:, 1]
            best_threshold, val_roi = find_best_threshold_on_val(train_val, proba_val)
            logger.info("Best threshold from val: %.2f, val ROI=%.2f%%", best_threshold, val_roi)

            # Apply to test
            proba_test = model.predict_proba(x_test)[:, 1]
            roi_result = calc_roi(test, proba_test, threshold=best_threshold)

            auc = roc_auc_score(y_test, proba_test)
            preds_binary = (proba_test >= 0.5).astype(int)
            f1 = f1_score(y_test, preds_binary)
            precision = precision_score(y_test, preds_binary)
            recall = recall_score(y_test, preds_binary)
            logger.info("AUC=%.4f, F1=%.4f, P=%.4f, R=%.4f", auc, f1, precision, recall)

            roi_default = calc_roi(test, proba_test, threshold=0.5)

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roi_t050": roi_default["roi"],
                    "roc_auc": auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "best_threshold": best_threshold,
                    "val_roi_at_threshold": val_roi,
                    "n_bets_selected": roi_result["n_bets"],
                    "pct_selected": roi_result["pct_selected"],
                    "win_rate_selected": roi_result["win_rate"],
                }
            )

            # Feature coefficients
            for fname, coef in zip(feature_cols, model.coef_[0], strict=True):
                mlflow.log_metric(f"coef_{fname}", coef)
                logger.info("  %s: %.4f", fname, coef)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info(
                "ROI: %.2f%% at threshold=%.2f (n=%d, WR=%.4f)",
                roi_result["roi"],
                best_threshold,
                roi_result["n_bets"],
                roi_result["win_rate"],
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.3")
            raise


if __name__ == "__main__":
    main()
