"""Step 1.4: Non-linear baseline -- CatBoost с дефолтами."""

import logging
import os
import traceback

import mlflow
from catboost import CatBoostClassifier
from common import (
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

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
    logger.info("Step 1.4: CatBoost baseline")
    df = load_data()
    train, test = time_series_split(df)

    feature_cols = get_feature_columns()

    with mlflow.start_run(run_name="phase1/step1.4_catboost_default") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.4")
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
                    "method": "catboost_default",
                    "test_size": 0.2,
                    "features": ",".join(feature_cols),
                    "n_features": len(feature_cols),
                    "iterations": 1000,
                    "depth": 6,
                }
            )

            x_fit = train_fit[feature_cols].copy().values.astype(float)
            y_fit = train_fit["target"].values
            x_val = train_val[feature_cols].copy().values.astype(float)
            y_val = train_val["target"].values
            x_test = test[feature_cols].copy().values.astype(float)
            y_test = test["target"].values

            model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                random_seed=42,
                verbose=100,
                eval_metric="AUC",
                auto_class_weights="Balanced",
            )

            model.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=100)

            # Threshold on val (anti-leakage)
            proba_val = model.predict_proba(x_val)[:, 1]
            best_threshold, val_roi = find_best_threshold_on_val(train_val, proba_val)
            logger.info("Best threshold from val: %.2f, val ROI=%.2f%%", best_threshold, val_roi)

            # Apply to test once
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
                    "best_iteration": model.get_best_iteration(),
                }
            )

            importances = model.get_feature_importance()
            for fname, imp in zip(feature_cols, importances, strict=True):
                mlflow.log_metric(f"importance_{fname}", imp)
            logger.info("Feature importances:")
            ranked = sorted(zip(feature_cols, importances, strict=True), key=lambda x: -x[1])
            for fname, imp in ranked:
                logger.info("  %s: %.2f", fname, imp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")

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
            mlflow.set_tag("failure_reason", "exception in step 1.4")
            raise


if __name__ == "__main__":
    main()
