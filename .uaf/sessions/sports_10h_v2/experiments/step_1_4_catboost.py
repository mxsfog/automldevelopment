"""Step 1.4: Non-linear baseline -- CatBoost с дефолтами."""

import logging
import os
import traceback

import mlflow
from catboost import CatBoostClassifier
from common import (
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
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
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "catboost_default",
                    "test_size": 0.2,
                    "features": ",".join(feature_cols),
                    "n_features": len(feature_cols),
                    "iterations": 1000,
                    "learning_rate": "auto",
                    "depth": 6,
                }
            )

            x_train = train[feature_cols].copy().values.astype(float)
            y_train = train["target"].values
            x_test = test[feature_cols].copy().values.astype(float)
            y_test = test["target"].values

            # NaN handling is built into CatBoost
            model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                random_seed=42,
                verbose=100,
                eval_metric="AUC",
                auto_class_weights="Balanced",
            )

            model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=100)

            proba = model.predict_proba(x_test)[:, 1]

            # Classification metrics
            auc = roc_auc_score(y_test, proba)
            preds_binary = (proba >= 0.5).astype(int)
            f1 = f1_score(y_test, preds_binary)
            precision = precision_score(y_test, preds_binary)
            recall = recall_score(y_test, preds_binary)
            logger.info("AUC=%.4f, F1=%.4f, P=%.4f, R=%.4f", auc, f1, precision, recall)

            # ROI at various thresholds
            roi_results = calc_roi_at_thresholds(test, proba)
            best_roi = -999.0
            best_threshold = 0.5
            for thresh, result in roi_results.items():
                logger.info(
                    "Threshold %.2f: ROI=%.2f%%, n=%d, WR=%.4f, selected=%.1f%%",
                    thresh,
                    result["roi"],
                    result["n_bets"],
                    result["win_rate"],
                    result["pct_selected"],
                )
                if result["n_bets"] >= 50 and result["roi"] > best_roi:
                    best_roi = result["roi"]
                    best_threshold = thresh

            roi_default = calc_roi(test, proba, threshold=0.5)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_t050": roi_default["roi"],
                    "roc_auc": auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "best_threshold": best_threshold,
                    "n_bets_selected": roi_results[best_threshold]["n_bets"],
                    "pct_selected": roi_results[best_threshold]["pct_selected"],
                    "best_iteration": model.get_best_iteration(),
                }
            )

            # Feature importances
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
            logger.info("Best ROI: %.2f%% at threshold=%.2f", best_roi, best_threshold)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.4")
            raise


if __name__ == "__main__":
    main()
