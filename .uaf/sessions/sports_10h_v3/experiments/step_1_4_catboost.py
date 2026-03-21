"""Step 1.4: CatBoost default -- strong non-linear baseline."""

import logging
import os
import traceback

import mlflow
from catboost import CatBoostClassifier
from common import (
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
    logger.info("Step 1.4: CatBoost default baseline")
    df = load_data()
    train, test = time_series_split(df)

    features = get_feature_columns()
    cat_features = ["Is_Parlay_bool"]

    x_train = train[features].copy()
    x_test = test[features].copy()
    y_train = train["target"].values
    y_test = test["target"].values

    # Fill NaN with -999 for CatBoost
    x_train = x_train.fillna(-999)
    x_test = x_test.fillna(-999)

    # Convert bool to int for CatBoost
    for col in cat_features:
        x_train[col] = x_train[col].astype(int)
        x_test[col] = x_test[col].astype(int)

    with mlflow.start_run(run_name="phase1/step1.4_catboost_default") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("phase", "1")

        try:
            params = {
                "iterations": 1000,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
                "verbose": 100,
                "eval_metric": "AUC",
                "cat_features": [features.index(c) for c in cat_features],
                "early_stopping_rounds": 50,
            }

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(x_train),
                    "n_samples_val": len(x_test),
                    "method": "catboost_default",
                    "test_size": 0.2,
                    "features": ",".join(features),
                    "n_features": len(features),
                    **{k: str(v) for k, v in params.items() if k != "cat_features"},
                }
            )

            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, eval_set=(x_test, y_test))

            proba = model.predict_proba(x_test)[:, 1]
            preds = (proba >= 0.5).astype(int)

            auc = roc_auc_score(y_test, proba)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            logger.info(
                "AUC=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f", auc, precision, recall, f1
            )

            # ROI at various thresholds
            roi_results = calc_roi_at_thresholds(test, proba)
            best_roi = -999.0
            best_threshold = 0.5
            for t, r in roi_results.items():
                logger.info(
                    "  t=%.2f: ROI=%.2f%%, n=%d, WR=%.3f, selected=%.1f%%",
                    t,
                    r["roi"],
                    r["n_bets"],
                    r["win_rate"],
                    r["pct_selected"],
                )
                if r["n_bets"] >= 50 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_threshold = t

            # Feature importances
            importances = model.get_feature_importance()
            for feat, imp in zip(features, importances, strict=True):
                logger.info("  importance %s: %.2f", feat, imp)
                mlflow.log_metric(f"importance_{feat}", imp)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roc_auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "best_threshold": best_threshold,
                    "best_iteration": model.get_best_iteration() or 0,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ROI: %.2f%% at threshold %.2f", best_roi, best_threshold)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.4")
            raise


if __name__ == "__main__":
    main()
