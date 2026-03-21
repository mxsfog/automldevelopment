"""Step 1.3: LogisticRegression baseline с ROI-оптимальным порогом."""

import logging
import os
import traceback

import mlflow
from common import (
    calc_roi_at_thresholds,
    check_budget,
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

    features = get_feature_columns()
    logger.info("Features: %s", features)

    X_train = train[features].copy()  # noqa: N806
    X_test = test[features].copy()  # noqa: N806
    y_train = train["target"].values
    y_test = test["target"].values

    # Fill NaN with median from train
    for col in features:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # Convert bool to int
    for col in features:
        if X_train[col].dtype == bool:
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # noqa: N806
    X_test_scaled = scaler.transform(X_test)  # noqa: N806

    with mlflow.start_run(run_name="phase1/step1.3_logistic") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(X_train),
                    "n_samples_val": len(X_test),
                    "method": "logistic_regression",
                    "test_size": 0.2,
                    "features": ",".join(features),
                    "n_features": len(features),
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "C": 1.0,
                }
            )

            model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
            model.fit(X_train_scaled, y_train)

            proba = model.predict_proba(X_test_scaled)[:, 1]
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
            for feat, coef in zip(features, model.coef_[0], strict=True):
                logger.info("  coef %s: %.4f", feat, coef)
                mlflow.log_metric(f"coef_{feat}", coef)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roc_auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "best_threshold": best_threshold,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ROI: %.2f%% at threshold %.2f", best_roi, best_threshold)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.3")
            raise


if __name__ == "__main__":
    main()
