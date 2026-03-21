"""Step 1.4: Non-linear baseline (CatBoost default)."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    calc_roi,
    check_budget,
    find_best_threshold,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uaf/sports_10h_v4")
SESSION_ID = os.environ.get("UAF_SESSION_ID", "sports_10h_v4")

NUM_COLS = [
    "Odds",
    "USD",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Odds_outcome",
    "n_outcomes",
]
CAT_COLS = ["Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found"]


def prepare_catboost_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    """Подготовка данных для CatBoost."""
    features = []
    cat_indices = []

    for c in NUM_COLS:
        if c in train.columns:
            features.append(c)

    for c in CAT_COLS:
        if c in train.columns:
            cat_indices.append(len(features))
            features.append(c)

    X_train = train[features].copy()
    X_test = test[features].copy()

    # CatBoost handles NaN for numeric, but cats need string
    for idx in cat_indices:
        col = features[idx]
        X_train[col] = X_train[col].fillna("_missing_").astype(str)
        X_test[col] = X_test[col].fillna("_missing_").astype(str)

    return X_train, X_test, features, cat_indices


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step_1_4_catboost") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.4")
            mlflow.set_tag("phase", "1")

            df = load_data()
            train, test = time_series_split(df)

            # Val split для threshold
            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            X_train, X_test, features, cat_indices = prepare_catboost_data(train_inner, test)
            X_val = val[features].copy()
            for idx in cat_indices:
                col = features[idx]
                X_val[col] = X_val[col].fillna("_missing_").astype(str)

            y_train = train_inner["target"].values
            y_val = val["target"].values
            y_test = test["target"].values

            params = {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3,
                "random_seed": 42,
                "verbose": 100,
                "eval_metric": "AUC",
                "cat_features": cat_indices,
                "early_stopping_rounds": 50,
                "task_type": "CPU",
            }

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "catboost_default",
                    "n_features": len(features),
                    "features": str(features),
                    "gap_days": 7,
                    **{k: v for k, v in params.items() if k != "cat_features"},
                }
            )

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

            # Val threshold
            val_probas = model.predict_proba(X_val)[:, 1]
            best_thr = find_best_threshold(val, val_probas)

            # Test
            test_probas = model.predict_proba(X_test)[:, 1]
            test_result = calc_roi(test, test_probas, threshold=best_thr)
            test_result_50 = calc_roi(test, test_probas, threshold=0.5)

            auc = roc_auc_score(y_test, test_probas)

            logger.info(
                "Test (thr=%.2f): ROI=%.2f%%, bets=%d/%d, prec=%.3f, sel=%.3f",
                best_thr,
                test_result["roi"],
                test_result["n_bets"],
                len(test),
                test_result["precision"],
                test_result["selectivity"],
            )
            logger.info(
                "Test (thr=0.50): ROI=%.2f%%, bets=%d/%d, prec=%.3f",
                test_result_50["roi"],
                test_result_50["n_bets"],
                len(test),
                test_result_50["precision"],
            )
            logger.info("AUC: %.4f", auc)

            mlflow.log_metrics(
                {
                    "roi": test_result["roi"],
                    "roi_thr_50": test_result_50["roi"],
                    "best_threshold": best_thr,
                    "n_bets": test_result["n_bets"],
                    "precision": test_result["precision"],
                    "selectivity": test_result["selectivity"],
                    "n_won": test_result["n_won"],
                    "n_lost": test_result["n_lost"],
                    "roc_auc": auc,
                    "best_iteration": model.best_iteration_,
                }
            )

            # Feature importance
            fi = model.get_feature_importance()
            for i, name in enumerate(features):
                mlflow.log_metric(f"fi_{name}", float(fi[i]))
            fi_sorted = sorted(zip(features, fi), key=lambda x: x[1], reverse=True)
            logger.info("Feature importance: %s", fi_sorted[:10])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.4")
            logger.exception("Step 1.4 failed")
            raise


if __name__ == "__main__":
    main()
