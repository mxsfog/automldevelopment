"""Step 1.3: Linear baseline (LogisticRegression)."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

CAT_COLS = ["Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found"]
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


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Подготовка фичей для LogisticRegression."""
    feature_names = []

    # Numeric
    num_available = [c for c in NUM_COLS if c in train.columns]
    X_train_num = train[num_available].fillna(0).values
    X_test_num = test[num_available].fillna(0).values
    feature_names.extend(num_available)

    # Scale
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    # Categorical -> label encode
    X_train_cat_list = []
    X_test_cat_list = []
    for col in CAT_COLS:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        train_vals = train[col].fillna("_missing_").astype(str)
        test_vals = test[col].fillna("_missing_").astype(str)
        all_vals = pd.concat([train_vals, test_vals])
        le.fit(all_vals)
        X_train_cat_list.append(le.transform(train_vals).reshape(-1, 1))
        X_test_cat_list.append(le.transform(test_vals).reshape(-1, 1))
        feature_names.append(col)

    parts_train = [X_train_num]
    parts_test = [X_test_num]
    if X_train_cat_list:
        parts_train.extend(X_train_cat_list)
        parts_test.extend(X_test_cat_list)

    return np.hstack(parts_train), np.hstack(parts_test), feature_names


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step_1_3_logistic") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.3")
            mlflow.set_tag("phase", "1")

            df = load_data()
            train, test = time_series_split(df)

            # Val split для подбора threshold (последние 20% train)
            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            X_train, X_test, feature_names = prepare_features(train_inner, test)
            _, X_val, _ = prepare_features(train_inner, val)

            y_train = train_inner["target"].values
            y_val = val["target"].values
            y_test = test["target"].values

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "logistic_regression",
                    "n_features": len(feature_names),
                    "features": str(feature_names),
                    "gap_days": 7,
                    "C": 1.0,
                    "max_iter": 1000,
                }
            )

            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
            model.fit(X_train, y_train)

            # Val predictions -> threshold
            val_probas = model.predict_proba(X_val)[:, 1]
            best_thr = find_best_threshold(val, val_probas)

            # Test evaluation
            test_probas = model.predict_proba(X_test)[:, 1]
            test_result = calc_roi(test, test_probas, threshold=best_thr)

            # Also default threshold
            test_result_50 = calc_roi(test, test_probas, threshold=0.5)

            logger.info(
                "Test (thr=%.2f): ROI=%.2f%%, bets=%d/%d, prec=%.3f",
                best_thr,
                test_result["roi"],
                test_result["n_bets"],
                len(test),
                test_result["precision"],
            )
            logger.info(
                "Test (thr=0.50): ROI=%.2f%%, bets=%d/%d, prec=%.3f",
                test_result_50["roi"],
                test_result_50["n_bets"],
                len(test),
                test_result_50["precision"],
            )

            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_test, test_probas)
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
                }
            )

            # Feature importance (coefficients)
            for i, name in enumerate(feature_names):
                if i < model.coef_.shape[1]:
                    mlflow.log_metric(f"coef_{name}", float(model.coef_[0][i]))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.3")
            logger.exception("Step 1.3 failed")
            raise


if __name__ == "__main__":
    main()
