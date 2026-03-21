"""Phase 1: All baselines (1.1 Dummy, 1.2 Rule, 1.3 LogReg, 1.4 CatBoost)."""

import logging
import os

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def run_step_1_1(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.1: DummyClassifier baseline."""
    logger.info("Step 1.1: DummyClassifier (most_frequent)")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(train[["Odds"]], train["target"])
    proba = dummy.predict_proba(test[["Odds"]])[:, 1]

    _ = calc_roi(test, proba, threshold=0.5)
    roi_all_no_filter = calc_roi(test, np.ones(len(test)), threshold=0.5)

    with mlflow.start_run(run_name="phase1/step1.1_dummy") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("phase", "1")
        mlflow.log_params(
            {
                "method": "DummyClassifier",
                "strategy": "most_frequent",
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train),
                "n_samples_val": len(test),
            }
        )
        mlflow.log_metrics(
            {
                "roi": roi_all_no_filter["roi"],
                "n_bets": roi_all_no_filter["n_bets"],
                "win_rate": roi_all_no_filter["win_rate"],
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.1")
        logger.info(
            "Step 1.1: ROI=%.2f%% n=%d run=%s",
            roi_all_no_filter["roi"],
            roi_all_no_filter["n_bets"],
            run.info.run_id,
        )
        return {
            "run_id": run.info.run_id,
            "roi": roi_all_no_filter["roi"],
            "n_bets": roi_all_no_filter["n_bets"],
        }


def run_step_1_2(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.2: Rule-based (ML_Edge threshold)."""
    logger.info("Step 1.2: Rule-based ML_Edge threshold")

    val_split = int(len(train) * 0.8)
    val_df = train.iloc[val_split:]

    proba_proxy = (train["ML_Edge"] - train["ML_Edge"].min()) / (
        train["ML_Edge"].max() - train["ML_Edge"].min()
    )
    val_proxy = proba_proxy.iloc[val_split:]
    test_proxy = (test["ML_Edge"] - train["ML_Edge"].min()) / (
        train["ML_Edge"].max() - train["ML_Edge"].min()
    )

    best_t, _ = find_best_threshold_on_val(val_df, val_proxy.values, min_bets=30)
    roi_result = calc_roi(test, test_proxy.values, threshold=best_t)

    with mlflow.start_run(run_name="phase1/step1.2_rule") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")
        mlflow.log_params(
            {
                "method": "rule_ml_edge",
                "threshold": best_t,
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train),
                "n_samples_val": len(test),
            }
        )
        mlflow.log_metrics(
            {
                "roi": roi_result["roi"],
                "n_bets": roi_result["n_bets"],
                "win_rate": roi_result["win_rate"],
                "best_threshold": best_t,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.2")
        logger.info(
            "Step 1.2: ROI=%.2f%% t=%.2f n=%d run=%s",
            roi_result["roi"],
            best_t,
            roi_result["n_bets"],
            run.info.run_id,
        )
        return {
            "run_id": run.info.run_id,
            "roi": roi_result["roi"],
            "threshold": best_t,
            "n_bets": roi_result["n_bets"],
        }


def run_step_1_3(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.3: LogisticRegression baseline."""
    logger.info("Step 1.3: LogisticRegression")

    feat_list = get_base_features() + get_engineered_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_fit = scaler.fit_transform(imp.fit_transform(train_fit[feat_list]))
    x_val = scaler.transform(imp.transform(val_df[feat_list]))
    x_test = scaler.transform(imp.transform(test[feat_list]))

    lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr.fit(x_fit, train_fit["target"])

    proba_val = lr.predict_proba(x_val)[:, 1]
    proba_test = lr.predict_proba(x_test)[:, 1]

    best_t, _ = find_best_threshold_on_val(val_df, proba_val, min_bets=30)
    roi_result = calc_roi(test, proba_test, threshold=best_t)
    auc = roc_auc_score(test["target"], proba_test)

    with mlflow.start_run(run_name="phase1/step1.3_logreg") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")
        mlflow.log_params(
            {
                "method": "LogisticRegression",
                "n_features": len(feat_list),
                "threshold": best_t,
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_fit),
                "n_samples_val": len(test),
            }
        )
        mlflow.log_metrics(
            {
                "roi": roi_result["roi"],
                "roc_auc": auc,
                "n_bets": roi_result["n_bets"],
                "win_rate": roi_result["win_rate"],
                "best_threshold": best_t,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.3")
        logger.info(
            "Step 1.3: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
            roi_result["roi"],
            auc,
            best_t,
            roi_result["n_bets"],
            run.info.run_id,
        )
        return {
            "run_id": run.info.run_id,
            "roi": roi_result["roi"],
            "auc": auc,
            "threshold": best_t,
            "n_bets": roi_result["n_bets"],
        }


def run_step_1_4(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.4: CatBoost default baseline."""
    logger.info("Step 1.4: CatBoost default")

    feat_list = get_base_features() + get_engineered_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test[feat_list])

    cb = CatBoostClassifier(
        iterations=1000, random_seed=42, verbose=0, eval_metric="AUC", early_stopping_rounds=50
    )
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

    proba_val = cb.predict_proba(x_val)[:, 1]
    proba_test = cb.predict_proba(x_test)[:, 1]

    best_t, _ = find_best_threshold_on_val(val_df, proba_val, min_bets=30)
    roi_result = calc_roi(test, proba_test, threshold=best_t)
    auc = roc_auc_score(test["target"], proba_test)

    roi_thresholds = calc_roi_at_thresholds(test, proba_test)
    logger.info("ROI at thresholds:")
    for t, r in roi_thresholds.items():
        logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])

    with mlflow.start_run(run_name="phase1/step1.4_catboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("phase", "1")
        mlflow.log_params(
            {
                "method": "CatBoost_default",
                "n_features": len(feat_list),
                "threshold": best_t,
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_fit),
                "n_samples_val": len(test),
            }
        )
        mlflow.log_metrics(
            {
                "roi": roi_result["roi"],
                "roc_auc": auc,
                "n_bets": roi_result["n_bets"],
                "win_rate": roi_result["win_rate"],
                "best_threshold": best_t,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.4")
        logger.info(
            "Step 1.4: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
            roi_result["roi"],
            auc,
            best_t,
            roi_result["n_bets"],
            run.info.run_id,
        )
        return {
            "run_id": run.info.run_id,
            "roi": roi_result["roi"],
            "auc": auc,
            "threshold": best_t,
            "n_bets": roi_result["n_bets"],
        }


def main() -> None:
    """Запуск всех baseline-ов Phase 1."""

    df = load_data()
    df = add_engineered_features(df)
    train, test = time_series_split(df)

    logger.info("Data: %d total, train=%d, test=%d", len(df), len(train), len(test))

    results = {}
    results["1.1"] = run_step_1_1(train, test)
    check_budget()
    results["1.2"] = run_step_1_2(train, test)
    check_budget()
    results["1.3"] = run_step_1_3(train, test)
    check_budget()
    results["1.4"] = run_step_1_4(train, test)

    logger.info("Phase 1 complete:")
    for step, r in results.items():
        logger.info("  Step %s: ROI=%.2f%% run=%s", step, r["roi"], r["run_id"])


if __name__ == "__main__":
    main()
