"""Step 1.1-1.4: Все baselines одним скриптом.

1.1 DummyClassifier (most_frequent)
1.2 Rule-based (ML_Edge threshold)
1.3 LogisticRegression
1.4 CatBoost default
"""

import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
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


def run_step_1_1(train, test, feat_list: list[str]) -> dict:
    """Step 1.1: DummyClassifier baseline."""
    check_budget()
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(train[feat_list], train["target"])

    roi_all = calc_roi(test, np.ones(len(test)), threshold=0.5)

    with mlflow.start_run(run_name="phase1/step1.1_dummy") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("status", "running")
        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "DummyClassifier",
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": roi_all["roi"],
                    "n_bets": roi_all["n_bets"],
                    "win_rate": roi_all["win_rate"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            logger.info(
                "Step 1.1: Dummy ROI=%.2f%% n=%d run=%s",
                roi_all["roi"],
                roi_all["n_bets"],
                run.info.run_id,
            )
            return {"roi": roi_all["roi"], "n_bets": roi_all["n_bets"], "run_id": run.info.run_id}
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


def run_step_1_2(train, test) -> dict:
    """Step 1.2: Rule-based baseline (ML_Edge threshold)."""
    check_budget()
    val_split = int(len(train) * 0.8)
    val_df = train.iloc[val_split:]

    edge = val_df["ML_Edge"].values
    best_roi = -999.0
    best_t = 0.0
    for t in np.arange(0.0, 30.0, 0.5):
        mask = edge >= t
        if mask.sum() < 30:
            continue
        roi = calc_roi(val_df, mask.astype(float), threshold=0.5)
        if roi["roi"] > best_roi:
            best_roi = roi["roi"]
            best_t = t

    test_mask = (test["ML_Edge"].values >= best_t).astype(float)
    test_roi = calc_roi(test, test_mask, threshold=0.5)

    with mlflow.start_run(run_name="phase1/step1.2_rule") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("status", "running")
        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "rule_ML_Edge",
                    "threshold": best_t,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": test_roi["roi"],
                    "n_bets": test_roi["n_bets"],
                    "win_rate": test_roi["win_rate"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")
            logger.info(
                "Step 1.2: Rule(ML_Edge>=%.1f) ROI=%.2f%% n=%d run=%s",
                best_t,
                test_roi["roi"],
                test_roi["n_bets"],
                run.info.run_id,
            )
            return {
                "roi": test_roi["roi"],
                "n_bets": test_roi["n_bets"],
                "threshold": best_t,
                "run_id": run.info.run_id,
            }
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


def run_step_1_3(train, test, feat_list: list[str]) -> dict:
    """Step 1.3: LogisticRegression baseline."""
    check_budget()
    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(imp.fit_transform(train_fit[feat_list]))
    x_val = scaler.transform(imp.transform(val_df[feat_list]))
    x_test = scaler.transform(imp.transform(test[feat_list]))

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(x_fit, train_fit["target"])

    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    best_t, _ = find_best_threshold_on_val(val_df, p_val, min_bets=30)
    auc = roc_auc_score(test["target"], p_test)
    test_roi = calc_roi(test, p_test, threshold=best_t)

    with mlflow.start_run(run_name="phase1/step1.3_logreg") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("status", "running")
        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "LogisticRegression",
                    "threshold": best_t,
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(test),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": test_roi["roi"],
                    "roc_auc": auc,
                    "n_bets": test_roi["n_bets"],
                    "win_rate": test_roi["win_rate"],
                    "best_threshold": best_t,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")
            logger.info(
                "Step 1.3: LogReg ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                test_roi["roi"],
                auc,
                best_t,
                test_roi["n_bets"],
                run.info.run_id,
            )
            return {
                "roi": test_roi["roi"],
                "auc": auc,
                "threshold": best_t,
                "n_bets": test_roi["n_bets"],
                "run_id": run.info.run_id,
            }
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


def run_step_1_4(train, test, feat_list: list[str]) -> dict:
    """Step 1.4: CatBoost default."""
    check_budget()
    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test[feat_list])

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=50,
    )
    model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    best_t, _ = find_best_threshold_on_val(val_df, p_val, min_bets=30)
    auc = roc_auc_score(test["target"], p_test)
    test_roi = calc_roi(test, p_test, threshold=best_t)

    with mlflow.start_run(run_name="phase1/step1.4_catboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("status", "running")
        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "CatBoost_default",
                    "threshold": best_t,
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(test),
                    "best_iteration": model.get_best_iteration(),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": test_roi["roi"],
                    "roc_auc": auc,
                    "n_bets": test_roi["n_bets"],
                    "win_rate": test_roi["win_rate"],
                    "best_threshold": best_t,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")
            logger.info(
                "Step 1.4: CatBoost ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                test_roi["roi"],
                auc,
                best_t,
                test_roi["n_bets"],
                run.info.run_id,
            )
            return {
                "roi": test_roi["roi"],
                "auc": auc,
                "threshold": best_t,
                "n_bets": test_roi["n_bets"],
                "run_id": run.info.run_id,
            }
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


def main() -> None:
    """Запуск всех baselines."""
    df = load_data()
    df = add_engineered_features(df)
    train, test = time_series_split(df)

    feat_base = get_base_features()
    feat_all = feat_base + get_engineered_features()

    logger.info("Data: total=%d, train=%d, test=%d", len(df), len(train), len(test))

    r11 = run_step_1_1(train, test, feat_base)
    r12 = run_step_1_2(train, test)
    r13 = run_step_1_3(train, test, feat_all)
    r14 = run_step_1_4(train, test, feat_all)

    logger.info("Phase 1 Summary:")
    logger.info("  1.1 Dummy:    ROI=%.2f%% n=%d", r11["roi"], r11["n_bets"])
    logger.info("  1.2 Rule:     ROI=%.2f%% n=%d", r12["roi"], r12["n_bets"])
    logger.info("  1.3 LogReg:   ROI=%.2f%% n=%d", r13["roi"], r13["n_bets"])
    logger.info("  1.4 CatBoost: ROI=%.2f%% n=%d", r14["roi"], r14["n_bets"])


if __name__ == "__main__":
    main()
