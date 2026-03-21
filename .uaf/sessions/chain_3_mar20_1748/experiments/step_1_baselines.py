"""Phase 1: All baselines (1.1-1.4) in one script."""

import logging
import os
import traceback

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
    """Step 1.1: DummyClassifier (most_frequent) -- lower bound."""
    logger.info("Step 1.1: DummyClassifier")

    with mlflow.start_run(run_name="phase1/step1.1_dummy") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("phase", "1")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "dummy_most_frequent",
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                }
            )

            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            feature_cols = get_base_features()
            imp = SimpleImputer(strategy="median")
            x_train = imp.fit_transform(train[feature_cols])
            x_test = imp.transform(test[feature_cols])

            dummy.fit(x_train, train["target"])
            dummy.predict_proba(x_test)[:, 1]

            roi_all = calc_roi(test, np.ones(len(test)), threshold=0.5)
            logger.info("All bets ROI: %.2f%% n=%d", roi_all["roi"], roi_all["n_bets"])

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

            logger.info("Step 1.1 run_id: %s", run.info.run_id)
            return {"run_id": run.info.run_id, "roi": roi_all["roi"], "n_bets": roi_all["n_bets"]}

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.1")
            raise


def run_step_1_2(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.2: Rule-based (ML_Edge threshold)."""
    logger.info("Step 1.2: Rule-based ML_Edge threshold")
    check_budget()

    with mlflow.start_run(run_name="phase1/step1.2_rule") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "rule_ml_edge",
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                }
            )

            val_split = int(len(train) * 0.8)
            val_df = train.iloc[val_split:]

            edge_val = val_df["ML_Edge"].fillna(0).values
            edge_test = test["ML_Edge"].fillna(0).values

            edge_max = max(edge_val.max(), edge_test.max(), 1.0)
            proba_val = np.clip(edge_val / edge_max, 0, 1)
            proba_test = np.clip(edge_test / edge_max, 0, 1)

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=50)
            roi_result = calc_roi(test, proba_test, threshold=best_t)

            for edge_t in [0, 2, 5, 8, 10, 15, 20]:
                rule_proba = (edge_test >= edge_t).astype(float)
                r = calc_roi(test, rule_proba, threshold=0.5)
                logger.info("  Edge>=%d: ROI=%.2f%% n=%d", edge_t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_edge_{edge_t}", r["roi"])

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
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info(
                "Step 1.2: ROI=%.2f%% t=%.2f n=%d run_id=%s",
                roi_result["roi"],
                best_t,
                roi_result["n_bets"],
                run.info.run_id,
            )
            return {
                "run_id": run.info.run_id,
                "roi": roi_result["roi"],
                "n_bets": roi_result["n_bets"],
            }

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            raise


def run_step_1_3(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.3: LogisticRegression с engineered features."""
    logger.info("Step 1.3: LogisticRegression")
    check_budget()

    with mlflow.start_run(run_name="phase1/step1.3_logreg") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")

        try:
            feature_cols = get_base_features() + get_engineered_features()
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val_df = train.iloc[val_split:]

            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            x_fit = scaler.fit_transform(imp.fit_transform(train_fit[feature_cols]))
            x_val = scaler.transform(imp.transform(val_df[feature_cols]))
            x_test = scaler.transform(imp.transform(test[feature_cols]))

            model = LogisticRegression(
                C=0.01, penalty="l1", solver="saga", random_state=42, max_iter=2000
            )
            model.fit(x_fit, train_fit["target"])

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(test["target"], proba_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "logistic_regression",
                    "C": 0.01,
                    "penalty": "l1",
                    "n_features": len(feature_cols),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                }
            )

            roi_thresholds = calc_roi_at_thresholds(test, proba_test)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

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
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info(
                "Step 1.3: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run_id=%s",
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

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.3")
            raise


def run_step_1_4(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Step 1.4: CatBoost с engineered features."""
    logger.info("Step 1.4: CatBoost default")
    check_budget()

    with mlflow.start_run(run_name="phase1/step1.4_catboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("phase", "1")

        try:
            feature_cols = get_base_features() + get_engineered_features()
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val_df = train.iloc[val_split:]

            imp = SimpleImputer(strategy="median")
            x_fit = imp.fit_transform(train_fit[feature_cols])
            x_val = imp.transform(val_df[feature_cols])
            x_test = imp.transform(test[feature_cols])

            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=50,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(test["target"], proba_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "catboost_default",
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "n_features": len(feature_cols),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "best_iteration": model.best_iteration_,
                }
            )

            roi_thresholds = calc_roi_at_thresholds(test, proba_test)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            fi = dict(zip(feature_cols, model.feature_importances_, strict=False))
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted:
                logger.info("  FI: %s = %.3f", fname, fval)
                mlflow.log_metric(f"fi_{fname}", fval)

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
                "Step 1.4: ROI=%.2f%% AUC=%.4f t=%.2f n=%d best_iter=%d run_id=%s",
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                model.best_iteration_,
                run.info.run_id,
            )
            return {
                "run_id": run.info.run_id,
                "roi": roi_result["roi"],
                "auc": auc,
                "threshold": best_t,
                "n_bets": roi_result["n_bets"],
                "best_iteration": model.best_iteration_,
            }

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.4")
            raise


def main() -> None:
    """Запуск всех baseline шагов Phase 1."""
    logger.info("Phase 1: Loading data")
    df = load_data()
    df = add_engineered_features(df)
    logger.info("Total rows after filtering: %d", len(df))
    logger.info("Target distribution: %s", df["target"].value_counts().to_dict())
    logger.info("Date range: %s to %s", df["Created_At"].min(), df["Created_At"].max())

    train, test = time_series_split(df)

    results = {}
    results["1.1"] = run_step_1_1(train, test)
    results["1.2"] = run_step_1_2(train, test)
    results["1.3"] = run_step_1_3(train, test)
    results["1.4"] = run_step_1_4(train, test)

    logger.info("Phase 1 results:")
    for step, res in results.items():
        logger.info(
            "  Step %s: ROI=%.2f%% n=%d run_id=%s",
            step,
            res["roi"],
            res["n_bets"],
            res["run_id"],
        )


if __name__ == "__main__":
    main()
