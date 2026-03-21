"""Step 2.2: Selective Feature Engineering -- только odds-derived и value фичи."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
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
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Фичи без target encoding -- только трансформации существующих колонок."""
    df = df.copy()

    # Odds-derived
    df["log_odds"] = np.log1p(df["Odds"])
    df["implied_prob"] = 1.0 / df["Odds"]
    df["value_ratio"] = df["ML_P_Model"] / 100.0 / df["implied_prob"]
    df["value_ratio"] = df["value_ratio"].clip(0, 10).fillna(1.0)

    # Edge / EV interactions
    df["edge_x_ev"] = df["ML_Edge"] * df["ML_EV"]
    df["edge_abs"] = df["ML_Edge"].abs()
    df["ev_positive"] = (df["ML_EV"] > 0).astype(float)

    # Model vs implied divergence
    df["model_implied_diff"] = df["ML_P_Model"] - df["ML_P_Implied"]

    # Bet sizing
    df["log_usd"] = np.log1p(df["USD"])
    df["usd_per_outcome"] = df["USD"] / df["Outcomes_Count"].clip(lower=1)
    df["log_usd_per_outcome"] = np.log1p(df["usd_per_outcome"])

    # Parlay complexity
    df["is_parlay_float"] = (df["Is_Parlay"] == "t").astype(float)
    df["parlay_complexity"] = df["Outcomes_Count"] * df["is_parlay_float"]

    return df


def main() -> None:
    logger.info("Step 2.2: Selective features (no target encoding)")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    with mlflow.start_run(run_name="phase2/step2.2_selective_features") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "2.2")
        mlflow.set_tag("phase", "2")

        try:
            # Baseline
            baseline_cols = get_feature_columns()

            # Candidate
            train_fit_fe = add_safe_features(train_fit)
            train_val_fe = add_safe_features(train_val)
            test_fe = add_safe_features(test)

            candidate_cols = [
                *baseline_cols,
                "log_odds",
                "implied_prob",
                "value_ratio",
                "edge_x_ev",
                "edge_abs",
                "ev_positive",
                "model_implied_diff",
                "log_usd",
                "log_usd_per_outcome",
                "parlay_complexity",
            ]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "shadow_feature_trick",
                    "baseline_n_features": len(baseline_cols),
                    "candidate_n_features": len(candidate_cols),
                    "new_features": ",".join(
                        [f for f in candidate_cols if f not in baseline_cols]
                    ),
                }
            )

            # Train baseline
            x_fit_b = train_fit[baseline_cols].values.astype(float)
            y_fit = train_fit["target"].values
            x_val_b = train_val[baseline_cols].values.astype(float)
            y_val = train_val["target"].values
            x_test_b = test[baseline_cols].values.astype(float)
            y_test = test["target"].values

            model_b = CatBoostClassifier(
                iterations=1000,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                auto_class_weights="Balanced",
            )
            model_b.fit(x_fit_b, y_fit, eval_set=(x_val_b, y_val), early_stopping_rounds=100)

            proba_val_b = model_b.predict_proba(x_val_b)[:, 1]
            thr_b, _val_roi_b = find_best_threshold_on_val(train_val, proba_val_b)
            proba_test_b = model_b.predict_proba(x_test_b)[:, 1]
            roi_b = calc_roi(test, proba_test_b, threshold=thr_b)
            auc_b = roc_auc_score(y_test, proba_test_b)
            logger.info(
                "[baseline] ROI=%.2f%%, AUC=%.4f, thr=%.2f, n=%d",
                roi_b["roi"],
                auc_b,
                thr_b,
                roi_b["n_bets"],
            )

            # Train candidate
            x_fit_c = np.nan_to_num(train_fit_fe[candidate_cols].values.astype(float), nan=0.0)
            x_val_c = np.nan_to_num(train_val_fe[candidate_cols].values.astype(float), nan=0.0)
            x_test_c = np.nan_to_num(test_fe[candidate_cols].values.astype(float), nan=0.0)

            model_c = CatBoostClassifier(
                iterations=1000,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                auto_class_weights="Balanced",
            )
            model_c.fit(x_fit_c, y_fit, eval_set=(x_val_c, y_val), early_stopping_rounds=100)

            proba_val_c = model_c.predict_proba(x_val_c)[:, 1]
            thr_c, _val_roi_c = find_best_threshold_on_val(train_val_fe, proba_val_c)
            proba_test_c = model_c.predict_proba(x_test_c)[:, 1]
            roi_c = calc_roi(test_fe, proba_test_c, threshold=thr_c)
            auc_c = roc_auc_score(y_test, proba_test_c)

            logger.info(
                "[candidate] ROI=%.2f%%, AUC=%.4f, thr=%.2f, n=%d",
                roi_c["roi"],
                auc_c,
                thr_c,
                roi_c["n_bets"],
            )

            importances = model_c.get_feature_importance()
            ranked = sorted(zip(candidate_cols, importances, strict=True), key=lambda x: -x[1])
            for fname, imp in ranked:
                logger.info("  %s: %.2f", fname, imp)
                mlflow.log_metric(f"importance_{fname}", imp)

            delta_roi = roi_c["roi"] - roi_b["roi"]
            delta_auc = auc_c - auc_b

            if delta_roi > 0.002:
                decision = "accepted"
            elif delta_roi <= 0:
                decision = "rejected"
            else:
                decision = "marginal"

            logger.info(
                "Delta ROI=%.4f%%, Delta AUC=%.4f, Decision=%s",
                delta_roi,
                delta_auc,
                decision,
            )

            mlflow.log_metrics(
                {
                    "roi": roi_c["roi"],
                    "roi_baseline": roi_b["roi"],
                    "roi_candidate": roi_c["roi"],
                    "delta_roi": delta_roi,
                    "auc_baseline": auc_b,
                    "auc_candidate": auc_c,
                    "delta_auc": delta_auc,
                    "threshold_baseline": thr_b,
                    "threshold_candidate": thr_c,
                    "n_bets_baseline": roi_b["n_bets"],
                    "n_bets_candidate": roi_c["n_bets"],
                    "best_iteration_b": model_b.get_best_iteration(),
                    "best_iteration_c": model_c.get_best_iteration(),
                }
            )

            mlflow.set_tag("fe_decision", decision)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.2")
            raise


if __name__ == "__main__":
    main()
