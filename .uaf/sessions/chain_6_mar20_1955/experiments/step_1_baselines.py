"""Phase 1: Baselines (steps 1.1-1.4) для chain_6_mar20_1955."""

import logging
import os
import sys
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CB_BEST_PARAMS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_all_features,
    get_base_features,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def run_step_1_1(train, test, features):
    """Step 1.1: DummyClassifier baseline."""
    check_budget()
    with mlflow.start_run(run_name="phase1/step_1.1_dummy") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
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

        try:
            model = DummyClassifier(strategy="most_frequent", random_state=42)
            model.fit(train[features], train["target"])
            proba = model.predict_proba(test[features])[:, 1]
            roi_result = calc_roi(test, proba, threshold=0.5)

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            logger.info(
                "Step 1.1 DummyClassifier: ROI=%.2f%%, N=%d",
                roi_result["roi"],
                roi_result["n_bets"],
            )
            return run.info.run_id, roi_result
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "DummyClassifier failed")
            raise


def run_step_1_2(train, test, features):
    """Step 1.2: Rule-based baseline (ML_Edge threshold)."""
    check_budget()
    with mlflow.start_run(run_name="phase1/step_1.2_rule") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            val_split = int(len(train) * 0.8)
            val_df = train.iloc[val_split:]
            best_edge, best_roi_val = -999.0, -999.0
            for edge_thresh in np.arange(0.0, 30.0, 0.5):
                mask = val_df["ML_Edge"].values >= edge_thresh
                n_sel = int(mask.sum())
                if n_sel < 30:
                    continue
                sel = val_df.iloc[np.where(mask)[0]]
                total_staked = sel["USD"].sum()
                total_payout = sel["Payout_USD"].sum()
                roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0
                if roi > best_roi_val:
                    best_roi_val = roi
                    best_edge = edge_thresh

            proba_test = (test["ML_Edge"].values >= best_edge).astype(float)
            roi_result = calc_roi(test, proba_test, threshold=0.5)

            mlflow.log_params(
                {
                    "method": "threshold_rule",
                    "feature": "ML_Edge",
                    "threshold": best_edge,
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
                    "val_best_roi": best_roi_val,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")
            logger.info(
                "Step 1.2 Rule ML_Edge>=%.1f: ROI=%.2f%%, N=%d",
                best_edge,
                roi_result["roi"],
                roi_result["n_bets"],
            )
            return run.info.run_id, roi_result, best_edge
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "Rule baseline failed")
            raise


def run_step_1_3(train, test, features):
    """Step 1.3: LogisticRegression baseline."""
    check_budget()
    with mlflow.start_run(run_name="phase1/step_1.3_logreg") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            x_train = train[features].fillna(0).values
            x_test = test[features].fillna(0).values
            y_train = train["target"].values

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)

            model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            model.fit(x_train_s, y_train)
            proba = model.predict_proba(x_test_s)[:, 1]

            auc = roc_auc_score(test["target"].values, proba)

            val_split = int(len(train) * 0.8)
            val_df = train.iloc[val_split:]
            x_val_s = scaler.transform(val_df[features].fillna(0).values)
            proba_val = model.predict_proba(x_val_s)[:, 1]
            best_t, _ = find_best_threshold_on_val(val_df, proba_val, min_bets=30)

            roi_result = calc_roi(test, proba, threshold=best_t)
            ev_result = calc_ev_roi(test, proba, ev_threshold=0.0, min_prob=best_t)

            mlflow.log_params(
                {
                    "method": "LogisticRegression",
                    "C": 1.0,
                    "threshold": best_t,
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "n_features": len(features),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "auc": auc,
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                    "ev_roi": ev_result["roi"],
                    "ev_n_bets": ev_result["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")
            logger.info(
                "Step 1.3 LogReg: ROI=%.2f%%, AUC=%.4f, t=%.2f, N=%d",
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
            )
            return run.info.run_id, roi_result, auc, best_t
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "LogReg failed")
            raise


def run_step_1_4(train, test, features):
    """Step 1.4: CatBoost default baseline."""
    check_budget()
    with mlflow.start_run(run_name="phase1/step_1.4_catboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            x_train = train[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train["target"]

            val_split = int(len(train) * 0.8)
            x_tr = x_train.iloc[:val_split]
            y_tr = y_train.iloc[:val_split]
            x_val = x_train.iloc[val_split:]
            y_val = y_train.iloc[val_split:]
            val_df = train.iloc[val_split:]

            model = CatBoostClassifier(**CB_BEST_PARAMS)
            model.fit(x_tr, y_tr, eval_set=(x_val, y_val), use_best_model=True)

            proba_test = model.predict_proba(x_test)[:, 1]
            proba_val = model.predict_proba(x_val)[:, 1]
            auc = roc_auc_score(test["target"].values, proba_test)

            best_t, _ = find_best_threshold_on_val(val_df, proba_val, min_bets=30)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            ev_result = calc_ev_roi(test, proba_test, ev_threshold=0.0, min_prob=0.77)

            mlflow.log_params(
                {
                    "method": "CatBoost_default",
                    "threshold": best_t,
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "n_features": len(features),
                    "best_iteration": model.get_best_iteration(),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "auc": auc,
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                    "ev_roi": ev_result["roi"],
                    "ev_n_bets": ev_result["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")
            logger.info(
                "Step 1.4 CatBoost: ROI=%.2f%%, AUC=%.4f, t=%.2f, N=%d | EV ROI=%.2f%% N=%d",
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                ev_result["roi"],
                ev_result["n_bets"],
            )
            return run.info.run_id, roi_result, auc, best_t, ev_result
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "CatBoost failed")
            raise


def main():
    set_seed()
    logger.info("Loading data...")
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    features = get_all_features()
    logger.info("Features (%d): %s", len(features), features)

    train, test = time_series_split(df)
    logger.info("Train: %d, Test: %d", len(train), len(test))

    base_feats = get_base_features()

    logger.info("=== Step 1.1: DummyClassifier ===")
    run_id_11, roi_11 = run_step_1_1(train, test, base_feats)
    logger.info("Run ID: %s", run_id_11)

    logger.info("=== Step 1.2: Rule-based ===")
    run_id_12, roi_12, _edge_t = run_step_1_2(train, test, base_feats)
    logger.info("Run ID: %s", run_id_12)

    logger.info("=== Step 1.3: LogisticRegression ===")
    run_id_13, roi_13, auc_13, _t_13 = run_step_1_3(train, test, features)
    logger.info("Run ID: %s", run_id_13)

    logger.info("=== Step 1.4: CatBoost ===")
    run_id_14, roi_14, auc_14, _t_14, ev_14 = run_step_1_4(train, test, features)
    logger.info("Run ID: %s", run_id_14)

    logger.info("=== Phase 1 Summary ===")
    logger.info("1.1 Dummy:    ROI=%.2f%% N=%d  [%s]", roi_11["roi"], roi_11["n_bets"], run_id_11)
    logger.info("1.2 Rule:     ROI=%.2f%% N=%d  [%s]", roi_12["roi"], roi_12["n_bets"], run_id_12)
    logger.info(
        "1.3 LogReg:   ROI=%.2f%% AUC=%.4f N=%d  [%s]",
        roi_13["roi"],
        auc_13,
        roi_13["n_bets"],
        run_id_13,
    )
    logger.info(
        "1.4 CatBoost: ROI=%.2f%% AUC=%.4f N=%d | EV: ROI=%.2f%% N=%d  [%s]",
        roi_14["roi"],
        auc_14,
        roi_14["n_bets"],
        ev_14["roi"],
        ev_14["n_bets"],
        run_id_14,
    )


if __name__ == "__main__":
    main()
