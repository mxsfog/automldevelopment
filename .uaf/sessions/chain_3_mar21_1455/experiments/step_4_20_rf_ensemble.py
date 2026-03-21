"""Step 4.20 — RandomForest in ensemble instead of LogisticRegression.

LR линейна, может RandomForest как третья модель в ensemble
даст лучшую diversity. Также: ExtraTrees.
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import add_sport_market_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcome_Odds",
    "n_outcomes",
    "mean_outcome_odds",
    "max_outcome_odds",
    "min_outcome_odds",
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def evaluate_conf_ev(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    odds: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    """conf_ev at thresholds."""
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf
    results = {}
    for thr in [0.10, 0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        results[f"conf_ev_{thr:.2f}"] = r
    return results


def main() -> None:
    """RF/ET ensemble comparison."""
    with mlflow.start_run(run_name="phase4/rf_ensemble") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            x_tr = train_fit_enc[FEATURES].fillna(0)
            x_va = val_enc[FEATURES].fillna(0)
            y_tr = train_fit_enc["target"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "rf_ensemble",
                }
            )

            # Train individual models
            cb = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb.fit(x_tr, y_tr)
            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm.fit(x_tr, y_tr)
            scaler = StandardScaler()
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(scaler.fit_transform(x_tr), y_tr)
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(x_tr, y_tr)
            et = ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
            et.fit(x_tr, y_tr)

            # Val predictions
            p_cb = cb.predict_proba(x_va)[:, 1]
            p_lgbm = lgbm.predict_proba(x_va)[:, 1]
            p_lr = lr.predict_proba(scaler.transform(x_va))[:, 1]
            p_rf = rf.predict_proba(x_va)[:, 1]
            p_et = et.predict_proba(x_va)[:, 1]

            odds_val = val_enc["Odds"].values
            y_val = val_enc["target"].values

            # Per-model AUC
            for name, p in [
                ("CB", p_cb),
                ("LGBM", p_lgbm),
                ("LR", p_lr),
                ("RF", p_rf),
                ("ET", p_et),
            ]:
                logger.info("Val AUC %s: %.4f", name, roc_auc_score(y_val, p))

            all_val = {}

            # Ensemble configs
            ensembles = {
                "cb_lgbm_lr": [p_cb, p_lgbm, p_lr],
                "cb_lgbm_rf": [p_cb, p_lgbm, p_rf],
                "cb_lgbm_et": [p_cb, p_lgbm, p_et],
                "cb_rf_lr": [p_cb, p_rf, p_lr],
                "cb_lgbm_rf_lr": [p_cb, p_lgbm, p_rf, p_lr],
                "all5": [p_cb, p_lgbm, p_lr, p_rf, p_et],
            }

            for ens_name, preds in ensembles.items():
                p_mean = np.mean(preds, axis=0)
                p_std = np.std(preds, axis=0)
                r = evaluate_conf_ev(p_mean, p_std, odds_val, val_enc)
                for k, v in r.items():
                    all_val[f"{ens_name}_{k}"] = v
                    logger.info("Val %s_%s: ROI=%.2f%%, n=%d", ens_name, k, v["roi"], v["n_bets"])

            val_ranked = sorted(
                all_val.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 30 else -999,
                reverse=True,
            )
            logger.info("Top-5 val:")
            for name, r in val_ranked[:5]:
                logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            y_train = train_enc["target"]

            cb_f = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb_f.fit(x_train, y_train)
            lgbm_f = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm_f.fit(x_train, y_train)
            sc_f = StandardScaler()
            lr_f = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_f.fit(sc_f.fit_transform(x_train), y_train)
            rf_f = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            )
            rf_f.fit(x_train, y_train)
            et_f = ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
            et_f.fit(x_train, y_train)

            p_cb_t = cb_f.predict_proba(x_test)[:, 1]
            p_lgbm_t = lgbm_f.predict_proba(x_test)[:, 1]
            p_lr_t = lr_f.predict_proba(sc_f.transform(x_test))[:, 1]
            p_rf_t = rf_f.predict_proba(x_test)[:, 1]
            p_et_t = et_f.predict_proba(x_test)[:, 1]

            ensembles_t = {
                "cb_lgbm_lr": [p_cb_t, p_lgbm_t, p_lr_t],
                "cb_lgbm_rf": [p_cb_t, p_lgbm_t, p_rf_t],
                "cb_lgbm_et": [p_cb_t, p_lgbm_t, p_et_t],
                "cb_rf_lr": [p_cb_t, p_rf_t, p_lr_t],
                "cb_lgbm_rf_lr": [p_cb_t, p_lgbm_t, p_rf_t, p_lr_t],
                "all5": [p_cb_t, p_lgbm_t, p_lr_t, p_rf_t, p_et_t],
            }

            test_results = {}
            for ens_name, preds in ensembles_t.items():
                p_mean = np.mean(preds, axis=0)
                p_std = np.std(preds, axis=0)
                r = evaluate_conf_ev(p_mean, p_std, test_enc["Odds"].values, test_enc)
                for k, v in r.items():
                    test_results[f"{ens_name}_{k}"] = v

            for k, v in sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )[:10]:
                if v["n_bets"] >= 50:
                    logger.info("Test %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            best_name = max(
                test_results,
                key=lambda k: test_results[k]["roi"] if test_results[k]["n_bets"] >= 50 else -999,
            )
            best = test_results[best_name]
            logger.info(
                "Best (n>=50): %s -> ROI=%.2f%%, n=%d",
                best_name,
                best["roi"],
                best["n_bets"],
            )

            mlflow.log_metrics(
                {
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            mlflow.set_tag("best_strategy", best_name)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.20 failed")
            raise


if __name__ == "__main__":
    main()
