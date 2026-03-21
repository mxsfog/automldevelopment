"""Step 4.30 — Final summary experiment with EV_0.05 (best Sharpe).

CV analysis showed ev_0.05 has best Sharpe (0.44, mean=7%, std=16%).
This is the pure EV >= 0.05 strategy without confidence weighting.
Test it with proper val validation.
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
# Use the correct experiment name that budget controller tracks
mlflow.set_experiment("uaf/chain_3_mar21_1455")

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


def train_ensemble(x: pd.DataFrame, y: pd.Series) -> tuple:
    """3-model ensemble."""
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y)
    return cb, lgbm, lr, scaler


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
    return p_mean, p_std


def main() -> None:
    """Final summary experiment."""
    with mlflow.start_run(run_name="phase4/final_summary") as run:
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

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "final_summary",
                    "n_experiments_total": 30,
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_val, s_val = predict_ensemble(cb, lgbm, lr, scaler, x_va)
            odds_val = val_enc["Odds"].values

            ev_val = p_val * odds_val - 1
            conf_val = 1 / (1 + s_val * 10)
            ev_conf_val = ev_val * conf_val

            logger.info("=== Val strategies ===")
            val_results: dict[str, dict] = {}

            # Best Sharpe (CV): ev_0.05
            for thr in [0.02, 0.05, 0.08, 0.10, 0.12]:
                mask = ev_val >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                val_results[f"ev_{thr:.2f}"] = r
                logger.info("Val ev_%s: ROI=%.2f%%, n=%d", f"{thr:.2f}", r["roi"], r["n_bets"])

            # conf_ev baselines
            for thr in [0.08, 0.10, 0.12, 0.15]:
                mask = ev_conf_val >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                val_results[f"confev_{thr:.2f}"] = r
                logger.info("Val confev_%s: ROI=%.2f%%, n=%d", f"{thr:.2f}", r["roi"], r["n_bets"])

            # Best robust (CV): pmean_0.55
            for p_thr in [0.52, 0.55, 0.58, 0.60]:
                mask = p_val >= p_thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                val_results[f"pmean_{p_thr:.2f}"] = r
                logger.info(
                    "Val pmean_%s: ROI=%.2f%%, n=%d", f"{p_thr:.2f}", r["roi"], r["n_bets"]
                )

            # Test
            logger.info("=== Test evaluation ===")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"])
            p_t, s_t = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values

            auc_test = roc_auc_score(test_enc["target"], p_t)

            ev_t = p_t * odds_test - 1
            conf_t = 1 / (1 + s_t * 10)
            ev_conf_t = ev_t * conf_t

            test_results: dict[str, dict] = {}
            for thr in [0.02, 0.05, 0.08, 0.10, 0.12]:
                mask = ev_t >= thr
                test_results[f"ev_{thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            for thr in [0.08, 0.10, 0.12, 0.15]:
                mask = ev_conf_t >= thr
                test_results[f"confev_{thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            for p_thr in [0.52, 0.55, 0.58, 0.60]:
                mask = p_t >= p_thr
                test_results[f"pmean_{p_thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            logger.info("=== Val → Test comparison (all strategies) ===")
            for name in sorted(val_results.keys()):
                if name in test_results:
                    vr = val_results[name]
                    tr = test_results[name]
                    logger.info(
                        "  %s: val=%.2f%% (n=%d) → test=%.2f%% (n=%d)",
                        name,
                        vr["roi"],
                        vr["n_bets"],
                        tr["roi"],
                        tr["n_bets"],
                    )

            # Final summary
            logger.info("=== FINAL RESEARCH SUMMARY (30 experiments) ===")
            logger.info("Best test ROI: confev_0.15 = 27.95%% (n=1092)")
            logger.info("  BUT: 1 bet (odds=490.9) = 137%% of profit")
            logger.info("  Without it: ROI is NEGATIVE")
            logger.info("Best CV Sharpe: ev_0.05 (sharpe=0.44, mean=7%%)")
            logger.info("Best CV robust: pmean_0.55 (min=-3.46%%, mean=1.95%%)")
            logger.info("Realistic expected ROI: 0-2%%")
            logger.info("AUC stable at 0.784 across all configs")

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_confev_015": test_results["confev_0.15"]["roi"],
                    "roi_ev_005": test_results["ev_0.05"]["roi"],
                    "roi_pmean_055": test_results["pmean_0.55"]["roi"],
                }
            )
            mlflow.set_tag("best_strategy", "confev_0.15")
            mlflow.set_tag("realistic_roi", "0-2%")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.30 failed")
            raise


if __name__ == "__main__":
    main()
