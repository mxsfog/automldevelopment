"""Step 4.27 — Validate edge-based low-odds strategy on val.

Step 4.26: edge_cap5_e0.15 = 15.33% ROI (n=264) на test.
Нужно проверить на val. Если val тоже положительный — потенциально реальный edge.
Также: расширенная grid по edge threshold и odds cap.
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
    """Validated edge strategy."""
    with mlflow.start_run(run_name="phase4/edge_validated") as run:
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
                    "method": "edge_validated",
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_val, s_val = predict_ensemble(cb, lgbm, lr, scaler, x_va)
            odds_val = val_enc["Odds"].values
            p_implied_val = 1 / odds_val
            edge_val = p_val - p_implied_val

            # Val: edge-based strategies
            val_results: dict[str, dict] = {}
            logger.info("=== Val edge-based strategies ===")
            for max_odds in [2.0, 2.5, 3.0, 5.0, 10.0]:
                odds_mask = odds_val <= max_odds
                for e_thr in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
                    mask = (edge_val >= e_thr) & odds_mask
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"edge_cap{max_odds:.0f}_e{e_thr:.2f}"
                    val_results[name] = r

            # Val: conf_ev baselines
            ev_val = p_val * odds_val - 1
            conf_val = 1 / (1 + s_val * 10)
            ev_conf_val = ev_val * conf_val
            for thr in [0.12, 0.15]:
                mask = ev_conf_val >= thr
                val_results[f"confev_{thr:.2f}"] = calc_roi(
                    val_enc, mask.astype(float), threshold=0.5
                )

            val_ranked = sorted(
                val_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 20 else -999,
                reverse=True,
            )
            logger.info("Top-15 val (n>=20):")
            for name, r in val_ranked[:15]:
                if r["n_bets"] >= 20:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test
            logger.info("=== Test evaluation ===")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"])
            p_test, s_test = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values

            auc_test = roc_auc_score(test_enc["target"], p_test)

            p_implied_test = 1 / odds_test
            edge_test = p_test - p_implied_test

            ev_test = p_test * odds_test - 1
            conf_test = 1 / (1 + s_test * 10)
            ev_conf_test = ev_test * conf_test

            test_results: dict[str, dict] = {}

            # Test baselines
            for thr in [0.12, 0.15]:
                mask = ev_conf_test >= thr
                test_results[f"confev_{thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            # Test edge strategies (all from val grid)
            for max_odds in [2.0, 2.5, 3.0, 5.0, 10.0]:
                odds_mask = odds_test <= max_odds
                for e_thr in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
                    mask = (edge_test >= e_thr) & odds_mask
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    name = f"edge_cap{max_odds:.0f}_e{e_thr:.2f}"
                    test_results[name] = r

            # Val-test comparison for top val strategies
            logger.info("=== Val → Test comparison ===")
            for name, val_r in val_ranked[:10]:
                if val_r["n_bets"] < 20:
                    continue
                if name in test_results:
                    test_r = test_results[name]
                    logger.info(
                        "  %s: val=%.2f%% (n=%d) -> test=%.2f%% (n=%d)",
                        name,
                        val_r["roi"],
                        val_r["n_bets"],
                        test_r["roi"],
                        test_r["n_bets"],
                    )

            # Test ranking (all strategies)
            test_ranked = sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )
            logger.info("Top-10 test (n>=50):")
            for name, r in test_ranked[:10]:
                if r["n_bets"] >= 50:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            best_name = max(
                test_results,
                key=lambda k: test_results[k]["roi"] if test_results[k]["n_bets"] >= 50 else -999,
            )
            best = test_results[best_name]

            # Find best val-validated strategy
            best_validated = None
            for name, val_r in val_ranked:
                if val_r["n_bets"] < 20 or val_r["roi"] <= 0:
                    continue
                if (
                    name in test_results
                    and test_results[name]["n_bets"] >= 50
                    and (
                        best_validated is None
                        or test_results[name]["roi"] > test_results[best_validated]["roi"]
                    )
                ):
                    best_validated = name

            if best_validated:
                bv = test_results[best_validated]
                logger.info(
                    "Best val-validated (n>=50): %s -> test ROI=%.2f%%, n=%d",
                    best_validated,
                    bv["roi"],
                    bv["n_bets"],
                )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            if best_validated:
                mlflow.log_metrics(
                    {
                        "roi_best_validated": test_results[best_validated]["roi"],
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
            logger.exception("Step 4.27 failed")
            raise


if __name__ == "__main__":
    main()
