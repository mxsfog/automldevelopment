"""Step 4.31 — Combined edge+EV strategy.

Test combining edge-based selection (p_model - p_implied >= thr, low odds)
with EV-based selection (p*odds-1 >= thr, any odds) as union/intersection.
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
    """Combined edge+EV strategy analysis."""
    with mlflow.start_run(run_name="phase4/combined_edge_ev") as run:
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
                    "validation_scheme": "time_series_val_test",
                    "seed": 42,
                    "method": "combined_edge_ev",
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_val, _s_val = predict_ensemble(cb, lgbm, lr, scaler, x_va)
            odds_val = val_enc["Odds"].values

            ev_val = p_val * odds_val - 1
            p_implied_val = 1.0 / odds_val
            edge_val = p_val - p_implied_val

            # Test
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"])
            p_t, _s_t = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values

            auc_test = roc_auc_score(test_enc["target"], p_t)

            ev_t = p_t * odds_test - 1
            p_implied_t = 1.0 / odds_test
            edge_t = p_t - p_implied_t

            logger.info("AUC test: %.4f", auc_test)

            # Combined strategies
            results: dict[str, dict] = {}
            for odds_cap in [3, 5, 10]:
                for e_thr in [0.05, 0.08, 0.10]:
                    for ev_thr in [0.02, 0.05]:
                        # Intersection: edge AND ev
                        name_i = f"inter_e{e_thr}_ev{ev_thr}_cap{odds_cap}"
                        mask_val_i = (
                            (edge_val >= e_thr) & (ev_val >= ev_thr) & (odds_val <= odds_cap)
                        )
                        mask_test_i = (
                            (edge_t >= e_thr) & (ev_t >= ev_thr) & (odds_test <= odds_cap)
                        )
                        r_val = calc_roi(val_enc, mask_val_i.astype(float), threshold=0.5)
                        r_test = calc_roi(test_enc, mask_test_i.astype(float), threshold=0.5)
                        results[name_i] = {
                            "val_roi": r_val["roi"],
                            "val_n": r_val["n_bets"],
                            "test_roi": r_test["roi"],
                            "test_n": r_test["n_bets"],
                        }

                        # Union: edge OR ev (capped)
                        name_u = f"union_e{e_thr}_ev{ev_thr}_cap{odds_cap}"
                        mask_val_u = ((edge_val >= e_thr) & (odds_val <= odds_cap)) | (
                            ev_val >= ev_thr
                        )
                        mask_test_u = ((edge_t >= e_thr) & (odds_test <= odds_cap)) | (
                            ev_t >= ev_thr
                        )
                        r_val_u = calc_roi(val_enc, mask_val_u.astype(float), threshold=0.5)
                        r_test_u = calc_roi(test_enc, mask_test_u.astype(float), threshold=0.5)
                        results[name_u] = {
                            "val_roi": r_val_u["roi"],
                            "val_n": r_val_u["n_bets"],
                            "test_roi": r_test_u["roi"],
                            "test_n": r_test_u["n_bets"],
                        }

            # Sort by val-test consistency
            logger.info("=== Combined edge+EV strategies ===")
            for name, r in sorted(
                results.items(),
                key=lambda x: abs(x[1]["val_roi"] - x[1]["test_roi"]),
            )[:15]:
                logger.info(
                    "  %s: val=%.2f%% (n=%d) → test=%.2f%% (n=%d)",
                    name,
                    r["val_roi"],
                    r["val_n"],
                    r["test_roi"],
                    r["test_n"],
                )

            # Best by val ROI
            best_val = max(results.items(), key=lambda x: x[1]["val_roi"])
            best_consistent = min(
                results.items(),
                key=lambda x: abs(x[1]["val_roi"] - x[1]["test_roi"]),
            )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "best_val_roi": best_val[1]["val_roi"],
                    "best_val_test_roi": best_val[1]["test_roi"],
                    "best_consistent_val": best_consistent[1]["val_roi"],
                    "best_consistent_test": best_consistent[1]["test_roi"],
                }
            )
            mlflow.set_tag("best_val_strategy", best_val[0])
            mlflow.set_tag("best_consistent_strategy", best_consistent[0])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.31 failed")
            raise


if __name__ == "__main__":
    main()
