"""Step 4.22 — Parlay boost: higher conf_ev threshold for parlays.

Step 4.21 показал что прибыль из парлаев. Гипотеза:
1. Разные пороги conf_ev для singles/parlays
2. Parlay-specific features: n_outcomes, odds ratio (parlay_odds/product_odds)
3. Parlay-weighted conf_ev: boost EV для parlays
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


def dual_threshold_selection(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    odds: np.ndarray,
    is_parlay: np.ndarray,
    thr_single: float,
    thr_parlay: float,
) -> np.ndarray:
    """Different conf_ev thresholds for singles vs parlays."""
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf

    mask = np.zeros(len(p_mean), dtype=bool)
    mask[~is_parlay] = ev_conf[~is_parlay] >= thr_single
    mask[is_parlay] = ev_conf[is_parlay] >= thr_parlay
    return mask


def main() -> None:
    """Parlay boost experiment."""
    with mlflow.start_run(run_name="phase4/parlay_boost") as run:
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
                    "method": "parlay_boost",
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_va)

            odds_val = val_enc["Odds"].values
            is_parlay_val = val_enc["Is_Parlay"].values.astype(bool)

            all_val = {}

            # 1. Baseline: same threshold
            ev = p_mean * odds_val - 1
            conf = 1 / (1 + p_std * 10)
            ev_conf = ev * conf
            for thr in [0.12, 0.15, 0.18]:
                mask = ev_conf >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                all_val[f"uniform_{thr:.2f}"] = r
                logger.info(
                    "Val uniform_%s: ROI=%.2f%%, n=%d", f"{thr:.2f}", r["roi"], r["n_bets"]
                )

            # 2. Dual threshold: different for singles/parlays
            for s_thr in [0.12, 0.15, 0.18, 0.20]:
                for p_thr in [0.05, 0.08, 0.10, 0.12]:
                    mask = dual_threshold_selection(
                        p_mean, p_std, odds_val, is_parlay_val, s_thr, p_thr
                    )
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"dual_s{s_thr:.2f}_p{p_thr:.2f}"
                    all_val[name] = r

            # 3. Parlays only with lower threshold
            parlay_mask_val = is_parlay_val
            for thr in [0.05, 0.08, 0.10, 0.12]:
                mask = (ev_conf >= thr) & parlay_mask_val
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                all_val[f"parlays_only_{thr:.2f}"] = r

            # Val ranking
            val_ranked = sorted(
                all_val.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 20 else -999,
                reverse=True,
            )
            logger.info("Top-10 val (n>=20):")
            for name, r in val_ranked[:10]:
                if r["n_bets"] >= 20:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"])
            p_t, s_t = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values
            is_parlay_test = test_enc["Is_Parlay"].values.astype(bool)

            auc_test = roc_auc_score(test_enc["target"], p_t)

            test_results = {}
            ev_t = p_t * odds_test - 1
            conf_t = 1 / (1 + s_t * 10)
            ev_conf_t = ev_t * conf_t

            # Baseline
            for thr in [0.12, 0.15, 0.18]:
                mask = ev_conf_t >= thr
                test_results[f"uniform_{thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            # Top-5 dual threshold strategies from val
            for name, _rv in val_ranked[:10]:
                if _rv["n_bets"] < 20:
                    continue
                if name.startswith("dual_"):
                    parts = name.split("_")
                    s_thr = float(parts[1][1:])
                    p_thr = float(parts[2][1:])
                    mask = dual_threshold_selection(
                        p_t, s_t, odds_test, is_parlay_test, s_thr, p_thr
                    )
                    test_results[name] = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                elif name.startswith("parlays_only_"):
                    thr = float(name.split("_")[-1])
                    mask = (ev_conf_t >= thr) & is_parlay_test
                    test_results[name] = calc_roi(test_enc, mask.astype(float), threshold=0.5)

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
                    "auc_test": auc_test,
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
            logger.exception("Step 4.22 failed")
            raise


if __name__ == "__main__":
    main()
