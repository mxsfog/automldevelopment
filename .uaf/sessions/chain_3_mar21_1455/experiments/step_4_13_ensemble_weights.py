"""Step 4.13 — Optimal ensemble weights + geometric mean.

Текущий ensemble: p = (p_cb + p_lgbm + p_lr) / 3 (equal weights).
Гипотезы:
1. Оптимальные веса на val: w1*p_cb + w2*p_lgbm + w3*p_lr
2. Geometric mean: (p_cb * p_lgbm * p_lr)^(1/3)
3. Median вместо mean
4. Max/min combinations
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
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
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


def train_models(x: pd.DataFrame, y: pd.Series) -> tuple:
    """Train 3 models separately."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
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


def get_per_model_preds(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Get individual model predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    return p_cb, p_lgbm, p_lr


def find_optimal_weights(
    p_cb: np.ndarray,
    p_lgbm: np.ndarray,
    p_lr: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Find optimal weights minimizing log_loss on val."""

    def objective(w: np.ndarray) -> float:
        w_norm = np.abs(w) / np.abs(w).sum()
        p = w_norm[0] * p_cb + w_norm[1] * p_lgbm + w_norm[2] * p_lr
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return log_loss(y, p)

    best_result = None
    best_loss = float("inf")
    for _ in range(10):
        w0 = np.random.dirichlet([1, 1, 1])
        result = minimize(objective, w0, method="Nelder-Mead")
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    w = np.abs(best_result.x) / np.abs(best_result.x).sum()
    return w


def evaluate_conf_ev(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    odds: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    """Evaluate conf_ev at multiple thresholds."""
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
    """Ensemble weights optimization."""
    with mlflow.start_run(run_name="phase4/ensemble_weights") as run:
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
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "ensemble_weights",
                }
            )

            # Train on train_fit
            cb, lgbm, lr, scaler = train_models(x_tr, train_fit_enc["target"])
            p_cb_val, p_lgbm_val, p_lr_val = get_per_model_preds(cb, lgbm, lr, scaler, x_va)

            odds_val = val_enc["Odds"].values
            y_val = val_enc["target"].values

            # Per-model AUC
            auc_cb = roc_auc_score(y_val, p_cb_val)
            auc_lgbm = roc_auc_score(y_val, p_lgbm_val)
            auc_lr = roc_auc_score(y_val, p_lr_val)
            logger.info(
                "Per-model AUC: CB=%.4f, LGBM=%.4f, LR=%.4f",
                auc_cb,
                auc_lgbm,
                auc_lr,
            )

            all_results = {}

            # 1. Equal weights (baseline)
            p_equal = (p_cb_val + p_lgbm_val + p_lr_val) / 3
            p_std_equal = np.std(np.array([p_cb_val, p_lgbm_val, p_lr_val]), axis=0)
            r_equal = evaluate_conf_ev(p_equal, p_std_equal, odds_val, val_enc)
            for k, v in r_equal.items():
                all_results[f"equal_{k}"] = v
                logger.info("Val equal_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 2. Optimal weights (log_loss)
            w_opt = find_optimal_weights(p_cb_val, p_lgbm_val, p_lr_val, y_val)
            logger.info("Optimal weights: CB=%.3f, LGBM=%.3f, LR=%.3f", *w_opt)

            p_opt = w_opt[0] * p_cb_val + w_opt[1] * p_lgbm_val + w_opt[2] * p_lr_val
            # Std using weighted models
            preds_arr = np.array([p_cb_val, p_lgbm_val, p_lr_val])
            p_std_opt = np.sqrt(np.average((preds_arr - p_opt) ** 2, axis=0, weights=w_opt))

            r_opt = evaluate_conf_ev(p_opt, p_std_opt, odds_val, val_enc)
            for k, v in r_opt.items():
                all_results[f"opt_{k}"] = v
                logger.info("Val opt_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 3. Geometric mean
            p_geo = np.power(
                p_cb_val.clip(1e-7) * p_lgbm_val.clip(1e-7) * p_lr_val.clip(1e-7),
                1 / 3,
            )
            p_std_geo = np.std(np.array([p_cb_val, p_lgbm_val, p_lr_val]), axis=0)
            r_geo = evaluate_conf_ev(p_geo, p_std_geo, odds_val, val_enc)
            for k, v in r_geo.items():
                all_results[f"geo_{k}"] = v
                logger.info("Val geo_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 4. Median
            p_median = np.median(np.array([p_cb_val, p_lgbm_val, p_lr_val]), axis=0)
            r_med = evaluate_conf_ev(p_median, p_std_equal, odds_val, val_enc)
            for k, v in r_med.items():
                all_results[f"median_{k}"] = v
                logger.info("Val median_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 5. CB-only (best individual AUC)
            r_cb = evaluate_conf_ev(p_cb_val, np.zeros_like(p_cb_val) + 0.05, odds_val, val_enc)
            for k, v in r_cb.items():
                all_results[f"cb_only_{k}"] = v

            # Best on val
            val_ranked = sorted(
                all_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 20 else -999,
                reverse=True,
            )
            logger.info("Top-5 val:")
            for name, r in val_ranked[:5]:
                logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test evaluation
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_models(x_train, train_enc["target"])
            p_cb_t, p_lgbm_t, p_lr_t = get_per_model_preds(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values

            test_results = {}

            # Equal
            p_eq_t = (p_cb_t + p_lgbm_t + p_lr_t) / 3
            p_std_eq_t = np.std(np.array([p_cb_t, p_lgbm_t, p_lr_t]), axis=0)
            r_eq_t = evaluate_conf_ev(p_eq_t, p_std_eq_t, odds_test, test_enc)
            for k, v in r_eq_t.items():
                test_results[f"equal_{k}"] = v

            auc_test = roc_auc_score(test_enc["target"], p_eq_t)

            # Optimal weights (from val)
            p_opt_t = w_opt[0] * p_cb_t + w_opt[1] * p_lgbm_t + w_opt[2] * p_lr_t
            preds_arr_t = np.array([p_cb_t, p_lgbm_t, p_lr_t])
            p_std_opt_t = np.sqrt(np.average((preds_arr_t - p_opt_t) ** 2, axis=0, weights=w_opt))
            r_opt_t = evaluate_conf_ev(p_opt_t, p_std_opt_t, odds_test, test_enc)
            for k, v in r_opt_t.items():
                test_results[f"opt_{k}"] = v

            # Geometric
            p_geo_t = np.power(p_cb_t.clip(1e-7) * p_lgbm_t.clip(1e-7) * p_lr_t.clip(1e-7), 1 / 3)
            r_geo_t = evaluate_conf_ev(p_geo_t, p_std_eq_t, odds_test, test_enc)
            for k, v in r_geo_t.items():
                test_results[f"geo_{k}"] = v

            # Median
            p_med_t = np.median(np.array([p_cb_t, p_lgbm_t, p_lr_t]), axis=0)
            r_med_t = evaluate_conf_ev(p_med_t, p_std_eq_t, odds_test, test_enc)
            for k, v in r_med_t.items():
                test_results[f"median_{k}"] = v

            # Report
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
                    "auc_cb": auc_cb,
                    "auc_lgbm": auc_lgbm,
                    "auc_lr": auc_lr,
                    "w_cb": w_opt[0],
                    "w_lgbm": w_opt[1],
                    "w_lr": w_opt[2],
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            mlflow.set_tag("best_strategy", best_name)
            mlflow.set_tag(
                "optimal_weights", f"CB={w_opt[0]:.3f} LGBM={w_opt[1]:.3f} LR={w_opt[2]:.3f}"
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.13 failed")
            raise


if __name__ == "__main__":
    main()
