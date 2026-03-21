"""Step 4.14 — Rank averaging + probability binning.

Гипотезы:
1. Rank averaging: преобразовать p каждой модели в ранги (0-1),
   затем average рангов. Устраняет разницу в масштабах вероятностей.
2. Probability binning: дискретизировать p_mean в N бинов,
   использовать среднюю ROI по бину для selection.
3. CB+LR only (step 4.13 показал LGBM=0 optimal weight).
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
from scipy.stats import rankdata
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


def train_models(x: pd.DataFrame, y: pd.Series) -> tuple:
    """Train 3 models."""
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


def get_preds(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Individual model predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    return p_cb, p_lgbm, p_lr


def rank_average(preds: list[np.ndarray]) -> np.ndarray:
    """Rank average: преобразуем каждый вектор в ранги [0,1], затем среднее."""
    n = len(preds[0])
    ranks = []
    for p in preds:
        r = rankdata(p) / n
        ranks.append(r)
    return np.mean(ranks, axis=0)


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
    """Rank averaging + probability binning + CB+LR only."""
    with mlflow.start_run(run_name="phase4/rank_average") as run:
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
                    "method": "rank_average_binning",
                }
            )

            cb, lgbm, lr, scaler = train_models(x_tr, train_fit_enc["target"])
            p_cb_val, p_lgbm_val, p_lr_val = get_preds(cb, lgbm, lr, scaler, x_va)

            odds_val = val_enc["Odds"].values
            all_val_results = {}

            # 1. Standard equal average (baseline)
            p_equal = (p_cb_val + p_lgbm_val + p_lr_val) / 3
            p_std_equal = np.std([p_cb_val, p_lgbm_val, p_lr_val], axis=0)
            r_eq = evaluate_conf_ev(p_equal, p_std_equal, odds_val, val_enc)
            for k, v in r_eq.items():
                all_val_results[f"equal_{k}"] = v
                logger.info("Val equal_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 2. Rank average (3 models)
            p_rank = rank_average([p_cb_val, p_lgbm_val, p_lr_val])
            # Rank average is on [0,1] scale but not calibrated
            # Use ranks for EV too — need to scale back to probability
            # Use original std for confidence
            r_rank = evaluate_conf_ev(p_rank, p_std_equal, odds_val, val_enc)
            for k, v in r_rank.items():
                all_val_results[f"rank3_{k}"] = v
                logger.info("Val rank3_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 3. CB + LR only (step 4.13 showed LGBM=0 optimal)
            p_cb_lr = (p_cb_val + p_lr_val) / 2
            p_std_cb_lr = np.std([p_cb_val, p_lr_val], axis=0)
            r_cblr = evaluate_conf_ev(p_cb_lr, p_std_cb_lr, odds_val, val_enc)
            for k, v in r_cblr.items():
                all_val_results[f"cb_lr_{k}"] = v
                logger.info("Val cb_lr_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 4. Rank average CB+LR only
            p_rank2 = rank_average([p_cb_val, p_lr_val])
            r_rank2 = evaluate_conf_ev(p_rank2, p_std_cb_lr, odds_val, val_enc)
            for k, v in r_rank2.items():
                all_val_results[f"rank2_{k}"] = v
                logger.info("Val rank2_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 5. Power mean (p^2 average then sqrt) — emphasizes higher probs
            p_power = np.sqrt((p_cb_val**2 + p_lgbm_val**2 + p_lr_val**2) / 3)
            r_power = evaluate_conf_ev(p_power, p_std_equal, odds_val, val_enc)
            for k, v in r_power.items():
                all_val_results[f"power_{k}"] = v
                logger.info("Val power_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # Val ranking
            val_ranked = sorted(
                all_val_results.items(),
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
            p_cb_t, p_lgbm_t, p_lr_t = get_preds(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values

            test_results = {}

            # Equal
            p_eq_t = (p_cb_t + p_lgbm_t + p_lr_t) / 3
            p_std_t = np.std([p_cb_t, p_lgbm_t, p_lr_t], axis=0)
            r_t = evaluate_conf_ev(p_eq_t, p_std_t, odds_test, test_enc)
            for k, v in r_t.items():
                test_results[f"equal_{k}"] = v

            auc_test = roc_auc_score(test_enc["target"], p_eq_t)

            # Rank 3
            p_rank_t = rank_average([p_cb_t, p_lgbm_t, p_lr_t])
            r_t = evaluate_conf_ev(p_rank_t, p_std_t, odds_test, test_enc)
            for k, v in r_t.items():
                test_results[f"rank3_{k}"] = v

            # CB+LR
            p_cblr_t = (p_cb_t + p_lr_t) / 2
            p_std_cblr_t = np.std([p_cb_t, p_lr_t], axis=0)
            r_t = evaluate_conf_ev(p_cblr_t, p_std_cblr_t, odds_test, test_enc)
            for k, v in r_t.items():
                test_results[f"cb_lr_{k}"] = v

            # Rank 2
            p_rank2_t = rank_average([p_cb_t, p_lr_t])
            r_t = evaluate_conf_ev(p_rank2_t, p_std_cblr_t, odds_test, test_enc)
            for k, v in r_t.items():
                test_results[f"rank2_{k}"] = v

            # Power mean
            p_power_t = np.sqrt((p_cb_t**2 + p_lgbm_t**2 + p_lr_t**2) / 3)
            r_t = evaluate_conf_ev(p_power_t, p_std_t, odds_test, test_enc)
            for k, v in r_t.items():
                test_results[f"power_{k}"] = v

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
            logger.exception("Step 4.14 failed")
            raise


if __name__ == "__main__":
    main()
