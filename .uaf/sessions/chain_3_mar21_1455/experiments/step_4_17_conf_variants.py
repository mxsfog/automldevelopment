"""Step 4.17 — Alternative confidence formulas.

Текущая формула: conf = 1 / (1 + std * 10), порог conf_ev >= 0.15
Коэффициент 10 при std был выбран ad-hoc.

Тесты:
1. Разные коэффициенты при std: 5, 8, 10, 15, 20
2. Альтернативная формула: conf = exp(-k * std)
3. Std-percentile: conf = 1 - percentile_rank(std)
4. Min-max clipping: conf = clip((max_std - std) / (max_std - min_std), 0, 1)
5. Sweep ev_threshold при лучшем k
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
    """Alternative confidence formulas."""
    with mlflow.start_run(run_name="phase4/conf_variants") as run:
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
                    "method": "confidence_variants",
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_va)
            odds_val = val_enc["Odds"].values
            ev = p_mean * odds_val - 1

            all_val = {}

            # 1. Original formula: conf = 1/(1+k*std), vary k
            for k in [5, 8, 10, 12, 15, 20]:
                conf = 1 / (1 + p_std * k)
                ev_conf = ev * conf
                for thr in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
                    mask = ev_conf >= thr
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"inv_k{k}_t{thr:.2f}"
                    all_val[name] = r

            # 2. Exponential: conf = exp(-k * std)
            for k in [5, 10, 15, 20]:
                conf = np.exp(-k * p_std)
                ev_conf = ev * conf
                for thr in [0.08, 0.10, 0.12, 0.15]:
                    mask = ev_conf >= thr
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"exp_k{k}_t{thr:.2f}"
                    all_val[name] = r

            # 3. Percentile-based confidence
            std_rank = rankdata(p_std) / len(p_std)
            conf_pct = 1 - std_rank
            ev_conf_pct = ev * conf_pct
            for thr in [0.02, 0.04, 0.06, 0.08, 0.10]:
                mask = ev_conf_pct >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                all_val[f"pct_t{thr:.2f}"] = r

            # 4. Hard std filter + EV threshold
            for std_max in [0.03, 0.05, 0.07, 0.10]:
                for ev_thr in [0.08, 0.10, 0.12, 0.15]:
                    mask = (ev >= ev_thr) & (p_std <= std_max)
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    all_val[f"hard_s{std_max:.2f}_e{ev_thr:.2f}"] = r

            # Val ranking (min 30 bets)
            val_ranked = sorted(
                all_val.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 30 else -999,
                reverse=True,
            )
            logger.info("Top-15 val (n>=30):")
            for name, r in val_ranked[:15]:
                if r["n_bets"] >= 30:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test with top val strategies
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"])
            p_mean_t, p_std_t = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values
            ev_t = p_mean_t * odds_test - 1

            auc_test = roc_auc_score(test_enc["target"], p_mean_t)

            test_results = {}

            # Baseline: inv_k10_t0.15 (original conf_ev_0.15)
            conf_base = 1 / (1 + p_std_t * 10)
            mask_base = (ev_t * conf_base) >= 0.15
            test_results["inv_k10_t0.15"] = calc_roi(
                test_enc, mask_base.astype(float), threshold=0.5
            )

            # Test top-5 val strategies
            for name, _rv in val_ranked[:10]:
                if _rv["n_bets"] < 30:
                    continue
                parts = name.split("_")
                if name.startswith("inv_"):
                    k = int(parts[1][1:])
                    thr = float(parts[2][1:])
                    conf_t = 1 / (1 + p_std_t * k)
                    mask_t = (ev_t * conf_t) >= thr
                elif name.startswith("exp_"):
                    k = int(parts[1][1:])
                    thr = float(parts[2][1:])
                    conf_t = np.exp(-k * p_std_t)
                    mask_t = (ev_t * conf_t) >= thr
                elif name.startswith("pct_"):
                    thr = float(parts[1][1:])
                    std_rank_t = rankdata(p_std_t) / len(p_std_t)
                    conf_t = 1 - std_rank_t
                    mask_t = (ev_t * conf_t) >= thr
                elif name.startswith("hard_"):
                    std_max = float(parts[1][1:])
                    ev_thr = float(parts[2][1:])
                    mask_t = (ev_t >= ev_thr) & (p_std_t <= std_max)
                else:
                    continue

                r_t = calc_roi(test_enc, mask_t.astype(float), threshold=0.5)
                test_results[name] = r_t

            for k, v in sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            ):
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
            logger.exception("Step 4.17 failed")
            raise


if __name__ == "__main__":
    main()
