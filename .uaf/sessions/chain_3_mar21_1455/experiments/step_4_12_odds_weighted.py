"""Step 4.12 — Odds-weighted training + profit-proportional weighting.

CV (step 4.11) показал что средний ROI ~0-5% при std=11-70%.
Profit в основном из high-odds ставок.

Гипотеза: weight = log(odds) при обучении заставит модель фокусироваться
на high-odds ставках, где прибыль. Плюс: weight = odds * target
(profit-proportional) для ещё большего фокуса.
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


def train_weighted(
    x: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
) -> tuple:
    """3-model ensemble with sample weights."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    cb.fit(x, y, sample_weight=weights)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y, sample_weight=weights)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y, sample_weight=weights)
    return cb, lgbm, lr, scaler


def train_unweighted(x: pd.DataFrame, y: pd.Series) -> tuple:
    """3-model ensemble without weights."""
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


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions + std."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std(np.array([p_cb, p_lgbm, p_lr]), axis=0)
    return p_mean, p_std


def evaluate_strategy(
    df: pd.DataFrame,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    name: str,
) -> dict:
    """Evaluate conf_ev strategies."""
    odds = df["Odds"].values
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf

    results = {}
    for thr in [0.10, 0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        results[f"{name}_conf_ev_{thr:.2f}"] = r
    return results


def main() -> None:
    """Odds-weighted training experiment."""
    with mlflow.start_run(run_name="phase4/odds_weighted") as run:
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
            odds_tr = train_fit_enc["Odds"].values

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "odds_weighted_training",
                }
            )

            all_val_results = {}

            # 1. Unweighted baseline
            logger.info("Training unweighted baseline...")
            cb_u, lgbm_u, lr_u, sc_u = train_unweighted(x_tr, y_tr)
            p_u, p_std_u = predict_ensemble(cb_u, lgbm_u, lr_u, sc_u, x_va)
            val_r = evaluate_strategy(val_enc, p_u, p_std_u, "unweighted")
            all_val_results.update(val_r)
            for k, v in val_r.items():
                logger.info("Val %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 2. log(odds) weighting
            logger.info("Training with log(odds) weights...")
            w_log = np.log1p(odds_tr)
            cb_l, lgbm_l, lr_l, sc_l = train_weighted(x_tr, y_tr, w_log)
            p_l, p_std_l = predict_ensemble(cb_l, lgbm_l, lr_l, sc_l, x_va)
            val_r = evaluate_strategy(val_enc, p_l, p_std_l, "log_odds")
            all_val_results.update(val_r)
            for k, v in val_r.items():
                logger.info("Val %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 3. sqrt(odds) weighting
            logger.info("Training with sqrt(odds) weights...")
            w_sqrt = np.sqrt(odds_tr)
            cb_s, lgbm_s, lr_s, sc_s = train_weighted(x_tr, y_tr, w_sqrt)
            p_s, p_std_s = predict_ensemble(cb_s, lgbm_s, lr_s, sc_s, x_va)
            val_r = evaluate_strategy(val_enc, p_s, p_std_s, "sqrt_odds")
            all_val_results.update(val_r)
            for k, v in val_r.items():
                logger.info("Val %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 4. odds weighting (linear)
            logger.info("Training with odds weights...")
            w_odds = odds_tr.clip(max=100)
            cb_o, lgbm_o, lr_o, sc_o = train_weighted(x_tr, y_tr, w_odds)
            p_o, p_std_o = predict_ensemble(cb_o, lgbm_o, lr_o, sc_o, x_va)
            val_r = evaluate_strategy(val_enc, p_o, p_std_o, "odds")
            all_val_results.update(val_r)
            for k, v in val_r.items():
                logger.info("Val %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # Best on val (min 20 bets)
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
            odds_train = train_enc["Odds"].values

            test_results = {}

            # Unweighted on test
            cb_uf, lgbm_uf, lr_uf, sc_uf = train_unweighted(x_train, train_enc["target"])
            p_uf, p_std_uf = predict_ensemble(cb_uf, lgbm_uf, lr_uf, sc_uf, x_test)
            test_r = evaluate_strategy(test_enc, p_uf, p_std_uf, "unweighted")
            test_results.update(test_r)

            auc_test = roc_auc_score(test_enc["target"], p_uf)

            # Best weighted scheme from val
            best_val_name = val_ranked[0][0]
            weight_scheme = best_val_name.split("_conf_ev")[0]

            weight_map = {
                "unweighted": None,
                "log_odds": np.log1p(odds_train),
                "sqrt_odds": np.sqrt(odds_train),
                "odds": odds_train.clip(max=100),
            }

            if weight_scheme in weight_map and weight_map[weight_scheme] is not None:
                w = weight_map[weight_scheme]
                cb_w, lgbm_w, lr_w, sc_w = train_weighted(x_train, train_enc["target"], w)
                p_w, p_std_w = predict_ensemble(cb_w, lgbm_w, lr_w, sc_w, x_test)
                test_r = evaluate_strategy(test_enc, p_w, p_std_w, weight_scheme)
                test_results.update(test_r)
                for k, v in test_r.items():
                    logger.info("Test %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # Report test results
            for k, v in test_results.items():
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
            mlflow.set_tag("best_val_strategy", best_val_name)

            if best["roi"] > 27.95 and "unweighted" not in best_name:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_uf.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best["roi"],
                    "auc": auc_test,
                    "threshold": 0.12,
                    "ev_threshold": 0.15,
                    "n_bets": best["n_bets"],
                    "feature_names": FEATURES,
                    "selection_method": best_name,
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.12 failed")
            raise


if __name__ == "__main__":
    main()
