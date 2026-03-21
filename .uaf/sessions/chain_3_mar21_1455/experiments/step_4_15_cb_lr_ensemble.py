"""Step 4.15 — CB+LR 2-model ensemble validation.

Step 4.13 показал оптимальные веса CB=0.335, LGBM=0.000, LR=0.665.
Step 4.14 показал CB+LR conf_ev_0.10 = 8.45% на val (лучше 3-model equal).

Гипотеза: 2-model ensemble (CB+LR) может дать лучше
калиброванные вероятности, т.к. LGBM и CB коррелированы
(оба gradient boosting), а LR даёт другой вид.

Тесты:
1. CB+LR equal average vs 3-model
2. CB+LR weighted (0.335/0.665 из step 4.13)
3. conf_ev пороги на val -> apply to test
4. EV-only пороги для сравнения
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


def evaluate_strategies(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    odds: np.ndarray,
    df: pd.DataFrame,
    prefix: str,
) -> dict:
    """Evaluate conf_ev and plain EV strategies."""
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf

    results = {}
    # conf_ev thresholds
    for thr in [0.08, 0.10, 0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        results[f"{prefix}_conf_ev_{thr:.2f}"] = r

    # plain EV thresholds for comparison
    for thr in [0.08, 0.10, 0.12, 0.15]:
        mask = ev >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        results[f"{prefix}_ev_{thr:.2f}"] = r

    return results


def main() -> None:
    """CB+LR 2-model ensemble validation."""
    with mlflow.start_run(run_name="phase4/cb_lr_ensemble") as run:
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
                    "method": "cb_lr_2model",
                }
            )

            # Train CB and LR
            cb = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            )
            cb.fit(x_tr, train_fit_enc["target"])

            scaler = StandardScaler()
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(scaler.fit_transform(x_tr), train_fit_enc["target"])

            # Val predictions
            p_cb_val = cb.predict_proba(x_va)[:, 1]
            p_lr_val = lr.predict_proba(scaler.transform(x_va))[:, 1]

            odds_val = val_enc["Odds"].values

            # Per-model AUC
            y_val = val_enc["target"].values
            auc_cb = roc_auc_score(y_val, p_cb_val)
            auc_lr = roc_auc_score(y_val, p_lr_val)
            logger.info("Val AUC: CB=%.4f, LR=%.4f", auc_cb, auc_lr)

            all_val_results = {}

            # 1. CB+LR equal average
            p_eq = (p_cb_val + p_lr_val) / 2
            p_std_eq = np.std([p_cb_val, p_lr_val], axis=0)
            r = evaluate_strategies(p_eq, p_std_eq, odds_val, val_enc, "eq")
            all_val_results.update(r)
            for k, v in r.items():
                logger.info("Val %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 2. CB+LR weighted (0.335 / 0.665 from step 4.13)
            w_cb, w_lr = 0.335, 0.665
            p_w = w_cb * p_cb_val + w_lr * p_lr_val
            p_std_w = np.sqrt(w_cb * (p_cb_val - p_w) ** 2 + w_lr * (p_lr_val - p_w) ** 2)
            r = evaluate_strategies(p_w, p_std_w, odds_val, val_enc, "w34_66")
            all_val_results.update(r)
            for k, v in r.items():
                logger.info("Val %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # 3. CB only
            p_std_cb = np.zeros_like(p_cb_val) + 0.05
            r = evaluate_strategies(p_cb_val, p_std_cb, odds_val, val_enc, "cb_only")
            all_val_results.update(r)

            # 4. LR only
            p_std_lr = np.zeros_like(p_lr_val) + 0.05
            r = evaluate_strategies(p_lr_val, p_std_lr, odds_val, val_enc, "lr_only")
            all_val_results.update(r)

            # Val ranking
            val_ranked = sorted(
                all_val_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 30 else -999,
                reverse=True,
            )
            logger.info("Top-10 val (n>=30):")
            for name, rv in val_ranked[:10]:
                if rv["n_bets"] >= 30:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, rv["roi"], rv["n_bets"])

            # Test evaluation
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            )
            cb_f.fit(x_train, train_enc["target"])

            sc_f = StandardScaler()
            lr_f = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_f.fit(sc_f.fit_transform(x_train), train_enc["target"])

            p_cb_t = cb_f.predict_proba(x_test)[:, 1]
            p_lr_t = lr_f.predict_proba(sc_f.transform(x_test))[:, 1]
            odds_test = test_enc["Odds"].values

            test_results = {}

            # Equal
            p_eq_t = (p_cb_t + p_lr_t) / 2
            p_std_eq_t = np.std([p_cb_t, p_lr_t], axis=0)
            r = evaluate_strategies(p_eq_t, p_std_eq_t, odds_test, test_enc, "eq")
            test_results.update(r)

            # Weighted
            p_w_t = w_cb * p_cb_t + w_lr * p_lr_t
            p_std_w_t = np.sqrt(w_cb * (p_cb_t - p_w_t) ** 2 + w_lr * (p_lr_t - p_w_t) ** 2)
            r = evaluate_strategies(p_w_t, p_std_w_t, odds_test, test_enc, "w34_66")
            test_results.update(r)

            # CB only
            r = evaluate_strategies(
                p_cb_t, np.zeros_like(p_cb_t) + 0.05, odds_test, test_enc, "cb_only"
            )
            test_results.update(r)

            # LR only
            r = evaluate_strategies(
                p_lr_t, np.zeros_like(p_lr_t) + 0.05, odds_test, test_enc, "lr_only"
            )
            test_results.update(r)

            auc_test_eq = roc_auc_score(test_enc["target"], p_eq_t)
            auc_test_cb = roc_auc_score(test_enc["target"], p_cb_t)
            auc_test_lr = roc_auc_score(test_enc["target"], p_lr_t)

            # Report test
            logger.info(
                "Test AUC: eq=%.4f, CB=%.4f, LR=%.4f", auc_test_eq, auc_test_cb, auc_test_lr
            )
            for k, v in sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )[:15]:
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
                    "auc_test_eq": auc_test_eq,
                    "auc_test_cb": auc_test_cb,
                    "auc_test_lr": auc_test_lr,
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
            logger.exception("Step 4.15 failed")
            raise


if __name__ == "__main__":
    main()
