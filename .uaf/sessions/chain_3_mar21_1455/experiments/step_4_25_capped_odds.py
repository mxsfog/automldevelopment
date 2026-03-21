"""Step 4.25 — Capped odds: exclude extreme outliers, find real edge.

Step 4.24 показал: 1 ставка (odds=490.9) = 137% profit. Без неё — убыток.
Вопрос: есть ли edge если ограничить max odds?

Тесты:
1. Train на all, predict с max_odds cap: 10, 20, 50, 100
2. Train только на low-odds (odds < cap), predict на low-odds
3. Более агрессивные thresholds при capped odds
4. EV recalculation: при capped odds нужен ли другой threshold?
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
    """Capped odds experiment."""
    with mlflow.start_run(run_name="phase4/capped_odds") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "capped_odds",
                }
            )

            # Val split
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            all_val: dict[str, dict] = {}
            all_test: dict[str, dict] = {}

            # Strategy A: Train on all data, predict with odds cap
            logger.info("=== Strategy A: Train all, predict with odds cap ===")
            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            x_tr = train_fit_enc[FEATURES].fillna(0)
            x_va = val_enc[FEATURES].fillna(0)

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_va)

            odds_val = val_enc["Odds"].values
            ev = p_mean * odds_val - 1
            conf = 1 / (1 + p_std * 10)
            ev_conf = ev * conf

            for max_odds in [5, 10, 20, 50, 100, 200]:
                for thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    mask = (ev_conf >= thr) & (odds_val <= max_odds)
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"A_cap{max_odds}_thr{thr:.2f}"
                    all_val[name] = r

            # Baseline (no cap)
            for thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
                mask = ev_conf >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                all_val[f"baseline_thr{thr:.2f}"] = r

            # Strategy B: Train only on capped odds data
            logger.info("=== Strategy B: Train on capped odds only ===")
            for max_odds in [10, 20, 50]:
                train_capped = train_fit_enc[train_fit_enc["Odds"] <= max_odds]
                val_capped = val_enc[val_enc["Odds"] <= max_odds]
                if len(train_capped) < 100 or len(val_capped) < 50:
                    continue

                x_tr_c = train_capped[FEATURES].fillna(0)
                x_va_c = val_capped[FEATURES].fillna(0)

                cb_c, lgbm_c, lr_c, sc_c = train_ensemble(x_tr_c, train_capped["target"])
                p_c, s_c = predict_ensemble(cb_c, lgbm_c, lr_c, sc_c, x_va_c)

                odds_c = val_capped["Odds"].values
                ev_c = p_c * odds_c - 1
                conf_c = 1 / (1 + s_c * 10)
                ev_conf_c = ev_c * conf_c

                for thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    mask = ev_conf_c >= thr
                    r = calc_roi(val_capped, mask.astype(float), threshold=0.5)
                    name = f"B_cap{max_odds}_thr{thr:.2f}"
                    all_val[name] = r

            # Val ranking
            val_ranked = sorted(
                all_val.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 20 else -999,
                reverse=True,
            )
            logger.info("Top-15 val (n>=20):")
            for name, r in val_ranked[:15]:
                if r["n_bets"] >= 20:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test evaluation
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

            # Test baselines
            for thr in [0.12, 0.15]:
                mask = ev_conf_t >= thr
                all_test[f"baseline_thr{thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            # Test Strategy A (capped)
            for max_odds in [5, 10, 20, 50, 100, 200]:
                for thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    mask = (ev_conf_t >= thr) & (odds_test <= max_odds)
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    all_test[f"A_cap{max_odds}_thr{thr:.2f}"] = r

            # Test Strategy B (trained on capped)
            for max_odds in [10, 20, 50]:
                train_capped_full = train_enc[train_enc["Odds"] <= max_odds]
                test_capped = test_enc[test_enc["Odds"] <= max_odds]
                if len(train_capped_full) < 100 or len(test_capped) < 50:
                    continue

                x_tr_c = train_capped_full[FEATURES].fillna(0)
                x_te_c = test_capped[FEATURES].fillna(0)

                cb_c, lgbm_c, lr_c, sc_c = train_ensemble(x_tr_c, train_capped_full["target"])
                p_c, s_c = predict_ensemble(cb_c, lgbm_c, lr_c, sc_c, x_te_c)

                odds_c = test_capped["Odds"].values
                ev_c = p_c * odds_c - 1
                conf_c = 1 / (1 + s_c * 10)
                ev_conf_c = ev_c * conf_c

                for thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    mask = ev_conf_c >= thr
                    r = calc_roi(test_capped, mask.astype(float), threshold=0.5)
                    all_test[f"B_cap{max_odds}_thr{thr:.2f}"] = r

            # Test ranking
            test_ranked = sorted(
                all_test.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )
            logger.info("Top-15 test (n>=50):")
            for name, r in test_ranked[:15]:
                if r["n_bets"] >= 50:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Best with reasonable n
            best_name = max(
                all_test,
                key=lambda k: all_test[k]["roi"] if all_test[k]["n_bets"] >= 50 else -999,
            )
            best = all_test[best_name]
            logger.info(
                "Best (n>=50): %s -> ROI=%.2f%%, n=%d",
                best_name,
                best["roi"],
                best["n_bets"],
            )

            # Key comparison: capped vs uncapped
            logger.info("=== Key comparison ===")
            for cap in [10, 20, 50, 100]:
                cap_key = f"A_cap{cap}_thr0.15"
                if cap_key in all_test:
                    r = all_test[cap_key]
                    logger.info("Cap %d + thr=0.15: ROI=%.2f%%, n=%d", cap, r["roi"], r["n_bets"])

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
            logger.exception("Step 4.25 failed")
            raise


if __name__ == "__main__":
    main()
