"""Step 4.23 — Odds floor + conf_ev: min odds filter.

Step 4.21-22 показали что profit из high-odds (парлаи). Гипотеза:
вместо dual thresholds, простой фильтр odds >= K в комбинации с conf_ev.
Также тестируем odds-band selection: выбираем только определенный диапазон odds.
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
    """Odds floor experiment."""
    with mlflow.start_run(run_name="phase4/odds_floor") as run:
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
                    "method": "odds_floor",
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_va)

            odds_val = val_enc["Odds"].values
            ev = p_mean * odds_val - 1
            conf = 1 / (1 + p_std * 10)
            ev_conf = ev * conf

            all_val: dict[str, dict] = {}

            # 1. Baseline conf_ev
            for thr in [0.12, 0.15, 0.18]:
                mask = ev_conf >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                all_val[f"baseline_{thr:.2f}"] = r
                logger.info(
                    "Val baseline_%s: ROI=%.2f%%, n=%d", f"{thr:.2f}", r["roi"], r["n_bets"]
                )

            # 2. Odds floor + conf_ev: select only bets with odds >= min_odds AND conf_ev >= thr
            for min_odds in [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
                for thr in [0.08, 0.10, 0.12, 0.15]:
                    mask = (ev_conf >= thr) & (odds_val >= min_odds)
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"floor_{min_odds:.0f}_thr_{thr:.2f}"
                    all_val[name] = r

            # 3. Odds band: select only bets within odds range
            bands = [(2.0, 10.0), (10.0, 50.0), (50.0, 500.0), (3.0, 30.0), (5.0, 100.0)]
            for lo, hi in bands:
                for thr in [0.08, 0.10, 0.12, 0.15]:
                    mask = (ev_conf >= thr) & (odds_val >= lo) & (odds_val <= hi)
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"band_{lo:.0f}_{hi:.0f}_thr_{thr:.2f}"
                    all_val[name] = r

            # 4. Odds ceiling: exclude very high odds (potential outliers)
            for max_odds in [100.0, 200.0, 500.0]:
                for thr in [0.12, 0.15]:
                    mask = (ev_conf >= thr) & (odds_val <= max_odds)
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    name = f"ceil_{max_odds:.0f}_thr_{thr:.2f}"
                    all_val[name] = r

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

            auc_test = roc_auc_score(test_enc["target"], p_t)

            ev_t = p_t * odds_test - 1
            conf_t = 1 / (1 + s_t * 10)
            ev_conf_t = ev_t * conf_t

            test_results: dict[str, dict] = {}

            # Baseline
            for thr in [0.12, 0.15, 0.18]:
                mask = ev_conf_t >= thr
                test_results[f"baseline_{thr:.2f}"] = calc_roi(
                    test_enc, mask.astype(float), threshold=0.5
                )

            # Top val strategies on test
            for name, _rv in val_ranked[:15]:
                if _rv["n_bets"] < 20 or name.startswith("baseline_"):
                    continue
                if name.startswith("floor_"):
                    parts = name.split("_")
                    min_odds = float(parts[1])
                    thr = float(parts[3])
                    mask = (ev_conf_t >= thr) & (odds_test >= min_odds)
                    test_results[name] = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                elif name.startswith("band_"):
                    parts = name.split("_")
                    lo = float(parts[1])
                    hi = float(parts[2])
                    thr = float(parts[4])
                    mask = (ev_conf_t >= thr) & (odds_test >= lo) & (odds_test <= hi)
                    test_results[name] = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                elif name.startswith("ceil_"):
                    parts = name.split("_")
                    max_odds = float(parts[1])
                    thr = float(parts[3])
                    mask = (ev_conf_t >= thr) & (odds_test <= max_odds)
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

            # Odds distribution analysis
            for _thr_name in ["baseline_0.15"]:
                mask = ev_conf_t >= 0.15
                selected = test_enc[mask]
                if len(selected) > 0:
                    odds_sel = selected["Odds"].values
                    logger.info(
                        "Selected odds stats (conf_ev>=0.15): "
                        "median=%.1f, mean=%.1f, min=%.1f, max=%.1f",
                        np.median(odds_sel),
                        np.mean(odds_sel),
                        np.min(odds_sel),
                        np.max(odds_sel),
                    )
                    for pct in [25, 50, 75, 90, 95]:
                        logger.info("  P%d odds = %.1f", pct, np.percentile(odds_sel, pct))

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
            logger.exception("Step 4.23 failed")
            raise


if __name__ == "__main__":
    main()
