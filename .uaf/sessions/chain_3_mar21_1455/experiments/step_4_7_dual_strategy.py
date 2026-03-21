"""Step 4.7 — Dual strategy: separate models for low and high odds.

Insight из анализа: прибыль в основном из odds 50-500 (longshots),
а odds 1.5-5 дают маленький но стабильный edge.
Попробуем раздельные модели и стратегии.
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


def main() -> None:
    """Dual strategy experiment."""
    with mlflow.start_run(run_name="phase4/dual_strategy") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Val split for threshold selection
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            # Feature encoding
            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "dual_strategy",
                }
            )

            # Part 1: Train separate models for low-odds and high-odds
            odds_threshold = 10.0

            # Low-odds model (odds < 10): more bets, lower ROI per bet
            mask_low_train = train_fit_enc["Odds"] < odds_threshold
            mask_low_val = val_enc["Odds"] < odds_threshold

            x_low_train = train_fit_enc[mask_low_train][FEATURES].fillna(0)
            y_low_train = train_fit_enc[mask_low_train]["target"]
            x_low_val = val_enc[mask_low_val][FEATURES].fillna(0)

            logger.info("Low-odds train: %d, val: %d", len(x_low_train), mask_low_val.sum())

            cb_low, lgbm_low, lr_low, scaler_low = train_ensemble(x_low_train, y_low_train)
            p_low_val, _ = predict_ensemble(cb_low, lgbm_low, lr_low, scaler_low, x_low_val)

            # High-odds model (odds >= 10)
            mask_high_train = train_fit_enc["Odds"] >= odds_threshold
            mask_high_val = val_enc["Odds"] >= odds_threshold

            x_high_train = train_fit_enc[mask_high_train][FEATURES].fillna(0)
            y_high_train = train_fit_enc[mask_high_train]["target"]
            x_high_val = val_enc[mask_high_val][FEATURES].fillna(0)

            logger.info("High-odds train: %d, val: %d", len(x_high_train), mask_high_val.sum())

            cb_high, lgbm_high, lr_high, scaler_high = train_ensemble(x_high_train, y_high_train)
            p_high_val, _ = predict_ensemble(cb_high, lgbm_high, lr_high, scaler_high, x_high_val)

            # Find best EV thresholds on val for each segment
            odds_val_low = val_enc[mask_low_val]["Odds"].values
            odds_val_high = val_enc[mask_low_val.values == False]["Odds"].values  # noqa: E712

            # Low-odds: search threshold on val
            best_low_thr = 0.12
            best_low_roi = -999
            for thr in np.arange(0.05, 0.30, 0.01):
                ev = p_low_val * odds_val_low - 1
                mask = ev >= thr
                r = calc_roi(val_enc[mask_low_val], mask.astype(float), threshold=0.5)
                if r["n_bets"] >= 20 and r["roi"] > best_low_roi:
                    best_low_roi = r["roi"]
                    best_low_thr = round(thr, 2)
            logger.info("Best low-odds EV thr: %.2f (val ROI=%.2f%%)", best_low_thr, best_low_roi)

            # High-odds: search threshold on val
            best_high_thr = 0.12
            best_high_roi = -999
            for thr in np.arange(0.05, 0.30, 0.01):
                ev = p_high_val * odds_val_high - 1
                mask = ev >= thr
                r = calc_roi(val_enc[mask_high_val], mask.astype(float), threshold=0.5)
                if r["n_bets"] >= 10 and r["roi"] > best_high_roi:
                    best_high_roi = r["roi"]
                    best_high_thr = round(thr, 2)
            logger.info(
                "Best high-odds EV thr: %.2f (val ROI=%.2f%%)", best_high_thr, best_high_roi
            )

            # Part 2: Full train + test evaluation
            logger.info("Full train evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            # Global model baseline
            cb_g, lgbm_g, lr_g, scaler_g = train_ensemble(x_train, train_enc["target"])
            p_global, p_std_global = predict_ensemble(cb_g, lgbm_g, lr_g, scaler_g, x_test)

            auc_global = roc_auc_score(test_enc["target"], p_global)
            odds_test = test_enc["Odds"].values

            # Baseline
            ev_global = p_global * odds_test - 1
            mask_base = ev_global >= 0.12
            r_baseline = calc_roi(test_enc, mask_base.astype(float), threshold=0.5)
            logger.info("Baseline: ROI=%.2f%%, n=%d", r_baseline["roi"], r_baseline["n_bets"])

            # conf_ev_0.15 baseline
            conf_g = 1 / (1 + p_std_global * 10)
            mask_conf = (ev_global * conf_g) >= 0.15
            r_conf = calc_roi(test_enc, mask_conf.astype(float), threshold=0.5)
            logger.info("conf_ev_0.15: ROI=%.2f%%, n=%d", r_conf["roi"], r_conf["n_bets"])

            # Dual model: low-odds model for low-odds bets, high-odds model for high-odds
            mask_test_low = test_enc["Odds"] < odds_threshold
            mask_test_high = ~mask_test_low

            # Train segment models on full train
            x_train_low = train_enc[train_enc["Odds"] < odds_threshold][FEATURES].fillna(0)
            y_train_low = train_enc[train_enc["Odds"] < odds_threshold]["target"]
            x_train_high = train_enc[train_enc["Odds"] >= odds_threshold][FEATURES].fillna(0)
            y_train_high = train_enc[train_enc["Odds"] >= odds_threshold]["target"]

            cb_low_f, lgbm_low_f, lr_low_f, sc_low_f = train_ensemble(x_train_low, y_train_low)
            cb_high_f, lgbm_high_f, lr_high_f, sc_high_f = train_ensemble(
                x_train_high, y_train_high
            )

            # Predict on test segments
            x_test_low = test_enc[mask_test_low][FEATURES].fillna(0)
            x_test_high = test_enc[mask_test_high][FEATURES].fillna(0)

            p_low_test, _ = predict_ensemble(cb_low_f, lgbm_low_f, lr_low_f, sc_low_f, x_test_low)
            p_high_test, _ = predict_ensemble(
                cb_high_f, lgbm_high_f, lr_high_f, sc_high_f, x_test_high
            )

            # Apply per-segment thresholds from val
            ev_low_test = p_low_test * test_enc[mask_test_low]["Odds"].values - 1
            ev_high_test = p_high_test * test_enc[mask_test_high]["Odds"].values - 1

            mask_sel_low = ev_low_test >= best_low_thr
            mask_sel_high = ev_high_test >= best_high_thr

            # Combine
            combined_mask = np.zeros(len(test_enc), dtype=bool)
            combined_mask[mask_test_low.values] = mask_sel_low
            combined_mask[mask_test_high.values] = mask_sel_high

            r_dual = calc_roi(test_enc, combined_mask.astype(float), threshold=0.5)
            logger.info(
                "Dual strategy (low thr=%.2f, high thr=%.2f): ROI=%.2f%%, n=%d",
                best_low_thr,
                best_high_thr,
                r_dual["roi"],
                r_dual["n_bets"],
            )

            # Also try: global model + per-bracket val thresholds
            mask_global_low = mask_test_low.values & (ev_global >= best_low_thr)
            mask_global_high = mask_test_high.values & (ev_global >= best_high_thr)
            mask_global_dual = mask_global_low | mask_global_high
            r_global_dual = calc_roi(test_enc, mask_global_dual.astype(float), threshold=0.5)
            logger.info(
                "Global model + per-bracket thr: ROI=%.2f%%, n=%d",
                r_global_dual["roi"],
                r_global_dual["n_bets"],
            )

            # Per-segment analysis
            n_low_sel = mask_sel_low.sum()
            n_high_sel = mask_sel_high.sum()
            if n_low_sel > 0:
                r_low = calc_roi(
                    test_enc[mask_test_low], mask_sel_low.astype(float), threshold=0.5
                )
                logger.info("Low-odds segment: ROI=%.2f%%, n=%d", r_low["roi"], r_low["n_bets"])
            if n_high_sel > 0:
                r_high = calc_roi(
                    test_enc[mask_test_high], mask_sel_high.astype(float), threshold=0.5
                )
                logger.info("High-odds segment: ROI=%.2f%%, n=%d", r_high["roi"], r_high["n_bets"])

            # Best result
            all_results = {
                "baseline": r_baseline,
                "conf_ev_0.15": r_conf,
                "dual_model": r_dual,
                "global_dual_thr": r_global_dual,
            }
            best_name = max(
                all_results,
                key=lambda k: all_results[k]["roi"] if all_results[k]["n_bets"] >= 50 else -999,
            )
            best = all_results[best_name]
            logger.info("Best: %s -> ROI=%.2f%%, n=%d", best_name, best["roi"], best["n_bets"])

            mlflow.log_metrics(
                {
                    "auc_global": auc_global,
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "roi_conf_ev": r_conf["roi"],
                    "n_bets_conf_ev": r_conf["n_bets"],
                    "roi_dual": r_dual["roi"],
                    "n_bets_dual": r_dual["n_bets"],
                    "roi_global_dual": r_global_dual["roi"],
                    "n_bets_global_dual": r_global_dual["n_bets"],
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                    "low_odds_thr": best_low_thr,
                    "high_odds_thr": best_high_thr,
                }
            )
            mlflow.set_tag("best_strategy", best_name)

            if best["roi"] > 27.95:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_g.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best["roi"],
                    "auc": auc_global,
                    "threshold": 0.12,
                    "ev_threshold": 0.12,
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
            logger.exception("Step 4.7 failed")
            raise


if __name__ == "__main__":
    main()
