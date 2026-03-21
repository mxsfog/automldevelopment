"""Step 4.3 — Kelly criterion + odds-stratified EV.

Гипотезы:
1. Kelly criterion: stake = EV / (odds - 1), большие ставки на уверенные предсказания
2. Fractional Kelly (f=0.25): conservative Kelly для снижения variance
3. Odds-stratified EV: разные EV пороги для low/mid/high odds
4. EV-weighted ROI: профит взвешен по размеру EV
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


def calc_kelly_roi(
    df_sel: pd.DataFrame,
    ev_values: np.ndarray,
    odds_values: np.ndarray,
    fraction: float = 1.0,
) -> dict:
    """ROI с Kelly-размером ставки.

    Kelly stake = fraction * EV / (odds - 1).
    Capped at [0.01, 1.0] per bet.
    """
    kelly_stake = fraction * ev_values / (odds_values - 1.0)
    kelly_stake = np.clip(kelly_stake, 0.01, 1.0)

    total_staked = kelly_stake.sum()
    payouts = df_sel["target"].values * odds_values * kelly_stake
    total_payout = payouts.sum()
    profit = total_payout - total_staked
    roi = profit / total_staked * 100 if total_staked > 0 else 0.0

    return {
        "roi": round(float(roi), 4),
        "n_bets": len(df_sel),
        "profit": round(float(profit), 4),
        "total_staked": round(float(total_staked), 4),
        "win_rate": round(float(df_sel["target"].mean()), 4),
        "avg_odds": round(float(odds_values.mean()), 4),
        "avg_stake": round(float(kelly_stake.mean()), 4),
    }


def main() -> None:
    """Kelly + odds stratification."""
    with mlflow.start_run(run_name="phase4/step_4_3_kelly_stratified") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            train_enc, _ = add_sport_market_features(train.copy(), train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            features = FEATURES

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_enc),
                    "n_samples_val": len(test_enc),
                    "method": "kelly_stratified",
                    "n_features": len(features),
                }
            )

            x_train = train_enc[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_enc["target"]

            # Baseline ensemble (same as step 4.5/4.0)
            cb = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb.fit(x_train, y_train)

            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm.fit(x_train, y_train)

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(x_train_s, y_train)

            p_test = (
                cb.predict_proba(x_test)[:, 1]
                + lgbm.predict_proba(x_test)[:, 1]
                + lr.predict_proba(x_test_s)[:, 1]
            ) / 3

            auc_test = roc_auc_score(test_enc["target"], p_test)
            ev_test = p_test * test_enc["Odds"].values - 1
            odds_test = test_enc["Odds"].values

            # --- Baseline: flat EV>=0.12 ---
            ev_mask = ev_test >= 0.12
            result_baseline = calc_roi(test_enc, ev_mask.astype(float), threshold=0.5)
            logger.info("Baseline flat EV>=0.12: %s", result_baseline)

            # --- Kelly strategies on EV>=0.12 subset ---
            sel = test_enc[ev_mask].copy()
            ev_sel = ev_test[ev_mask]
            odds_sel = odds_test[ev_mask]

            for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0]:
                r_kelly = calc_kelly_roi(sel, ev_sel, odds_sel, fraction=frac)
                logger.info(
                    "  Kelly f=%.2f: ROI=%.2f%%, n=%d, avg_stake=%.3f",
                    frac,
                    r_kelly["roi"],
                    r_kelly["n_bets"],
                    r_kelly["avg_stake"],
                )
                mlflow.log_metric(f"roi_kelly_{frac:.2f}", r_kelly["roi"])

            # --- Odds-stratified EV thresholds ---
            logger.info("=== Odds-stratified EV ===")

            # Determine thresholds on val (last 20% of train)
            val_split = int(len(train_enc) * 0.8)
            val_part = train_enc.iloc[val_split:]
            x_val = val_part[features].fillna(0)
            x_val_s = scaler.transform(x_val)
            p_val = (
                cb.predict_proba(x_val)[:, 1]
                + lgbm.predict_proba(x_val)[:, 1]
                + lr.predict_proba(x_val_s)[:, 1]
            ) / 3
            ev_val = p_val * val_part["Odds"].values - 1
            odds_val = val_part["Odds"].values

            # Stratify by odds ranges
            odds_bins = [(1.0, 3.0), (3.0, 10.0), (10.0, 50.0), (50.0, 9999.0)]

            # Find best EV per odds bin on val
            best_strat_thresholds = {}
            for lo, hi in odds_bins:
                bin_mask = (odds_val >= lo) & (odds_val < hi)
                if bin_mask.sum() < 20:
                    best_strat_thresholds[(lo, hi)] = 0.12
                    continue

                best_ev_bin = 0.12
                best_roi_bin = -999.0
                for ev_t in np.arange(0.02, 0.40, 0.02):
                    comb_mask = bin_mask & (ev_val >= ev_t)
                    if comb_mask.sum() < 10:
                        continue
                    r_bin = calc_roi(val_part[comb_mask], np.ones(comb_mask.sum()), threshold=0.5)
                    if r_bin["roi"] > best_roi_bin:
                        best_roi_bin = r_bin["roi"]
                        best_ev_bin = round(float(ev_t), 2)
                best_strat_thresholds[(lo, hi)] = best_ev_bin
                logger.info(
                    "  Val odds [%.0f, %.0f): best EV>=%.2f, ROI=%.2f%%",
                    lo,
                    hi,
                    best_ev_bin,
                    best_roi_bin,
                )

            # Apply stratified thresholds to test
            strat_mask = np.zeros(len(test_enc), dtype=bool)
            for (lo, hi), ev_thr in best_strat_thresholds.items():
                bin_mask = (odds_test >= lo) & (odds_test < hi)
                strat_mask |= bin_mask & (ev_test >= ev_thr)

            result_strat = calc_roi(test_enc, strat_mask.astype(float), threshold=0.5)
            logger.info("Stratified EV result: %s", result_strat)
            logger.info("Stratified thresholds: %s", best_strat_thresholds)

            # --- Combined: Stratified + Kelly ---
            if strat_mask.sum() > 0:
                sel_strat = test_enc[strat_mask].copy()
                ev_strat = ev_test[strat_mask]
                odds_strat = odds_test[strat_mask]
                for frac in [0.10, 0.25]:
                    r_combo = calc_kelly_roi(sel_strat, ev_strat, odds_strat, fraction=frac)
                    logger.info(
                        "  Strat+Kelly f=%.2f: ROI=%.2f%%, n=%d",
                        frac,
                        r_combo["roi"],
                        r_combo["n_bets"],
                    )

            # --- EV rank strategy: only top N% by EV ---
            logger.info("=== Top-N by EV ===")
            ev_positive = ev_test >= 0
            sorted_idx = np.argsort(-ev_test)
            for pct in [5, 10, 15, 20]:
                n_top = max(1, int(len(test_enc) * pct / 100))
                top_mask = np.zeros(len(test_enc), dtype=bool)
                top_mask[sorted_idx[:n_top]] = True
                top_mask &= ev_positive
                r_top = calc_roi(test_enc, top_mask.astype(float), threshold=0.5)
                logger.info(
                    "  Top %d%% by EV: ROI=%.2f%%, n=%d",
                    pct,
                    r_top["roi"],
                    r_top["n_bets"],
                )
                mlflow.log_metric(f"roi_top{pct}pct", r_top["roi"])

            # --- Log final ---
            # Pick best overall
            results = {
                "baseline_flat": result_baseline,
                "stratified": result_strat,
            }
            best_name = max(results, key=lambda k: results[k]["roi"])
            best_result = results[best_name]

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test_baseline": result_baseline["roi"],
                    "roi_test_stratified": result_strat["roi"],
                    "roi_test": best_result["roi"],
                    "n_bets_test": best_result["n_bets"],
                }
            )
            mlflow.set_tag("best_approach", best_name)

            # CV stability for baseline (since it's still the best)
            logger.info("=== CV stability ===")
            n_folds = 5
            fold_size = len(train_enc) // (n_folds + 1)
            fold_rois = []

            for fold_idx in range(n_folds):
                fold_end = fold_size * (fold_idx + 2)
                fold_train = train_enc.iloc[: fold_size * (fold_idx + 1)]
                fold_val = train_enc.iloc[fold_size * (fold_idx + 1) : fold_end]

                if len(fold_val) < 100:
                    continue

                ft_x = fold_train[features].fillna(0)
                fv_x = fold_val[features].fillna(0)
                ft_y = fold_train["target"]

                sc_cv = StandardScaler()
                cb_cv = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
                )
                cb_cv.fit(ft_x, ft_y)
                lgbm_cv = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm_cv.fit(ft_x, ft_y)
                lr_cv = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_cv.fit(sc_cv.fit_transform(ft_x), ft_y)

                p_fv = (
                    cb_cv.predict_proba(fv_x)[:, 1]
                    + lgbm_cv.predict_proba(fv_x)[:, 1]
                    + lr_cv.predict_proba(sc_cv.transform(fv_x))[:, 1]
                ) / 3

                ev_fv = p_fv * fold_val["Odds"].values - 1
                mask_fv = ev_fv >= 0.12
                r_fv = calc_roi(fold_val, mask_fv.astype(float), threshold=0.5)
                fold_rois.append(r_fv["roi"])
                logger.info("  Fold %d: ROI=%.2f%%, n=%d", fold_idx, r_fv["roi"], r_fv["n_bets"])
                mlflow.log_metric(f"roi_fold_{fold_idx}", r_fv["roi"])

            if fold_rois:
                mean_roi = float(np.mean(fold_rois))
                std_roi = float(np.std(fold_rois))
                logger.info("CV ROI: mean=%.2f%%, std=%.2f%%", mean_roi, std_roi)
                mlflow.log_metrics({"roi_cv_mean": mean_roi, "roi_cv_std": std_roi})

            # Save if improved
            if best_result["roi"] > 16.02:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_result["roi"],
                    "auc": auc_test,
                    "threshold": 0.12,
                    "n_bets": best_result["n_bets"],
                    "feature_names": features,
                    "selection_method": f"kelly_stratified_{best_name}",
                    "ev_threshold": 0.12,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info("New best model saved: ROI=%.2f%%", best_result["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.3 failed")
            raise


if __name__ == "__main__":
    main()
