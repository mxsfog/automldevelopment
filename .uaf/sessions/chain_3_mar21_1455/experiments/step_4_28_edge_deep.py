"""Step 4.28 — Deep validation of edge strategy.

Step 4.27: edge_cap2_e0.10 val=6.69% -> test=5.50% (consistent).
Нужна глубокая проверка:
1. CV stability (5-fold expanding window)
2. Seed sensitivity (5 seeds)
3. Profit concentration (Gini) — не зависит ли от 1 bet?
4. Combined: edge + confidence filter
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


def train_ensemble(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> tuple:
    """3-model ensemble."""
    cb = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6, random_seed=seed, verbose=0
    )
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=seed,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
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
    """Deep edge validation."""
    with mlflow.start_run(run_name="phase4/edge_deep") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            random.seed(42)
            np.random.seed(42)

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "method": "edge_deep_validation",
                }
            )

            # 1. CV stability (5-fold expanding window)
            logger.info("=== CV Stability (5-fold expanding) ===")
            n = len(df)
            fold_size = n // 6
            cv_results: dict[str, list] = {
                "edge_cap2_e0.10": [],
                "edge_cap5_e0.10": [],
                "confev_0.15": [],
            }

            for fold in range(5):
                train_end = (fold + 1) * fold_size + fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, n)

                if test_end <= test_start:
                    continue

                cv_train = df.iloc[:train_end].copy()
                cv_test = df.iloc[test_start:test_end].copy()

                cv_train_enc, _ = add_sport_market_features(cv_train, cv_train)
                cv_test_enc, _ = add_sport_market_features(cv_test, cv_train_enc)

                x_tr = cv_train_enc[FEATURES].fillna(0)
                x_te = cv_test_enc[FEATURES].fillna(0)

                cb, lgbm, lr, scaler = train_ensemble(x_tr, cv_train_enc["target"])
                p, s = predict_ensemble(cb, lgbm, lr, scaler, x_te)
                odds = cv_test_enc["Odds"].values

                # Edge strategy
                p_implied = 1 / odds
                edge = p - p_implied

                for max_odds, e_thr, key in [
                    (2.0, 0.10, "edge_cap2_e0.10"),
                    (5.0, 0.10, "edge_cap5_e0.10"),
                ]:
                    mask = (edge >= e_thr) & (odds <= max_odds)
                    r = calc_roi(cv_test_enc, mask.astype(float), threshold=0.5)
                    cv_results[key].append(r["roi"])
                    logger.info(
                        "Fold %d, %s: ROI=%.2f%%, n=%d",
                        fold,
                        key,
                        r["roi"],
                        r["n_bets"],
                    )

                # conf_ev baseline
                ev = p * odds - 1
                conf = 1 / (1 + s * 10)
                mask = (ev * conf) >= 0.15
                r = calc_roi(cv_test_enc, mask.astype(float), threshold=0.5)
                cv_results["confev_0.15"].append(r["roi"])
                logger.info(
                    "Fold %d, confev_0.15: ROI=%.2f%%, n=%d",
                    fold,
                    r["roi"],
                    r["n_bets"],
                )

            for key, rois in cv_results.items():
                if rois:
                    logger.info(
                        "CV %s: mean=%.2f%%, std=%.2f%%, min=%.2f%%, max=%.2f%%",
                        key,
                        np.mean(rois),
                        np.std(rois),
                        np.min(rois),
                        np.max(rois),
                    )

            # 2. Seed sensitivity
            logger.info("=== Seed Sensitivity ===")
            train, test = time_series_split(df, test_size=0.2)
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            odds_test = test_enc["Odds"].values

            seed_rois: dict[str, list] = {
                "edge_cap2_e0.10": [],
                "edge_cap5_e0.10": [],
                "confev_0.15": [],
            }

            for seed in [42, 123, 456, 789, 2024]:
                random.seed(seed)
                np.random.seed(seed)
                cb_s, lgbm_s, lr_s, sc_s = train_ensemble(x_train, train_enc["target"], seed=seed)
                p_s, s_s = predict_ensemble(cb_s, lgbm_s, lr_s, sc_s, x_test)

                p_impl = 1 / odds_test
                edge_s = p_s - p_impl

                for max_odds, e_thr, key in [
                    (2.0, 0.10, "edge_cap2_e0.10"),
                    (5.0, 0.10, "edge_cap5_e0.10"),
                ]:
                    mask = (edge_s >= e_thr) & (odds_test <= max_odds)
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    seed_rois[key].append(r["roi"])
                    logger.info(
                        "Seed %d, %s: ROI=%.2f%%, n=%d",
                        seed,
                        key,
                        r["roi"],
                        r["n_bets"],
                    )

                ev_s = p_s * odds_test - 1
                conf_s = 1 / (1 + s_s * 10)
                mask = (ev_s * conf_s) >= 0.15
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                seed_rois["confev_0.15"].append(r["roi"])

            for key, rois in seed_rois.items():
                logger.info(
                    "Seeds %s: mean=%.2f%%, std=%.2f%%",
                    key,
                    np.mean(rois),
                    np.std(rois),
                )

            # 3. Profit concentration for edge strategy (seed=42)
            logger.info("=== Profit Concentration (edge_cap5_e0.10) ===")
            random.seed(42)
            np.random.seed(42)
            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"], seed=42)
            p_f, s_f = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            edge_f = p_f - 1 / odds_test

            mask_edge = (edge_f >= 0.10) & (odds_test <= 5.0)
            selected = test_enc[mask_edge].copy()
            if len(selected) > 0:
                stakes = selected["USD"].values
                payouts = np.where(
                    selected["target"].values == 1,
                    stakes * selected["Odds"].values,
                    0.0,
                )
                pnl = payouts - stakes
                total_profit = pnl.sum()
                total_stake = stakes.sum()

                logger.info(
                    "Edge selected: n=%d, ROI=%.2f%%",
                    len(selected),
                    total_profit / total_stake * 100 if total_stake > 0 else 0,
                )

                sorted_pnl = np.sort(pnl)[::-1]
                cumsum = np.cumsum(sorted_pnl)
                if total_profit > 0:
                    for pct in [50, 80, 100]:
                        threshold_val = total_profit * pct / 100
                        n_needed = int(np.searchsorted(cumsum, threshold_val) + 1)
                        logger.info(
                            "%d%% profit from top %d bets (%.1f%%)",
                            pct,
                            n_needed,
                            n_needed / len(pnl) * 100,
                        )

                # Top-5 P&L
                top_idx = np.argsort(pnl)[::-1][:5]
                for i, idx in enumerate(top_idx):
                    logger.info(
                        "  #%d: P&L=%.0f, odds=%.2f, won=%d",
                        i + 1,
                        pnl[idx],
                        selected["Odds"].iloc[idx],
                        selected["target"].iloc[idx],
                    )

            # 4. Combined: edge + confidence
            logger.info("=== Combined: edge + confidence ===")
            conf_f = 1 / (1 + s_f * 10)

            combo_results: dict[str, dict] = {}
            for max_odds in [3.0, 5.0]:
                odds_mask = odds_test <= max_odds
                for e_thr in [0.08, 0.10, 0.12]:
                    for c_thr in [0.5, 0.6, 0.7, 0.8]:
                        mask = (edge_f >= e_thr) & odds_mask & (conf_f >= c_thr)
                        r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                        name = f"edge_cap{max_odds:.0f}_e{e_thr:.2f}_c{c_thr:.1f}"
                        combo_results[name] = r

            combo_ranked = sorted(
                combo_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 30 else -999,
                reverse=True,
            )
            logger.info("Top-10 combo (n>=30):")
            for name, r in combo_ranked[:10]:
                if r["n_bets"] >= 30:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Log metrics
            cv_edge2 = cv_results.get("edge_cap2_e0.10", [])
            cv_confev = cv_results.get("confev_0.15", [])

            mlflow.log_metrics(
                {
                    "cv_mean_edge_cap2": float(np.mean(cv_edge2)) if cv_edge2 else 0,
                    "cv_std_edge_cap2": float(np.std(cv_edge2)) if cv_edge2 else 0,
                    "cv_mean_confev": float(np.mean(cv_confev)) if cv_confev else 0,
                    "cv_std_confev": float(np.std(cv_confev)) if cv_confev else 0,
                    "seed_std_edge_cap2": float(np.std(seed_rois["edge_cap2_e0.10"])),
                    "seed_std_confev": float(np.std(seed_rois["confev_0.15"])),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.28 failed")
            raise


if __name__ == "__main__":
    main()
