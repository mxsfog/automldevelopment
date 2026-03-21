"""Step 4.24 — Robustness analysis: profit concentration + seed sensitivity.

Step 4.23 показал что profit из extreme high-odds (band 50-500 = 124% ROI).
Вопросы:
1. Сколько отдельных ставок создают всю прибыль? (Gini concentration)
2. Если изменить seed модели (42 -> другие), ROI стабилен?
3. Какие конкретные ставки дают максимальный P&L?
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
    """3-model ensemble with configurable seed."""
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


def conf_ev_select(
    p_mean: np.ndarray, p_std: np.ndarray, odds: np.ndarray, threshold: float
) -> np.ndarray:
    """conf_ev selection mask."""
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    return (ev * conf) >= threshold


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of values (0=equal, 1=concentrated)."""
    if len(values) == 0:
        return 0.0
    sorted_v = np.sort(np.abs(values))
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_v) / (n * np.sum(sorted_v))) - (n + 1) / n)


def main() -> None:
    """Robustness analysis."""
    with mlflow.start_run(run_name="phase4/robustness") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            random.seed(42)
            np.random.seed(42)

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "method": "robustness_analysis",
                }
            )

            # 1. Profit concentration analysis (seed=42)
            logger.info("=== Profit Concentration Analysis ===")
            cb, lgbm, lr, scaler = train_ensemble(x_train, train_enc["target"], seed=42)
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_test)
            odds_test = test_enc["Odds"].values

            mask = conf_ev_select(p_mean, p_std, odds_test, 0.15)
            selected = test_enc[mask].copy()

            # P&L per bet
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
                "Selected: n=%d, total_stake=%.0f, total_profit=%.0f",
                len(selected),
                total_stake,
                total_profit,
            )
            logger.info("ROI=%.2f%%", total_profit / total_stake * 100 if total_stake > 0 else 0)

            # Top winners
            top_idx = np.argsort(pnl)[::-1]
            logger.info("Top-10 individual bet P&L:")
            cumulative_pct = 0.0
            for i, idx in enumerate(top_idx[:10]):
                pct = pnl[idx] / total_profit * 100 if total_profit > 0 else 0
                cumulative_pct += pct
                logger.info(
                    "  #%d: P&L=%.0f (%.1f%% of total), odds=%.1f, won=%d, cumul=%.1f%%",
                    i + 1,
                    pnl[idx],
                    pct,
                    selected["Odds"].iloc[idx],
                    selected["target"].iloc[idx],
                    cumulative_pct,
                )

            # How many bets account for 50%, 80%, 100% of profit
            sorted_pnl = np.sort(pnl)[::-1]
            cumsum = np.cumsum(sorted_pnl)
            for pct_target in [50, 80, 90, 100]:
                if total_profit > 0:
                    threshold_val = total_profit * pct_target / 100
                    n_needed = int(np.searchsorted(cumsum, threshold_val) + 1)
                    logger.info(
                        "%d%% of profit from top %d bets (%.1f%% of selected)",
                        pct_target,
                        n_needed,
                        n_needed / len(pnl) * 100,
                    )

            gini = gini_coefficient(pnl)
            logger.info("Gini coefficient of P&L: %.3f", gini)

            # Win rate by odds brackets
            logger.info("Win rate and ROI by odds bracket:")
            brackets = [(1, 3), (3, 10), (10, 50), (50, 200), (200, 10000)]
            for lo, hi in brackets:
                bracket_mask = (selected["Odds"].values >= lo) & (selected["Odds"].values < hi)
                n_bracket = bracket_mask.sum()
                if n_bracket > 0:
                    wr = selected["target"].values[bracket_mask].mean()
                    bracket_pnl = pnl[bracket_mask].sum()
                    bracket_stake = stakes[bracket_mask].sum()
                    bracket_roi = bracket_pnl / bracket_stake * 100 if bracket_stake > 0 else 0
                    logger.info(
                        "  odds [%d,%d): n=%d, wr=%.1f%%, ROI=%.1f%%, P&L=%.0f",
                        lo,
                        hi,
                        n_bracket,
                        wr * 100,
                        bracket_roi,
                        bracket_pnl,
                    )

            # 2. Seed sensitivity
            logger.info("=== Seed Sensitivity (5 seeds) ===")
            seed_results = []
            for seed in [42, 123, 456, 789, 2024]:
                random.seed(seed)
                np.random.seed(seed)
                cb_s, lgbm_s, lr_s, sc_s = train_ensemble(x_train, train_enc["target"], seed=seed)
                p_s, s_s = predict_ensemble(cb_s, lgbm_s, lr_s, sc_s, x_test)

                auc = roc_auc_score(test_enc["target"], p_s)
                for thr in [0.12, 0.15, 0.18]:
                    m = conf_ev_select(p_s, s_s, odds_test, thr)
                    r = calc_roi(test_enc, m.astype(float), threshold=0.5)
                    seed_results.append(
                        {
                            "seed": seed,
                            "threshold": thr,
                            "roi": r["roi"],
                            "n_bets": r["n_bets"],
                            "auc": auc,
                        }
                    )
                    logger.info(
                        "Seed %d, thr=%.2f: ROI=%.2f%%, n=%d, AUC=%.4f",
                        seed,
                        thr,
                        r["roi"],
                        r["n_bets"],
                        auc,
                    )

            # Summary stats per threshold
            for thr in [0.12, 0.15, 0.18]:
                rois = [r["roi"] for r in seed_results if r["threshold"] == thr]
                logger.info(
                    "Threshold %.2f across seeds: mean=%.2f%%, std=%.2f%%, min=%.2f%%, max=%.2f%%",
                    thr,
                    np.mean(rois),
                    np.std(rois),
                    np.min(rois),
                    np.max(rois),
                )

            # Log best results
            best_seed42 = next(
                r for r in seed_results if r["seed"] == 42 and r["threshold"] == 0.15
            )

            mlflow.log_metrics(
                {
                    "auc_test": best_seed42["auc"],
                    "roi_best": best_seed42["roi"],
                    "n_bets_best": best_seed42["n_bets"],
                    "gini_pnl": gini,
                    "roi_seed_std_015": float(
                        np.std([r["roi"] for r in seed_results if r["threshold"] == 0.15])
                    ),
                }
            )
            mlflow.set_tag("best_strategy", "conf_ev_0.15_seed_analysis")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.24 failed")
            raise


if __name__ == "__main__":
    main()
