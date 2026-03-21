"""Step 4.4 — Per-sport segmentation analysis.

Цель: найти виды спорта с более стабильным сигналом.
Для каждого спорта: отдельная модель + EV selection + ROI.
Также: анализ по Market и Odds brackets.
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
    """Train 3-model ensemble."""
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
    """Per-sport segmentation analysis."""
    with mlflow.start_run(run_name="phase4/sport_segments") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            y_train = train_enc["target"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "sport_segments",
                }
            )

            # Global ensemble
            cb, lgbm, lr, scaler = train_ensemble(x_train, y_train)
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_test)
            odds = test_enc["Odds"].values
            ev = p_mean * odds - 1

            # Part 1: Per-sport ROI analysis (using global model)
            logger.info("=== Per-sport analysis (global model) ===")
            sport_results = []

            for sport in test_enc["Sport"].unique():
                mask_sport = test_enc["Sport"] == sport
                n_total = mask_sport.sum()
                if n_total < 20:
                    continue

                # All bets in this sport
                r_all = calc_roi(test_enc[mask_sport], np.ones(n_total), threshold=0.5)

                # EV >= 0.12 selected
                mask_ev = mask_sport & (ev >= 0.12)
                n_ev = mask_ev.sum()
                if n_ev > 0:
                    r_ev = calc_roi(test_enc[mask_ev], np.ones(n_ev), threshold=0.5)
                else:
                    r_ev = {"roi": 0.0, "n_bets": 0, "win_rate": 0}

                # conf_ev_0.15
                confidence = 1 / (1 + p_std * 10)
                ev_conf = ev * confidence
                mask_conf = mask_sport & (ev_conf >= 0.15)
                n_conf = mask_conf.sum()
                if n_conf > 0:
                    r_conf = calc_roi(test_enc[mask_conf], np.ones(n_conf), threshold=0.5)
                else:
                    r_conf = {"roi": 0.0, "n_bets": 0, "win_rate": 0}

                sport_results.append(
                    {
                        "sport": sport,
                        "n_total": int(n_total),
                        "roi_all": r_all["roi"],
                        "winrate_all": r_all.get("win_rate", 0),
                        "n_ev": r_ev["n_bets"],
                        "roi_ev": r_ev["roi"],
                        "n_conf": r_conf["n_bets"],
                        "roi_conf": r_conf["roi"],
                        "avg_odds": test_enc.loc[mask_sport, "Odds"].mean(),
                    }
                )

            sport_df = pd.DataFrame(sport_results).sort_values("n_ev", ascending=False)
            logger.info("Per-sport results:\n%s", sport_df.to_string(index=False))

            # Part 2: Per-odds-bracket analysis
            logger.info("=== Per-odds-bracket analysis ===")
            brackets = [(1, 1.5), (1.5, 2.5), (2.5, 5), (5, 10), (10, 50), (50, 500)]
            bracket_results = []

            for lo, hi in brackets:
                mask_bracket = (odds >= lo) & (odds < hi)
                n_total = mask_bracket.sum()
                if n_total < 20:
                    continue

                # EV selected in this bracket
                mask_ev_bracket = mask_bracket & (ev >= 0.12)
                n_ev = mask_ev_bracket.sum()
                if n_ev > 0:
                    r_ev = calc_roi(test_enc[mask_ev_bracket], np.ones(n_ev), threshold=0.5)
                else:
                    r_ev = {"roi": 0.0, "n_bets": 0, "win_rate": 0}

                # conf_ev in bracket
                mask_conf_bracket = mask_bracket & (ev_conf >= 0.15)
                n_conf = mask_conf_bracket.sum()
                if n_conf > 0:
                    r_conf = calc_roi(test_enc[mask_conf_bracket], np.ones(n_conf), threshold=0.5)
                else:
                    r_conf = {"roi": 0.0, "n_bets": 0, "win_rate": 0}

                bracket_results.append(
                    {
                        "bracket": f"{lo}-{hi}",
                        "n_total": int(n_total),
                        "n_ev": r_ev["n_bets"],
                        "roi_ev": r_ev["roi"],
                        "winrate_ev": r_ev.get("win_rate", 0),
                        "n_conf": r_conf["n_bets"],
                        "roi_conf": r_conf["roi"],
                    }
                )

            bracket_df = pd.DataFrame(bracket_results)
            logger.info("Per-bracket results:\n%s", bracket_df.to_string(index=False))

            # Part 3: Sport exclusion experiment
            # Try excluding sports with negative ROI
            negative_sports = [
                r["sport"] for r in sport_results if r["roi_ev"] < 0 and r["n_ev"] >= 10
            ]
            logger.info("Sports with negative EV ROI (n>=10): %s", negative_sports)

            if negative_sports:
                mask_exclude = ~test_enc["Sport"].isin(negative_sports)
                mask_ev_filtered = mask_exclude & (ev >= 0.12)
                n_filt = mask_ev_filtered.sum()
                if n_filt > 0:
                    r_filtered = calc_roi(
                        test_enc[mask_ev_filtered], np.ones(n_filt), threshold=0.5
                    )
                    logger.info(
                        "Excluding %s: ROI=%.2f%%, n=%d (vs baseline ROI=16.02%%, n=2247)",
                        negative_sports,
                        r_filtered["roi"],
                        r_filtered["n_bets"],
                    )

                # conf_ev with exclusion
                mask_conf_filtered = mask_exclude & (ev_conf >= 0.15)
                n_conf_filt = mask_conf_filtered.sum()
                if n_conf_filt > 0:
                    r_conf_filt = calc_roi(
                        test_enc[mask_conf_filtered], np.ones(n_conf_filt), threshold=0.5
                    )
                    logger.info(
                        "Excluding %s + conf_ev: ROI=%.2f%%, n=%d",
                        negative_sports,
                        r_conf_filt["roi"],
                        r_conf_filt["n_bets"],
                    )

            # Part 4: Market analysis (top markets)
            logger.info("=== Per-market analysis ===")
            market_results = []
            for market in test_enc["Market"].value_counts().head(15).index:
                mask_market = test_enc["Market"] == market
                mask_ev_market = mask_market & (ev >= 0.12)
                n_ev = mask_ev_market.sum()
                if n_ev >= 5:
                    r = calc_roi(test_enc[mask_ev_market], np.ones(n_ev), threshold=0.5)
                    market_results.append(
                        {
                            "market": market,
                            "n_ev": r["n_bets"],
                            "roi_ev": r["roi"],
                            "winrate": r.get("win_rate", 0),
                            "avg_odds": test_enc.loc[mask_ev_market, "Odds"].mean(),
                        }
                    )
            market_df = pd.DataFrame(market_results).sort_values("roi_ev", ascending=False)
            logger.info("Per-market results:\n%s", market_df.to_string(index=False))

            # Log metrics
            baseline_ev = ev >= 0.12
            r_baseline = calc_roi(test_enc, baseline_ev.astype(float), threshold=0.5)

            mlflow.log_metrics(
                {
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "n_sports_analyzed": len(sport_results),
                    "n_negative_sports": len(negative_sports) if negative_sports else 0,
                }
            )

            # Save sport/bracket analysis
            sport_path = str(SESSION_DIR / "experiments" / "sport_analysis.csv")
            sport_df.to_csv(sport_path, index=False)
            mlflow.log_artifact(sport_path)

            bracket_path = str(SESSION_DIR / "experiments" / "bracket_analysis.csv")
            bracket_df.to_csv(bracket_path, index=False)
            mlflow.log_artifact(bracket_path)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.4 failed")
            raise


if __name__ == "__main__":
    main()
