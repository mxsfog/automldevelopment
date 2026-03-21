"""Step 4.25 -- Kelly criterion bet sizing.

Flat bets give ROI=13.80%. Kelly sizes bets proportionally to edge/odds,
potentially improving bankroll growth. Test: full Kelly, fractional Kelly (25%, 50%),
and EV-proportional sizing.
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
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
)

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

FEATURES_ENC = [
    "Odds", "USD", "Is_Parlay", "Outcomes_Count",
    "ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV",
    "ML_Winrate_Diff", "ML_Rating_Diff",
    "Outcome_Odds", "n_outcomes", "mean_outcome_odds",
    "max_outcome_odds", "min_outcome_odds",
    "Sport_target_enc", "Sport_count_enc",
    "Market_target_enc", "Market_count_enc",
]


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion: f = (p*b - q) / b where b = odds-1, q = 1-p."""
    b = odds - 1
    q = 1 - p
    f = (p * b - q) / b
    return np.clip(f, 0, 1)


def main() -> None:
    """Kelly sizing walk-forward."""
    with mlflow.start_run(run_name="phase4/kelly_sizing") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            df["year_week"] = df["Created_At"].dt.year * 100 + df[
                "Created_At"
            ].dt.isocalendar().week.astype(int)
            unique_weeks = sorted(df["year_week"].unique())
            n_weeks = len(unique_weeks)
            min_train_weeks = int(n_weeks * 0.6)
            retrain_schedule = unique_weeks[min_train_weeks:]

            sizing_methods = ["flat", "kelly_full", "kelly_50", "kelly_25", "ev_prop"]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "kelly_sizing",
                    "n_methods": len(sizing_methods),
                }
            )

            # Track bankroll evolution: start with 1000 units
            bankroll: dict[str, float] = {m: 1000.0 for m in sizing_methods}
            cumulative: dict[str, dict] = {
                m: {"wagered": 0.0, "profit": 0.0} for m in sizing_methods
            }
            all_blocks: list[dict] = []

            for i, test_week in enumerate(retrain_schedule):
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train = train_enc[FEATURES_ENC].fillna(0)
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                cb = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6,
                    random_seed=42, verbose=0,
                )
                cb.fit(x_train, y_train)

                lgbm = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=42, verbose=-1, min_child_samples=50,
                )
                lgbm.fit(x_train, y_train)

                scaler = StandardScaler()
                x_train_s = scaler.fit_transform(x_train)
                x_test_s = scaler.transform(x_test)
                lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr.fit(x_train_s, y_train)

                p_cb = cb.predict_proba(x_test)[:, 1]
                p_lgbm = lgbm.predict_proba(x_test)[:, 1]
                p_lr = lr.predict_proba(x_test_s)[:, 1]
                p_mean = (p_cb + p_lgbm + p_lr) / 3
                p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)

                odds = test_enc["Odds"].values
                ev = p_mean * odds - 1
                mask = (ev >= 0.05) & (p_std <= 0.02)

                n = mask.sum()
                block_info: dict = {"block": i, "week": test_week, "n_selected": int(n)}

                if n > 0:
                    sel_idx = np.where(mask)[0]
                    sel_p = p_mean[sel_idx]
                    sel_odds = odds[sel_idx]
                    sel_ev = ev[sel_idx]
                    sel_target = test_enc["target"].values[sel_idx]
                    sel_won = sel_target.astype(bool)

                    kf = kelly_fraction(sel_p, sel_odds)

                    for method in sizing_methods:
                        if method == "flat":
                            stakes = np.ones(n)
                        elif method == "kelly_full":
                            stakes = kf * bankroll[method] / 100
                        elif method == "kelly_50":
                            stakes = 0.5 * kf * bankroll[method] / 100
                        elif method == "kelly_25":
                            stakes = 0.25 * kf * bankroll[method] / 100
                        elif method == "ev_prop":
                            stakes = sel_ev / sel_ev.mean()
                        else:
                            stakes = np.ones(n)

                        stakes = np.clip(stakes, 0.01, 50)
                        total_wagered = stakes.sum()
                        payout = (sel_won * sel_odds * stakes).sum()
                        profit = payout - total_wagered
                        roi = profit / total_wagered * 100 if total_wagered > 0 else 0.0

                        bankroll[method] += profit
                        cumulative[method]["wagered"] += total_wagered
                        cumulative[method]["profit"] += profit

                        block_info[f"{method}_roi"] = round(roi, 2)
                        block_info[f"{method}_profit"] = round(profit, 2)
                        block_info[f"{method}_bankroll"] = round(bankroll[method], 2)
                else:
                    for method in sizing_methods:
                        block_info[f"{method}_roi"] = 0.0
                        block_info[f"{method}_profit"] = 0.0
                        block_info[f"{method}_bankroll"] = round(bankroll[method], 2)

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nKelly sizing results (ev005_agree_p02):")
            for method in sizing_methods:
                total_w = cumulative[method]["wagered"]
                total_p = cumulative[method]["profit"]
                overall_roi = (total_p / total_w * 100) if total_w > 0 else 0.0

                col = f"{method}_roi"
                n_pos = 0
                if col in results_df.columns:
                    n_pos = (results_df[col].values > 0).sum()

                logger.info(
                    "  %s: overall_roi=%.2f%% bankroll=%.0f profit=%.1f pos=%d/%d",
                    method,
                    overall_roi,
                    bankroll[method],
                    total_p,
                    n_pos,
                    len(all_blocks),
                )
                mlflow.log_metrics(
                    {
                        f"wf_{method}_roi": round(overall_roi, 2),
                        f"wf_{method}_bankroll": round(bankroll[method], 2),
                        f"wf_{method}_profit": round(total_p, 2),
                        f"wf_{method}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "kelly_results.csv")
            results_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.25 failed")
            raise


if __name__ == "__main__":
    main()
