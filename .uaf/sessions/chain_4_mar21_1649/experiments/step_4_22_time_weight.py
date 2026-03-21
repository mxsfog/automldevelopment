"""Step 4.22 -- Time-weighted training (address covariate shift).

Step 4.21 показал adv AUC=0.878 (значительный shift). Гипотеза:
придание большего веса недавним данным при обучении уменьшит shift
и улучшит предсказания для test блока.

Варианты: exponential decay, linear decay, last-N-weeks upweight.
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


def compute_sample_weights(
    train_dates: pd.Series, test_date: pd.Timestamp, method: str
) -> np.ndarray:
    """Compute sample weights based on temporal proximity to test."""
    days_diff = (test_date - train_dates).dt.days.values.astype(float)
    max_days = days_diff.max()

    if method == "uniform":
        return np.ones(len(days_diff))
    elif method == "linear":
        # Linear decay: recent data weight=1, oldest weight=0.1
        return 0.1 + 0.9 * (1 - days_diff / max_days)
    elif method == "exp_fast":
        # Exponential: half-life 14 days
        return np.exp(-0.693 * days_diff / 14)
    elif method == "exp_slow":
        # Exponential: half-life 30 days
        return np.exp(-0.693 * days_diff / 30)
    elif method == "last_2w_2x":
        # Last 2 weeks get 2x weight
        return np.where(days_diff <= 14, 2.0, 1.0)
    else:
        return np.ones(len(days_diff))


def main() -> None:
    """Time-weighted training walk-forward."""
    with mlflow.start_run(run_name="phase4/time_weighted") as run:
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

            weight_methods = ["uniform", "linear", "exp_fast", "exp_slow", "last_2w_2x"]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "time_weighted_training",
                    "n_weight_methods": len(weight_methods),
                }
            )

            cumulative: dict[str, dict] = {
                m: {"n": 0, "profit": 0.0} for m in weight_methods
            }
            all_blocks = []

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

                test_date = test_df["Created_At"].max()

                block_info = {"block": i, "week": test_week, "n_test": len(test_df)}

                for w_method in weight_methods:
                    weights = compute_sample_weights(
                        train_enc["Created_At"], test_date, w_method
                    )
                    weights = weights / weights.mean()  # normalize

                    cb = CatBoostClassifier(
                        iterations=200,
                        learning_rate=0.05,
                        depth=6,
                        random_seed=42,
                        verbose=0,
                    )
                    cb.fit(x_train, y_train, sample_weight=weights)

                    lgbm = LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                        verbose=-1,
                        min_child_samples=50,
                    )
                    lgbm.fit(x_train, y_train, sample_weight=weights)

                    scaler = StandardScaler()
                    x_train_s = scaler.fit_transform(x_train)
                    x_test_s = scaler.transform(x_test)
                    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                    lr.fit(x_train_s, y_train, sample_weight=weights)

                    p_cb = cb.predict_proba(x_test)[:, 1]
                    p_lgbm = lgbm.predict_proba(x_test)[:, 1]
                    p_lr = lr.predict_proba(x_test_s)[:, 1]
                    p_mean = (p_cb + p_lgbm + p_lr) / 3
                    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)

                    odds = test_enc["Odds"].values
                    ev = p_mean * odds - 1
                    mask = (ev >= 0.05) & (p_std <= 0.02)

                    n = mask.sum()
                    roi = 0.0
                    if n > 0:
                        sel = test_enc[mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                        cumulative[w_method]["n"] += n
                        cumulative[w_method]["profit"] += n * roi / 100

                    block_info[f"{w_method}_roi"] = round(roi, 2)
                    block_info[f"{w_method}_n"] = int(n)

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nTime-weighted training results (ev005_agree_p02):")
            for w_method in weight_methods:
                total_n = cumulative[w_method]["n"]
                total_profit = cumulative[w_method]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"{w_method}_roi"
                n_pos = 0
                if col in results_df.columns:
                    n_pos = (results_df[col].values > 0).sum()

                logger.info(
                    "  %s: overall=%.2f%% (n=%d) pos=%d/%d",
                    w_method,
                    overall_roi,
                    total_n,
                    n_pos,
                    len(all_blocks),
                )
                mlflow.log_metrics(
                    {
                        f"wf_{w_method}_roi": round(overall_roi, 2),
                        f"wf_{w_method}_n": total_n,
                        f"wf_{w_method}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "time_weight_results.csv")
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
            logger.exception("Step 4.22 failed")
            raise


if __name__ == "__main__":
    main()
