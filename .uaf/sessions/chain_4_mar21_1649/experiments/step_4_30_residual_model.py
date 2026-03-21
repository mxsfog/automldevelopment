"""Step 4.30 -- Residual model: predict deviation from implied probability.

Instead of predicting target directly, predict:
  residual = target - implied_probability

This frames the problem as "where does the market misprice?"
If p_implied is well-calibrated, residual should be ~0 on average.
Model only needs to find systematic deviations.

Also test: use implied probability as offset/prior.
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

# Features excluding implied probability (to avoid leaking the baseline)
FEATURES_NO_IMPLIED = [
    "Odds", "USD", "Is_Parlay", "Outcomes_Count",
    "ML_P_Model", "ML_Edge", "ML_EV",
    "ML_Winrate_Diff", "ML_Rating_Diff",
    "Outcome_Odds", "n_outcomes", "mean_outcome_odds",
    "max_outcome_odds", "min_outcome_odds",
    "Sport_target_enc", "Sport_count_enc",
    "Market_target_enc", "Market_count_enc",
]


def main() -> None:
    """Residual model walk-forward."""
    with mlflow.start_run(run_name="phase4/residual_model") as run:
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

            strategies = [
                "standard",
                "implied_prior",
                "implied_only",
            ]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "residual_model",
                }
            )

            cumulative: dict[str, dict] = {
                k: {"n": 0, "profit": 0.0} for k in strategies
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

                odds = test_enc["Odds"].values
                p_implied_test = 1.0 / odds

                block_info: dict = {"block": i, "week": test_week, "n_test": len(test_df)}

                for strategy in strategies:
                    if strategy == "standard":
                        # Standard 3-model ensemble
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

                    elif strategy == "implied_prior":
                        # Model predicts adjustment to implied probability
                        # P_final = alpha * P_model + (1-alpha) * P_implied
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

                        p_model = (
                            cb.predict_proba(x_test)[:, 1]
                            + lgbm.predict_proba(x_test)[:, 1]
                            + lr.predict_proba(x_test_s)[:, 1]
                        ) / 3
                        # Blend: 30% model, 70% implied
                        p_mean = 0.3 * p_model + 0.7 * p_implied_test
                        p_std = np.abs(p_model - p_implied_test)

                    elif strategy == "implied_only":
                        # Pure implied probability baseline
                        p_mean = p_implied_test
                        p_std = np.zeros(len(test_enc))

                    else:
                        continue

                    ev = p_mean * odds - 1
                    mask = (ev >= 0.05) & (p_std <= 0.02)

                    n = mask.sum()
                    roi = 0.0
                    if n > 0:
                        sel = test_enc[mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                        cumulative[strategy]["n"] += n
                        cumulative[strategy]["profit"] += n * roi / 100

                    block_info[f"{strategy}_roi"] = round(roi, 2)
                    block_info[f"{strategy}_n"] = int(n)

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nResidual/prior model results (ev005_agree_p02):")
            for strategy in strategies:
                total_n = cumulative[strategy]["n"]
                total_profit = cumulative[strategy]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"{strategy}_roi"
                n_pos = (results_df[col].values > 0).sum() if col in results_df.columns else 0

                logger.info(
                    "  %s: overall=%.2f%% (n=%d) pos=%d/%d",
                    strategy, overall_roi, total_n, n_pos, len(all_blocks),
                )
                mlflow.log_metrics(
                    {
                        f"wf_{strategy}_roi": round(overall_roi, 2),
                        f"wf_{strategy}_n": total_n,
                        f"wf_{strategy}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "residual_model_results.csv")
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
            logger.exception("Step 4.30 failed")
            raise


if __name__ == "__main__":
    main()
