"""Step 4.19 -- Random selection baseline.

Вопрос: какой ROI дает случайный отбор ставок (без модели)?
Если random selection дает ~10% ROI, то модель не добавляет ценности.

Метод: для каждого WF блока, случайно отобрать N ставок (N = то же что EV-selected),
повторить 100 раз, взять среднее.
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

N_RANDOM = 100


def main() -> None:
    """Random selection baseline."""
    with mlflow.start_run(run_name="phase4/random_baseline") as run:
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

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "random_selection_baseline",
                    "n_random_trials": N_RANDOM,
                }
            )

            # First: model-based walk-forward to get selection sizes
            model_n_per_block: list[int] = []
            model_cum = {"n": 0, "profit": 0.0}
            all_test_dfs: list[pd.DataFrame] = []

            for test_week in retrain_schedule:
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    model_n_per_block.append(0)
                    all_test_dfs.append(pd.DataFrame())
                    continue

                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train = train_enc[FEATURES_ENC].fillna(0)
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

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

                p_cb = cb.predict_proba(x_test)[:, 1]
                p_lgbm = lgbm.predict_proba(x_test)[:, 1]
                p_lr = lr.predict_proba(x_test_s)[:, 1]
                p_mean = (p_cb + p_lgbm + p_lr) / 3
                p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)

                ev = p_mean * test_enc["Odds"].values - 1
                mask = (ev >= 0.05) & (p_std <= 0.02)
                n_sel = mask.sum()
                model_n_per_block.append(int(n_sel))
                all_test_dfs.append(test_enc.copy())

                if n_sel > 0:
                    sel = test_enc[mask]
                    payout = (sel["target"] * sel["Odds"]).sum()
                    roi = (payout - n_sel) / n_sel * 100
                    model_cum["n"] += n_sel
                    model_cum["profit"] += n_sel * roi / 100

            model_roi = (model_cum["profit"] / model_cum["n"] * 100) if model_cum["n"] > 0 else 0.0
            logger.info("Model ROI (ev005_agree_p02): %.2f%% (n=%d)", model_roi, model_cum["n"])

            # Random baseline: select same N bets randomly per block
            random_rois: list[float] = []
            rng = np.random.RandomState(42)

            for _trial in range(N_RANDOM):
                cum_n = 0
                cum_profit = 0.0
                for block_i, test_enc in enumerate(all_test_dfs):
                    n_sel = model_n_per_block[block_i]
                    if n_sel == 0 or len(test_enc) == 0:
                        continue
                    n_avail = len(test_enc)
                    n_pick = min(n_sel, n_avail)
                    idx = rng.choice(n_avail, size=n_pick, replace=False)
                    sel = test_enc.iloc[idx]
                    payout = (sel["target"] * sel["Odds"]).sum()
                    roi = (payout - n_pick) / n_pick * 100
                    cum_n += n_pick
                    cum_profit += n_pick * roi / 100

                overall = (cum_profit / cum_n * 100) if cum_n > 0 else 0.0
                random_rois.append(overall)

            random_arr = np.array(random_rois)
            p_value = (np.sum(random_arr >= model_roi) + 1) / (N_RANDOM + 1)

            logger.info("\nRandom baseline results:")
            logger.info("  Model ROI: %.2f%%", model_roi)
            logger.info(
                "  Random mean: %.2f%% std: %.2f%% [%.2f, %.2f]",
                random_arr.mean(),
                random_arr.std(),
                random_arr.min(),
                random_arr.max(),
            )
            logger.info("  p-value: %.3f %s", p_value, "SIG" if p_value < 0.05 else "NOT SIG")

            # Also: random at odds<=10
            random_rois_capped: list[float] = []
            for _trial in range(N_RANDOM):
                cum_n = 0
                cum_profit = 0.0
                for block_i, test_enc in enumerate(all_test_dfs):
                    n_sel = model_n_per_block[block_i]
                    if n_sel == 0 or len(test_enc) == 0:
                        continue
                    capped = test_enc[test_enc["Odds"] <= 10]
                    if len(capped) == 0:
                        continue
                    n_pick = min(n_sel, len(capped))
                    idx = rng.choice(len(capped), size=n_pick, replace=False)
                    sel = capped.iloc[idx]
                    payout = (sel["target"] * sel["Odds"]).sum()
                    roi = (payout - n_pick) / n_pick * 100
                    cum_n += n_pick
                    cum_profit += n_pick * roi / 100

                overall = (cum_profit / cum_n * 100) if cum_n > 0 else 0.0
                random_rois_capped.append(overall)

            random_capped = np.array(random_rois_capped)
            logger.info("\nRandom baseline (odds<=10):")
            logger.info(
                "  Random mean: %.2f%% std: %.2f%% [%.2f, %.2f]",
                random_capped.mean(),
                random_capped.std(),
                random_capped.min(),
                random_capped.max(),
            )

            mlflow.log_metrics(
                {
                    "model_roi": round(model_roi, 2),
                    "random_mean": round(float(random_arr.mean()), 2),
                    "random_std": round(float(random_arr.std()), 2),
                    "random_p_value": round(float(p_value), 4),
                    "random_capped_mean": round(float(random_capped.mean()), 2),
                    "random_capped_std": round(float(random_capped.std()), 2),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.19 failed")
            raise


if __name__ == "__main__":
    main()
