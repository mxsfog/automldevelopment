"""Step 4.35 -- Seed stability for 5-model blend50.

Quick check: is 5m_blend50 (30.37%) robust across seeds?
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
)

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

FEATURES_ENC = [
    "Odds", "USD", "Is_Parlay", "Outcomes_Count",
    "ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV",
    "ML_Winrate_Diff", "ML_Rating_Diff",
    "Outcome_Odds", "n_outcomes", "mean_outcome_odds",
    "max_outcome_odds", "min_outcome_odds",
    "Sport_target_enc", "Sport_count_enc",
    "Market_target_enc", "Market_count_enc",
]

SEEDS = [42, 123, 777, 2024, 31337]


def run_wf(df: pd.DataFrame, retrain_schedule: list, seed: int) -> dict:
    """Single WF with 5m_blend50."""
    random.seed(seed)
    np.random.seed(seed)

    cum_n = 0
    cum_profit = 0.0
    n_pos = 0

    for test_week in retrain_schedule:
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
            random_seed=seed, verbose=0,
        )
        cb.fit(x_train, y_train)

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=seed, verbose=-1, min_child_samples=50,
        )
        lgbm.fit(x_train, y_train)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        lr.fit(x_train_s, y_train)

        xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=seed, verbosity=0, min_child_weight=50,
            use_label_encoder=False, eval_metric="logloss",
        )
        xgb.fit(x_train, y_train)

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=500, random_state=seed, early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(x_train_s, y_train)

        preds = [
            cb.predict_proba(x_test)[:, 1],
            lgbm.predict_proba(x_test)[:, 1],
            lr.predict_proba(x_test_s)[:, 1],
            xgb.predict_proba(x_test)[:, 1],
            mlp.predict_proba(x_test_s)[:, 1],
        ]
        p_model = np.mean(preds, axis=0)
        p_std = np.std(preds, axis=0)

        odds = test_enc["Odds"].values
        p_implied = 1.0 / odds
        p_final = 0.5 * p_model + 0.5 * p_implied

        ev = p_final * odds - 1
        mask = (ev >= 0.05) & (p_std <= 0.02)

        n = mask.sum()
        if n > 0:
            sel = test_enc[mask]
            payout = (sel["target"] * sel["Odds"]).sum()
            roi = (payout - n) / n * 100
            cum_n += n
            cum_profit += n * roi / 100
            if roi > 0:
                n_pos += 1

    overall_roi = (cum_profit / cum_n * 100) if cum_n > 0 else 0.0
    return {"roi": round(overall_roi, 2), "n": cum_n, "n_pos": n_pos}


def main() -> None:
    """5m_blend50 seed stability."""
    with mlflow.start_run(run_name="phase4/5model_seeds") as run:
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

            mlflow.log_params({
                "validation_scheme": "walk_forward",
                "method": "5model_seed_stability",
                "seeds": str(SEEDS),
            })

            rois = []
            for seed in SEEDS:
                res = run_wf(df, retrain_schedule, seed)
                rois.append(res["roi"])
                logger.info(
                    "  seed=%d: roi=%.2f%% n=%d pos=%d",
                    seed, res["roi"], res["n"], res["n_pos"],
                )

            mean_roi = np.mean(rois)
            std_roi = np.std(rois)
            logger.info(
                "\n5m_blend50: mean=%.2f%% std=%.2f%% [%.1f - %.1f]",
                mean_roi, std_roi, min(rois), max(rois),
            )

            mlflow.log_metrics({
                "seed_5m_blend50_mean": round(mean_roi, 2),
                "seed_5m_blend50_std": round(std_roi, 2),
                "seed_5m_blend50_min": round(min(rois), 2),
                "seed_5m_blend50_max": round(max(rois), 2),
            })

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.35 failed")
            raise


if __name__ == "__main__":
    main()
