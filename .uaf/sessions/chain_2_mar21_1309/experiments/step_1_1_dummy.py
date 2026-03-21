"""Step 1.1 — Constant baseline (DummyClassifier).

Устанавливает lower bound ROI: что будет, если ставить на все подряд.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка данных и time-series split."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(Sport=("Sport", "first"), Market=("Market", "first"))
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()
    df = df.sort_values("Created_At").reset_index(drop=True)

    for col in ["USD", "Payout_USD", "Odds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def calc_roi(df: pd.DataFrame) -> dict:
    """ROI на всех ставках (без фильтрации)."""
    total_staked = df["USD"].sum()
    total_payout = df.loc[df["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(df)}


def main():
    train, test = load_and_split()
    logger.info("Train: %d, Test: %d", len(train), len(test))

    with mlflow.start_run(run_name="phase1/step_1_1_dummy") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.1")
            mlflow.set_tag("phase", "1")

            train_roi = calc_roi(train)
            test_roi = calc_roi(test)

            win_rate_train = (train["Status"] == "won").mean()
            win_rate_test = (test["Status"] == "won").mean()

            logger.info(
                "Train ROI: %.2f%% (%d bets), winrate: %.4f",
                train_roi["roi"],
                train_roi["n_bets"],
                win_rate_train,
            )
            logger.info(
                "Test ROI: %.2f%% (%d bets), winrate: %.4f",
                test_roi["roi"],
                test_roi["n_bets"],
                win_rate_test,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "model": "DummyClassifier",
                    "method": "predict_all_positive",
                }
            )

            mlflow.log_metrics(
                {
                    "roi": test_roi["roi"],
                    "n_bets": test_roi["n_bets"],
                    "win_rate_train": win_rate_train,
                    "win_rate_test": win_rate_test,
                    "roi_train": train_roi["roi"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={test_roi['roi']}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.1")
            logger.exception("Step 1.1 failed")
            raise


if __name__ == "__main__":
    main()
