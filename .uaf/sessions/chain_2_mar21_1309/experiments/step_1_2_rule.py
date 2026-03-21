"""Step 1.2 — Rule-based baseline.

Простое правило: ставить на исходы с ML_P_Model >= порог.
Порог подбирается на val (последние 20% train).
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

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загрузка и split: train, val, test."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    outcomes_agg = outcomes.groupby("Bet_ID").agg(Sport=("Sport", "first")).reset_index()
    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()
    df = df.sort_values("Created_At").reset_index(drop=True)

    for col in ["USD", "Payout_USD", "Odds", "ML_P_Model"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    split_idx = int(len(df) * 0.8)
    train_full = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()

    return train, val, test


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sel = df[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def main():
    train, val, test = load_and_split()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    with mlflow.start_run(run_name="phase1/step_1_2_rule") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.2")
            mlflow.set_tag("phase", "1")

            best_roi = -999.0
            best_thr = 0.5

            for thr in np.arange(0.30, 0.85, 0.05):
                mask = val["ML_P_Model"] >= thr
                r = calc_roi(val, mask.values)
                logger.info("Val thr=%.2f: ROI=%.2f%% (%d bets)", thr, r["roi"], r["n_bets"])
                if r["n_bets"] >= 20 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_thr = thr

            logger.info("Best val threshold: %.2f => ROI=%.2f%%", best_thr, best_roi)

            test_mask = test["ML_P_Model"] >= best_thr
            test_result = calc_roi(test, test_mask.values)
            logger.info(
                "Test: thr=%.2f => ROI=%.2f%% (%d bets)",
                best_thr,
                test_result["roi"],
                test_result["n_bets"],
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "RuleBaseline",
                    "threshold": best_thr,
                    "rule": f"ML_P_Model >= {best_thr:.2f}",
                }
            )

            mlflow.log_metrics(
                {
                    "roi": test_result["roi"],
                    "n_bets": test_result["n_bets"],
                    "threshold": best_thr,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={test_result['roi']}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            logger.exception("Step 1.2 failed")
            raise


if __name__ == "__main__":
    main()
