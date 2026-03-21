"""Step 1.2 — Rule-based baseline: ML_Edge > threshold."""

import json
import logging
import os
import random
import sys
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
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass


def load_data() -> pd.DataFrame:
    """Загрузка данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)
    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """ROI на выбранных ставках."""
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def find_threshold_on_val(val_df: pd.DataFrame, min_bets: int = 200) -> tuple[float, float]:
    """Подбор порога ML_Edge на val."""
    best_roi, best_t = -999.0, 0.0
    for t in np.arange(0.0, 50.0, 0.5):
        mask = (val_df["ML_Edge"].fillna(-999) >= t) & (val_df["lead_hours"] > 0)
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def main() -> None:
    """Rule: фильтр по ML_Edge с порогом из val."""
    with mlflow.start_run(run_name="phase1/step1.2_rule_edge") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            logger.info("Val: %d, Test: %d", len(val_df), len(test_df))

            t_best, roi_val = find_threshold_on_val(val_df)

            mask_test = (test_df["ML_Edge"].fillna(-999) >= t_best) & (test_df["lead_hours"] > 0)
            roi_test, n_bets = calc_roi(test_df, mask_test)

            logger.info(
                "Rule ML_Edge >= %.1f: val=%.2f%%, test=%.2f%% (%d bets)",
                t_best,
                roi_val,
                roi_test,
                n_bets,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "model": "rule_ml_edge",
                    "threshold": t_best,
                }
            )
            mlflow.log_metrics(
                {
                    "roi_val": float(roi_val),
                    "roi_test": float(roi_test),
                    "n_bets": n_bets,
                    "edge_threshold": float(t_best),
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT roi_val={roi_val:.2f}% roi_test={roi_test:.2f}% "
                f"n_bets={n_bets} threshold={t_best:.1f} run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
