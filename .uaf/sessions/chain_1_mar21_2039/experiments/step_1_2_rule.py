"""Step 1.2 — Rule-based baseline: ML_Edge threshold."""

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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}


def load_data() -> pd.DataFrame:
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Market"]
    outcomes_first = outcomes_first[safe_cols]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df = df.sort_values("Created_At").reset_index(drop=True)
    return df


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


with mlflow.start_run(run_name="phase1/step1.2_rule") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")

    try:
        budget_file_path = os.environ.get("UAF_BUDGET_STATUS_FILE", "")
        if budget_file_path:
            try:
                budget_status = json.loads(Path(budget_file_path).read_text())
                if budget_status.get("hard_stop"):
                    mlflow.set_tag("status", "budget_stopped")
                    sys.exit(0)
            except FileNotFoundError:
                pass

        df = load_data()
        n = len(df)
        train_end = int(n * 0.8)
        val_start = int(n * 0.64)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:train_end]
        test_df = df.iloc[train_end:]

        # Подбор порога ML_Edge на val
        best_roi_val, best_threshold = -999.0, 8.0
        for t in np.arange(0, 30, 0.5):
            mask_v = (val_df["ML_Edge"] >= t).values
            if mask_v.sum() < 100:
                break
            roi_v, _ = calc_roi(val_df, mask_v)
            if roi_v > best_roi_val:
                best_roi_val = roi_v
                best_threshold = t

        logger.info("Best ML_Edge threshold=%.1f (val ROI=%.2f%%)", best_threshold, best_roi_val)

        mask_test = (test_df["ML_Edge"] >= best_threshold).values
        roi_test, n_test = calc_roi(test_df, mask_test)
        logger.info("Test ROI=%.2f%% (%d ставок)", roi_test, n_test)

        # CV
        cv_rois = []
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n
            fold_val = df.iloc[fold_start:fold_end]
            mask_f = (fold_val["ML_Edge"] >= best_threshold).values
            roi_f, _ = calc_roi(fold_val, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_f)

        roi_mean = float(np.mean(cv_rois))
        roi_std = float(np.std(cv_rois))

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "rule": "ML_Edge >= threshold",
                "threshold": best_threshold,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "roi_test": roi_test,
                "roi_val": best_roi_val,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_test": n_test,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.1")

        print(f"\n=== Step 1.2 Rule-based (ML_Edge >= {best_threshold:.1f}) ===")
        print(f"ROI test: {roi_test:.2f}% ({n_test} ставок)")
        print(f"ROI val:  {best_roi_val:.2f}%")
        print(f"CV ROI:   {roi_mean:.2f}% +/- {roi_std:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
