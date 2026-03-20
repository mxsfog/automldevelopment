"""Step 1.2: Rule-based baseline.

Гипотеза: Простое пороговое правило по ML_Edge > 0 отбирает
прибыльные ставки лучше, чем ставить на всё.
Также проверим несколько порогов: 0, 5, 10, 15.
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.warning("Budget hard stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass


def compute_roi_on_selected(df_val: pd.DataFrame, selected_mask: np.ndarray) -> dict[str, float]:
    """ROI и статистика на отобранных ставках."""
    if selected_mask.sum() == 0:
        return {"roi": 0.0, "n_selected": 0, "win_rate": 0.0, "coverage": 0.0}
    selected = df_val[selected_mask]
    total_staked = selected["USD"].sum()
    total_returned = selected.loc[selected["Status"] == "won", "Payout_USD"].sum()
    roi = (total_returned - total_staked) / total_staked * 100
    win_rate = (selected["Status"] == "won").mean()
    coverage = len(selected) / len(df_val)
    return {
        "roi": roi,
        "n_selected": int(selected_mask.sum()),
        "win_rate": win_rate,
        "coverage": coverage,
    }


def main() -> None:
    logger.info("Загрузка данных")
    bets = pd.read_csv(DATA_DIR / "bets.csv", low_memory=False)

    exclude = ["pending", "cancelled", "error", "cashout"]
    df = bets[~bets["Status"].isin(exclude)].copy()
    logger.info("После фильтрации: %d строк", len(df))

    df["Created_At"] = pd.to_datetime(df["Created_At"])
    df = df.sort_values("Created_At").reset_index(drop=True)

    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 1)

    thresholds = [0, 5, 10, 15]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step1.2_rule_based") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")

        try:
            best_threshold = None
            best_roi_mean = -999

            for threshold in thresholds:
                fold_rois = []
                fold_coverages = []
                fold_winrates = []

                for fold_idx in range(n_splits):
                    train_end = fold_size * (fold_idx + 1)
                    val_start = train_end
                    val_end = train_end + fold_size
                    if fold_idx == n_splits - 1:
                        val_end = n

                    val = df.iloc[val_start:val_end].copy()

                    # Правило: ставим где ML_Edge > threshold и ML_Edge не NaN
                    has_ml = val["ML_Edge"].notna()
                    selected = has_ml & (val["ML_Edge"] > threshold)

                    stats = compute_roi_on_selected(val, selected.values)
                    fold_rois.append(stats["roi"])
                    fold_coverages.append(stats["coverage"])
                    fold_winrates.append(stats["win_rate"])

                roi_mean = np.mean(fold_rois)
                roi_std = np.std(fold_rois)
                coverage_mean = np.mean(fold_coverages)
                winrate_mean = np.mean(fold_winrates)

                mlflow.log_metrics(
                    {
                        f"roi_mean_t{threshold}": round(roi_mean, 4),
                        f"roi_std_t{threshold}": round(roi_std, 4),
                        f"coverage_t{threshold}": round(coverage_mean, 4),
                        f"winrate_t{threshold}": round(winrate_mean, 4),
                    }
                )

                logger.info(
                    "Threshold=%d: roi=%.4f +/- %.4f, coverage=%.2f%%, winrate=%.2f%%",
                    threshold,
                    roi_mean,
                    roi_std,
                    coverage_mean * 100,
                    winrate_mean * 100,
                )

                if roi_mean > best_roi_mean:
                    best_roi_mean = roi_mean
                    best_threshold = threshold

            # Логируем лучший порог
            mlflow.log_params(
                {
                    "model": "rule_based",
                    "feature": "ML_Edge",
                    "best_threshold": best_threshold,
                    "seed": SEED,
                    "validation_scheme": "time_series",
                    "n_splits": n_splits,
                    "n_samples_total": n,
                }
            )
            mlflow.log_metrics(
                {
                    "roi_mean": round(best_roi_mean, 4),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Best threshold=%d, roi_mean=%.4f", best_threshold, best_roi_mean)
            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT: best_threshold={best_threshold}, roi_mean={best_roi_mean:.4f}")
            print(f"RUN_ID: {run.info.run_id}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception during rule-based baseline")
            logger.exception("Step 1.2 failed")
            raise


if __name__ == "__main__":
    main()
