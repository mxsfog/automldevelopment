"""Step 1.1: Constant baseline (DummyClassifier).

Гипотеза: DummyClassifier (most_frequent) задаёт lower bound для ROI.
При предсказании majority class (won) для всех ставок,
ROI = (sum(Payout_USD для won) - sum(USD)) / sum(USD) * 100.
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
from sklearn.dummy import DummyClassifier

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


def compute_roi(
    y_true: pd.Series, y_pred: np.ndarray, usd: pd.Series, payout_usd: pd.Series
) -> float:
    """ROI на отобранных ставках (где модель предсказала won)."""
    mask = y_pred == "won"
    if mask.sum() == 0:
        return 0.0
    selected_usd = usd[mask]
    selected_payout = payout_usd[mask]
    selected_actual = y_true.values[mask]
    total_staked = selected_usd.sum()
    total_returned = selected_payout[selected_actual == "won"].sum()
    return (total_returned - total_staked) / total_staked * 100


def main() -> None:
    logger.info("Загрузка данных")
    bets = pd.read_csv(DATA_DIR / "bets.csv", low_memory=False)
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv", low_memory=False)

    exclude = ["pending", "cancelled", "error", "cashout"]
    df = bets[~bets["Status"].isin(exclude)].copy()
    logger.info("После фильтрации: %d строк", len(df))

    df["Created_At"] = pd.to_datetime(df["Created_At"])
    df = df.sort_values("Created_At").reset_index(drop=True)

    # Join outcomes для получения Sport, Market
    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(Sport=("Sport", "first"), Market=("Market", "first"))
        .reset_index()
    )
    df = df.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")

    # Time series split: 5 фолдов
    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step1.1_constant_baseline") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("phase", "1")

        try:
            fold_rois = []
            fold_accuracies = []

            for fold_idx in range(n_splits):
                train_end = fold_size * (fold_idx + 1)
                val_start = train_end
                val_end = train_end + fold_size
                if fold_idx == n_splits - 1:
                    val_end = n

                train = df.iloc[:train_end]
                val = df.iloc[val_start:val_end]

                feat_train = train[["Odds"]].values
                y_train = train["Status"].values
                feat_val = val[["Odds"]].values
                y_val = val["Status"].values

                dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
                dummy.fit(feat_train, y_train)
                y_pred = dummy.predict(feat_val)

                roi = compute_roi(
                    val["Status"], y_pred, val["USD"].values, val["Payout_USD"].values
                )
                acc = (y_pred == y_val).mean()

                fold_rois.append(roi)
                fold_accuracies.append(acc)

                mlflow.log_metric(f"roi_fold_{fold_idx}", round(roi, 4))
                mlflow.log_metric(f"accuracy_fold_{fold_idx}", round(acc, 4))
                mlflow.set_tag("fold_idx", str(fold_idx))

                logger.info(
                    "Fold %d: train=%d, val=%d, roi=%.4f, acc=%.4f, pred=%s",
                    fold_idx,
                    len(train),
                    len(val),
                    roi,
                    acc,
                    np.unique(y_pred, return_counts=True),
                )

            roi_mean = np.mean(fold_rois)
            roi_std = np.std(fold_rois)
            acc_mean = np.mean(fold_accuracies)

            mlflow.log_params(
                {
                    "model": "DummyClassifier",
                    "strategy": "most_frequent",
                    "seed": SEED,
                    "validation_scheme": "time_series",
                    "n_splits": n_splits,
                    "n_samples_total": n,
                }
            )

            mlflow.log_metrics(
                {
                    "roi_mean": round(roi_mean, 4),
                    "roi_std": round(roi_std, 4),
                    "accuracy_mean": round(acc_mean, 4),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("ROI mean: %.4f +/- %.4f", roi_mean, roi_std)
            logger.info("Accuracy mean: %.4f", acc_mean)
            logger.info("Run ID: %s", run.info.run_id)

            print(f"RESULT: roi_mean={roi_mean:.4f}, roi_std={roi_std:.4f}")
            print(f"RUN_ID: {run.info.run_id}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception during training")
            logger.exception("Step 1.1 failed")
            raise


if __name__ == "__main__":
    main()
