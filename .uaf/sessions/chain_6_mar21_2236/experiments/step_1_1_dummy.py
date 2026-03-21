"""Step 1.1 — DummyClassifier: нижняя граница ROI."""

import json
import logging
import os
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

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
    """Загрузка и объединение данных."""
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


def main() -> None:
    """DummyClassifier: ставим на всё."""
    with mlflow.start_run(run_name="phase1/step1.1_dummy") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

            # DummyClassifier — ставим на все ставки
            y_tr = (train_df["Status"] == "won").astype(int)
            y_te = (test_df["Status"] == "won").astype(int)

            clf = DummyClassifier(strategy="most_frequent", random_state=42)
            clf.fit(train_df[["USD"]], y_tr)
            pt = clf.predict_proba(test_df[["USD"]])[:, 1]

            # Ставим на всё (нет фильтрации)
            mask_all = np.ones(len(test_df), dtype=bool)
            roi_all, n_bets_all = calc_roi(test_df, mask_all)

            # Pre-match только
            pm_test = (test_df["lead_hours"] > 0).values
            roi_pm, n_bets_pm = calc_roi(test_df, pm_test)

            try:
                auc = roc_auc_score(y_te, pt)
            except Exception:
                auc = 0.5

            logger.info(
                "Dummy: roi_all=%.2f%% (%d bets), roi_prematch=%.2f%% (%d bets)",
                roi_all,
                n_bets_all,
                roi_pm,
                n_bets_pm,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "model": "dummy_most_frequent",
                }
            )
            mlflow.log_metrics(
                {
                    "auc_test": float(auc),
                    "roi_test": float(roi_all),
                    "roi_prematch": float(roi_pm),
                    "n_bets": n_bets_all,
                    "n_bets_prematch": n_bets_pm,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT roi_test={roi_all:.2f}% n_bets={n_bets_all} "
                f"roi_pm={roi_pm:.2f}% n_bets_pm={n_bets_pm} run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
