"""Step 1.1 — DummyClassifier (most_frequent) — нижняя граница ROI.

Гипотеза: предсказание всех ставок как won / всех как lost задаёт lower bound.
"""

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
    df = df.sort_values("Created_At").reset_index(drop=True)
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
    """Основной эксперимент."""
    with mlflow.start_run(run_name="phase1/step1.1_dummy") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            logger.info("Train: %d, Test: %d", len(train_df), len(test_df))
            logger.info(
                "Win rate train: %.2f%%",
                (train_df["Status"] == "won").mean() * 100,
            )
            logger.info(
                "Win rate test: %.2f%%",
                (test_df["Status"] == "won").mean() * 100,
            )

            y_tr = (train_df["Status"] == "won").astype(int)

            # Dummy: all won (выбираем все ставки)
            all_mask = np.ones(len(test_df), dtype=bool)
            roi_all, n_all = calc_roi(test_df, all_mask)
            logger.info("ROI (select all): %.2f%% (%d bets)", roi_all, n_all)

            # Dummy: most_frequent
            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy.fit(train_df[["USD"]], y_tr)
            pred = dummy.predict(test_df[["USD"]])
            mask_mf = pred == 1
            roi_mf, n_mf = calc_roi(test_df, mask_mf)
            logger.info("ROI (most_frequent): %.2f%% (%d bets)", roi_mf, n_mf)

            # Winrate на test
            winrate_test = float((test_df["Status"] == "won").mean() * 100)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                    "strategy": "most_frequent",
                }
            )
            mlflow.log_metrics(
                {
                    "roi_select_all": roi_all,
                    "n_bets_all": n_all,
                    "roi_most_frequent": roi_mf,
                    "n_bets_mf": n_mf,
                    "winrate_test_pct": winrate_test,
                    "winrate_train_pct": float((train_df["Status"] == "won").mean() * 100),
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT roi_select_all={roi_all:.2f} roi_mf={roi_mf:.2f} run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
