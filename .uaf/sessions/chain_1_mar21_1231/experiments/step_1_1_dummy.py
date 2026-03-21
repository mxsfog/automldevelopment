"""Step 1.1 — Constant baseline (DummyClassifier).

Устанавливает lower bound: ROI при ставке на всё подряд и при DummyClassifier.
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

random.seed(42)
np.random.seed(42)

# UAF-SECTION: MLFLOW-INIT
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])

# Budget check
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


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    # Агрегация outcomes по Bet_ID — берём первый Sport/Market (для синглов)
    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(
            Sport=("Sport", "first"),
            Market=("Market", "first"),
            Selection=("Selection", "first"),
            Outcome_Odds=("Odds", "first"),
            Outcome_Status=("Status", "first"),
            Start_Time=("Start_Time", "first"),
        )
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")

    # Фильтрация: только won/lost
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()
    logger.info("После фильтрации: %d строк", len(df))
    logger.info("Распределение Status:\n%s", df["Status"].value_counts())

    return df


def calc_roi(df: pd.DataFrame, selected_mask: np.ndarray) -> dict:
    """Расчёт ROI на выбранных ставках."""
    sel = df[selected_mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0, "total_staked": 0.0, "total_payout": 0.0}

    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0

    return {
        "roi": round(roi, 4),
        "n_bets": len(sel),
        "total_staked": round(total_staked, 2),
        "total_payout": round(total_payout, 2),
    }


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split по индексу."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(
        "Split: train=%d (%s..%s), test=%d (%s..%s)",
        len(train),
        train["Created_At"].min().date(),
        train["Created_At"].max().date(),
        len(test),
        test["Created_At"].min().date(),
        test["Created_At"].max().date(),
    )
    return train, test


def main():
    df = load_data()
    train, test = time_series_split(df)
    logger.info("Train: %d, Test: %d", len(train), len(test))

    target = "Status"
    y_train = (train[target] == "won").astype(int)
    y_test = (test[target] == "won").astype(int)

    # Dummy features — просто один столбец
    x_train = np.zeros((len(train), 1))
    x_test = np.zeros((len(test), 1))

    with mlflow.start_run(run_name="phase1/step_1_1_dummy") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.1")
            mlflow.set_tag("phase", "1")

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "test_size": 0.2,
                    "gap_days": 7,
                    "model": "DummyClassifier",
                    "strategy": "most_frequent",
                }
            )

            # Baseline 1: ставим на всё (no model)
            all_selected = np.ones(len(test), dtype=bool)
            roi_all = calc_roi(test, all_selected)
            logger.info(
                "ROI (bet on everything): %.4f%% (%d bets)", roi_all["roi"], roi_all["n_bets"]
            )

            mlflow.log_metrics(
                {
                    "roi_bet_all": roi_all["roi"],
                    "n_bets_all": roi_all["n_bets"],
                }
            )

            # DummyClassifier — most_frequent
            dummy = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy.fit(x_train, y_train)
            y_pred = dummy.predict(x_test)

            # ROI на предсказанных won
            predicted_won = y_pred == 1
            roi_dummy = calc_roi(test, predicted_won)
            logger.info(
                "DummyClassifier (most_frequent) predicts class: %s",
                dummy.classes_[dummy.class_prior_.argmax()],
            )
            logger.info(
                "ROI (DummyClassifier): %.4f%% (%d bets)", roi_dummy["roi"], roi_dummy["n_bets"]
            )

            # DummyClassifier — stratified
            dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
            dummy_strat.fit(x_train, y_train)
            y_pred_strat = dummy_strat.predict(x_test)
            predicted_won_strat = y_pred_strat == 1
            roi_strat = calc_roi(test, predicted_won_strat)
            logger.info(
                "ROI (DummyClassifier stratified): %.4f%% (%d bets)",
                roi_strat["roi"],
                roi_strat["n_bets"],
            )

            # Winrate
            train_winrate = y_train.mean()
            test_winrate = y_test.mean()
            logger.info("Train winrate: %.4f, Test winrate: %.4f", train_winrate, test_winrate)

            mlflow.log_metrics(
                {
                    "roi": roi_dummy["roi"],
                    "roi_dummy_stratified": roi_strat["roi"],
                    "n_bets": roi_dummy["n_bets"],
                    "n_bets_stratified": roi_strat["n_bets"],
                    "train_winrate": round(train_winrate, 4),
                    "test_winrate": round(test_winrate, 4),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi_bet_all={roi_all['roi']}")
            print(f"RESULT:roi_dummy={roi_dummy['roi']}")
            print(f"RESULT:roi_stratified={roi_strat['roi']}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.1")
            logger.exception("Step 1.1 failed")
            raise


if __name__ == "__main__":
    main()
