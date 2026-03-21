"""Step 1.2 — Rule-based baseline: порог по ML_Edge.

Гипотеза: простая фильтрация ML_Edge > порог задаёт правило-baseline.
Порог выбирается на val (последние 20% train), применяется к test один раз.
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


def find_best_threshold(
    df: pd.DataFrame,
    signal: np.ndarray,
    thresholds: np.ndarray,
    min_bets: int = 200,
) -> tuple[float, float]:
    """Поиск лучшего порога на val по ROI."""
    best_roi, best_t = -999.0, thresholds[0]
    for t in thresholds:
        mask = signal >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def main() -> None:
    """Основной эксперимент."""
    with mlflow.start_run(run_name="phase1/step1.2_rule") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)  # последние 20% train = val

            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            logger.info("Val: %d, Test: %d", len(val_df), len(test_df))

            # Правило 1: порог по ML_Edge (raw edge платформы)
            val_edge = val_df["ML_Edge"].fillna(-999).values
            test_edge = test_df["ML_Edge"].fillna(-999).values

            thresholds_edge = np.arange(0.0, 20.0, 0.5)
            t_edge, roi_edge_val = find_best_threshold(val_df, val_edge, thresholds_edge)
            roi_edge_test, n_edge = calc_roi(test_df, test_edge >= t_edge)
            logger.info(
                "Rule ML_Edge >= %.1f: val=%.2f%%, test=%.2f%% (%d bets)",
                t_edge,
                roi_edge_val,
                roi_edge_test,
                n_edge,
            )

            # Правило 2: порог по ML_EV
            val_ev = val_df["ML_EV"].fillna(-999).values
            test_ev = test_df["ML_EV"].fillna(-999).values

            thresholds_ev = np.arange(0.0, 100.0, 2.0)
            t_ev, roi_ev_val = find_best_threshold(val_df, val_ev, thresholds_ev)
            roi_ev_test, n_ev = calc_roi(test_df, test_ev >= t_ev)
            logger.info(
                "Rule ML_EV >= %.1f: val=%.2f%%, test=%.2f%% (%d bets)",
                t_ev,
                roi_ev_val,
                roi_ev_test,
                n_ev,
            )

            # Правило 3: ML_Edge > 0 AND ML_EV > 0 (positive edge + positive EV)
            val_pos_mask = (val_df["ML_Edge"].fillna(-1) > 0) & (val_df["ML_EV"].fillna(-1) > 0)
            test_pos_mask = (test_df["ML_Edge"].fillna(-1) > 0) & (test_df["ML_EV"].fillna(-1) > 0)
            roi_pos_val, _ = calc_roi(val_df, val_pos_mask.values)
            roi_pos_test, n_pos_test = calc_roi(test_df, test_pos_mask.values)
            logger.info(
                "Rule edge>0 AND ev>0: val=%.2f%%, test=%.2f%% (%d bets)",
                roi_pos_val,
                roi_pos_test,
                n_pos_test,
            )

            # Лучший результат по val
            results = [
                ("ml_edge_thresh", t_edge, roi_edge_val, roi_edge_test, n_edge),
                ("ml_ev_thresh", t_ev, roi_ev_val, roi_ev_test, n_ev),
                ("pos_edge_ev", 0.0, roi_pos_val, roi_pos_test, n_pos_test),
            ]
            best = max(results, key=lambda x: x[2])
            logger.info("Лучшее правило (по val): %s, test_roi=%.2f%%", best[0], best[3])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "best_rule": best[0],
                    "best_threshold": best[1],
                }
            )
            mlflow.log_metrics(
                {
                    "roi_edge_val": roi_edge_val,
                    "roi_edge_test": roi_edge_test,
                    "n_bets_edge": n_edge,
                    "roi_ev_val": roi_ev_val,
                    "roi_ev_test": roi_ev_test,
                    "n_bets_ev": n_ev,
                    "roi_pos_val": roi_pos_val,
                    "roi_pos_test": roi_pos_test,
                    "n_bets_pos": n_pos_test,
                    "roi_best_test": best[3],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT best_rule={best[0]} roi_val={best[2]:.2f} "
                f"roi_test={best[3]:.2f} n_bets={best[4]} run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
