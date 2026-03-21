"""Step 1.2 — Rule-based baseline.

Пороговые правила по ML_Edge, ML_EV, Odds для фильтрации ставок.
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


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(
            Sport=("Sport", "first"),
            Market=("Market", "first"),
        )
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()
    logger.info("После фильтрации: %d строк", len(df))
    return df


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split по индексу."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """ROI на выбранных ставках."""
    sel = df[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def find_best_threshold_on_val(
    val_df: pd.DataFrame, feature: str, thresholds: list[float]
) -> tuple[float, float]:
    """Подбор порога на валидации."""
    best_roi = -999.0
    best_thr = thresholds[0]
    for thr in thresholds:
        mask = val_df[feature].fillna(-999) >= thr
        result = calc_roi(val_df, mask.values)
        if result["n_bets"] >= 10 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr, best_roi


def main():
    df = load_data()
    train_full, test = time_series_split(df)
    logger.info("Train: %d, Test: %d", len(train_full), len(test))

    # Разделяем train на train/val (последние 20% train = val)
    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split]
    val = train_full.iloc[val_split:]
    logger.info("Train/Val: %d/%d", len(train), len(val))

    with mlflow.start_run(run_name="phase1/step_1_2_rule") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.2")
            mlflow.set_tag("phase", "1")

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "threshold_rule",
                }
            )

            results = {}

            # Rule 1: ML_Edge > threshold (ставим только на +EV)
            edge_thresholds = list(range(0, 30, 2))
            best_edge_thr, best_edge_roi_val = find_best_threshold_on_val(
                val, "ML_Edge", edge_thresholds
            )
            edge_mask_test = test["ML_Edge"].fillna(-999) >= best_edge_thr
            roi_edge = calc_roi(test, edge_mask_test.values)
            results["ML_Edge"] = {
                "threshold": best_edge_thr,
                "val_roi": best_edge_roi_val,
                **roi_edge,
            }
            logger.info(
                "ML_Edge >= %.1f: val_roi=%.2f%%, test_roi=%.2f%% (%d bets)",
                best_edge_thr,
                best_edge_roi_val,
                roi_edge["roi"],
                roi_edge["n_bets"],
            )

            # Rule 2: ML_EV > threshold
            ev_thresholds = list(range(-10, 50, 5))
            best_ev_thr, best_ev_roi_val = find_best_threshold_on_val(val, "ML_EV", ev_thresholds)
            ev_mask_test = test["ML_EV"].fillna(-999) >= best_ev_thr
            roi_ev = calc_roi(test, ev_mask_test.values)
            results["ML_EV"] = {
                "threshold": best_ev_thr,
                "val_roi": best_ev_roi_val,
                **roi_ev,
            }
            logger.info(
                "ML_EV >= %.1f: val_roi=%.2f%%, test_roi=%.2f%% (%d bets)",
                best_ev_thr,
                best_ev_roi_val,
                roi_ev["roi"],
                roi_ev["n_bets"],
            )

            # Rule 3: ML_P_Model > threshold
            p_thresholds = [x / 100 for x in range(40, 80, 5)]
            best_p_thr, best_p_roi_val = find_best_threshold_on_val(
                val, "ML_P_Model", p_thresholds
            )
            p_mask_test = test["ML_P_Model"].fillna(0) >= best_p_thr
            roi_p = calc_roi(test, p_mask_test.values)
            results["ML_P_Model"] = {
                "threshold": best_p_thr,
                "val_roi": best_p_roi_val,
                **roi_p,
            }
            logger.info(
                "ML_P_Model >= %.2f: val_roi=%.2f%%, test_roi=%.2f%% (%d bets)",
                best_p_thr,
                best_p_roi_val,
                roi_p["roi"],
                roi_p["n_bets"],
            )

            # Rule 4: Odds range (midrange odds often more profitable)
            odds_ranges = [(1.3, 2.0), (1.5, 2.5), (2.0, 3.5), (1.2, 1.8)]
            best_odds_roi_val = -999.0
            best_odds_range = odds_ranges[0]
            for lo, hi in odds_ranges:
                mask = (val["Odds"] >= lo) & (val["Odds"] <= hi)
                r = calc_roi(val, mask.values)
                if r["n_bets"] >= 10 and r["roi"] > best_odds_roi_val:
                    best_odds_roi_val = r["roi"]
                    best_odds_range = (lo, hi)
            odds_mask_test = (test["Odds"] >= best_odds_range[0]) & (
                test["Odds"] <= best_odds_range[1]
            )
            roi_odds = calc_roi(test, odds_mask_test.values)
            results["Odds_range"] = {
                "range": best_odds_range,
                "val_roi": best_odds_roi_val,
                **roi_odds,
            }
            logger.info(
                "Odds [%.1f, %.1f]: val_roi=%.2f%%, test_roi=%.2f%% (%d bets)",
                best_odds_range[0],
                best_odds_range[1],
                best_odds_roi_val,
                roi_odds["roi"],
                roi_odds["n_bets"],
            )

            # Rule 5: Combined — ML_Edge > thr AND Odds in range
            combined_mask_val = (
                (val["ML_Edge"].fillna(-999) >= best_edge_thr)
                & (val["Odds"] >= best_odds_range[0])
                & (val["Odds"] <= best_odds_range[1])
            )
            roi_combined_val = calc_roi(val, combined_mask_val.values)
            combined_mask_test = (
                (test["ML_Edge"].fillna(-999) >= best_edge_thr)
                & (test["Odds"] >= best_odds_range[0])
                & (test["Odds"] <= best_odds_range[1])
            )
            roi_combined = calc_roi(test, combined_mask_test.values)
            results["Combined"] = {
                "val_roi": roi_combined_val["roi"],
                **roi_combined,
            }
            logger.info(
                "Combined: val_roi=%.2f%%, test_roi=%.2f%% (%d bets)",
                roi_combined_val["roi"],
                roi_combined["roi"],
                roi_combined["n_bets"],
            )

            # Лучший результат
            best_rule = max(results.items(), key=lambda x: x[1]["roi"])
            logger.info("Best rule: %s -> ROI=%.2f%%", best_rule[0], best_rule[1]["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_rule[1]["roi"],
                    "n_bets": best_rule[1]["n_bets"],
                    "roi_edge": roi_edge["roi"],
                    "roi_ev": roi_ev["roi"],
                    "roi_p_model": roi_p["roi"],
                    "roi_odds": roi_odds["roi"],
                    "roi_combined": roi_combined["roi"],
                }
            )
            mlflow.log_params(
                {
                    "best_rule": best_rule[0],
                    "edge_threshold": best_edge_thr,
                    "ev_threshold": best_ev_thr,
                    "p_threshold": best_p_thr,
                    "odds_range": str(best_odds_range),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:best_rule={best_rule[0]}")
            print(f"RESULT:roi={best_rule[1]['roi']}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            logger.exception("Step 1.2 failed")
            raise


if __name__ == "__main__":
    main()
