"""Step 1.2 - Rule-based baseline.

Hypothesis: Простое пороговое правило по ML_Edge (позитивный ожидаемый перевес)
задаёт upper bound для простых стратегий.
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
SESSION_DIR = os.environ["UAF_SESSION_DIR"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Selection", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df = df.sort_values("Created_At").reset_index(drop=True)
    return df


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """Вычислить ROI на выбранных ставках."""
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def time_series_split(df: pd.DataFrame):
    """Разбивка по времени: train/val/test."""
    n = len(df)
    train_end = int(n * 0.8)
    val_start = int(n * 0.64)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[val_start:train_end]
    test_df = df.iloc[train_end:]
    return train_df, val_df, test_df


def find_best_threshold_edge(val_df: pd.DataFrame, thresholds: list[float]) -> float:
    """Найти лучший порог ML_Edge на val-сете."""
    best_roi = -999.0
    best_threshold = 0.0
    for t in thresholds:
        has_edge = val_df["ML_Edge"].notna()
        mask = has_edge & (val_df["ML_Edge"] >= t)
        if mask.sum() < 50:
            continue
        roi, _ = calc_roi(val_df, mask.values)
        if roi > best_roi:
            best_roi = roi
            best_threshold = t
    return best_threshold


with mlflow.start_run(run_name="phase1/step1.2_rule_based") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")

    try:
        budget_file_path = os.environ.get("UAF_BUDGET_STATUS_FILE", "")
        if budget_file_path:
            try:
                status = json.loads(Path(budget_file_path).read_text())
                if status.get("hard_stop"):
                    mlflow.set_tag("status", "budget_stopped")
                    sys.exit(0)
            except FileNotFoundError:
                pass

        df = load_data()
        train_df, val_df, test_df = time_series_split(df)

        logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "rule_based_ML_Edge_threshold",
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )

        # Правило 1: ML_Edge > 0 (положительный edge)
        # Правило 2: ML_Edge > 5
        # Правило 3: ML_Edge > 10
        # Порог выбираем на val, применяем к test один раз

        # Сначала смотрим на val
        edge_thresholds = [-10, -5, 0, 5, 10, 15, 20]
        for t in edge_thresholds:
            has_edge = val_df["ML_Edge"].notna()
            mask = has_edge & (val_df["ML_Edge"] >= t)
            if mask.sum() < 50:
                continue
            roi_v, n_v = calc_roi(val_df, mask.values)
            logger.info("Val ML_Edge >= %d: ROI=%.2f%% (%d ставок)", t, roi_v, n_v)

        # Лучший порог по val
        best_threshold = find_best_threshold_edge(val_df, thresholds=list(range(-10, 30, 2)))
        logger.info("Лучший порог по val: ML_Edge >= %.1f", best_threshold)

        # Применяем к test
        has_edge_test = test_df["ML_Edge"].notna()
        mask_test = has_edge_test & (test_df["ML_Edge"] >= best_threshold)
        roi_test, n_test = calc_roi(test_df, mask_test.values)

        # ML_EV > 0 стратегия (дополнительно)
        has_ev = val_df["ML_EV"].notna()
        ev_thresholds = [0, 5, 10, 15]
        best_ev_roi = -999.0
        best_ev_threshold = 0.0
        for t in ev_thresholds:
            mask_ev = has_ev & (val_df["ML_EV"] >= t)
            if mask_ev.sum() < 50:
                continue
            roi_ev, n_ev = calc_roi(val_df, mask_ev.values)
            logger.info("Val ML_EV >= %d: ROI=%.2f%% (%d ставок)", t, roi_ev, n_ev)
            if roi_ev > best_ev_roi:
                best_ev_roi = roi_ev
                best_ev_threshold = t

        # CV по времени на test-стиль
        n = len(df)
        fold_size = n // 5
        cv_rois = []
        for fold_idx in range(5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n
            if fold_start < n // 10:
                continue
            fold_val = df.iloc[fold_start:fold_end]
            has_edge_fold = fold_val["ML_Edge"].notna()
            mask_fold = has_edge_fold & (fold_val["ML_Edge"] >= best_threshold)
            if mask_fold.sum() < 20:
                cv_rois.append(-100.0)
                continue
            roi_fold, _ = calc_roi(fold_val, mask_fold.values)
            cv_rois.append(roi_fold)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_fold)
            logger.info("CV Fold %d: ROI=%.2f%%", fold_idx, roi_fold)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else roi_test
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        # val ROI (для отчёта)
        has_edge_val = val_df["ML_Edge"].notna()
        mask_val = has_edge_val & (val_df["ML_Edge"] >= best_threshold)
        roi_val, n_val = calc_roi(val_df, mask_val.values)

        mlflow.log_params({"best_edge_threshold": best_threshold})
        mlflow.log_metrics(
            {
                "roi_val": roi_val,
                "roi_test": roi_test,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_val": n_val,
                "n_bets_test": n_test,
            }
        )

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.3")

        print("\n=== Step 1.2 Results ===")
        print(f"Best threshold (by val): ML_Edge >= {best_threshold:.1f}")
        print(f"ROI val:  {roi_val:.2f}% ({n_val} ставок)")
        print(f"ROI test: {roi_test:.2f}% ({n_test} ставок)")
        print(f"CV ROI:   {roi_mean:.2f}% ± {roi_std:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
