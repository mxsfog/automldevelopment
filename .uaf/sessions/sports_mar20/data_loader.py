"""Загрузка и подготовка данных для спортивных ставок."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = ["pending", "cancelled", "error", "cashout"]
SEED = 42


def load_bets(with_outcomes: bool = False) -> pd.DataFrame:
    """Загрузка и фильтрация ставок."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    bets["Created_At"] = pd.to_datetime(bets["Created_At"])
    bets["target"] = (bets["Status"] == "won").astype(int)
    bets["Is_Parlay"] = (bets["Is_Parlay"] == "t").astype(int)

    if with_outcomes:
        outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
        # Берем первый outcome для каждой ставки (для синглов = единственный)
        first_outcome = outcomes.groupby("Bet_ID").first().reset_index()
        bets = bets.merge(
            first_outcome[["Bet_ID", "Sport", "Market", "Selection", "Start_Time"]],
            left_on="ID",
            right_on="Bet_ID",
            how="left",
        )

    bets = bets.sort_values("Created_At").reset_index(drop=True)
    logger.info(
        "Загружено %d ставок (won=%d, lost=%d)",
        len(bets),
        (bets["target"] == 1).sum(),
        (bets["target"] == 0).sum(),
    )
    return bets


def time_series_split(
    df: pd.DataFrame, n_splits: int = 5, gap_days: int = 7
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time series split для валидации."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_splits + 1)
    splits = []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        # Gap: пропускаем gap_days после train_end
        train_cutoff = df.loc[train_end - 1, "Created_At"]
        gap_cutoff = train_cutoff + pd.Timedelta(days=gap_days)

        val_mask = df["Created_At"] > gap_cutoff
        val_start = val_mask.idxmax() if val_mask.any() else train_end
        val_end = min(val_start + fold_size, n)

        if val_start >= val_end:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
        logger.info(
            "Fold %d: train=%d [..%s], val=%d [%s..%s]",
            i,
            len(train_idx),
            df.loc[train_end - 1, "Created_At"].date(),
            len(val_idx),
            df.loc[val_start, "Created_At"].date(),
            df.loc[val_end - 1, "Created_At"].date(),
        )

    return splits


def compute_roi(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    stakes: np.ndarray,
    payouts: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Вычисление ROI на отобранных ставках."""
    selected = y_pred_proba >= threshold
    n_selected = selected.sum()

    if n_selected == 0:
        return {"roi": 0.0, "n_selected": 0, "n_total": len(y_true), "select_rate": 0.0}

    won_mask = (y_true == 1) & selected
    total_stake = stakes[selected].sum()
    total_payout = payouts[won_mask].sum()
    roi = (total_payout - total_stake) / total_stake * 100

    return {
        "roi": roi,
        "n_selected": int(n_selected),
        "n_total": len(y_true),
        "select_rate": n_selected / len(y_true),
        "total_stake": float(total_stake),
        "total_payout": float(total_payout),
        "profit": float(total_payout - total_stake),
        "win_rate_selected": float(y_true[selected].mean()),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    stakes: np.ndarray,
    payouts: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Поиск лучшего порога по ROI."""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95, 0.01)

    best_roi = -999.0
    best_threshold = 0.5
    best_result = {}

    for t in thresholds:
        result = compute_roi(y_true, y_pred_proba, stakes, payouts, threshold=t)
        if result["n_selected"] >= 50 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_threshold = t
            best_result = result

    return best_threshold, best_result
