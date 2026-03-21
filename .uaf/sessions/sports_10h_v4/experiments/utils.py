"""Общие утилиты для экспериментов sports_10h_v4."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
SEED = 42

# Колонки, которые нельзя использовать как фичи (leakage / идентификаторы)
LEAKAGE_COLS = {
    "ID",
    "IID",
    "User",
    "Status",
    "Payout",
    "Payout_USD",
    "Settled_At",
    "Last_Checked_At",
    "Check_Count",
    "ML_Predicted_At",
    "Detected_At",
    "Outcomes",
    "Created_At",
    "Bet_ID",
    "target",
    "Amount",
    "Currency",
}


def set_seed(seed: int = SEED) -> None:
    """Фиксация random seed."""
    random.seed(seed)
    np.random.seed(seed)


def check_budget() -> bool:
    """Проверка hard_stop в budget_status.json. Возвращает True если нужно остановиться."""
    import os

    budget_file = Path(
        os.environ.get("UAF_BUDGET_STATUS_FILE", "/mnt/d/automl-research/.uaf/budget_status.json")
    )
    try:
        status = json.loads(budget_file.read_text())
        if status.get("hard_stop"):
            logger.warning("hard_stop detected, stopping")
            return True
    except FileNotFoundError:
        pass
    return False


def load_data() -> pd.DataFrame:
    """Загрузка и подготовка датасета."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    # Фильтрация статусов
    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    # Target
    bets["target"] = (bets["Status"] == "won").astype(int)

    # Время
    bets["Created_At"] = pd.to_datetime(bets["Created_At"])

    # Агрегация outcomes
    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(
            Sport=("Sport", "first"),
            Market=("Market", "first"),
            Selection=("Selection", "first"),
            Odds_outcome=("Odds", "mean"),
            n_outcomes=("Bet_ID", "count"),
            Start_Time=("Start_Time", "first"),
        )
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    df = df.sort_values("Created_At").reset_index(drop=True)

    logger.info("Loaded %d rows, target mean=%.4f", len(df), df["target"].mean())
    return df


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2, gap_days: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time series split с gap."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_size))

    # gap: убираем gap_days перед test из train
    test_start_time = df.iloc[test_start]["Created_At"]
    gap_start = test_start_time - pd.Timedelta(days=gap_days)

    train = df[df["Created_At"] < gap_start].copy()
    test = df.iloc[test_start:].copy()

    logger.info(
        "Train: %d [%s .. %s], Test: %d [%s .. %s], gap=%d days",
        len(train),
        train["Created_At"].min(),
        train["Created_At"].max(),
        len(test),
        test["Created_At"].min(),
        test["Created_At"].max(),
        gap_days,
    )
    return train, test


def calc_roi(
    df: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float = 0.5,
    stake_col: str = "USD",
) -> dict:
    """Расчет ROI на ставках, где модель предсказала победу.

    ROI = (sum(Payout_USD для won) - sum(USD)) / sum(USD) * 100
    Считается flat betting (stake=1 на каждую отобранную ставку).
    """
    mask = predictions >= threshold
    selected = df[mask]

    if len(selected) == 0:
        return {
            "roi": 0.0,
            "n_bets": 0,
            "n_won": 0,
            "n_lost": 0,
            "precision": 0.0,
            "total_stake": 0.0,
            "total_payout": 0.0,
        }

    # Flat betting: каждая ставка = $1
    n_bets = len(selected)
    n_won = selected["target"].sum()
    n_lost = n_bets - n_won

    # Payout при flat betting: для выигранных ставок payout = odds * stake
    total_payout = selected.loc[selected["target"] == 1, "Odds"].sum()
    total_stake = float(n_bets)  # $1 per bet
    roi = (total_payout - total_stake) / total_stake * 100

    precision = n_won / n_bets if n_bets > 0 else 0.0

    return {
        "roi": roi,
        "n_bets": n_bets,
        "n_won": int(n_won),
        "n_lost": int(n_lost),
        "precision": precision,
        "total_stake": total_stake,
        "total_payout": total_payout,
        "selectivity": n_bets / len(df),
    }


def get_base_features(df: pd.DataFrame) -> list[str]:
    """Базовый набор фичей для моделирования."""
    numeric_candidates = [
        "Odds",
        "USD",
        "Outcomes_Count",
        "ML_P_Model",
        "ML_P_Implied",
        "ML_Edge",
        "ML_EV",
        "ML_Winrate_Diff",
        "ML_Rating_Diff",
        "Odds_outcome",
        "n_outcomes",
    ]
    cat_candidates = ["Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found"]

    features = []
    for c in numeric_candidates:
        if c in df.columns:
            features.append(c)
    for c in cat_candidates:
        if c in df.columns:
            features.append(c)

    return features


def find_best_threshold(
    df: pd.DataFrame, probas: np.ndarray, thresholds: np.ndarray | None = None
) -> float:
    """Поиск порога, максимизирующего ROI на валидационном наборе."""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95, 0.01)

    best_roi = -999.0
    best_thr = 0.5

    for thr in thresholds:
        result = calc_roi(df, probas, threshold=thr)
        if result["n_bets"] >= 20 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr

    logger.info("Best threshold: %.2f, ROI: %.2f%%", best_thr, best_roi)
    return best_thr
