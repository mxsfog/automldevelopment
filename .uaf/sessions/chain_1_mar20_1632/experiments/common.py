"""Общие утилиты для экспериментов chain_1_mar20_1632."""

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Фиксация seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)


def check_budget() -> None:
    """Проверка бюджета, остановка если hard_stop."""
    budget_file = Path(os.environ.get("UAF_BUDGET_STATUS_FILE", ""))
    try:
        status = json.loads(budget_file.read_text())
        if status.get("hard_stop"):
            logger.warning("Budget hard stop detected")
            sys.exit(0)
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def load_data() -> pd.DataFrame:
    """Загрузка и фильтрация данных ставок с join outcomes."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    exclude = ["pending", "cancelled", "error", "cashout"]
    bets = bets[~bets["Status"].isin(exclude)].copy()

    bets["Created_At"] = pd.to_datetime(bets["Created_At"])
    bets = bets.sort_values("Created_At").reset_index(drop=True)

    # Join sport/market from outcomes (first outcome per bet)
    outcome_agg = (
        outcomes.groupby("Bet_ID").first()[["Sport", "Market", "Tournament"]].reset_index()
    )
    bets = bets.merge(outcome_agg, left_on="ID", right_on="Bet_ID", how="left")

    bets["target"] = (bets["Status"] == "won").astype(int)
    bets["Is_Parlay_bool"] = bets["Is_Parlay"] == "t"

    logger.info(
        "Data loaded: %d rows, %d won (%.1f%%)",
        len(bets),
        bets["target"].sum(),
        bets["target"].mean() * 100,
    )
    return bets


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split по индексу (данные уже отсортированы по времени)."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(
        "Split: train=%d (%s to %s), test=%d (%s to %s)",
        len(train),
        train["Created_At"].min(),
        train["Created_At"].max(),
        len(test),
        test["Created_At"].min(),
        test["Created_At"].max(),
    )
    return train, test


def calc_roi(df: pd.DataFrame, predictions: np.ndarray, threshold: float = 0.5) -> dict:
    """Расчет ROI на ставках где модель предсказала won (prob >= threshold)."""
    mask = predictions >= threshold
    n_selected = int(mask.sum())
    if n_selected == 0:
        return {
            "roi": 0.0,
            "n_bets": 0,
            "n_won": 0,
            "total_staked": 0.0,
            "win_rate": 0.0,
            "pct_selected": 0.0,
        }

    selected = df.iloc[np.where(mask)[0]]
    total_staked = selected["USD"].sum()
    total_payout = selected["Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100
    n_won = (selected["Status"] == "won").sum()
    return {
        "roi": float(roi),
        "n_bets": n_selected,
        "n_won": int(n_won),
        "total_staked": float(total_staked),
        "win_rate": float(n_won / n_selected),
        "pct_selected": float(n_selected / len(df) * 100),
    }


def calc_roi_at_thresholds(
    df: pd.DataFrame, proba: np.ndarray, thresholds: list[float] | None = None
) -> dict[float, dict]:
    """ROI при разных порогах отсечения."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    results = {}
    for t in thresholds:
        results[t] = calc_roi(df, proba, threshold=t)
    return results


def find_best_threshold_on_val(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    thresholds: list[float] | None = None,
    min_bets: int = 30,
) -> tuple[float, float]:
    """Подбор лучшего порога на валидационной части train (anti-leakage)."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    best_roi = -999.0
    best_t = 0.5
    for t in thresholds:
        result = calc_roi(val_df, proba, threshold=t)
        if result["n_bets"] >= min_bets and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_t = t
    return best_t, best_roi


def get_feature_columns() -> list[str]:
    """Базовые числовые фичи для моделей."""
    return [
        "Odds",
        "USD",
        "ML_P_Model",
        "ML_P_Implied",
        "ML_Edge",
        "ML_EV",
        "Outcomes_Count",
        "Is_Parlay_bool",
    ]


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Фичи без target encoding -- трансформации существующих колонок."""
    df = df.copy()
    df["log_odds"] = np.log1p(df["Odds"])
    df["implied_prob"] = 1.0 / df["Odds"]
    df["value_ratio"] = (df["ML_P_Model"] / 100.0 / df["implied_prob"]).clip(0, 10).fillna(1.0)
    df["edge_x_ev"] = df["ML_Edge"] * df["ML_EV"]
    df["edge_abs"] = df["ML_Edge"].abs()
    df["ev_positive"] = (df["ML_EV"] > 0).astype(float)
    df["model_implied_diff"] = df["ML_P_Model"] - df["ML_P_Implied"]
    df["log_usd"] = np.log1p(df["USD"])
    df["usd_per_outcome"] = df["USD"] / df["Outcomes_Count"].clip(lower=1)
    df["log_usd_per_outcome"] = np.log1p(df["usd_per_outcome"])
    df["parlay_complexity"] = df["Outcomes_Count"] * (df["Is_Parlay"] == "t").astype(float)
    return df


def get_extended_feature_columns() -> list[str]:
    """Расширенный набор фичей (после Phase 2)."""
    return [
        *get_feature_columns(),
        "log_odds",
        "implied_prob",
        "value_ratio",
        "edge_x_ev",
        "edge_abs",
        "ev_positive",
        "model_implied_diff",
        "log_usd",
        "log_usd_per_outcome",
        "parlay_complexity",
    ]
