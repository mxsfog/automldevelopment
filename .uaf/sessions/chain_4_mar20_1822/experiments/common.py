"""Общие утилиты для экспериментов chain_4_mar20_1822."""

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

    outcome_agg = (
        outcomes.groupby("Bet_ID").first()[["Sport", "Market", "Tournament"]].reset_index()
    )
    bets = bets.merge(outcome_agg, left_on="ID", right_on="Bet_ID", how="left")

    bets["target"] = (bets["Status"] == "won").astype(int)
    bets["Is_Parlay_bool"] = bets["Is_Parlay"] == "t"
    return bets


def load_elo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка ELO истории и справочника команд."""
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    return elo, teams


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
        thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
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
        thresholds = np.arange(0.30, 0.90, 0.01).tolist()
    best_roi = -999.0
    best_t = 0.5
    for t in thresholds:
        result = calc_roi(val_df, proba, threshold=t)
        if result["n_bets"] >= min_bets and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_t = t
    return best_t, best_roi


def get_base_features() -> list[str]:
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


def get_engineered_features() -> list[str]:
    """Проверенные фичи из chain_1 (safe, без leakage)."""
    return [
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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление проверенных фичей из chain_1."""
    df = df.copy()
    df["log_odds"] = np.log1p(df["Odds"])
    df["implied_prob"] = 1.0 / df["Odds"]
    df["value_ratio"] = (df["ML_P_Model"] / 100.0) / df["implied_prob"]
    df["edge_x_ev"] = df["ML_Edge"] * df["ML_EV"]
    df["edge_abs"] = df["ML_Edge"].abs()
    df["ev_positive"] = (df["ML_EV"] > 0).astype(float)
    df["model_implied_diff"] = df["ML_P_Model"] - df["ML_P_Implied"]
    df["log_usd"] = np.log1p(df["USD"])
    df["log_usd_per_outcome"] = np.log1p(df["USD"] / df["Outcomes_Count"])
    df["parlay_complexity"] = df["Outcomes_Count"] * df["Is_Parlay_bool"].astype(float)
    return df


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление ELO фичей из elo_history + teams (safe, без leakage)."""
    elo, teams = load_elo_data()

    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            team_elo_mean=("Old_ELO", "mean"),
            team_elo_max=("Old_ELO", "max"),
            team_elo_min=("Old_ELO", "min"),
            k_factor_mean=("K_Factor", "mean"),
            n_elo_records=("ID", "count"),
        )
        .reset_index()
    )

    elo_with_teams = elo.merge(
        teams[["ID", "Winrate", "Total_Games", "Current_ELO"]],
        left_on="Team_ID",
        right_on="ID",
        how="left",
        suffixes=("", "_team"),
    )

    team_agg = (
        elo_with_teams.groupby("Bet_ID")
        .agg(
            team_winrate_mean=("Winrate", "mean"),
            team_winrate_max=("Winrate", "max"),
            team_total_games_mean=("Total_Games", "mean"),
            team_current_elo_mean=("Current_ELO", "mean"),
        )
        .reset_index()
    )

    elo_sorted = elo.sort_values(["Bet_ID", "Old_ELO"], ascending=[True, False])
    elo_first = elo_sorted.groupby("Bet_ID").first().reset_index()
    elo_last = elo_sorted.groupby("Bet_ID").last().reset_index()

    elo_diff_df = pd.DataFrame(
        {
            "Bet_ID": elo_first["Bet_ID"],
            "elo_diff": elo_first["Old_ELO"].values - elo_last["Old_ELO"].values,
        }
    )

    wr_sorted = elo_with_teams.sort_values(["Bet_ID", "Winrate"], ascending=[True, False])
    wr_first = wr_sorted.groupby("Bet_ID").first().reset_index()
    wr_last = wr_sorted.groupby("Bet_ID").last().reset_index()

    wr_diff_df = pd.DataFrame(
        {
            "Bet_ID": wr_first["Bet_ID"],
            "team_winrate_diff": wr_first["Winrate"].values - wr_last["Winrate"].values,
        }
    )

    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    df = df.merge(team_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_ta"))
    df = df.merge(elo_diff_df, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_ed"))
    df = df.merge(wr_diff_df, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_wd"))

    df["elo_diff_abs"] = df["elo_diff"].abs()
    df["has_elo"] = df["team_elo_mean"].notna().astype(float)
    df["elo_spread"] = df["team_elo_max"] - df["team_elo_min"]
    df["elo_mean_vs_1500"] = df["team_elo_mean"] - 1500.0

    drop_cols = [c for c in df.columns if c.startswith("Bet_ID_") or (c == "Bet_ID" and c != "ID")]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def get_elo_features() -> list[str]:
    """Проверенные ELO фичи из chain_2 (safe, без leakage)."""
    return [
        "team_elo_mean",
        "team_elo_max",
        "team_elo_min",
        "k_factor_mean",
        "n_elo_records",
        "elo_diff",
        "elo_diff_abs",
        "has_elo",
        "team_winrate_mean",
        "team_winrate_max",
        "team_winrate_diff",
        "team_total_games_mean",
        "team_current_elo_mean",
        "elo_spread",
        "elo_mean_vs_1500",
    ]


UNPROFITABLE_SPORTS = ["Basketball", "MMA", "FIFA", "Snooker"]
