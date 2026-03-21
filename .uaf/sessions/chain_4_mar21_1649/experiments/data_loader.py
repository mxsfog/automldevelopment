"""Загрузка и подготовка данных для sports betting prediction."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

EXCLUDE_CLASSES = ["pending", "cancelled", "error", "cashout"]

LEAKAGE_COLS = ["Payout", "Payout_USD", "Settled_At", "Last_Checked_At", "Check_Count"]

CAT_COLS_OUTCOMES = ["Sport", "Market", "Is_Parlay"]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загрузка сырых данных из CSV."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    elo_history = pd.read_csv(DATA_DIR / "elo_history.csv")
    logger.info(
        "Loaded: bets=%d, outcomes=%d, teams=%d, elo=%d",
        len(bets),
        len(outcomes),
        len(teams),
        len(elo_history),
    )
    return bets, outcomes, teams, elo_history


def prepare_dataset(
    bets: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Подготовка основного датасета: join, фильтрация, фичи."""
    df = bets[~bets["Status"].isin(EXCLUDE_CLASSES)].copy()
    logger.info("After status filter: %d rows", len(df))

    df["target"] = (df["Status"] == "won").astype(int)
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)

    outcome_agg = _aggregate_outcomes(outcomes)
    df = df.merge(outcome_agg, left_on="ID", right_on="Bet_ID", how="left")

    df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], inplace=True)

    df["Is_Parlay"] = df["Is_Parlay"].map({"t": 1, "f": 0}).fillna(0).astype(int)

    df.sort_values("Created_At", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Final dataset: %d rows, target mean=%.4f", len(df), df["target"].mean())
    return df


def _aggregate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Агрегация outcomes до уровня ставки."""
    first_outcome = (
        outcomes.sort_values("Bet_ID")
        .groupby("Bet_ID")
        .first()
        .reset_index()[["Bet_ID", "Sport", "Market", "Odds"]]
    )
    first_outcome.rename(columns={"Odds": "Outcome_Odds"}, inplace=True)

    agg = (
        outcomes.groupby("Bet_ID")
        .agg(
            n_outcomes=("Sport", "size"),
            mean_outcome_odds=("Odds", "mean"),
            max_outcome_odds=("Odds", "max"),
            min_outcome_odds=("Odds", "min"),
        )
        .reset_index()
    )

    result = first_outcome.merge(agg, on="Bet_ID", how="left")
    return result


def get_base_features() -> list[str]:
    """Базовый набор фичей (без leakage)."""
    return [
        "Odds",
        "USD",
        "Is_Parlay",
        "Outcomes_Count",
        "ML_P_Model",
        "ML_P_Implied",
        "ML_Edge",
        "ML_EV",
        "ML_Winrate_Diff",
        "ML_Rating_Diff",
        "Outcome_Odds",
        "n_outcomes",
        "mean_outcome_odds",
        "max_outcome_odds",
        "min_outcome_odds",
    ]


def add_sport_market_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Sport/Market target encoding (fit on train only)."""
    df = df.copy()
    new_feats = []

    for col in ["Sport", "Market"]:
        if col not in df.columns:
            continue
        means = train_df.groupby(col)["target"].mean()
        global_mean = train_df["target"].mean()
        counts = train_df.groupby(col)["target"].count()
        smooth = 50
        smoothed = (means * counts + global_mean * smooth) / (counts + smooth)

        feat_name = f"{col}_target_enc"
        df[feat_name] = df[col].map(smoothed).fillna(global_mean)
        new_feats.append(feat_name)

        count_map = train_df[col].value_counts()
        feat_name_cnt = f"{col}_count_enc"
        df[feat_name_cnt] = df[col].map(count_map).fillna(0)
        new_feats.append(feat_name_cnt)

    return df, new_feats


def time_series_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time series split по строкам (хронологический порядок)."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info(
        "Split: train=%d [%s..%s], test=%d [%s..%s]",
        len(train),
        train["Created_At"].min().date(),
        train["Created_At"].max().date(),
        len(test),
        test["Created_At"].min().date(),
        test["Created_At"].max().date(),
    )
    return train, test


def calc_roi(
    df: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float = 0.5,
    stake: float = 1.0,
) -> dict:
    """Расчет ROI на выбранных ставках."""
    mask = predictions >= threshold
    n_bets = mask.sum()
    if n_bets == 0:
        return {"roi": 0.0, "n_bets": 0, "profit": 0.0, "total_staked": 0.0}

    selected = df[mask].copy()
    total_staked = n_bets * stake
    payouts = selected["target"] * selected["Odds"] * stake
    total_payout = payouts.sum()
    profit = total_payout - total_staked
    roi = profit / total_staked * 100

    return {
        "roi": round(roi, 4),
        "n_bets": int(n_bets),
        "profit": round(profit, 4),
        "total_staked": round(total_staked, 4),
        "win_rate": round(selected["target"].mean(), 4),
        "avg_odds": round(selected["Odds"].mean(), 4),
    }


def find_best_threshold(
    df: pd.DataFrame,
    probas: np.ndarray,
    thresholds: np.ndarray | None = None,
    min_bets: int = 50,
) -> tuple[float, dict]:
    """Поиск лучшего порога по ROI на валидации."""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95, 0.01)

    best_roi = -999.0
    best_threshold = 0.5
    best_result = {}

    for t in thresholds:
        result = calc_roi(df, probas, threshold=t)
        if result["n_bets"] >= min_bets and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_threshold = t
            best_result = result

    return round(best_threshold, 2), best_result
