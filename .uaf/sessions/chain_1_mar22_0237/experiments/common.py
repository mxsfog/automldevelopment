"""Общие утилиты для экспериментов chain_1_mar22_0237.

Данные: sports_betting (bets + outcomes + elo_history)
Метрика: ROI = (payout_won - stake) / stake * 100
Задача: максимизировать ROI на отобранных ставках.
"""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
SEED = 42
BASELINE_ROI = 33.35382490813881  # chain_9 best: p80 Kelly + 1x2
EXCLUDE_CLASSES = {"pending", "cancelled", "error", "cashout"}


def load_raw_data() -> pd.DataFrame:
    """Загрузка и объединение bets + outcomes + elo.

    Returns:
        DataFrame отсортированный по Created_At, без excluded классов.
    """
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_CLASSES)].copy()

    # Для каждого бета берём первый ряд outcomes (для парлаев — первую ногу)
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time", "Fixture_Status"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")

    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    # ELO агрегация по Bet_ID
    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            elo_max=("Old_ELO", "max"),
            elo_min=("Old_ELO", "min"),
            elo_mean=("Old_ELO", "mean"),
            elo_std=("Old_ELO", "std"),
            elo_count=("Old_ELO", "count"),
            k_factor_mean=("K_Factor", "mean"),
        )
        .reset_index()
    )
    elo_agg["elo_diff"] = elo_agg["elo_max"] - elo_agg["elo_min"]
    elo_agg["elo_ratio"] = elo_agg["elo_max"] / elo_agg["elo_min"].clip(1.0)
    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))

    logger.info(
        "Загружено %d строк (won=%d, lost=%d)",
        len(df),
        (df["Status"] == "won").sum(),
        (df["Status"] == "lost").sum(),
    )
    return df


def build_features_base(df: pd.DataFrame) -> pd.DataFrame:
    """Базовый feature set (совместимый с chain_8/chain_9).

    34 признака без новых гипотез.
    """
    feats = pd.DataFrame(index=df.index)
    feats["Odds"] = df["Odds"]
    feats["USD"] = df["USD"]
    feats["log_odds"] = np.log(df["Odds"].clip(1.001))
    feats["log_usd"] = np.log1p(df["USD"].clip(0))
    feats["implied_prob"] = 1.0 / df["Odds"].clip(1.001)
    feats["is_parlay"] = (df["Is_Parlay"] == "t").astype(int)
    feats["outcomes_count"] = df["Outcomes_Count"].fillna(1)
    feats["ml_p_model"] = df["ML_P_Model"].fillna(-1)
    feats["ml_p_implied"] = df["ML_P_Implied"].fillna(-1)
    feats["ml_edge"] = df["ML_Edge"].fillna(0.0)
    feats["ml_ev"] = df["ML_EV"].clip(-100, 1000).fillna(0.0)
    feats["ml_team_stats_found"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
    feats["ml_winrate_diff"] = df["ML_Winrate_Diff"].fillna(0.0)
    feats["ml_rating_diff"] = df["ML_Rating_Diff"].fillna(0.0)
    feats["hour"] = df["Created_At"].dt.hour
    feats["day_of_week"] = df["Created_At"].dt.dayofweek
    feats["month"] = df["Created_At"].dt.month
    feats["odds_times_stake"] = feats["Odds"] * feats["USD"]
    feats["ml_edge_pos"] = feats["ml_edge"].clip(0)
    feats["ml_ev_pos"] = feats["ml_ev"].clip(0)
    feats["elo_max"] = df["elo_max"].fillna(-1)
    feats["elo_min"] = df["elo_min"].fillna(-1)
    feats["elo_diff"] = df["elo_diff"].fillna(0.0)
    feats["elo_ratio"] = df["elo_ratio"].fillna(1.0)
    feats["elo_mean"] = df["elo_mean"].fillna(-1)
    feats["elo_std"] = df["elo_std"].fillna(0.0)
    feats["k_factor_mean"] = df["k_factor_mean"].fillna(-1)
    feats["has_elo"] = df["elo_count"].notna().astype(int)
    feats["elo_count"] = df["elo_count"].fillna(0)
    feats["ml_edge_x_elo_diff"] = feats["ml_edge"] * feats["elo_diff"].clip(0, 500) / 500
    feats["elo_implied_agree"] = (
        feats["implied_prob"] - 1.0 / feats["elo_ratio"].clip(0.5, 2.0)
    ).abs()
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    return feats


def build_features_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Расширенный feature set с новыми признаками (Fixture_Status, lead_hours, etc.)."""
    feats = build_features_base(df)

    # Fixture_Status: live bet или pre-match
    feats["is_live"] = (df["Fixture_Status"] == "live").astype(int)

    # lead_hours: сколько часов от создания ставки до старта матча (отрицательное = live)
    lead_td = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    feats["lead_hours"] = lead_td.fillna(0.0).clip(-48, 168)

    # log1p(|lead_hours|) as magnitude feature
    feats["log_lead_abs"] = np.log1p(feats["lead_hours"].abs())

    return feats


def time_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Временной сплит: первые train_frac по времени = train, остальное = test.

    Args:
        df: DataFrame отсортированный по Created_At.
        train_frac: доля тренировочных данных.

    Returns:
        (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info("Сплит: train=%d, test=%d", len(train), len(test))
    return train, test


def time_split_val(
    df: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Временной сплит: train / val / test.

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """ROI на выбранных ставках.

    Args:
        df: DataFrame с колонками Status и Payout_USD, USD.
        mask: boolean array.

    Returns:
        (roi_percent, n_selected)
    """
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion: f* = (p*b - q) / b, где b = odds - 1."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def p80_kelly_threshold(
    train: pd.DataFrame, model, feature_names: list[str], cat_features: list[str] | None = None
) -> float:
    """Порог Kelly p80 на тренировочных LOW-odds ставках (anti-leakage).

    Берём ставки с Odds < 2.5 из train,
    вычисляем Kelly и возвращаем 80-й перцентиль.
    """
    low_mask = train["Odds"].values < 2.5
    low_df = train[low_mask].copy()
    if len(low_df) == 0:
        return 0.5

    x_low = build_features_base(low_df)[feature_names] if cat_features is None else \
        build_features_extended(low_df)[feature_names]

    proba_low = model.predict_proba(x_low)[:, 1]
    kelly_low = compute_kelly(proba_low, low_df["Odds"].values)
    threshold = float(np.percentile(kelly_low, 80))
    logger.info("p80 Kelly threshold (train LOW, n=%d): %.4f", len(low_df), threshold)
    return threshold


def already_done(experiment_name: str, step: str) -> bool:
    """Проверить, выполнялся ли шаг ранее (success run в MLflow).

    Args:
        experiment_name: название эксперимента.
        step: тег step (например '1.1', '4.0').

    Returns:
        True если уже есть успешный run с этим step.
    """
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"tags.step = '{step}' AND tags.status = 'success'",
            max_results=1,
        )
        return len(runs) > 0
    except Exception:
        return False


def check_budget(budget_file: Path) -> bool:
    """Проверить бюджет. Returns True если hard_stop."""
    import json

    try:
        status = json.loads(budget_file.read_text())
        return bool(status.get("hard_stop", False))
    except FileNotFoundError:
        return False
