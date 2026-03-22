"""Общие утилиты для экспериментов chain_4_mar22_1623.

Данные: sports_betting (bets + outcomes + elo_history + teams)
Метрика: ROI = (payout_won - stake) / stake * 100
Задача: максимизировать ROI на отобранных ставках.

Базируется на chain_2_mar22_1516/experiments/common.py.
Лучший предыдущий результат: ROI=31.41% (chain_2_mar22_1516, step 4.2, V3 features, 1x2).

Запрещено повторять (chain_2_mar22_1516):
  - 4.1: Extended features + CatBoost 90%, 1x2, p80 Kelly
  - 4.2: V3 features + CatBoost 90%, 1x2, p80 Kelly
  - 4.3: XGBoost + V3 features
  - 4.4: Multi-market analysis
  - 4.5: CatBoost + LightGBM stacking (L1/L2/test)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
SEED = 42
BASELINE_ROI = 31.407652637647125
EXCLUDE_CLASSES = {"pending", "cancelled", "error", "cashout"}

PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_2_mar22_1516/models/best")


def load_raw_data() -> pd.DataFrame:
    """Загрузка и объединение bets + outcomes + elo.

    Returns:
        DataFrame отсортированный по Created_At, без excluded классов.
    """
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_CLASSES)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time", "Fixture_Status"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")

    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            elo_max=("Old_ELO", "max"),
            elo_min=("Old_ELO", "min"),
            elo_mean=("Old_ELO", "mean"),
            elo_std=("Old_ELO", "std"),
            elo_count=("Old_ELO", "count"),
            k_factor_mean=("K_Factor", "mean"),
            elo_change_sum=("ELO_Change", "sum"),
            elo_change_max=("ELO_Change", "max"),
            elo_change_min=("ELO_Change", "min"),
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


def load_team_stats() -> pd.DataFrame:
    """Загрузка teams.csv для обогащения данных командными статами."""
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    teams["Normalized_Name"] = teams["Normalized_Name"].str.lower().str.strip()
    return teams


def build_features_base(df: pd.DataFrame) -> pd.DataFrame:
    """Базовый feature set (совместимый с предыдущими сессиями).

    44 признака без новых гипотез.
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
    """Расширенный feature set: добавляет Fixture_Status и lead_hours."""
    feats = build_features_base(df)
    feats["is_live"] = (df["Fixture_Status"] == "live").astype(int)
    lead_td = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    feats["lead_hours"] = lead_td.fillna(0.0).clip(-48, 168)
    feats["log_lead_abs"] = np.log1p(feats["lead_hours"].abs())
    feats["edge_x_lead"] = feats["ml_edge"].clip(-1, 5) * feats["lead_hours"].clip(0, 48) / 48
    feats["elo_x_live"] = feats["elo_diff"] * feats["is_live"]
    return feats


def build_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """V3: расширенный + дополнительные ratio-фичи и взаимодействия.

    Это best feature set из chain_2_mar22_1516.
    """
    feats = build_features_extended(df)
    feats["p_model_vs_implied"] = df["ML_P_Model"].fillna(0.5) - feats["implied_prob"]
    feats["edge_squared"] = feats["ml_edge"] ** 2
    feats["ev_per_odd"] = feats["ml_ev"] / df["Odds"].clip(1.001)
    feats["elo_mean_norm"] = feats["elo_mean"] / 2000.0
    feats["stake_log_odds"] = feats["log_usd"] * feats["log_odds"]
    feats["kelly_approx"] = (
        df["ML_P_Model"].fillna(0.5) * (df["Odds"].clip(1.001) - 1)
        - (1 - df["ML_P_Model"].fillna(0.5))
    ) / (df["Odds"].clip(1.001) - 1).clip(0.001)
    return feats


def build_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """V4: V3 + ELO momentum/trend features.

    Новые признаки в этой сессии (chain_4_mar22_1623).
    """
    feats = build_features_v3(df)
    # ELO momentum: сумма изменений (положительная = набирающая форма команда)
    feats["elo_change_sum"] = df["elo_change_sum"].fillna(0.0).clip(-200, 200)
    feats["elo_change_max"] = df["elo_change_max"].fillna(0.0)
    feats["elo_change_min"] = df["elo_change_min"].fillna(0.0)
    # Асимметрия изменений ELO между командами
    feats["elo_momentum_diff"] = feats["elo_change_max"].abs() - feats["elo_change_min"].abs()
    # Ratio ставки к ELO разнице (ценность относительно известной силы команд)
    feats["stake_elo_ratio"] = feats["log_usd"] / feats["elo_diff"].clip(1, 500)
    # Kelly x ELO momentum
    feats["kelly_x_momentum"] = feats["kelly_approx"] * feats["elo_change_sum"].clip(-1, 1) / 100
    return feats


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Временной сплит: первые train_frac по времени = train, остальное = test."""
    n = len(df)
    split_idx = int(n * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info("Сплит: train=%d, test=%d", len(train), len(test))
    return train, test


def time_split_val(
    df: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Временной сплит: train / val / test."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


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


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion: f* = (p*b - q) / b, где b = odds - 1."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def p80_kelly_threshold(
    train: pd.DataFrame,
    model,
    feature_names: list[str],
    build_fn=None,
) -> float:
    """Порог Kelly p80 на val части train (anti-leakage).

    Берёт последние 20% train как val, вычисляет Kelly p80.
    """
    if build_fn is None:
        build_fn = build_features_base
    val_inner = train.iloc[int(len(train) * 0.8) :].copy()
    if len(val_inner) == 0:
        return 0.5
    x_val = build_fn(val_inner)[feature_names]
    proba_val = model.predict_proba(x_val)[:, 1]
    kelly_val = compute_kelly(proba_val, val_inner["Odds"].values)
    threshold = float(np.percentile(kelly_val, 80))
    logger.info("p80 Kelly threshold (val=%d): %.4f", len(val_inner), threshold)
    return threshold


def check_budget(budget_file: Path) -> bool:
    """Проверить бюджет. Returns True если hard_stop."""
    try:
        status = json.loads(budget_file.read_text())
        return bool(status.get("hard_stop", False))
    except FileNotFoundError:
        return False
