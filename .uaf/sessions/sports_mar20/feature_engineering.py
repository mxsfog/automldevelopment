"""Feature engineering функции для спортивных ставок.

Фичи основаны на EDA: нелинейная связь Edge-ROI, спортивная сегментация,
value-betting паттерны.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROFITABLE_SPORTS = {"Tennis", "Dota 2", "League of Legends", "CS2", "Table Tennis", "Volleyball"}
LOSING_SPORTS = {"Soccer", "MMA", "Ice Hockey", "FIFA", "Super Bowl LX"}


def add_odds_features(df: pd.DataFrame) -> list[str]:
    """Odds decomposition и value-betting features."""
    new_cols = []

    df["implied_prob"] = 1.0 / df["Odds"]
    new_cols.append("implied_prob")

    df["log_odds"] = np.log(df["Odds"])
    new_cols.append("log_odds")

    # Value = model probability - implied probability
    df["value"] = df["ML_P_Model"].fillna(50.0) / 100.0 - df["implied_prob"]
    new_cols.append("value")

    # Odds в прибыльном диапазоне (1.35-2.15 из EDA)
    df["odds_sweet_spot"] = ((df["Odds"] >= 1.35) & (df["Odds"] <= 2.15)).astype(int)
    new_cols.append("odds_sweet_spot")

    # Odds bucket
    df["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[0, 1.5, 2.5, 5.0, 1000],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    new_cols.append("odds_bucket")

    logger.info("Odds features: %s", new_cols)
    return new_cols


def add_edge_nonlinear_features(df: pd.DataFrame) -> list[str]:
    """Edge non-linearity features.

    EDA показал: Edge 14-31 = best ROI (+17.9%), Edge > 31 = negative.
    Связь нелинейная — нужны полиномиальные и bin-фичи.
    """
    new_cols = []

    edge = df["ML_Edge"].fillna(0.0)

    # Edge sweet spot (14-31 — лучший ROI из EDA)
    df["edge_sweet_spot"] = ((edge >= 14.0) & (edge <= 31.0)).astype(int)
    new_cols.append("edge_sweet_spot")

    # Edge positive zone (4-31 — все с положительным ROI)
    df["edge_positive_zone"] = ((edge >= 4.0) & (edge <= 31.0)).astype(int)
    new_cols.append("edge_positive_zone")

    # Edge squared (captures non-monotonic relationship)
    df["edge_sq"] = edge**2
    new_cols.append("edge_sq")

    # Clipped edge (cap extreme values that hurt ROI)
    df["edge_clipped"] = edge.clip(-50, 35)
    new_cols.append("edge_clipped")

    # Edge * Odds interaction
    df["edge_x_odds"] = edge * df["Odds"]
    new_cols.append("edge_x_odds")

    # Edge bins для CatBoost
    df["edge_bin"] = pd.cut(
        edge,
        bins=[-100, -20, 0, 10, 20, 35, 100],
        labels=["very_neg", "neg", "low_pos", "mid_pos", "sweet", "over"],
    ).astype(str)
    new_cols.append("edge_bin")

    logger.info("Edge nonlinear features: %s", new_cols)
    return new_cols


def add_sport_profitability_features(df: pd.DataFrame) -> list[str]:
    """Sport-level profitability features.

    EDA: Tennis (+10%), Dota2/LoL (+11%), CS2 (+5%), Soccer (-8.6%).
    """
    new_cols = []

    sport = df["Sport"].fillna("unknown")

    # Binary: profitable sport (из EDA)
    df["is_profitable_sport"] = sport.isin(PROFITABLE_SPORTS).astype(int)
    new_cols.append("is_profitable_sport")

    # Binary: losing sport
    df["is_losing_sport"] = sport.isin(LOSING_SPORTS).astype(int)
    new_cols.append("is_losing_sport")

    # Singles flag (parlays = -19.6% ROI)
    df["is_single"] = (df["Is_Parlay"] == 0).astype(int)
    new_cols.append("is_single")

    # Interaction: profitable sport + single + positive edge
    edge = df["ML_Edge"].fillna(0.0)
    df["profitable_single_posedge"] = (
        df["is_profitable_sport"] * df["is_single"] * (edge > 10).astype(int)
    )
    new_cols.append("profitable_single_posedge")

    # Expanding win rate по спорту (leak-safe)
    df["sport_winrate"] = df.groupby("Sport")["target"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["sport_winrate"] = df["sport_winrate"].fillna(0.5)
    new_cols.append("sport_winrate")

    logger.info("Sport profitability features: %s", new_cols)
    return new_cols


def add_ml_calibration_features(df: pd.DataFrame) -> list[str]:
    """ML model calibration и value features.

    EDA: P_Model 9-24% дает ROI +20.3%, P_Model 49-56% дает +7.95%.
    Зоны с лучшим ROI — где модель не совпадает с implied.
    """
    new_cols = []

    p_model = df["ML_P_Model"].fillna(50.0) / 100.0
    p_implied = df["ML_P_Implied"].fillna(50.0) / 100.0

    # Model confidence (distance from 0.5)
    df["model_confidence"] = (p_model - 0.5).abs()
    new_cols.append("model_confidence")

    # Calibration gap: model vs implied
    df["calibration_gap"] = p_model - p_implied
    new_cols.append("calibration_gap")

    # Value ratio: model/implied
    df["value_ratio"] = p_model / p_implied.clip(lower=0.01)
    new_cols.append("value_ratio")

    # ML_EV normalized by odds
    df["ev_per_dollar"] = df["ML_EV"].fillna(0.0) / df["Odds"].clip(lower=1.0)
    new_cols.append("ev_per_dollar")

    # P_Model in sweet spot (9-56% from EDA)
    df["pmodel_value_zone"] = (
        (df["ML_P_Model"].fillna(50.0) >= 9) & (df["ML_P_Model"].fillna(50.0) <= 56)
    ).astype(int)
    new_cols.append("pmodel_value_zone")

    # Edge positive flag
    df["edge_positive"] = (df["ML_Edge"].fillna(0.0) > 0).astype(int)
    new_cols.append("edge_positive")

    logger.info("ML calibration features: %s", new_cols)
    return new_cols


def add_stake_features(df: pd.DataFrame) -> list[str]:
    """Stake и potential payout features."""
    new_cols = []

    df["log_usd"] = np.log1p(df["USD"])
    new_cols.append("log_usd")

    # Potential payout
    df["log_potential_payout"] = np.log1p(df["USD"] * df["Odds"])
    new_cols.append("log_potential_payout")

    # Odds-weighted edge (higher odds amplify edge)
    df["edge_payout_potential"] = df["ML_Edge"].fillna(0.0) * np.log1p(df["Odds"])
    new_cols.append("edge_payout_potential")

    logger.info("Stake features: %s", new_cols)
    return new_cols
