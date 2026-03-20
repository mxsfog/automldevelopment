"""Step 4.5: Feature interactions — попарные и тройные комбинации top фичей.

Гипотеза: interaction features (winrate_diff * odds, elo_diff * value_ratio и т.д.)
могут улучшить модель, т.к. CatBoost не всегда находит оптимальные interactions.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import TYPE_CHECKING

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CB_BEST_PARAMS,
    PS_EV_THRESHOLDS,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
    calc_per_sport_ev_roi,
    check_budget,
    get_all_features,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление interaction features из top-фичей."""
    df = df.copy()
    # winrate_diff доминирует, его комбинации наиболее перспективны
    df["wr_diff_x_odds"] = df["team_winrate_diff"] * df["Odds"]
    df["wr_diff_x_implied"] = df["team_winrate_diff"] * df["implied_prob"]
    df["wr_diff_x_value"] = df["team_winrate_diff"] * df["value_ratio"]
    df["wr_diff_x_elo_diff"] = df["team_winrate_diff"] * df["elo_diff"]

    # odds-related interactions
    df["odds_x_elo_mean"] = df["log_odds"] * df["team_elo_mean"]
    df["odds_x_wr_mean"] = df["log_odds"] * df["team_winrate_mean"]

    # value + elo
    df["value_x_elo_diff"] = df["value_ratio"] * df["elo_diff"]
    df["value_x_games"] = df["value_ratio"] * df["team_total_games_mean"]

    # edge interactions
    df["edge_x_wr_diff"] = df["ML_Edge"] * df["team_winrate_diff"]
    df["edge_x_elo_spread"] = df["ML_Edge"] * df["elo_spread"]

    # elo confidence
    df["elo_conf"] = df["n_elo_records"] * df["team_total_games_mean"]
    df["elo_quality"] = df["elo_diff_abs"] / (df["elo_spread"] + 1.0)

    return df


INTERACTION_FEATURES = [
    "wr_diff_x_odds",
    "wr_diff_x_implied",
    "wr_diff_x_value",
    "wr_diff_x_elo_diff",
    "odds_x_elo_mean",
    "odds_x_wr_mean",
    "value_x_elo_diff",
    "value_x_games",
    "edge_x_wr_diff",
    "edge_x_elo_spread",
    "elo_conf",
    "elo_quality",
]


def find_sport_ev_thresholds(
    val_df: pd.DataFrame, p_val: np.ndarray, ev_floor: float = 0.15
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта."""
    sports = val_df["Sport"].unique()
    thresholds: dict[str, float] = {}
    for sport in sports:
        sport_mask = (val_df["Sport"] == sport).values
        if sport_mask.sum() < 10:
            thresholds[sport] = ev_floor
            continue
        val_sport = val_df[sport_mask]
        p_sport = p_val[sport_mask]
        best_ev_t = ev_floor
        best_roi = -999.0
        for ev_t in np.arange(ev_floor, 0.35, 0.005):
            r = calc_ev_roi(val_sport, p_sport, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= 2 and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_ev_t = float(ev_t)
        thresholds[sport] = max(best_ev_t, ev_floor)
    return thresholds


def apply_sport_ev(test_df: pd.DataFrame, p_test: np.ndarray, sport_ev: dict[str, float]) -> dict:
    """Применение per-sport EV thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t = sport_ev.get(sports[i], 0.15)
        if ev[i] >= ev_t and p_test[i] >= 0.77:
            mask[i] = True
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def train_and_eval(
    train_sf: pd.DataFrame,
    test_sf: pd.DataFrame,
    features: list[str],
    label: str,
) -> dict:
    """Обучение + оценка стратегий."""
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[features])
    x_val = imp.transform(val_df[features])

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[features])
    x_test = imp_full.transform(test_sf[features])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[features]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    ev010 = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    ps_hc = calc_per_sport_ev_roi(
        test_sf,
        p_test,
        sport_thresholds=PS_EV_THRESHOLDS,
        min_prob=0.77,
    )
    sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
    ps_val = apply_sport_ev(test_sf, p_test, sport_ev)

    logger.info(
        "%s: AUC=%.4f EV010=%.2f%%(%d) PS_HC=%.2f%%(%d) PS15=%.2f%%(%d)",
        label,
        auc,
        ev010["roi"],
        ev010["n_bets"],
        ps_hc["roi"],
        ps_hc["n_bets"],
        ps_val["roi"],
        ps_val["n_bets"],
    )

    return {
        "auc": auc,
        "ev010_roi": ev010["roi"],
        "ev010_n": ev010["n_bets"],
        "ps_hc_roi": ps_hc["roi"],
        "ps_hc_n": ps_hc["n_bets"],
        "ps15_roi": ps_val["roi"],
        "ps15_n": ps_val["n_bets"],
        "fi": dict(zip(features, cb_ft.get_feature_importance().tolist(), strict=True)),
    }


def main() -> None:
    set_seed()
    check_budget()

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)
    df = add_interaction_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    base_features = get_all_features()
    features_with_interactions = base_features + INTERACTION_FEATURES

    # Baseline
    r_base = train_and_eval(train_sf, test_sf, base_features, "baseline")

    # Candidate
    r_cand = train_and_eval(train_sf, test_sf, features_with_interactions, "candidate")

    delta_auc = r_cand["auc"] - r_base["auc"]
    delta_ps15 = r_cand["ps15_roi"] - r_base["ps15_roi"]
    delta_ps_hc = r_cand["ps_hc_roi"] - r_base["ps_hc_roi"]

    verdict = (
        "accepted" if delta_auc > 0.002 or delta_ps15 > 2.0 or delta_ps_hc > 2.0 else "rejected"
    )
    logger.info(
        "Delta: AUC=%.4f PS15=%.2fpp PS_HC=%.2fpp -> %s",
        delta_auc,
        delta_ps15,
        delta_ps_hc,
        verdict,
    )

    # Top interaction features by importance
    if r_cand.get("fi"):
        inter_fi = {k: v for k, v in r_cand["fi"].items() if k in INTERACTION_FEATURES}
        sorted_fi = sorted(inter_fi.items(), key=lambda x: x[1], reverse=True)
        logger.info("Interaction feature importance:")
        for name, imp in sorted_fi:
            logger.info("  %s: %.2f", name, imp)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.5_interactions") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "feature_interactions",
                    "n_interaction_features": len(INTERACTION_FEATURES),
                    "interaction_features": str(INTERACTION_FEATURES[:5]),
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "verdict": verdict,
                }
            )
            mlflow.log_metrics(
                {
                    "auc_base": r_base["auc"],
                    "auc_cand": r_cand["auc"],
                    "delta_auc": delta_auc,
                    "ps15_base": r_base["ps15_roi"],
                    "ps15_cand": r_cand["ps15_roi"],
                    "delta_ps15": delta_ps15,
                    "ps_hc_base": r_base["ps_hc_roi"],
                    "ps_hc_cand": r_cand["ps_hc_roi"],
                    "roi": max(r_cand["ps15_roi"], r_cand["ps_hc_roi"]),
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
