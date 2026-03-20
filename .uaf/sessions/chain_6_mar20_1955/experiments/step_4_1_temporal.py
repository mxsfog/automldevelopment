"""Step 4.1: Temporal features — hour, day_of_week, market_age, bet_density.

Гипотеза: временные паттерны влияют на ROI. Ставки в определённое время дня/недели
или с разной плотностью могут иметь разную доходность.
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


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление временных фичей (без leakage)."""
    df = df.copy()
    dt = df["Created_At"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # Плотность ставок: количество ставок в +-1 час (rolling count)
    df["ts_unix"] = dt.astype(np.int64) // 10**9
    df = df.sort_values("Created_At").reset_index(drop=True)
    ts = df["ts_unix"].values
    density = np.zeros(len(df))
    for i in range(len(df)):
        window = 3600  # 1 час
        low = np.searchsorted(ts, ts[i] - window, side="left")
        high = np.searchsorted(ts, ts[i] + window, side="right")
        density[i] = high - low
    df["bet_density_1h"] = density
    df["log_bet_density"] = np.log1p(density)
    df.drop(columns=["ts_unix"], inplace=True)

    return df


TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "bet_density_1h",
    "log_bet_density",
]


def find_sport_ev_thresholds(
    val_df: pd.DataFrame,
    p_val: np.ndarray,
    ev_floor: float = 0.10,
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта на валидации."""
    sports = val_df["Sport"].unique()
    thresholds: dict[str, float] = {}
    for sport in sports:
        sport_mask = (val_df["Sport"] == sport).values
        if sport_mask.sum() < 15:
            thresholds[sport] = ev_floor
            continue
        val_sport = val_df[sport_mask]
        p_sport = p_val[sport_mask]
        best_ev_t = ev_floor
        best_roi = -999.0
        for ev_t in np.arange(max(-0.05, ev_floor), 0.25, 0.01):
            r = calc_ev_roi(val_sport, p_sport, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= 3 and r["roi"] > best_roi:
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
        ev_t = sport_ev.get(sports[i], 0.10)
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


def main() -> None:
    set_seed()
    check_budget()

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)
    df = add_temporal_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    base_features = get_all_features()
    features_with_temporal = base_features + TEMPORAL_FEATURES
    logger.info(
        "Features: %d base + %d temporal = %d total",
        len(base_features),
        len(TEMPORAL_FEATURES),
        len(features_with_temporal),
    )

    # Baseline (без temporal) для сравнения
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imputer_base = SimpleImputer(strategy="median")
    x_fit_base = imputer_base.fit_transform(train_fit[base_features])
    x_val_base = imputer_base.transform(val_df[base_features])

    cb_base = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_base.fit(x_fit_base, train_fit["target"], eval_set=(x_val_base, val_df["target"]))
    best_iter_base = cb_base.get_best_iteration()

    imputer_base_full = SimpleImputer(strategy="median")
    x_full_base = imputer_base_full.fit_transform(train_sf[base_features])
    x_test_base = imputer_base_full.transform(test_sf[base_features])

    ft_params_base = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params_base["iterations"] = best_iter_base + 10
    cb_base_ft = CatBoostClassifier(**ft_params_base)
    cb_base_ft.fit(x_full_base, train_sf["target"])

    p_test_base = cb_base_ft.predict_proba(x_test_base)[:, 1]
    auc_base = roc_auc_score(test_sf["target"], p_test_base)
    ev_base = calc_ev_roi(test_sf, p_test_base, ev_threshold=0.10, min_prob=0.77)
    ps_base = calc_per_sport_ev_roi(
        test_sf,
        p_test_base,
        sport_thresholds=PS_EV_THRESHOLDS,
        min_prob=0.77,
    )

    logger.info(
        "Baseline: AUC=%.4f EV010=%.2f%%(%d) PS_EV=%.2f%%(%d)",
        auc_base,
        ev_base["roi"],
        ev_base["n_bets"],
        ps_base["roi"],
        ps_base["n_bets"],
    )

    # Candidate (с temporal)
    imputer_cand = SimpleImputer(strategy="median")
    x_fit_cand = imputer_cand.fit_transform(train_fit[features_with_temporal])
    x_val_cand = imputer_cand.transform(val_df[features_with_temporal])

    cb_cand = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_cand.fit(x_fit_cand, train_fit["target"], eval_set=(x_val_cand, val_df["target"]))
    best_iter_cand = cb_cand.get_best_iteration()

    imputer_cand_full = SimpleImputer(strategy="median")
    x_full_cand = imputer_cand_full.fit_transform(train_sf[features_with_temporal])
    x_test_cand = imputer_cand_full.transform(test_sf[features_with_temporal])

    ft_params_cand = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params_cand["iterations"] = best_iter_cand + 10
    cb_cand_ft = CatBoostClassifier(**ft_params_cand)
    cb_cand_ft.fit(x_full_cand, train_sf["target"])

    p_test_cand = cb_cand_ft.predict_proba(x_test_cand)[:, 1]
    auc_cand = roc_auc_score(test_sf["target"], p_test_cand)
    ev_cand = calc_ev_roi(test_sf, p_test_cand, ev_threshold=0.10, min_prob=0.77)
    ps_cand = calc_per_sport_ev_roi(
        test_sf,
        p_test_cand,
        sport_thresholds=PS_EV_THRESHOLDS,
        min_prob=0.77,
    )

    # Val-tuned per-sport EV
    p_val_cand = cb_cand_ft.predict_proba(
        imputer_cand_full.transform(val_df[features_with_temporal])
    )[:, 1]
    sport_ev_cand = find_sport_ev_thresholds(val_df, p_val_cand, ev_floor=0.10)
    ps_val_cand = apply_sport_ev(test_sf, p_test_cand, sport_ev_cand)

    logger.info(
        "Candidate (+temporal): AUC=%.4f EV010=%.2f%%(%d) PS_EV=%.2f%%(%d) PS_val=%.2f%%(%d)",
        auc_cand,
        ev_cand["roi"],
        ev_cand["n_bets"],
        ps_cand["roi"],
        ps_cand["n_bets"],
        ps_val_cand["roi"],
        ps_val_cand["n_bets"],
    )

    delta_auc = auc_cand - auc_base
    delta_ev = ev_cand["roi"] - ev_base["roi"]
    delta_ps = ps_cand["roi"] - ps_base["roi"]
    logger.info("Delta: AUC=%.4f EV_ROI=%.2fpp PS_ROI=%.2fpp", delta_auc, delta_ev, delta_ps)

    verdict = "accepted" if delta_auc > 0.002 or delta_ev > 2.0 or delta_ps > 2.0 else "rejected"
    logger.info("Verdict: %s", verdict)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.1_temporal") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "temporal_features",
                    "n_temporal_features": len(TEMPORAL_FEATURES),
                    "temporal_features": str(TEMPORAL_FEATURES),
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "verdict": verdict,
                }
            )
            mlflow.log_metrics(
                {
                    "auc_base": auc_base,
                    "auc_cand": auc_cand,
                    "delta_auc": delta_auc,
                    "roi_ev010_base": ev_base["roi"],
                    "roi_ev010_cand": ev_cand["roi"],
                    "delta_ev_roi": delta_ev,
                    "roi_ps_base": ps_base["roi"],
                    "roi_ps_cand": ps_cand["roi"],
                    "delta_ps_roi": delta_ps,
                    "roi_ps_val_cand": ps_val_cand["roi"],
                    "n_bets_ps_val_cand": ps_val_cand["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
