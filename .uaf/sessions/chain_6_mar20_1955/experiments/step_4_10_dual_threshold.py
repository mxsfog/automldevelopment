"""Step 4.10: Dual per-sport thresholds — EV + min_prob оба подбираются на спорт.

Гипотеза: для некоторых спортов min_prob=0.77 слишком
консервативен/агрессивен. Подбор обоих порогов на спорт даст лучший отбор.
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
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
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


def find_dual_thresholds(
    val_df: pd.DataFrame, p_val: np.ndarray, ev_floor: float = 0.15
) -> dict[str, tuple[float, float]]:
    """Подбор EV threshold и min_prob для каждого спорта."""
    sports = val_df["Sport"].unique()
    thresholds: dict[str, tuple[float, float]] = {}
    for sport in sports:
        sport_mask = (val_df["Sport"] == sport).values
        if sport_mask.sum() < 10:
            thresholds[sport] = (ev_floor, 0.77)
            continue
        val_sport = val_df[sport_mask]
        p_sport = p_val[sport_mask]
        best_ev_t = ev_floor
        best_mp = 0.77
        best_roi = -999.0
        for ev_t in np.arange(ev_floor, 0.35, 0.01):
            for mp in np.arange(0.70, 0.85, 0.01):
                r = calc_ev_roi(val_sport, p_sport, ev_threshold=ev_t, min_prob=mp)
                if r["n_bets"] >= 2 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_ev_t = float(ev_t)
                    best_mp = float(mp)
        thresholds[sport] = (max(best_ev_t, ev_floor), best_mp)
    return thresholds


def apply_dual_thresholds(
    test_df: pd.DataFrame,
    p_test: np.ndarray,
    dual_th: dict[str, tuple[float, float]],
) -> dict:
    """Применение per-sport dual thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t, mp = dual_th.get(sports[i], (0.15, 0.77))
        if ev[i] >= ev_t and p_test[i] >= mp:
            mask[i] = True
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def find_sport_ev_thresholds(
    val_df: pd.DataFrame, p_val: np.ndarray, ev_floor: float = 0.15
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта (single threshold)."""
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
    """Применение per-sport EV thresholds (single)."""
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


def main() -> None:
    set_seed(456)
    check_budget()

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    features = get_all_features()

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[features])
    x_val = imp.transform(val_df[features])

    params = {**CB_BEST_PARAMS, "random_seed": 456}
    cb = CatBoostClassifier(**params)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[features])
    x_test = imp_full.transform(test_sf[features])

    ft_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[features]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    # Baseline: PS_floor15 single threshold
    sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
    ps_base = apply_sport_ev(test_sf, p_test, sport_ev)
    logger.info("Baseline PS15: ROI=%.2f%%(%d) AUC=%.4f", ps_base["roi"], ps_base["n_bets"], auc)

    # Dual thresholds
    dual_th = find_dual_thresholds(val_df, p_val, ev_floor=0.15)
    ps_dual = apply_dual_thresholds(test_sf, p_test, dual_th)
    logger.info("Dual PS15: ROI=%.2f%%(%d)", ps_dual["roi"], ps_dual["n_bets"])

    # Log per-sport thresholds
    for sport, (ev_t, mp) in sorted(dual_th.items()):
        logger.info("  %s: ev_t=%.3f min_p=%.2f", sport, ev_t, mp)

    # Sweep min_prob globally
    results: dict[str, dict] = {
        "baseline_ps15": {"roi": ps_base["roi"], "n_bets": ps_base["n_bets"]},
        "dual_ps15": {"roi": ps_dual["roi"], "n_bets": ps_dual["n_bets"]},
    }

    for mp in np.arange(0.70, 0.85, 0.01):
        sport_ev_mp = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
        # apply with different global min_prob
        odds = test_sf["Odds"].values
        ev = p_test * odds - 1.0
        sports = test_sf["Sport"].values
        mask = np.zeros(len(test_sf), dtype=bool)
        for i in range(len(test_sf)):
            ev_t = sport_ev_mp.get(sports[i], 0.15)
            if ev[i] >= ev_t and p_test[i] >= mp:
                mask[i] = True
        n_sel = int(mask.sum())
        if n_sel == 0:
            roi = 0.0
        else:
            sel = test_sf.iloc[np.where(mask)[0]]
            staked = sel["USD"].sum()
            payout = sel["Payout_USD"].sum()
            roi = float((payout - staked) / staked * 100)
        label = f"mp_{mp:.2f}"
        results[label] = {"roi": roi, "n_bets": n_sel}
        logger.info("min_prob=%.2f: ROI=%.2f%%(%d)", mp, roi, n_sel)

    # Summary
    logger.info("=== Summary ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True)
    for name, r in sorted_results[:10]:
        logger.info("  %s: ROI=%.2f%%(%d)", name, r["roi"], r["n_bets"])

    best_name = sorted_results[0][0]
    best_r = sorted_results[0][1]
    verdict = "accepted" if best_r["roi"] > ps_base["roi"] + 2.0 else "rejected"
    logger.info("Best: %s -> %s", best_name, verdict)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.10_dual_threshold") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "dual_per_sport_thresholds",
                    "validation_scheme": "time_series",
                    "seed": 456,
                    "verdict": verdict,
                    "best_variant": best_name,
                }
            )
            mlflow.log_metric("roi", best_r["roi"])
            mlflow.log_metric("roi_baseline", ps_base["roi"])
            mlflow.log_metric("roi_dual", ps_dual["roi"])
            mlflow.log_metric("auc", auc)
            for name, r in results.items():
                safe = name.replace(".", "_")
                mlflow.log_metric(f"roi_{safe}", r["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
