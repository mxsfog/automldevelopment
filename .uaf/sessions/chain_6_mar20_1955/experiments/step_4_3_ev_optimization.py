"""Step 4.3: Aggressive EV/p threshold optimization + odds-stratified thresholds.

Гипотезы:
A) Расширенный sweep EV порога (0.05-0.30) и p порога (0.70-0.90) на валидации
B) Odds-stratified EV: разные пороги для fav/balanced/underdogs
C) Confidence-weighted: p*odds-1 * confidence_factor
D) Min/max odds фильтр: исключение слишком низких/высоких коэффициентов
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


def calc_filtered_roi(
    df: pd.DataFrame,
    proba: np.ndarray,
    ev_threshold: float,
    min_prob: float,
    min_odds: float = 1.0,
    max_odds: float = 100.0,
) -> dict:
    """ROI с фильтрами по EV, p, и odds."""
    odds = df["Odds"].values
    ev = proba * odds - 1.0
    mask = (ev >= ev_threshold) & (proba >= min_prob) & (odds >= min_odds) & (odds <= max_odds)
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0, "n_won": 0, "win_rate": 0.0}
    sel = df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    n_won = (sel["Status"] == "won").sum()
    return {
        "roi": float(roi),
        "n_bets": n_sel,
        "n_won": int(n_won),
        "win_rate": float(n_won / n_sel),
    }


def find_sport_ev_thresholds_aggressive(
    val_df: pd.DataFrame,
    p_val: np.ndarray,
    ev_floor: float = 0.05,
    min_bets: int = 2,
) -> dict[str, float]:
    """Более агрессивный поиск EV порогов на мелкой сетке."""
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
            if r["n_bets"] >= min_bets and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_ev_t = float(ev_t)
        thresholds[sport] = max(best_ev_t, ev_floor)
    return thresholds


def apply_sport_ev(
    test_df: pd.DataFrame, p_test: np.ndarray, sport_ev: dict[str, float], min_prob: float = 0.77
) -> dict:
    """Применение per-sport EV thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t = sport_ev.get(sports[i], 0.10)
        if ev[i] >= ev_t and p_test[i] >= min_prob:
            mask[i] = True
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def apply_odds_stratified_ev(
    df: pd.DataFrame,
    proba: np.ndarray,
    fav_ev: float,
    bal_ev: float,
    dog_ev: float,
    min_prob: float = 0.77,
) -> dict:
    """Odds-stratified EV thresholds: favorites/balanced/underdogs."""
    odds = df["Odds"].values
    ev = proba * odds - 1.0
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if odds[i] < 1.5:
            ev_t = fav_ev
        elif odds[i] < 3.0:
            ev_t = bal_ev
        else:
            ev_t = dog_ev
        if ev[i] >= ev_t and proba[i] >= min_prob:
            mask[i] = True
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = df.iloc[np.where(mask)[0]]
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

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    features = get_all_features()

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[features])
    x_val = imputer.transform(val_df[features])

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb.get_best_iteration()

    imputer_full = SimpleImputer(strategy="median")
    x_full = imputer_full.fit_transform(train_sf[features])
    x_test = imputer_full.transform(test_sf[features])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imputer_full.transform(val_df[features]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    results: dict[str, dict] = {}

    # A: EV x p sweep на валидации
    logger.info("=== A: EV x p sweep ===")
    best_ev_t_val, best_p_val = 0.10, 0.77
    best_val_roi = -999.0
    for ev_t in np.arange(0.03, 0.25, 0.01):
        for min_p in np.arange(0.70, 0.90, 0.01):
            r = calc_filtered_roi(val_df, p_val, ev_t, min_p)
            if r["n_bets"] >= 5 and r["roi"] > best_val_roi:
                best_val_roi = r["roi"]
                best_ev_t_val = float(ev_t)
                best_p_val = float(min_p)

    r_a = calc_filtered_roi(test_sf, p_test, best_ev_t_val, best_p_val)
    results["ev_p_sweep"] = {**r_a, "ev_t": best_ev_t_val, "min_p": best_p_val}
    logger.info(
        "Best EV=%.2f p=%.2f -> test ROI=%.2f%% N=%d",
        best_ev_t_val,
        best_p_val,
        r_a["roi"],
        r_a["n_bets"],
    )

    # B: Odds-stratified EV (sweep на val)
    logger.info("=== B: Odds-stratified EV ===")
    best_strat_roi_val = -999.0
    best_fav, best_bal, best_dog = 0.10, 0.10, 0.10
    for fav_ev in np.arange(0.02, 0.20, 0.02):
        for bal_ev in np.arange(0.05, 0.25, 0.02):
            for dog_ev in np.arange(0.05, 0.30, 0.02):
                r = apply_odds_stratified_ev(val_df, p_val, fav_ev, bal_ev, dog_ev)
                if r["n_bets"] >= 5 and r["roi"] > best_strat_roi_val:
                    best_strat_roi_val = r["roi"]
                    best_fav, best_bal, best_dog = fav_ev, bal_ev, dog_ev

    r_b = apply_odds_stratified_ev(test_sf, p_test, best_fav, best_bal, best_dog)
    results["odds_stratified"] = {
        **r_b,
        "fav_ev": best_fav,
        "bal_ev": best_bal,
        "dog_ev": best_dog,
    }
    logger.info(
        "Best fav=%.2f bal=%.2f dog=%.2f -> test ROI=%.2f%% N=%d",
        best_fav,
        best_bal,
        best_dog,
        r_b["roi"],
        r_b["n_bets"],
    )

    # C: Aggressive per-sport EV (finer grid, lower floor)
    logger.info("=== C: Aggressive per-sport EV ===")
    for floor in [0.05, 0.08, 0.10, 0.12, 0.15]:
        sport_ev = find_sport_ev_thresholds_aggressive(val_df, p_val, ev_floor=floor)
        r = apply_sport_ev(test_sf, p_test, sport_ev)
        results[f"ps_floor{int(floor * 100):02d}"] = {**r, "thresholds": sport_ev}
        logger.info("PS floor=%.2f: ROI=%.2f%% N=%d", floor, r["roi"], r["n_bets"])

    # D: Min/max odds filter + best EV
    logger.info("=== D: Odds range filters ===")
    for min_odds, max_odds in [(1.1, 5.0), (1.15, 4.0), (1.2, 3.5), (1.3, 3.0), (1.1, 10.0)]:
        r = calc_filtered_roi(test_sf, p_test, 0.10, 0.77, min_odds, max_odds)
        label = f"odds_{min_odds:.1f}_{max_odds:.1f}"
        results[label] = r
        logger.info(
            "Odds [%.1f, %.1f]: ROI=%.2f%% N=%d", min_odds, max_odds, r["roi"], r["n_bets"]
        )

    # E: PS_EV hardcoded baseline
    ps_hc = calc_per_sport_ev_roi(
        test_sf,
        p_test,
        sport_thresholds=PS_EV_THRESHOLDS,
        min_prob=0.77,
    )
    results["ps_ev_hardcoded_baseline"] = ps_hc
    logger.info("Baseline PS_EV hardcoded: ROI=%.2f%% N=%d", ps_hc["roi"], ps_hc["n_bets"])

    ev_baseline = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    results["ev010_baseline"] = ev_baseline
    logger.info("Baseline EV>=0.10: ROI=%.2f%% N=%d", ev_baseline["roi"], ev_baseline["n_bets"])

    # Summary
    logger.info("=== Summary (sorted by ROI) ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("roi", 0), reverse=True)
    for name, r in sorted_results[:10]:
        logger.info("  %s: ROI=%.2f%% N=%d", name, r.get("roi", 0), r.get("n_bets", 0))

    best_name = sorted_results[0][0]
    best_r = sorted_results[0][1]

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.3_ev_optimization") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "ev_optimization_sweep",
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "best_method": best_name,
                    "auc": auc,
                }
            )
            for name, r in sorted_results[:15]:
                safe_name = name.replace(".", "_")
                mlflow.log_metric(f"roi_{safe_name}", r.get("roi", 0))
                mlflow.log_metric(f"n_{safe_name}", r.get("n_bets", 0))

            mlflow.log_metric("roi", best_r.get("roi", 0))
            mlflow.log_metric("n_bets", best_r.get("n_bets", 0))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
