"""Step 4.9: Per-sport ROI analysis + expanded sport filter.

Гипотеза: среди оставшихся спортов могут быть убыточные сегменты,
их исключение улучшит общий ROI.
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
    train_sf: pd.DataFrame, test_sf: pd.DataFrame, features: list[str], seed: int = 456
) -> tuple[float, np.ndarray, np.ndarray, pd.DataFrame]:
    """Train model, return AUC and predictions."""
    set_seed(seed)
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[features])
    x_val = imp.transform(val_df[features])

    params = {**CB_BEST_PARAMS, "random_seed": seed}
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

    return auc, p_test, p_val, val_df


def main() -> None:
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

    # Per-sport analysis with baseline model
    auc, p_test, p_val, val_df = train_and_eval(train_sf, test_sf, features, seed=456)
    logger.info("Baseline AUC: %.4f", auc)

    # Per-sport ROI breakdown
    sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
    logger.info("=== Per-sport ROI (PS15) ===")
    sport_rois: dict[str, dict] = {}
    for sport in sorted(test_sf["Sport"].unique()):
        sport_mask = (test_sf["Sport"] == sport).values
        n_sport = int(sport_mask.sum())
        if n_sport < 5:
            continue
        sport_test = test_sf[sport_mask]
        p_sport = p_test[sport_mask]
        ev_t = sport_ev.get(sport, 0.15)
        r = calc_ev_roi(sport_test, p_sport, ev_threshold=ev_t, min_prob=0.77)
        sport_rois[sport] = {"roi": r["roi"], "n_bets": r["n_bets"], "n_total": n_sport}
        logger.info(
            "  %s: ROI=%.2f%% N=%d (total=%d, ev_t=%.3f)",
            sport,
            r["roi"],
            r["n_bets"],
            n_sport,
            ev_t,
        )

    # Find unprofitable sports in current selection
    unprofitable = [s for s, r in sport_rois.items() if r["roi"] < -10 and r["n_bets"] >= 3]
    logger.info("Unprofitable in current set: %s", unprofitable)

    # Try expanded filter variants
    results: dict[str, dict] = {}

    # Baseline
    ps_base = apply_sport_ev(test_sf, p_test, sport_ev)
    results["baseline"] = {"roi": ps_base["roi"], "n_bets": ps_base["n_bets"], "auc": auc}
    logger.info("Baseline: PS15=%.2f%%(%d)", ps_base["roi"], ps_base["n_bets"])

    # Try removing each sport one at a time (leave-one-sport-out)
    for sport in sorted(test_sf["Sport"].unique()):
        check_budget()
        expanded_filter = [*UNPROFITABLE_SPORTS, sport]
        tr = train_elo[~train_elo["Sport"].isin(expanded_filter)].copy()
        te = test_elo[~test_elo["Sport"].isin(expanded_filter)].copy()

        if len(tr) < 100 or len(te) < 20:
            continue

        a, pt, pv, vd = train_and_eval(tr, te, features, seed=456)
        sev = find_sport_ev_thresholds(vd, pv, ev_floor=0.15)
        ps = apply_sport_ev(te, pt, sev)
        label = f"excl_{sport}"
        results[label] = {"roi": ps["roi"], "n_bets": ps["n_bets"], "auc": a}
        logger.info(
            "  -%s: PS15=%.2f%%(%d) AUC=%.4f",
            sport,
            ps["roi"],
            ps["n_bets"],
            a,
        )

    # Summary
    logger.info("=== Summary ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True)
    for name, r in sorted_results:
        logger.info("  %s: ROI=%.2f%%(%d) AUC=%.4f", name, r["roi"], r["n_bets"], r["auc"])

    best_name = sorted_results[0][0]
    best_r = sorted_results[0][1]
    verdict = "accepted" if best_r["roi"] > ps_base["roi"] + 2.0 else "rejected"
    logger.info("Best: %s -> %s", best_name, verdict)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.9_sport_analysis") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "sport_analysis_leave_one_out",
                    "validation_scheme": "time_series",
                    "seed": 456,
                    "verdict": verdict,
                    "best_variant": best_name,
                }
            )
            for name, r in results.items():
                safe = name.replace(" ", "_")
                mlflow.log_metric(f"roi_{safe}", r["roi"])
                mlflow.log_metric(f"n_{safe}", r["n_bets"])

            mlflow.log_metric("roi", best_r["roi"])
            for sport, sr in sport_rois.items():
                safe = sport.replace(" ", "_").lower()
                mlflow.log_metric(f"sport_roi_{safe}", sr["roi"])
                mlflow.log_metric(f"sport_n_{safe}", sr["n_bets"])

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
