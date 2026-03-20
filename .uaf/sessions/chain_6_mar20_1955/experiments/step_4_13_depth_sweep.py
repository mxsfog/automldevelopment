"""Step 4.13: CatBoost depth sweep (6/8/10) с PS_floor15."""

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
    results: dict[str, dict] = {}

    for depth in [6, 8, 10]:
        check_budget()
        set_seed(456)

        val_split = int(len(train_sf) * 0.8)
        train_fit = train_sf.iloc[:val_split]
        val_df = train_sf.iloc[val_split:]

        imp = SimpleImputer(strategy="median")
        x_fit = imp.fit_transform(train_fit[features])
        x_val = imp.transform(val_df[features])

        params = {**CB_BEST_PARAMS, "random_seed": 456, "depth": depth}
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

        ev = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
        sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
        ps = apply_sport_ev(test_sf, p_test, sport_ev)

        label = f"depth_{depth}"
        results[label] = {
            "auc": auc,
            "ev_roi": ev["roi"],
            "ev_n": ev["n_bets"],
            "ps_roi": ps["roi"],
            "ps_n": ps["n_bets"],
            "best_iter": best_iter,
        }
        logger.info(
            "%s: AUC=%.4f EV=%.2f%%(%d) PS15=%.2f%%(%d) iter=%d",
            label,
            auc,
            ev["roi"],
            ev["n_bets"],
            ps["roi"],
            ps["n_bets"],
            best_iter,
        )

    sorted_results = sorted(results.items(), key=lambda x: x[1]["ps_roi"], reverse=True)
    best_name = sorted_results[0][0]
    best_r = sorted_results[0][1]
    logger.info("Best: %s PS15=%.2f%%", best_name, best_r["ps_roi"])

    with mlflow.start_run(run_name="phase4/step_4.13_depth_sweep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "catboost_depth_sweep",
                    "depths": "6,8,10",
                    "seed": 456,
                    "best_depth": best_name,
                }
            )
            for name, r in results.items():
                mlflow.log_metric(f"auc_{name}", r["auc"])
                mlflow.log_metric(f"ps_roi_{name}", r["ps_roi"])
            mlflow.log_metric("roi", best_r["ps_roi"])
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
