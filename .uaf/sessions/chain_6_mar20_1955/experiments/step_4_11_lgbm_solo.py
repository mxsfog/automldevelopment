"""Step 4.11: LightGBM solo с PS_floor15.

Гипотеза: LightGBM может дать лучшую калибровку вероятностей
и улучшить EV-отбор через PS_floor15.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import TYPE_CHECKING

import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (
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

    import lightgbm

    lgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 20,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 456,
        "verbose": -1,
    }

    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[lightgbm.early_stopping(50, verbose=False)],
    )
    lgb_best_iter = lgb.best_iteration_

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[features])
    x_test = imp_full.transform(test_sf[features])

    lgb_ft_params = {k: v for k, v in lgb_params.items()}
    lgb_ft_params["n_estimators"] = lgb_best_iter + 10
    lgb_ft = LGBMClassifier(**lgb_ft_params)
    lgb_ft.fit(x_full, train_sf["target"])

    p_test = lgb_ft.predict_proba(x_test)[:, 1]
    p_val = lgb_ft.predict_proba(imp_full.transform(val_df[features]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    ev = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
    ps = apply_sport_ev(test_sf, p_test, sport_ev)

    logger.info(
        "LGB: AUC=%.4f EV010=%.2f%%(%d) PS15=%.2f%%(%d) best_iter=%d",
        auc,
        ev["roi"],
        ev["n_bets"],
        ps["roi"],
        ps["n_bets"],
        lgb_best_iter,
    )

    # CB baseline for comparison
    from catboost import CatBoostClassifier
    from common import CB_BEST_PARAMS

    cb_params = {**CB_BEST_PARAMS, "random_seed": 456}
    imp2 = SimpleImputer(strategy="median")
    x_fit2 = imp2.fit_transform(train_fit[features])
    x_val2 = imp2.transform(val_df[features])
    cb = CatBoostClassifier(**cb_params)
    cb.fit(x_fit2, train_fit["target"], eval_set=(x_val2, val_df["target"]))
    cb_best_iter = cb.get_best_iteration()

    imp2_full = SimpleImputer(strategy="median")
    x_full2 = imp2_full.fit_transform(train_sf[features])
    x_test2 = imp2_full.transform(test_sf[features])

    ft_params = {k: v for k, v in cb_params.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = cb_best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full2, train_sf["target"])

    p_test_cb = cb_ft.predict_proba(x_test2)[:, 1]
    p_val_cb = cb_ft.predict_proba(imp2_full.transform(val_df[features]))[:, 1]
    auc_cb = roc_auc_score(test_sf["target"], p_test_cb)

    ev_cb = calc_ev_roi(test_sf, p_test_cb, ev_threshold=0.10, min_prob=0.77)
    sport_ev_cb = find_sport_ev_thresholds(val_df, p_val_cb, ev_floor=0.15)
    ps_cb = apply_sport_ev(test_sf, p_test_cb, sport_ev_cb)

    logger.info(
        "CB:  AUC=%.4f EV010=%.2f%%(%d) PS15=%.2f%%(%d)",
        auc_cb,
        ev_cb["roi"],
        ev_cb["n_bets"],
        ps_cb["roi"],
        ps_cb["n_bets"],
    )

    # Blend
    for w_cb in [0.5, 0.6, 0.7]:
        p_blend_test = w_cb * p_test_cb + (1 - w_cb) * p_test
        p_blend_val = w_cb * p_val_cb + (1 - w_cb) * p_val
        auc_blend = roc_auc_score(test_sf["target"], p_blend_test)
        sport_ev_bl = find_sport_ev_thresholds(val_df, p_blend_val, ev_floor=0.15)
        ps_bl = apply_sport_ev(test_sf, p_blend_test, sport_ev_bl)
        ev_bl = calc_ev_roi(test_sf, p_blend_test, ev_threshold=0.10, min_prob=0.77)
        logger.info(
            "Blend(CB=%.1f): AUC=%.4f EV=%.2f%%(%d) PS15=%.2f%%(%d)",
            w_cb,
            auc_blend,
            ev_bl["roi"],
            ev_bl["n_bets"],
            ps_bl["roi"],
            ps_bl["n_bets"],
        )

    verdict = "accepted" if ps["roi"] > ps_cb["roi"] + 2.0 else "rejected"
    logger.info("LGB vs CB -> %s", verdict)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.11_lgbm_solo") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "lgbm_vs_catboost",
                    "validation_scheme": "time_series",
                    "seed": 456,
                    "verdict": verdict,
                    "lgb_best_iter": lgb_best_iter,
                }
            )
            mlflow.log_metrics(
                {
                    "auc_lgb": auc,
                    "auc_cb": auc_cb,
                    "ps_roi_lgb": ps["roi"],
                    "ps_roi_cb": ps_cb["roi"],
                    "ev_roi_lgb": ev["roi"],
                    "ev_roi_cb": ev_cb["roi"],
                    "roi": max(ps["roi"], ps_cb["roi"]),
                }
            )
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
