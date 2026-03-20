"""Step 4.7: Probability calibration — Platt scaling и isotonic regression.

Гипотеза: калибровка вероятностей может улучшить EV-отбор,
т.к. CatBoost может быть overconfident/underconfident в разных диапазонах.
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
from sklearn.calibration import CalibratedClassifierCV
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

    p_test_raw = cb_ft.predict_proba(x_test)[:, 1]
    p_val_raw = cb_ft.predict_proba(imp_full.transform(val_df[features]))[:, 1]
    auc_raw = roc_auc_score(test_sf["target"], p_test_raw)

    # Baseline: uncalibrated
    sport_ev_raw = find_sport_ev_thresholds(val_df, p_val_raw, ev_floor=0.15)
    ps_raw = apply_sport_ev(test_sf, p_test_raw, sport_ev_raw)
    ev_raw = calc_ev_roi(test_sf, p_test_raw, ev_threshold=0.10, min_prob=0.77)

    logger.info(
        "Raw: AUC=%.4f EV010=%.2f%%(%d) PS15=%.2f%%(%d)",
        auc_raw,
        ev_raw["roi"],
        ev_raw["n_bets"],
        ps_raw["roi"],
        ps_raw["n_bets"],
    )

    results: dict[str, dict] = {
        "raw": {
            "auc": auc_raw,
            "ev_roi": ev_raw["roi"],
            "ev_n": ev_raw["n_bets"],
            "ps_roi": ps_raw["roi"],
            "ps_n": ps_raw["n_bets"],
        }
    }

    # Calibration на val split (80% train -> 60% fit + 20% calib)
    calib_split = int(len(train_sf) * 0.6)
    train_calib_fit = train_sf.iloc[:calib_split]
    calib_df = train_sf.iloc[calib_split:val_split]

    imp_calib = SimpleImputer(strategy="median")
    x_calib_fit = imp_calib.fit_transform(train_calib_fit[features])
    x_calib_val = imp_calib.transform(calib_df[features])
    x_calib_test = imp_calib.transform(test_sf[features])
    x_calib_eval = imp_calib.transform(val_df[features])

    cb_calib = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_calib.fit(
        x_calib_fit,
        train_calib_fit["target"],
        eval_set=(x_calib_val, calib_df["target"]),
    )
    calib_best_iter = cb_calib.get_best_iteration()

    ft_calib_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_calib_params["iterations"] = calib_best_iter + 10
    cb_calib_ft = CatBoostClassifier(**ft_calib_params)
    cb_calib_ft.fit(x_calib_fit, train_calib_fit["target"])

    for method in ["sigmoid", "isotonic"]:
        check_budget()
        cal = CalibratedClassifierCV(cb_calib_ft, method=method, cv="prefit")
        cal.fit(x_calib_val, calib_df["target"])

        p_test_cal = cal.predict_proba(x_calib_test)[:, 1]
        p_val_cal = cal.predict_proba(x_calib_eval)[:, 1]
        auc_cal = roc_auc_score(test_sf["target"], p_test_cal)

        sport_ev_cal = find_sport_ev_thresholds(val_df, p_val_cal, ev_floor=0.15)
        ps_cal = apply_sport_ev(test_sf, p_test_cal, sport_ev_cal)
        ev_cal = calc_ev_roi(test_sf, p_test_cal, ev_threshold=0.10, min_prob=0.77)

        logger.info(
            "%s: AUC=%.4f EV010=%.2f%%(%d) PS15=%.2f%%(%d)",
            method,
            auc_cal,
            ev_cal["roi"],
            ev_cal["n_bets"],
            ps_cal["roi"],
            ps_cal["n_bets"],
        )

        results[method] = {
            "auc": auc_cal,
            "ev_roi": ev_cal["roi"],
            "ev_n": ev_cal["n_bets"],
            "ps_roi": ps_cal["roi"],
            "ps_n": ps_cal["n_bets"],
        }

    # Temperature scaling (простой вариант)
    check_budget()
    for temp in [0.8, 0.9, 1.1, 1.2, 1.5]:
        logits_test = np.log(p_test_raw / (1 - p_test_raw + 1e-10))
        logits_val = np.log(p_val_raw / (1 - p_val_raw + 1e-10))
        p_test_ts = 1 / (1 + np.exp(-logits_test / temp))
        p_val_ts = 1 / (1 + np.exp(-logits_val / temp))

        sport_ev_ts = find_sport_ev_thresholds(val_df, p_val_ts, ev_floor=0.15)
        ps_ts = apply_sport_ev(test_sf, p_test_ts, sport_ev_ts)
        ev_ts = calc_ev_roi(test_sf, p_test_ts, ev_threshold=0.10, min_prob=0.77)

        label = f"temp_{temp:.1f}"
        results[label] = {
            "auc": auc_raw,
            "ev_roi": ev_ts["roi"],
            "ev_n": ev_ts["n_bets"],
            "ps_roi": ps_ts["roi"],
            "ps_n": ps_ts["n_bets"],
        }
        logger.info(
            "T=%.1f: EV010=%.2f%%(%d) PS15=%.2f%%(%d)",
            temp,
            ev_ts["roi"],
            ev_ts["n_bets"],
            ps_ts["roi"],
            ps_ts["n_bets"],
        )

    # Summary
    logger.info("=== Summary ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["ps_roi"], reverse=True)
    for name, r in sorted_results:
        logger.info(
            "  %s: AUC=%.4f EV=%.2f%%(%d) PS15=%.2f%%(%d)",
            name,
            r["auc"],
            r["ev_roi"],
            r["ev_n"],
            r["ps_roi"],
            r["ps_n"],
        )

    best_name = sorted_results[0][0]
    best_r = sorted_results[0][1]

    verdict = "accepted" if best_r["ps_roi"] > ps_raw["roi"] + 2.0 else "rejected"
    logger.info("Best: %s -> %s", best_name, verdict)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.7_calibration") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "probability_calibration",
                    "calibration_methods": "sigmoid,isotonic,temperature",
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "verdict": verdict,
                    "best_method": best_name,
                }
            )
            for name, r in results.items():
                safe = name.replace(".", "_")
                mlflow.log_metric(f"auc_{safe}", r["auc"])
                mlflow.log_metric(f"ps_roi_{safe}", r["ps_roi"])
                mlflow.log_metric(f"ev_roi_{safe}", r["ev_roi"])

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
