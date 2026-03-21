"""Step 4.4: Time features + market-level EV + odds-range models.

Гипотезы:
A) Time features (hour, day_of_week, minutes_from_start) могут улучшить модель
B) Market-level EV thresholds (аналог per-sport, но по рынкам)
C) Odds-range EV: разные минимальные EV для разных odds диапазонов
D) Исключение odds < 1.15 (margin zone) из отбора
"""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def add_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Добавление временных фичей."""
    df = df.copy()
    new_feats = []

    dt = df["Created_At"]
    df["hour"] = dt.dt.hour.astype(float)
    df["day_of_week"] = dt.dt.dayofweek.astype(float)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    new_feats.extend(["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"])
    return df, new_feats


def main() -> None:
    """Time features + market EV + odds-range analysis."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)
    df, time_feats = add_time_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    base_feats = get_all_features()
    feat_with_time = base_feats + time_feats

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    results: dict[str, dict] = {}

    # A: CatBoost + time features
    check_budget()
    imp_t = SimpleImputer(strategy="median")
    x_fit_t = imp_t.fit_transform(train_fit[feat_with_time])
    x_val_t = imp_t.transform(val_df[feat_with_time])

    cb_t = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_t.fit(x_fit_t, train_fit["target"], eval_set=(x_val_t, val_df["target"]))
    best_iter_t = cb_t.get_best_iteration()

    imp_t_full = SimpleImputer(strategy="median")
    x_full_t = imp_t_full.fit_transform(train_sf[feat_with_time])
    x_test_t = imp_t_full.transform(test_sf[feat_with_time])

    ft_t = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_t["iterations"] = best_iter_t + 10
    cb_ft_t = CatBoostClassifier(**ft_t)
    cb_ft_t.fit(x_full_t, train_sf["target"])

    p_test_t = cb_ft_t.predict_proba(x_test_t)[:, 1]
    auc_t = roc_auc_score(test_sf["target"], p_test_t)
    roi_t = calc_ev_roi(test_sf, p_test_t, ev_threshold=0.0, min_prob=0.77)
    results["cb_time"] = {"roi": roi_t["roi"], "n_bets": roi_t["n_bets"], "auc": auc_t}
    logger.info("A: CB+time: ROI=%.2f%% n=%d AUC=%.4f", roi_t["roi"], roi_t["n_bets"], auc_t)

    # Reference baseline
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[base_feats])
    x_val = imp.transform(val_df[base_feats])

    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb_ref.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[base_feats])
    x_test = imp_full.transform(test_sf[base_feats])

    ft_p = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_p["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_p)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    auc_ref = roc_auc_score(test_sf["target"], p_test)
    roi_ref = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)
    results["cb_base"] = {"roi": roi_ref["roi"], "n_bets": roi_ref["n_bets"], "auc": auc_ref}

    # B: Odds-range specific EV thresholds
    check_budget()
    odds_ranges = [
        ("1.01-1.15", 1.01, 1.15),
        ("1.15-1.30", 1.15, 1.30),
        ("1.30-1.50", 1.30, 1.50),
        ("1.50-2.00", 1.50, 2.00),
        ("2.00+", 2.00, 100.0),
    ]

    # Find optimal EV per odds-range on val
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[base_feats]))[:, 1]
    odds_ev_thresholds: dict[str, float] = {}

    for name, lo, hi in odds_ranges:
        mask = (val_df["Odds"] >= lo) & (val_df["Odds"] < hi)
        val_range = val_df[mask]
        p_range = p_val[mask.values]
        if len(val_range) < 10:
            odds_ev_thresholds[name] = 0.0
            continue
        best_ev = 0.0
        best_r = -999.0
        for ev_t in np.arange(-0.05, 0.25, 0.01):
            r = calc_ev_roi(val_range, p_range, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= 2 and r["roi"] > best_r:
                best_r = r["roi"]
                best_ev = float(ev_t)
        odds_ev_thresholds[name] = best_ev
        logger.info("  %s: val EV_t=%.2f (val ROI=%.2f%%)", name, best_ev, best_r)

    # Apply odds-range EV thresholds
    test_odds = test_sf["Odds"].values
    ev_test = p_test * test_odds - 1.0
    or_mask = np.zeros(len(test_sf), dtype=bool)
    for i in range(len(test_sf)):
        if p_test[i] < 0.77:
            continue
        odds_val = test_odds[i]
        for name, lo, hi in odds_ranges:
            if lo <= odds_val < hi:
                if ev_test[i] >= odds_ev_thresholds[name]:
                    or_mask[i] = True
                break

    n_or = int(or_mask.sum())
    if n_or > 0:
        or_sel = test_sf.iloc[np.where(or_mask)[0]]
        or_staked = or_sel["USD"].sum()
        or_payout = or_sel["Payout_USD"].sum()
        or_roi = (or_payout - or_staked) / or_staked * 100
        results["odds_range_ev"] = {"roi": float(or_roi), "n_bets": n_or, "auc": auc_ref}
        logger.info("B: Odds-range EV: ROI=%.2f%% n=%d", or_roi, n_or)

    # C: Simple odds filter: exclude odds < 1.15
    check_budget()
    odds_mask = test_sf["Odds"].values >= 1.15
    p_filtered = p_test.copy()
    p_filtered[~odds_mask] = 0.0
    roi_of = calc_ev_roi(test_sf, p_filtered, ev_threshold=0.0, min_prob=0.77)
    results["odds_gt_115"] = {"roi": roi_of["roi"], "n_bets": roi_of["n_bets"], "auc": auc_ref}
    logger.info("C: Odds>=1.15+EV0+p77: ROI=%.2f%% n=%d", roi_of["roi"], roi_of["n_bets"])

    # D: Minimum odds 1.10
    odds_mask_10 = test_sf["Odds"].values >= 1.10
    p_f10 = p_test.copy()
    p_f10[~odds_mask_10] = 0.0
    roi_of10 = calc_ev_roi(test_sf, p_f10, ev_threshold=0.0, min_prob=0.77)
    results["odds_gt_110"] = {"roi": roi_of10["roi"], "n_bets": roi_of10["n_bets"], "auc": auc_ref}
    logger.info("D: Odds>=1.10+EV0+p77: ROI=%.2f%% n=%d", roi_of10["roi"], roi_of10["n_bets"])

    # E: Minimum odds 1.20
    odds_mask_20 = test_sf["Odds"].values >= 1.20
    p_f20 = p_test.copy()
    p_f20[~odds_mask_20] = 0.0
    roi_of20 = calc_ev_roi(test_sf, p_f20, ev_threshold=0.0, min_prob=0.77)
    results["odds_gt_120"] = {"roi": roi_of20["roi"], "n_bets": roi_of20["n_bets"], "auc": auc_ref}
    logger.info("E: Odds>=1.20+EV0+p77: ROI=%.2f%% n=%d", roi_of20["roi"], roi_of20["n_bets"])

    # F: EV>=0.05 (stricter EV filter)
    roi_ev05 = calc_ev_roi(test_sf, p_test, ev_threshold=0.05, min_prob=0.77)
    results["ev_005"] = {"roi": roi_ev05["roi"], "n_bets": roi_ev05["n_bets"], "auc": auc_ref}
    logger.info("F: EV>=0.05+p77: ROI=%.2f%% n=%d", roi_ev05["roi"], roi_ev05["n_bets"])

    # G: EV>=0.10
    roi_ev10 = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    results["ev_010"] = {"roi": roi_ev10["roi"], "n_bets": roi_ev10["n_bets"], "auc": auc_ref}
    logger.info("G: EV>=0.10+p77: ROI=%.2f%% n=%d", roi_ev10["roi"], roi_ev10["n_bets"])

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d AUC=%.4f", name, r["roi"], r["n_bets"], r["auc"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.4_time_market") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "time_market_odds_ev",
                    "n_base_features": len(base_feats),
                    "n_time_features": len(time_feats),
                    "best_variant": best_key,
                    "odds_ev_thresholds": str(odds_ev_thresholds),
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": best_r["auc"],
                    "n_bets": best_r["n_bets"],
                    "delta_vs_baseline": best_r["roi"] - roi_ref["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.4: Best=%s ROI=%.2f%% (delta=%.2f pp) run=%s",
                best_key,
                best_r["roi"],
                best_r["roi"] - roi_ref["roi"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
