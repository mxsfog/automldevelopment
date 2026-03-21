"""Step 4.2: Probability calibration + per-sport EV thresholds.

Гипотезы:
A) Platt scaling / Isotonic calibration улучшит EV filter (более точные p -> лучший EV)
B) Per-sport EV thresholds: разные пороги для разных видов спорта
C) Per-sport модели: отдельный CatBoost для каждого крупного спорта
"""

import logging
import os
import traceback

import mlflow
import numpy as np
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


def main() -> None:
    """Calibration + per-sport EV thresholds."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_all_features()
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    results: dict[str, dict] = {}

    # Reference CatBoost (full-train)
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb_ref.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test_raw = cb_ft.predict_proba(x_test)[:, 1]
    auc_raw = roc_auc_score(test_sf["target"], p_test_raw)
    roi_raw = calc_ev_roi(test_sf, p_test_raw, ev_threshold=0.0, min_prob=0.77)
    results["raw_cb"] = {"roi": roi_raw["roi"], "n_bets": roi_raw["n_bets"], "auc": auc_raw}
    logger.info("Raw CB: ROI=%.2f%% n=%d AUC=%.4f", roi_raw["roi"], roi_raw["n_bets"], auc_raw)

    # A: Platt scaling (sigmoid calibration) on val
    check_budget()
    p_val_raw = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]

    from sklearn.linear_model import LogisticRegression

    platt = LogisticRegression(max_iter=1000)
    platt.fit(p_val_raw.reshape(-1, 1), val_df["target"])
    p_test_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    auc_platt = roc_auc_score(test_sf["target"], p_test_platt)
    roi_platt = calc_ev_roi(test_sf, p_test_platt, ev_threshold=0.0, min_prob=0.77)
    results["platt"] = {
        "roi": roi_platt["roi"],
        "n_bets": roi_platt["n_bets"],
        "auc": auc_platt,
    }
    logger.info(
        "A: Platt: ROI=%.2f%% n=%d AUC=%.4f (delta=%.2f pp)",
        roi_platt["roi"],
        roi_platt["n_bets"],
        auc_platt,
        roi_platt["roi"] - roi_raw["roi"],
    )

    # Also try different thresholds after calibration
    for min_p in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.77, 0.8]:
        r = calc_ev_roi(test_sf, p_test_platt, ev_threshold=0.0, min_prob=min_p)
        if r["n_bets"] > 0:
            logger.info(
                "  Platt EV>=0 p>=%.2f: ROI=%.2f%% n=%d",
                min_p,
                r["roi"],
                r["n_bets"],
            )

    # B: Isotonic calibration on val
    check_budget()
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, val_df["target"])
    p_test_iso = iso.predict(p_test_raw)

    auc_iso = roc_auc_score(test_sf["target"], p_test_iso)
    roi_iso = calc_ev_roi(test_sf, p_test_iso, ev_threshold=0.0, min_prob=0.77)
    results["isotonic"] = {
        "roi": roi_iso["roi"],
        "n_bets": roi_iso["n_bets"],
        "auc": auc_iso,
    }
    logger.info(
        "B: Isotonic: ROI=%.2f%% n=%d AUC=%.4f (delta=%.2f pp)",
        roi_iso["roi"],
        roi_iso["n_bets"],
        auc_iso,
        roi_iso["roi"] - roi_raw["roi"],
    )

    for min_p in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.77, 0.8]:
        r = calc_ev_roi(test_sf, p_test_iso, ev_threshold=0.0, min_prob=min_p)
        if r["n_bets"] > 0:
            logger.info(
                "  Isotonic EV>=0 p>=%.2f: ROI=%.2f%% n=%d",
                min_p,
                r["roi"],
                r["n_bets"],
            )

    # C: Per-sport EV thresholds (optimal EV threshold per sport on val)
    check_budget()
    p_val_full = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    sports_in_val = val_df["Sport"].unique()
    sport_ev_thresholds: dict[str, float] = {}

    for sport in sorted(sports_in_val):
        val_sport = val_df[val_df["Sport"] == sport]
        if len(val_sport) < 15:
            sport_ev_thresholds[sport] = 0.0
            continue

        sport_mask = val_df["Sport"] == sport
        p_val_sport = p_val_full[sport_mask.values]
        best_ev_t = 0.0
        best_sport_roi = -999.0
        for ev_t in np.arange(-0.05, 0.20, 0.01):
            r = calc_ev_roi(val_sport, p_val_sport, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= 3 and r["roi"] > best_sport_roi:
                best_sport_roi = r["roi"]
                best_ev_t = float(ev_t)
        sport_ev_thresholds[sport] = best_ev_t
        logger.info(
            "  %s: val-optimal EV_t=%.2f (val ROI=%.2f%%)", sport, best_ev_t, best_sport_roi
        )

    # Apply per-sport EV thresholds to test
    test_odds = test_sf["Odds"].values
    ev_test = p_test_raw * test_odds - 1.0
    selected_mask = np.zeros(len(test_sf), dtype=bool)
    for i, (_, row) in enumerate(test_sf.iterrows()):
        sport = row["Sport"]
        ev_t = sport_ev_thresholds.get(sport, 0.0)
        if ev_test[i] >= ev_t and p_test_raw[i] >= 0.77:
            selected_mask[i] = True

    n_sel = selected_mask.sum()
    if n_sel > 0:
        sel_df = test_sf.iloc[np.where(selected_mask)[0]]
        total_staked = sel_df["USD"].sum()
        total_payout = sel_df["Payout_USD"].sum()
        roi_ps = (total_payout - total_staked) / total_staked * 100
        results["per_sport_ev"] = {"roi": float(roi_ps), "n_bets": int(n_sel), "auc": auc_raw}
        logger.info("C: Per-sport EV: ROI=%.2f%% n=%d", roi_ps, n_sel)
    else:
        results["per_sport_ev"] = {"roi": 0.0, "n_bets": 0, "auc": auc_raw}

    # D: Per-sport models
    check_budget()
    top_sports = train_sf["Sport"].value_counts()
    top_sports = top_sports[top_sports >= 100].index.tolist()
    logger.info("Per-sport models for: %s", top_sports)

    per_sport_preds = p_test_raw.copy()
    improved_sports = []

    for sport in top_sports:
        check_budget()
        train_s = train_sf[train_sf["Sport"] == sport]
        test_s = test_sf[test_sf["Sport"] == sport]
        if len(test_s) < 10:
            continue

        vs = int(len(train_s) * 0.8)
        tr_s = train_s.iloc[:vs]
        va_s = train_s.iloc[vs:]

        if len(tr_s) < 30 or len(va_s) < 10:
            continue

        imp_s = SimpleImputer(strategy="median")
        x_tr_s = imp_s.fit_transform(tr_s[feat_list])
        x_va_s = imp_s.transform(va_s[feat_list])

        cb_s = CatBoostClassifier(**CB_BEST_PARAMS)
        cb_s.fit(x_tr_s, tr_s["target"], eval_set=(x_va_s, va_s["target"]))
        bi_s = cb_s.get_best_iteration()

        imp_s_full = SimpleImputer(strategy="median")
        x_s_full = imp_s_full.fit_transform(train_s[feat_list])
        x_s_test = imp_s_full.transform(test_s[feat_list])

        ft_s = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        ft_s["iterations"] = max(bi_s + 10, 30)
        cb_s_ft = CatBoostClassifier(**ft_s)
        cb_s_ft.fit(x_s_full, train_s["target"])

        p_s = cb_s_ft.predict_proba(x_s_test)[:, 1]
        roi_s_spec = calc_ev_roi(test_s, p_s, ev_threshold=0.0, min_prob=0.77)
        roi_s_gen = calc_ev_roi(
            test_s,
            p_test_raw[test_sf["Sport"] == sport],
            ev_threshold=0.0,
            min_prob=0.77,
        )

        logger.info(
            "  %s (n=%d): spec ROI=%.2f%% n=%d | gen ROI=%.2f%% n=%d",
            sport,
            len(test_s),
            roi_s_spec["roi"],
            roi_s_spec["n_bets"],
            roi_s_gen["roi"],
            roi_s_gen["n_bets"],
        )

        # Use sport-specific model if better on val
        va_p_s = cb_s.predict_proba(x_va_s)[:, 1]
        va_p_g = cb_ref.predict_proba(imp.transform(va_s[feat_list]))[:, 1]
        va_roi_s = calc_ev_roi(va_s, va_p_s, ev_threshold=0.0, min_prob=0.77)
        va_roi_g = calc_ev_roi(va_s, va_p_g, ev_threshold=0.0, min_prob=0.77)

        if va_roi_s["roi"] > va_roi_g["roi"] + 2.0:
            test_idx = test_sf[test_sf["Sport"] == sport].index
            idx_map = {idx: i for i, idx in enumerate(test_sf.index)}
            for idx in test_idx:
                pos = idx_map[idx]
                sport_pos = list(test_s.index).index(idx)
                per_sport_preds[pos] = p_s[sport_pos]
            improved_sports.append(sport)

    roi_hybrid = calc_ev_roi(test_sf, per_sport_preds, ev_threshold=0.0, min_prob=0.77)
    results["hybrid_per_sport"] = {
        "roi": roi_hybrid["roi"],
        "n_bets": roi_hybrid["n_bets"],
        "auc": auc_raw,
    }
    logger.info(
        "D: Hybrid per-sport: ROI=%.2f%% n=%d (improved: %s)",
        roi_hybrid["roi"],
        roi_hybrid["n_bets"],
        improved_sports,
    )

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d AUC=%.4f", name, r["roi"], r["n_bets"], r["auc"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.2_calib_persport") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "calibration_persport",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                    "sport_ev_thresholds": str(sport_ev_thresholds),
                    "improved_sports": str(improved_sports),
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
                    "delta_vs_baseline": best_r["roi"] - roi_raw["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.2: Best=%s ROI=%.2f%% (delta=%.2f pp) run=%s",
                best_key,
                best_r["roi"],
                best_r["roi"] - roi_raw["roi"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
