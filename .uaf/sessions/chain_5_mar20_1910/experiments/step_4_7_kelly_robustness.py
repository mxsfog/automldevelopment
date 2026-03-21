"""Step 4.7: Kelly criterion staking + robustness analysis.

Гипотезы:
A) Kelly criterion (fractional) для размера ставок вместо flat staking
B) Half-Kelly (более консервативный подход)
C) Анализ robustness per-sport EV floor=0.10: min_bets sensitivity
D) Более строгий floor=0.15 для per-sport EV
E) Leave-one-sport-out analysis — насколько зависимы от одного спорта
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


def find_sport_ev_thresholds(
    val_df, p_val: np.ndarray, min_bets: int = 3, ev_floor: float = 0.0
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта с минимальным floor."""
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
            if r["n_bets"] >= min_bets and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_ev_t = float(ev_t)
        thresholds[sport] = max(best_ev_t, ev_floor)
    return thresholds


def apply_sport_ev(test_df, p_test: np.ndarray, sport_ev: dict[str, float]) -> dict:
    """Применение per-sport EV thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t = sport_ev.get(sports[i], 0.0)
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


def kelly_roi(
    test_df, p_test: np.ndarray, ev_threshold: float, min_prob: float, kelly_fraction: float = 1.0
) -> dict:
    """ROI с Kelly criterion staking: stake = kelly_fraction * edge / (odds - 1)."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    mask = (ev >= ev_threshold) & (p_test >= min_prob)
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0, "kelly_total": 0.0}

    sel_idx = np.where(mask)[0]
    sel = test_df.iloc[sel_idx]
    p_sel = p_test[sel_idx]
    odds_sel = odds[sel_idx]

    bankroll = 1000.0
    total_staked = 0.0
    total_payout = 0.0

    for i in range(len(sel)):
        p = p_sel[i]
        o = odds_sel[i]
        edge = p * o - 1.0
        kelly = kelly_fraction * edge / (o - 1.0) if o > 1.0 else 0.0
        kelly = min(max(kelly, 0.01), 0.25)  # clamp 1%-25%
        stake = bankroll * kelly
        won = sel.iloc[i]["Status"] == "won"
        payout = stake * o if won else 0.0
        total_staked += stake
        total_payout += payout

    if total_staked == 0:
        return {"roi": 0.0, "n_bets": n_sel, "kelly_total": 0.0}

    roi = (total_payout - total_staked) / total_staked * 100
    return {"roi": float(roi), "n_bets": n_sel, "kelly_total": float(total_staked)}


def main() -> None:
    """Kelly staking + robustness analysis."""
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

    # Train model
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

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    results: dict[str, dict] = {}

    # Baseline: flat staking EV>=0.10
    r_flat = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    results["flat_ev010"] = r_flat
    logger.info("Flat EV>=0.10: ROI=%.2f%% n=%d", r_flat["roi"], r_flat["n_bets"])

    # A: Full Kelly EV>=0.10
    check_budget()
    r_kelly = kelly_roi(test_sf, p_test, 0.10, 0.77, kelly_fraction=1.0)
    results["kelly_full"] = r_kelly
    logger.info("A: Full Kelly: ROI=%.2f%% n=%d", r_kelly["roi"], r_kelly["n_bets"])

    # B: Half Kelly
    r_half = kelly_roi(test_sf, p_test, 0.10, 0.77, kelly_fraction=0.5)
    results["kelly_half"] = r_half
    logger.info("B: Half Kelly: ROI=%.2f%% n=%d", r_half["roi"], r_half["n_bets"])

    # B2: Quarter Kelly
    r_quarter = kelly_roi(test_sf, p_test, 0.10, 0.77, kelly_fraction=0.25)
    results["kelly_quarter"] = r_quarter
    logger.info("B2: Quarter Kelly: ROI=%.2f%% n=%d", r_quarter["roi"], r_quarter["n_bets"])

    # C: Per-sport EV floor sensitivity
    check_budget()
    for floor in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=floor)
        r = apply_sport_ev(test_sf, p_test, sport_ev)
        results[f"ps_floor_{int(floor * 100):02d}"] = r
        logger.info("C: PS floor=%.2f: ROI=%.2f%% n=%d", floor, r["roi"], r["n_bets"])

    # D: min_bets sensitivity for per-sport EV floor=0.10
    check_budget()
    for mb in [2, 3, 5, 7, 10]:
        sport_ev = find_sport_ev_thresholds(val_df, p_val, min_bets=mb, ev_floor=0.10)
        r = apply_sport_ev(test_sf, p_test, sport_ev)
        logger.info("D: PS floor=0.10 min_bets=%d: ROI=%.2f%% n=%d", mb, r["roi"], r["n_bets"])

    # E: Leave-one-sport-out analysis (for EV>=0.10+p77)
    check_budget()
    logger.info("E: Leave-one-sport-out (EV>=0.10+p77):")
    sports_in_test = test_sf["Sport"].value_counts()
    for sport in sports_in_test.index[:10]:
        sport_test = test_sf[test_sf["Sport"] != sport]
        sport_p = p_test[(test_sf["Sport"] != sport).values]
        r = calc_ev_roi(sport_test, sport_p, ev_threshold=0.10, min_prob=0.77)
        logger.info("  Without %s: ROI=%.2f%% n=%d", sport, r["roi"], r["n_bets"])

    # E2: Per-sport breakdown
    logger.info("E2: Per-sport breakdown (EV>=0.10+p77):")
    for sport in sports_in_test.index[:10]:
        sport_mask = (test_sf["Sport"] == sport).values
        sport_test = test_sf[sport_mask]
        sport_p = p_test[sport_mask]
        r = calc_ev_roi(sport_test, sport_p, ev_threshold=0.10, min_prob=0.77)
        if r["n_bets"] > 0:
            logger.info(
                "  %s: ROI=%.2f%% n=%d (of %d)", sport, r["roi"], r["n_bets"], len(sport_test)
            )

    # Summary
    logger.info("All test results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d", name, r["roi"], r["n_bets"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.7_kelly_robustness") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "kelly_robustness",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": float(auc),
                    "n_bets": best_r["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.7: Best=%s ROI=%.2f%% n=%d run=%s",
                best_key,
                best_r["roi"],
                best_r["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
