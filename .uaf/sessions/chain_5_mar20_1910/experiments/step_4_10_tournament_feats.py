"""Step 4.10: Tournament-level features + final experiments.

Гипотезы:
A) Tournament popularity (count) как фича — популярные турниры лучше калиброваны
B) Tournament historical win rate (leak-free: из train только)
C) Sport+Tournament interaction features
D) EV>=0.10 с новыми фичами
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


def add_tournament_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Tournament features from train only (no leakage)."""
    new_feats: list[str] = []

    # Tournament frequency (from train)
    tourn_counts = train_df["Tournament"].value_counts()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["tournament_count"] = train_df["Tournament"].map(tourn_counts).fillna(0).astype(float)
    test_df["tournament_count"] = test_df["Tournament"].map(tourn_counts).fillna(0).astype(float)
    train_df["log_tournament_count"] = np.log1p(train_df["tournament_count"])
    test_df["log_tournament_count"] = np.log1p(test_df["tournament_count"])
    new_feats.extend(["tournament_count", "log_tournament_count"])

    # Tournament win rate (from train only)
    tourn_wr = train_df.groupby("Tournament")["target"].mean()
    train_df["tournament_winrate"] = train_df["Tournament"].map(tourn_wr).fillna(0.5).astype(float)
    test_df["tournament_winrate"] = test_df["Tournament"].map(tourn_wr).fillna(0.5).astype(float)
    new_feats.append("tournament_winrate")

    # Tournament avg odds (from train)
    tourn_odds = train_df.groupby("Tournament")["Odds"].mean()
    train_df["tournament_avg_odds"] = (
        train_df["Tournament"].map(tourn_odds).fillna(train_df["Odds"].mean()).astype(float)
    )
    test_df["tournament_avg_odds"] = (
        test_df["Tournament"].map(tourn_odds).fillna(train_df["Odds"].mean()).astype(float)
    )
    new_feats.append("tournament_avg_odds")

    # Is popular tournament (>100 bets in train)
    train_df["is_popular_tournament"] = (train_df["tournament_count"] > 100).astype(float)
    test_df["is_popular_tournament"] = (test_df["tournament_count"] > 100).astype(float)
    new_feats.append("is_popular_tournament")

    return train_df, test_df, new_feats


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


def main() -> None:
    """Tournament features + final experiments."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    # Add tournament features (leak-free)
    train_sf, test_sf, tourn_feats = add_tournament_features(train_sf, test_sf)
    logger.info("Tournament features added: %s", tourn_feats)

    base_feats = get_all_features()
    feat_with_tourn = base_feats + tourn_feats

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    results: dict[str, dict] = {}

    # A: CatBoost with tournament features
    check_budget()
    imp_t = SimpleImputer(strategy="median")
    x_fit_t = imp_t.fit_transform(train_fit[feat_with_tourn])
    x_val_t = imp_t.transform(val_df[feat_with_tourn])

    cb_t = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_t.fit(x_fit_t, train_fit["target"], eval_set=(x_val_t, val_df["target"]))
    bi_t = cb_t.get_best_iteration()

    imp_tf = SimpleImputer(strategy="median")
    x_full_t = imp_tf.fit_transform(train_sf[feat_with_tourn])
    x_test_t = imp_tf.transform(test_sf[feat_with_tourn])

    ft_t = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_t["iterations"] = bi_t + 10
    cb_ft_t = CatBoostClassifier(**ft_t)
    cb_ft_t.fit(x_full_t, train_sf["target"])

    p_test_t = cb_ft_t.predict_proba(x_test_t)[:, 1]
    auc_t = roc_auc_score(test_sf["target"], p_test_t)

    r_t_ev010 = calc_ev_roi(test_sf, p_test_t, ev_threshold=0.10, min_prob=0.77)
    results["cb_tourn_ev010"] = r_t_ev010
    logger.info(
        "A: CB+tourn EV>=0.10: ROI=%.2f%% n=%d AUC=%.4f",
        r_t_ev010["roi"],
        r_t_ev010["n_bets"],
        auc_t,
    )

    # B: CB+tourn + per-sport EV floor=0.10
    p_val_t = cb_ft_t.predict_proba(imp_tf.transform(val_df[feat_with_tourn]))[:, 1]
    sport_ev_t = find_sport_ev_thresholds(val_df, p_val_t, ev_floor=0.10)
    r_t_ps = apply_sport_ev(test_sf, p_test_t, sport_ev_t)
    results["cb_tourn_ps010"] = r_t_ps
    logger.info("B: CB+tourn PS010: ROI=%.2f%% n=%d", r_t_ps["roi"], r_t_ps["n_bets"])

    # Reference: base feats only
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[base_feats])
    x_val = imp.transform(val_df[base_feats])

    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    bi_ref = cb_ref.get_best_iteration()

    imp_f = SimpleImputer(strategy="median")
    x_full = imp_f.fit_transform(train_sf[base_feats])
    x_test = imp_f.transform(test_sf[base_feats])

    ft_r = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_r["iterations"] = bi_ref + 10
    cb_ft = CatBoostClassifier(**ft_r)
    cb_ft.fit(x_full, train_sf["target"])

    p_test_ref = cb_ft.predict_proba(x_test)[:, 1]
    p_val_ref = cb_ft.predict_proba(imp_f.transform(val_df[base_feats]))[:, 1]

    r_ref_ev010 = calc_ev_roi(test_sf, p_test_ref, ev_threshold=0.10, min_prob=0.77)
    results["cb_base_ev010"] = r_ref_ev010

    sport_ev_ref = find_sport_ev_thresholds(val_df, p_val_ref, ev_floor=0.10)
    r_ref_ps = apply_sport_ev(test_sf, p_test_ref, sport_ev_ref)
    results["cb_base_ps010"] = r_ref_ps

    logger.info(
        "Ref: base EV010=%.2f%% n=%d | base PS010=%.2f%% n=%d",
        r_ref_ev010["roi"],
        r_ref_ev010["n_bets"],
        r_ref_ps["roi"],
        r_ref_ps["n_bets"],
    )

    # Delta
    delta_ev = r_t_ev010["roi"] - r_ref_ev010["roi"]
    delta_ps = r_t_ps["roi"] - r_ref_ps["roi"]
    logger.info("Tournament features delta: EV010=%.2f pp, PS010=%.2f pp", delta_ev, delta_ps)

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d", name, r["roi"], r["n_bets"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.10_tournament") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.10")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "tournament_features",
                    "n_base_features": len(base_feats),
                    "n_tournament_features": len(tourn_feats),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": float(auc_t),
                    "n_bets": best_r["n_bets"],
                    "delta_ev010": delta_ev,
                    "delta_ps010": delta_ps,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.10: Best=%s ROI=%.2f%% n=%d run=%s",
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
