"""Step 4.13: Bootstrap confidence intervals for final results.

Финальная валидация: bootstrap 95% CI для ROI всех основных стратегий.
Это даст статистическую значимость превышения ROI > 0.
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


def bootstrap_roi(test_df, p_test: np.ndarray, mask: np.ndarray, n_bootstrap: int = 1000) -> dict:
    """Bootstrap 95% CI for ROI on selected bets."""
    sel_idx = np.where(mask)[0]
    if len(sel_idx) == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "pct_positive": 0.0}

    sel = test_df.iloc[sel_idx]
    usd = sel["USD"].values
    payout = sel["Payout_USD"].values

    rng = np.random.RandomState(42)
    boot_rois: list[float] = []
    n = len(sel_idx)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        staked = usd[idx].sum()
        if staked > 0:
            roi = (payout[idx].sum() - staked) / staked * 100
            boot_rois.append(roi)

    boot_rois_arr = np.array(boot_rois)
    return {
        "mean": float(np.mean(boot_rois_arr)),
        "ci_lo": float(np.percentile(boot_rois_arr, 2.5)),
        "ci_hi": float(np.percentile(boot_rois_arr, 97.5)),
        "pct_positive": float(np.mean(boot_rois_arr > 0) * 100),
    }


def main() -> None:
    """Bootstrap CI for final results."""
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

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    bi = cb.get_best_iteration()

    imp_f = SimpleImputer(strategy="median")
    x_full = imp_f.fit_transform(train_sf[feat_list])
    x_test = imp_f.transform(test_sf[feat_list])

    ft_p = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_p["iterations"] = bi + 10
    cb_ft = CatBoostClassifier(**ft_p)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imp_f.transform(val_df[feat_list]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    # Strategies to validate
    odds = test_sf["Odds"].values
    ev = p_test * odds - 1.0

    strategies = {
        "t77": p_test >= 0.77,
        "ev0_p77": (ev >= 0) & (p_test >= 0.77),
        "ev010_p77": (ev >= 0.10) & (p_test >= 0.77),
    }

    # Per-sport EV
    sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.10)
    sports = test_sf["Sport"].values
    ps_mask = np.zeros(len(test_sf), dtype=bool)
    for i in range(len(test_sf)):
        ev_t = sport_ev.get(sports[i], 0.0)
        if ev[i] >= ev_t and p_test[i] >= 0.77:
            ps_mask[i] = True
    strategies["ps_ev010"] = ps_mask

    # Bootstrap all strategies
    logger.info("Bootstrap 95%% CI (1000 iterations):")
    for name, mask in strategies.items():
        n_sel = int(mask.sum())
        if n_sel == 0:
            continue

        sel = test_sf.iloc[np.where(mask)[0]]
        staked = sel["USD"].sum()
        payout = sel["Payout_USD"].sum()
        roi = (payout - staked) / staked * 100

        boot = bootstrap_roi(test_sf, p_test, mask)
        logger.info(
            "  %-12s: ROI=%.2f%% [%.2f%%, %.2f%%] n=%d P(ROI>0)=%.1f%%",
            name,
            roi,
            boot["ci_lo"],
            boot["ci_hi"],
            n_sel,
            boot["pct_positive"],
        )

    # Summary table
    logger.info("Final summary:")
    logger.info("  AUC: %.4f", auc)
    logger.info("  Model: CatBoost (depth=8, lr=0.08, l2=21.1) full-train")
    logger.info("  Features: %d (base + engineered + ELO)", len(feat_list))
    logger.info("  Sport filter: %s", UNPROFITABLE_SPORTS)
    logger.info("  Per-sport EV: %s", sport_ev)

    with mlflow.start_run(run_name="phase4/step4.13_bootstrap") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.13")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "bootstrap_ci",
                    "n_features": len(feat_list),
                    "n_bootstrap": 1000,
                }
            )

            for name, mask in strategies.items():
                n_sel = int(mask.sum())
                if n_sel == 0:
                    continue
                sel = test_sf.iloc[np.where(mask)[0]]
                staked = sel["USD"].sum()
                payout = sel["Payout_USD"].sum()
                roi = (payout - staked) / staked * 100
                boot = bootstrap_roi(test_sf, p_test, mask)

                mlflow.log_metric(f"roi_{name}", roi)
                mlflow.log_metric(f"ci_lo_{name}", boot["ci_lo"])
                mlflow.log_metric(f"ci_hi_{name}", boot["ci_hi"])
                mlflow.log_metric(f"pct_positive_{name}", boot["pct_positive"])
                mlflow.log_metric(f"nbets_{name}", n_sel)

            mlflow.log_metrics({"roi": roi, "roc_auc": float(auc), "n_bets": n_sel})
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")

            logger.info("Step 4.13: Bootstrap CI complete. run=%s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
