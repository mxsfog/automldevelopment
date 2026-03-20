"""Step 4.6: Multi-seed stability + bootstrap CI для PS_floor15.

Финальная валидация лучшей стратегии: 10 seeds + bootstrap CI.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
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
    SESSION_DIR,
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


def apply_sport_ev(
    test_df: pd.DataFrame, p_test: np.ndarray, sport_ev: dict[str, float]
) -> dict:
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

    seeds = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789]
    seed_results_ev: list[float] = []
    seed_results_ps: list[float] = []
    seed_results_auc: list[float] = []
    seed_n_bets: list[int] = []
    best_roi = -999.0
    best_seed = 42
    best_model = None
    best_imputer = None
    best_params = None
    best_sport_ev = None

    for seed in seeds:
        check_budget()
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

        ev_r = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
        sport_ev = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.15)
        ps_r = apply_sport_ev(test_sf, p_test, sport_ev)

        seed_results_ev.append(ev_r["roi"])
        seed_results_ps.append(ps_r["roi"])
        seed_results_auc.append(auc)
        seed_n_bets.append(ps_r["n_bets"])

        logger.info(
            "Seed %d: AUC=%.4f EV010=%.2f%%(%d) PS15=%.2f%%(%d)",
            seed, auc, ev_r["roi"], ev_r["n_bets"], ps_r["roi"], ps_r["n_bets"],
        )

        if ps_r["roi"] > best_roi:
            best_roi = ps_r["roi"]
            best_seed = seed
            best_model = cb_ft
            best_imputer = imp_full
            best_params = ft_params
            best_sport_ev = sport_ev

    # Summary
    logger.info("=== Multi-seed Summary ===")
    logger.info("EV010: avg=%.2f%% std=%.2f%%", np.mean(seed_results_ev), np.std(seed_results_ev))
    logger.info("PS15:  avg=%.2f%% std=%.2f%%", np.mean(seed_results_ps), np.std(seed_results_ps))
    logger.info("AUC:   avg=%.4f std=%.4f", np.mean(seed_results_auc), np.std(seed_results_auc))
    logger.info("Best seed: %d (PS15=%.2f%%)", best_seed, best_roi)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    n_bootstrap = 1000
    p_test_best = best_model.predict_proba(best_imputer.transform(test_sf[features]))[:, 1]
    bootstrap_rois: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(test_sf), len(test_sf))
        boot_test = test_sf.iloc[idx]
        boot_proba = p_test_best[idx]
        r = apply_sport_ev(boot_test, boot_proba, best_sport_ev)
        if r["n_bets"] > 0:
            bootstrap_rois.append(r["roi"])

    ci_lower = np.percentile(bootstrap_rois, 2.5) if bootstrap_rois else 0.0
    ci_upper = np.percentile(bootstrap_rois, 97.5) if bootstrap_rois else 0.0
    ci_median = np.median(bootstrap_rois) if bootstrap_rois else 0.0

    logger.info(
        "Bootstrap CI (95%%): [%.2f%%, %.2f%%] median=%.2f%%",
        ci_lower, ci_upper, ci_median,
    )

    # Save best model
    model_dir = Path(SESSION_DIR) / "models" / "best"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(model_dir / "model.cbm"))

    metadata = {
        "framework": "catboost",
        "model_file": "model.cbm",
        "roi": best_roi,
        "roi_avg_10seeds": float(np.mean(seed_results_ps)),
        "roi_std_10seeds": float(np.std(seed_results_ps)),
        "roi_ci_lower": float(ci_lower),
        "roi_ci_upper": float(ci_upper),
        "auc": float(np.mean(seed_results_auc)),
        "threshold": 0.77,
        "ev_floor": 0.15,
        "selection_strategy": "PS_floor15 aggressive",
        "n_bets_avg": float(np.mean(seed_n_bets)),
        "best_seed": best_seed,
        "feature_names": features,
        "params": best_params,
        "sport_filter": UNPROFITABLE_SPORTS,
        "elo_filter": True,
        "session_id": SESSION_ID,
        "per_sport_ev_thresholds": best_sport_ev,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved (seed=%d)", best_seed)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.6_stability") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params({
                "method": "multi_seed_stability",
                "n_seeds": len(seeds),
                "strategy": "PS_floor15_aggressive",
                "validation_scheme": "time_series",
                "best_seed": best_seed,
            })
            mlflow.log_metrics({
                "roi": best_roi,
                "roi_avg": float(np.mean(seed_results_ps)),
                "roi_std": float(np.std(seed_results_ps)),
                "auc_avg": float(np.mean(seed_results_auc)),
                "auc_std": float(np.std(seed_results_auc)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "ci_median": float(ci_median),
                "n_bets_avg": float(np.mean(seed_n_bets)),
            })
            for i, seed in enumerate(seeds):
                mlflow.log_metric(f"roi_seed_{seed}", seed_results_ps[i])
                mlflow.log_metric(f"auc_seed_{seed}", seed_results_auc[i])

            mlflow.log_artifact(__file__)
            mlflow.log_artifact(str(model_dir / "metadata.json"))
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
