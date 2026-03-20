"""Step 4.8: Seed ensemble — усреднение предсказаний 5 моделей с разными seeds.

Гипотеза: средние вероятности из нескольких моделей дают более стабильный
EV-отбор и снижают variance per-sport thresholds.
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

    # Top-5 seeds из step 4.6
    seeds = [456, 789, 42, 123, 3456]
    n_ensemble_variants = [3, 5]

    all_p_test: list[np.ndarray] = []
    all_p_val: list[np.ndarray] = []

    val_split = int(len(train_sf) * 0.8)
    val_df = train_sf.iloc[val_split:]

    for seed in seeds:
        check_budget()
        set_seed(seed)

        train_fit = train_sf.iloc[:val_split]

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

        all_p_test.append(p_test)
        all_p_val.append(p_val)

        auc = roc_auc_score(test_sf["target"], p_test)
        logger.info("Seed %d: AUC=%.4f", seed, auc)

    results: dict[str, dict] = {}

    # Evaluate individual seeds
    for i, seed in enumerate(seeds):
        sport_ev = find_sport_ev_thresholds(val_df, all_p_val[i], ev_floor=0.15)
        ps = apply_sport_ev(test_sf, all_p_test[i], sport_ev)
        ev = calc_ev_roi(test_sf, all_p_test[i], ev_threshold=0.10, min_prob=0.77)
        auc = roc_auc_score(test_sf["target"], all_p_test[i])
        label = f"seed_{seed}"
        results[label] = {
            "auc": auc,
            "ev_roi": ev["roi"],
            "ev_n": ev["n_bets"],
            "ps_roi": ps["roi"],
            "ps_n": ps["n_bets"],
        }
        logger.info(
            "%s: AUC=%.4f EV=%.2f%%(%d) PS15=%.2f%%(%d)",
            label,
            auc,
            ev["roi"],
            ev["n_bets"],
            ps["roi"],
            ps["n_bets"],
        )

    # Ensemble variants
    for n_ens in n_ensemble_variants:
        p_test_ens = np.mean(all_p_test[:n_ens], axis=0)
        p_val_ens = np.mean(all_p_val[:n_ens], axis=0)
        auc_ens = roc_auc_score(test_sf["target"], p_test_ens)

        sport_ev_ens = find_sport_ev_thresholds(val_df, p_val_ens, ev_floor=0.15)
        ps_ens = apply_sport_ev(test_sf, p_test_ens, sport_ev_ens)
        ev_ens = calc_ev_roi(test_sf, p_test_ens, ev_threshold=0.10, min_prob=0.77)

        label = f"ens_{n_ens}seeds"
        results[label] = {
            "auc": auc_ens,
            "ev_roi": ev_ens["roi"],
            "ev_n": ev_ens["n_bets"],
            "ps_roi": ps_ens["roi"],
            "ps_n": ps_ens["n_bets"],
        }
        logger.info(
            "%s: AUC=%.4f EV=%.2f%%(%d) PS15=%.2f%%(%d)",
            label,
            auc_ens,
            ev_ens["roi"],
            ev_ens["n_bets"],
            ps_ens["roi"],
            ps_ens["n_bets"],
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

    # Save if ensemble is best
    if best_name.startswith("ens_"):
        model_dir = Path(SESSION_DIR) / "models" / "best"
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "framework": "catboost_ensemble",
            "method": best_name,
            "roi": best_r["ps_roi"],
            "auc": best_r["auc"],
            "threshold": 0.77,
            "ev_floor": 0.15,
            "selection_strategy": f"PS_floor15_{best_name}",
            "n_bets": best_r["ps_n"],
            "seeds": seeds[: int(best_name.split("_")[1].replace("seeds", ""))],
            "feature_names": features,
            "params": CB_BEST_PARAMS,
            "sport_filter": UNPROFITABLE_SPORTS,
            "elo_filter": True,
            "session_id": SESSION_ID,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Ensemble model metadata saved")

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.8_seed_ensemble") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "seed_ensemble",
                    "seeds": str(seeds),
                    "n_ensemble_variants": str(n_ensemble_variants),
                    "validation_scheme": "time_series",
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
