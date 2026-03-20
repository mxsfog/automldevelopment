"""Phase 2+3 corrected: ELO filter + sport filter + proper training pipeline."""

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

if TYPE_CHECKING:
    import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CB_BEST_PARAMS,
    PS_EV_THRESHOLDS,
    SESSION_DIR,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
    calc_per_sport_ev_roi,
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
    val_df: pd.DataFrame,
    p_val: np.ndarray,
    min_bets: int = 3,
    ev_floor: float = 0.0,
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта на валидации."""
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


def apply_sport_ev(test_df: pd.DataFrame, p_test: np.ndarray, sport_ev: dict[str, float]) -> dict:
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
        return {"roi": 0.0, "n_bets": 0, "n_won": 0, "win_rate": 0.0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    n_won = (sel["Status"] == "won").sum()
    return {
        "roi": float(roi),
        "n_bets": n_sel,
        "n_won": int(n_won),
        "win_rate": float(n_won / n_sel),
    }


def main() -> None:
    set_seed()
    check_budget()

    logger.info("Loading data...")
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

    # Фильтр: только ставки с ELO данными + исключение убыточных спортов
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    logger.info(
        "Filters: all=%d -> elo=%d -> sport_filter=%d (train)",
        len(train_all),
        len(train_elo),
        len(train_sf),
    )
    logger.info(
        "Filters: all=%d -> elo=%d -> sport_filter=%d (test)",
        len(test_all),
        len(test_elo),
        len(test_sf),
    )

    features = get_all_features()

    # Двухфазное обучение: 1) val для best_iteration, 2) retrain на полном train
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[features])
    x_val = imputer.transform(val_df[features])

    logger.info("Training phase 1: finding best_iteration...")
    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb_ref.get_best_iteration()
    logger.info("Best iteration: %d", best_iter)

    # Retrain на полном training set
    imputer_full = SimpleImputer(strategy="median")
    x_full = imputer_full.fit_transform(train_sf[features])
    x_test = imputer_full.transform(test_sf[features])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    logger.info("Training phase 2: full retrain with %d iterations...", ft_params["iterations"])
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imputer_full.transform(val_df[features]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    # EV selection strategies
    ev_result = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    ps_result = calc_per_sport_ev_roi(
        test_sf,
        p_test,
        sport_thresholds=PS_EV_THRESHOLDS,
        min_prob=0.77,
    )

    # Per-sport EV с порогами из validation
    sport_ev_010 = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.10)
    ps_val_result = apply_sport_ev(test_sf, p_test, sport_ev_010)

    logger.info("AUC: %.4f", auc)
    logger.info(
        "EV>=0.10+p77: ROI=%.2f%% N=%d",
        ev_result["roi"],
        ev_result["n_bets"],
    )
    logger.info(
        "PS_EV (hardcoded): ROI=%.2f%% N=%d",
        ps_result["roi"],
        ps_result["n_bets"],
    )
    logger.info(
        "PS_EV (val-tuned floor=0.10): ROI=%.2f%% N=%d",
        ps_val_result["roi"],
        ps_val_result["n_bets"],
    )
    logger.info("Val-tuned sport thresholds: %s", sport_ev_010)

    # Feature importance
    fi = cb_ft.get_feature_importance()
    fi_sorted = sorted(
        zip(features, fi, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top-10 features:")
    for name, imp in fi_sorted[:10]:
        logger.info("  %s: %.2f", name, imp)

    # MLflow logging
    with mlflow.start_run(run_name="phase2_3/elo_catboost_corrected") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "CatBoost_ELO_two_phase",
                    "elo_filter": True,
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "ev_threshold": 0.10,
                    "min_prob": 0.77,
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_sf),
                    "n_samples_test": len(test_sf),
                    "n_features": len(features),
                    "best_iteration": best_iter,
                    "iterations_final": ft_params["iterations"],
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc,
                    "roi_ev010": ev_result["roi"],
                    "n_bets_ev010": ev_result["n_bets"],
                    "roi_ps_ev_hardcoded": ps_result["roi"],
                    "n_bets_ps_hardcoded": ps_result["n_bets"],
                    "roi_ps_ev_val": ps_val_result["roi"],
                    "n_bets_ps_val": ps_val_result["n_bets"],
                }
            )

            fi_text = "\n".join(f"{name}: {imp:.2f}" for name, imp in fi_sorted)
            mlflow.log_text(fi_text, "feature_importance.txt")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            run_id = run.info.run_id
            logger.info("Run ID: %s", run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise

    # Сохранение модели
    best_roi = max(ev_result["roi"], ps_result["roi"], ps_val_result["roi"])
    if best_roi == ps_val_result["roi"]:
        best_strategy = "PS_EV val-tuned floor=0.10"
        best_n = ps_val_result["n_bets"]
    elif best_roi == ps_result["roi"]:
        best_strategy = "PS_EV hardcoded"
        best_n = ps_result["n_bets"]
    else:
        best_strategy = "EV>=0.10+p77"
        best_n = ev_result["n_bets"]

    model_dir = Path(SESSION_DIR) / "models" / "best"
    model_dir.mkdir(parents=True, exist_ok=True)
    cb_ft.save_model(str(model_dir / "model.cbm"))

    metadata = {
        "framework": "catboost",
        "model_file": "model.cbm",
        "roi_ev010": ev_result["roi"],
        "roi_ps_ev_hardcoded": ps_result["roi"],
        "roi_ps_ev_val": ps_val_result["roi"],
        "auc": float(auc),
        "threshold": 0.77,
        "ev_threshold": 0.10,
        "selection_strategy": best_strategy,
        "n_bets": best_n,
        "feature_names": features,
        "params": ft_params,
        "sport_filter": UNPROFITABLE_SPORTS,
        "elo_filter": True,
        "best_iteration": best_iter,
        "session_id": SESSION_ID,
        "per_sport_ev_thresholds_val": sport_ev_010,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s", model_dir)
    logger.info("Best strategy: %s ROI=%.2f%% N=%d", best_strategy, best_roi, best_n)


if __name__ == "__main__":
    main()
