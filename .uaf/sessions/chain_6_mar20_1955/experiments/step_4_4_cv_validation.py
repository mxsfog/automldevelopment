"""Step 4.4: 5-fold time-series CV validation лучших стратегий + model save.

Валидация стратегий из step 4.3 на expanding window CV для подтверждения
робастности и сохранение лучшей модели.
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


def calc_filtered_roi(
    df: pd.DataFrame,
    proba: np.ndarray,
    ev_threshold: float,
    min_prob: float,
) -> dict:
    """ROI с фильтром по EV и p."""
    odds = df["Odds"].values
    ev = proba * odds - 1.0
    mask = (ev >= ev_threshold) & (proba >= min_prob)
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def find_sport_ev_thresholds(
    val_df: pd.DataFrame, p_val: np.ndarray, ev_floor: float = 0.10
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта на валидации."""
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
    test_df: pd.DataFrame,
    p_test: np.ndarray,
    sport_ev: dict[str, float],
    min_prob: float = 0.77,
) -> dict:
    """Применение per-sport EV thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t = sport_ev.get(sports[i], 0.10)
        if ev[i] >= ev_t and p_test[i] >= min_prob:
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

    # Стратегии для валидации
    strategies = {
        "ev024_p73": {"ev_t": 0.24, "min_p": 0.73},
        "ev010_p77": {"ev_t": 0.10, "min_p": 0.77},
        "ps_floor10": {"ps": True, "floor": 0.10, "min_p": 0.77},
        "ps_floor12": {"ps": True, "floor": 0.12, "min_p": 0.77},
        "ps_floor15": {"ps": True, "floor": 0.15, "min_p": 0.77},
        "ps_hc": {"ps_hc": True, "min_p": 0.77},
    }

    # 5-fold expanding window CV
    all_sf = train_sf.sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    cv_results: dict[str, list[float]] = {s: [] for s in strategies}
    cv_nbets: dict[str, list[int]] = {s: [] for s in strategies}
    cv_aucs: list[float] = []

    logger.info("5-fold CV (n=%d, block=%d)", n, block_size)

    for fold_idx in range(n_folds):
        check_budget()
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(train_end + block_size, n)

        fold_train = all_sf.iloc[:train_end].copy()
        fold_test = all_sf.iloc[test_start:test_end].copy()

        if len(fold_train) < 100 or len(fold_test) < 20:
            continue

        inner_val_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_val_split]
        inner_val = fold_train.iloc[inner_val_split:]

        imp = SimpleImputer(strategy="median")
        x_inner_train = imp.fit_transform(inner_train[features])
        x_inner_val = imp.transform(inner_val[features])

        cb = CatBoostClassifier(**CB_BEST_PARAMS)
        cb.fit(x_inner_train, inner_train["target"], eval_set=(x_inner_val, inner_val["target"]))
        fold_best_iter = cb.get_best_iteration()

        imp_full = SimpleImputer(strategy="median")
        x_fold_full = imp_full.fit_transform(fold_train[features])
        x_fold_test = imp_full.transform(fold_test[features])

        ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        ft_params["iterations"] = max(fold_best_iter + 10, 50)
        cb_ft = CatBoostClassifier(**ft_params)
        cb_ft.fit(x_fold_full, fold_train["target"])

        p_fold_test = cb_ft.predict_proba(x_fold_test)[:, 1]
        p_fold_val = cb_ft.predict_proba(imp_full.transform(inner_val[features]))[:, 1]

        try:
            auc = roc_auc_score(fold_test["target"], p_fold_test)
            cv_aucs.append(auc)
        except ValueError:
            auc = 0.0

        for strat_name, strat_cfg in strategies.items():
            if strat_cfg.get("ps"):
                sport_ev = find_sport_ev_thresholds(
                    inner_val,
                    p_fold_val,
                    ev_floor=strat_cfg["floor"],
                )
                r = apply_sport_ev(
                    fold_test,
                    p_fold_test,
                    sport_ev,
                    min_prob=strat_cfg["min_p"],
                )
            elif strat_cfg.get("ps_hc"):
                r = calc_per_sport_ev_roi(
                    fold_test,
                    p_fold_test,
                    sport_thresholds=PS_EV_THRESHOLDS,
                    min_prob=strat_cfg["min_p"],
                )
            else:
                r = calc_filtered_roi(
                    fold_test,
                    p_fold_test,
                    strat_cfg["ev_t"],
                    strat_cfg["min_p"],
                )
            cv_results[strat_name].append(r.get("roi", 0.0))
            cv_nbets[strat_name].append(r.get("n_bets", 0))

        logger.info(
            "Fold %d: AUC=%.4f | %s",
            fold_idx,
            auc,
            " | ".join(f"{s}={cv_results[s][-1]:.1f}%({cv_nbets[s][-1]})" for s in strategies),
        )

    # CV summary
    logger.info("=== CV Summary ===")
    cv_summary: dict[str, dict] = {}
    for s in strategies:
        if cv_results[s]:
            avg = np.mean(cv_results[s])
            std = np.std(cv_results[s])
            pos = sum(1 for r in cv_results[s] if r > 0)
            avg_n = np.mean(cv_nbets[s])
            cv_summary[s] = {
                "avg": float(avg),
                "std": float(std),
                "pos": pos,
                "total": len(cv_results[s]),
                "avg_n": float(avg_n),
            }
            logger.info(
                "  %-15s: avg=%.2f%% std=%.2f%% pos=%d/%d avg_n=%.0f",
                s,
                avg,
                std,
                pos,
                len(cv_results[s]),
                avg_n,
            )

    avg_auc = np.mean(cv_aucs) if cv_aucs else 0.0
    logger.info("  AUC: avg=%.4f std=%.4f", avg_auc, np.std(cv_aucs) if cv_aucs else 0.0)

    # Test set evaluation с лучшей моделью
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[features])
    x_val = imputer.transform(val_df[features])

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb.get_best_iteration()

    imputer_full = SimpleImputer(strategy="median")
    x_full = imputer_full.fit_transform(train_sf[features])
    x_test = imputer_full.transform(test_sf[features])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imputer_full.transform(val_df[features]))[:, 1]
    test_auc = roc_auc_score(test_sf["target"], p_test)

    test_results: dict[str, dict] = {}
    for strat_name, strat_cfg in strategies.items():
        if strat_cfg.get("ps"):
            sport_ev = find_sport_ev_thresholds(
                val_df,
                p_val,
                ev_floor=strat_cfg["floor"],
            )
            r = apply_sport_ev(test_sf, p_test, sport_ev, min_prob=strat_cfg["min_p"])
        elif strat_cfg.get("ps_hc"):
            r = calc_per_sport_ev_roi(
                test_sf,
                p_test,
                sport_thresholds=PS_EV_THRESHOLDS,
                min_prob=strat_cfg["min_p"],
            )
        else:
            r = calc_filtered_roi(test_sf, p_test, strat_cfg["ev_t"], strat_cfg["min_p"])
        test_results[strat_name] = r
        logger.info("Test %s: ROI=%.2f%% N=%d", strat_name, r["roi"], r["n_bets"])

    # Выбор лучшей стратегии по CV (avg ROI, pos > 3)
    best_strat = max(
        (s for s in cv_summary if cv_summary[s]["pos"] >= 3),
        key=lambda s: cv_summary[s]["avg"],
        default="ev010_p77",
    )
    logger.info("Best strategy by CV: %s", best_strat)

    # Save model
    model_dir = Path(SESSION_DIR) / "models" / "best"
    model_dir.mkdir(parents=True, exist_ok=True)
    cb_ft.save_model(str(model_dir / "model.cbm"))

    metadata = {
        "framework": "catboost",
        "model_file": "model.cbm",
        "roi": test_results[best_strat]["roi"],
        "auc": float(test_auc),
        "threshold": 0.77,
        "ev_threshold": 0.10,
        "selection_strategy": best_strat,
        "n_bets": test_results[best_strat]["n_bets"],
        "feature_names": features,
        "params": ft_params,
        "sport_filter": UNPROFITABLE_SPORTS,
        "elo_filter": True,
        "best_iteration": best_iter,
        "session_id": SESSION_ID,
        "cv_results": {
            s: {"avg": cv_summary[s]["avg"], "std": cv_summary[s]["std"]} for s in cv_summary
        },
        "test_results": {
            s: {"roi": r["roi"], "n_bets": r["n_bets"]} for s, r in test_results.items()
        },
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s", model_dir)

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.4_cv_validation") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "5fold_cv_validation",
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_folds": n_folds,
                    "best_strategy": best_strat,
                    "best_iteration": best_iter,
                }
            )
            mlflow.log_metrics(
                {
                    "auc": float(test_auc),
                    "cv_auc_avg": float(avg_auc),
                    "roi": test_results[best_strat]["roi"],
                    "n_bets": test_results[best_strat]["n_bets"],
                }
            )
            for s in cv_summary:
                mlflow.log_metric(f"cv_avg_{s}", cv_summary[s]["avg"])
                mlflow.log_metric(f"cv_std_{s}", cv_summary[s]["std"])
                mlflow.log_metric(f"cv_pos_{s}", cv_summary[s]["pos"])
                mlflow.log_metric(f"test_roi_{s}", test_results[s]["roi"])
                mlflow.log_metric(f"test_n_{s}", test_results[s]["n_bets"])

            for fold_idx in range(len(cv_aucs)):
                mlflow.log_metric(f"auc_fold_{fold_idx}", cv_aucs[fold_idx])

            mlflow.log_artifact(__file__)
            mlflow.log_artifact(str(model_dir / "metadata.json"))
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
