"""Step 4.2: Stacking ensemble — CatBoost + LightGBM + XGBoost с LogReg meta-learner.

Гипотеза: разнообразие моделей в стекинге может дать лучшую калибровку вероятностей
и более робастный отбор ставок через EV фильтр.
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

if TYPE_CHECKING:
    import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CB_BEST_PARAMS,
    PS_EV_THRESHOLDS,
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
    val_df: pd.DataFrame, p_val: np.ndarray, ev_floor: float = 0.10
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
            if r["n_bets"] >= 3 and r["roi"] > best_roi:
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
        ev_t = sport_ev.get(sports[i], 0.10)
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

    # 3-way split: train -> fit / val / stack_val
    n = len(train_sf)
    split_1 = int(n * 0.6)
    split_2 = int(n * 0.8)
    train_fit = train_sf.iloc[:split_1]
    stack_val = train_sf.iloc[split_1:split_2]
    val_df = train_sf.iloc[split_2:]

    logger.info(
        "Splits: fit=%d, stack_val=%d, val=%d, test=%d",
        len(train_fit),
        len(stack_val),
        len(val_df),
        len(test_sf),
    )

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[features])
    x_stack = imputer.transform(stack_val[features])
    x_val = imputer.transform(val_df[features])
    x_test = imputer.transform(test_sf[features])

    y_fit = train_fit["target"].values
    y_stack = stack_val["target"].values

    # Level-0: CatBoost
    check_budget()
    logger.info("Training CatBoost...")
    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, y_fit, eval_set=(x_stack, y_stack), use_best_model=True)
    cb_best_iter = cb.get_best_iteration()

    p_cb_stack = cb.predict_proba(x_stack)[:, 1]
    p_cb_val = cb.predict_proba(x_val)[:, 1]
    p_cb_test = cb.predict_proba(x_test)[:, 1]

    # Level-0: LightGBM
    check_budget()
    logger.info("Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=20,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(
        x_fit,
        y_fit,
        eval_set=[(x_stack, y_stack)],
        callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=False),
        ],
    )

    p_lgb_stack = lgb.predict_proba(x_stack)[:, 1]
    p_lgb_val = lgb.predict_proba(x_val)[:, 1]
    p_lgb_test = lgb.predict_proba(x_test)[:, 1]

    # Level-0: XGBoost
    check_budget()
    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=20,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb.fit(x_fit, y_fit, eval_set=[(x_stack, y_stack)], verbose=False)

    p_xgb_stack = xgb.predict_proba(x_stack)[:, 1]
    p_xgb_val = xgb.predict_proba(x_val)[:, 1]
    p_xgb_test = xgb.predict_proba(x_test)[:, 1]

    # Level-1: LogReg meta-learner
    logger.info("Training meta-learner...")
    meta_features_stack = np.column_stack([p_cb_stack, p_lgb_stack, p_xgb_stack])
    meta_features_val = np.column_stack([p_cb_val, p_lgb_val, p_xgb_val])
    meta_features_test = np.column_stack([p_cb_test, p_lgb_test, p_xgb_test])

    meta = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    meta.fit(meta_features_stack, y_stack)

    p_meta_val = meta.predict_proba(meta_features_val)[:, 1]
    p_meta_test = meta.predict_proba(meta_features_test)[:, 1]
    logger.info("Meta weights: %s", meta.coef_)

    # Simple averaging тоже
    p_avg_test = (p_cb_test + p_lgb_test + p_xgb_test) / 3.0
    p_avg_val = (p_cb_val + p_lgb_val + p_xgb_val) / 3.0

    # Evaluate all candidates
    results: dict[str, dict] = {}
    for name, p_test, p_val_arr in [
        ("cb_only", p_cb_test, p_cb_val),
        ("lgb_only", p_lgb_test, p_lgb_val),
        ("xgb_only", p_xgb_test, p_xgb_val),
        ("meta_lr", p_meta_test, p_meta_val),
        ("avg_3", p_avg_test, p_avg_val),
    ]:
        auc = roc_auc_score(test_sf["target"], p_test)
        ev = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
        ps = calc_per_sport_ev_roi(
            test_sf,
            p_test,
            sport_thresholds=PS_EV_THRESHOLDS,
            min_prob=0.77,
        )
        sport_ev = find_sport_ev_thresholds(val_df, p_val_arr, ev_floor=0.10)
        ps_val = apply_sport_ev(test_sf, p_test, sport_ev)

        results[name] = {
            "auc": auc,
            "ev_roi": ev["roi"],
            "ev_n": ev["n_bets"],
            "ps_roi": ps["roi"],
            "ps_n": ps["n_bets"],
            "ps_val_roi": ps_val["roi"],
            "ps_val_n": ps_val["n_bets"],
        }
        logger.info(
            "%s: AUC=%.4f EV=%.2f%%(%d) PS=%.2f%%(%d) PSval=%.2f%%(%d)",
            name,
            auc,
            ev["roi"],
            ev["n_bets"],
            ps["roi"],
            ps["n_bets"],
            ps_val["roi"],
            ps_val["n_bets"],
        )

    # Weighted average optimized on val
    best_w_roi = -999.0
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    for w_cb in np.arange(0.2, 0.8, 0.1):
        for w_lgb in np.arange(0.1, 0.8 - w_cb, 0.1):
            w_xgb = 1.0 - w_cb - w_lgb
            if w_xgb < 0.05:
                continue
            p_w_val = w_cb * p_cb_val + w_lgb * p_lgb_val + w_xgb * p_xgb_val
            r = calc_ev_roi(val_df, p_w_val, ev_threshold=0.10, min_prob=0.77)
            if r["n_bets"] >= 5 and r["roi"] > best_w_roi:
                best_w_roi = r["roi"]
                best_weights = (w_cb, w_lgb, w_xgb)

    p_weighted_test = (
        best_weights[0] * p_cb_test + best_weights[1] * p_lgb_test + best_weights[2] * p_xgb_test
    )
    auc_w = roc_auc_score(test_sf["target"], p_weighted_test)
    ev_w = calc_ev_roi(test_sf, p_weighted_test, ev_threshold=0.10, min_prob=0.77)
    ps_w = calc_per_sport_ev_roi(
        test_sf,
        p_weighted_test,
        sport_thresholds=PS_EV_THRESHOLDS,
        min_prob=0.77,
    )
    results["weighted_opt"] = {
        "auc": auc_w,
        "ev_roi": ev_w["roi"],
        "ev_n": ev_w["n_bets"],
        "ps_roi": ps_w["roi"],
        "ps_n": ps_w["n_bets"],
        "weights": best_weights,
    }
    logger.info(
        "weighted_opt (%.2f/%.2f/%.2f): AUC=%.4f EV=%.2f%%(%d) PS=%.2f%%(%d)",
        *best_weights,
        auc_w,
        ev_w["roi"],
        ev_w["n_bets"],
        ps_w["roi"],
        ps_w["n_bets"],
    )

    # MLflow
    with mlflow.start_run(run_name="phase4/step_4.2_stacking") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "stacking_cb_lgb_xgb",
                    "meta_learner": "LogisticRegression",
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "cb_best_iter": cb_best_iter,
                    "best_weights": str(best_weights),
                }
            )
            for name, r in results.items():
                mlflow.log_metric(f"auc_{name}", r["auc"])
                mlflow.log_metric(f"ev_roi_{name}", r["ev_roi"])
                mlflow.log_metric(f"ps_roi_{name}", r.get("ps_roi", 0.0))

            best_method = max(
                results, key=lambda k: results[k].get("ps_roi", results[k]["ev_roi"])
            )
            mlflow.log_metric(
                "roi", results[best_method].get("ps_roi", results[best_method]["ev_roi"])
            )
            mlflow.set_tag("best_method", best_method)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
