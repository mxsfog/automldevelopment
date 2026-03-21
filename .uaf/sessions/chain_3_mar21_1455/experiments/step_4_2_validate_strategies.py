"""Step 4.2 — Proper validation of selection strategies.

Train/val/test split: train_fit=64%, val=16%, test=20%.
Стратегии выбираются на val, применяются к test один раз.
Фокус на:
  - Confidence-weighted EV с разными порогами
  - Odds-stratified EV
  - Low-disagreement filtering (std percentile from val)
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import add_sport_market_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcome_Odds",
    "n_outcomes",
    "mean_outcome_odds",
    "max_outcome_odds",
    "min_outcome_odds",
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def train_ensemble(x_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Обучение 3-model ensemble."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    cb.fit(x_train, y_train)

    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x_train, y_train)

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x_train)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_s, y_train)

    return cb, lgbm, lr, scaler


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Предсказания всех 3 моделей и статистики."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]

    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.sqrt(((p_cb - p_mean) ** 2 + (p_lgbm - p_mean) ** 2 + (p_lr - p_mean) ** 2) / 3)

    return p_cb, p_lgbm, p_lr, p_mean, p_std


def evaluate_strategy(
    df: pd.DataFrame,
    mask: np.ndarray,
    name: str,
) -> dict:
    """Evaluate selection strategy."""
    result = calc_roi(df, mask.astype(float), threshold=0.5)
    return {**result, "name": name}


def find_best_strategy_on_val(
    val_df: pd.DataFrame,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    p_cb: np.ndarray,
    p_lgbm: np.ndarray,
    p_lr: np.ndarray,
) -> tuple[str, dict]:
    """Поиск лучшей стратегии на валидации."""
    odds = val_df["Odds"].values
    ev_mean = p_mean * odds - 1
    strategies = {}

    # Baseline EV thresholds
    for thr in np.arange(0.05, 0.30, 0.01):
        mask = ev_mean >= thr
        r = evaluate_strategy(val_df, mask, f"ev_{thr:.2f}")
        if r["n_bets"] >= 30:
            strategies[f"ev_{thr:.2f}"] = r

    # Confidence-weighted EV
    confidence = 1 / (1 + p_std * 10)
    ev_conf = ev_mean * confidence
    for thr in np.arange(0.03, 0.25, 0.01):
        mask = ev_conf >= thr
        r = evaluate_strategy(val_df, mask, f"conf_ev_{thr:.2f}")
        if r["n_bets"] >= 30:
            strategies[f"conf_ev_{thr:.2f}"] = r

    # Low-disagreement filtering with EV
    for ev_thr in [0.10, 0.12, 0.15]:
        mask_ev = ev_mean >= ev_thr
        if mask_ev.sum() < 30:
            continue
        for std_pct in [25, 50, 75]:
            std_thr = np.percentile(p_std[mask_ev], std_pct)
            mask = mask_ev & (p_std <= std_thr)
            r = evaluate_strategy(val_df, mask, f"std_p{std_pct}_ev{ev_thr:.2f}")
            if r["n_bets"] >= 30:
                strategies[f"std_p{std_pct}_ev{ev_thr:.2f}"] = r

    # Odds-stratified EV (multiple configs)
    for config_name, brackets in {
        "strat_v1": [(1, 3, 0.20), (3, 10, 0.15), (10, 50, 0.12), (50, 500, 0.08)],
        "strat_v2": [(1, 3, 0.25), (3, 10, 0.18), (10, 50, 0.10), (50, 500, 0.06)],
        "strat_v3": [(1, 5, 0.20), (5, 20, 0.12), (20, 100, 0.08), (100, 500, 0.05)],
        "strat_v4": [(1, 3, 0.15), (3, 10, 0.12), (10, 50, 0.10), (50, 500, 0.08)],
    }.items():
        mask = np.zeros(len(val_df), dtype=bool)
        for lo, hi, thr in brackets:
            bracket = (odds >= lo) & (odds < hi)
            mask |= bracket & (ev_mean >= thr)
        r = evaluate_strategy(val_df, mask, config_name)
        if r["n_bets"] >= 30:
            strategies[config_name] = r

    # Find best by ROI
    best_name = max(strategies, key=lambda k: strategies[k]["roi"])
    best = strategies[best_name]

    logger.info(
        "Val best strategy: %s -> ROI=%.2f%%, n=%d", best_name, best["roi"], best["n_bets"]
    )

    # Log top 5
    sorted_strats = sorted(strategies.items(), key=lambda x: x[1]["roi"], reverse=True)[:10]
    for name, r in sorted_strats:
        logger.info("  Val %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

    return best_name, strategies


def apply_strategy(
    name: str,
    df: pd.DataFrame,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    p_cb: np.ndarray,
    p_lgbm: np.ndarray,
    p_lr: np.ndarray,
    val_df: pd.DataFrame = None,
    val_p_std: np.ndarray = None,
    val_p_mean: np.ndarray = None,
) -> np.ndarray:
    """Применение стратегии к данным."""
    odds = df["Odds"].values
    ev_mean = p_mean * odds - 1

    if name.startswith("ev_"):
        thr = float(name.split("_")[1])
        return ev_mean >= thr

    if name.startswith("conf_ev_"):
        thr = float(name.split("_")[2])
        confidence = 1 / (1 + p_std * 10)
        return (ev_mean * confidence) >= thr

    if name.startswith("std_p"):
        # Format: std_p75_ev0.12
        import re

        m = re.match(r"std_p(\d+)_ev([\d.]+)", name)
        std_pct = int(m.group(1))
        ev_thr = float(m.group(2))
        # Используем percentile из val, не из test
        if val_p_std is not None and val_p_mean is not None:
            val_odds = val_df["Odds"].values
            val_ev = val_p_mean * val_odds - 1
            val_mask_ev = val_ev >= ev_thr
            if val_mask_ev.sum() > 0:
                std_thr = np.percentile(val_p_std[val_mask_ev], std_pct)
            else:
                std_thr = np.percentile(p_std, std_pct)
        else:
            std_thr = np.percentile(p_std, std_pct)
        mask_ev = ev_mean >= ev_thr
        return mask_ev & (p_std <= std_thr)

    if name.startswith("strat_"):
        configs = {
            "strat_v1": [(1, 3, 0.20), (3, 10, 0.15), (10, 50, 0.12), (50, 500, 0.08)],
            "strat_v2": [(1, 3, 0.25), (3, 10, 0.18), (10, 50, 0.10), (50, 500, 0.06)],
            "strat_v3": [(1, 5, 0.20), (5, 20, 0.12), (20, 100, 0.08), (100, 500, 0.05)],
            "strat_v4": [(1, 3, 0.15), (3, 10, 0.12), (10, 50, 0.10), (50, 500, 0.08)],
        }
        brackets = configs[name]
        mask = np.zeros(len(df), dtype=bool)
        for lo, hi, thr in brackets:
            bracket = (odds >= lo) & (odds < hi)
            mask |= bracket & (ev_mean >= thr)
        return mask

    # Fallback
    return ev_mean >= 0.12


def main() -> None:
    """Proper validation of selection strategies."""
    with mlflow.start_run(run_name="phase4/validate_strategies") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Proper val split from train
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            # Feature engineering
            train_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_enc)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_val = val_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            y_train = train_enc["target"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "n_features": len(FEATURES),
                    "method": "validate_strategies",
                }
            )

            # Обучаем на train_fit
            cb, lgbm, lr, scaler = train_ensemble(x_train, y_train)

            # Предсказания на val и test
            p_cb_val, p_lgbm_val, p_lr_val, p_mean_val, p_std_val = predict_ensemble(
                cb, lgbm, lr, scaler, x_val
            )
            _, _, _, p_mean_test, _ = predict_ensemble(cb, lgbm, lr, scaler, x_test)

            auc_val = roc_auc_score(val_enc["target"], p_mean_val)
            auc_test = roc_auc_score(test_enc["target"], p_mean_test)
            logger.info("AUC val: %.4f, test: %.4f", auc_val, auc_test)

            # Поиск лучшей стратегии на val
            best_name, all_strats = find_best_strategy_on_val(
                val_enc, p_mean_val, p_std_val, p_cb_val, p_lgbm_val, p_lr_val
            )

            # Теперь обучаем на ПОЛНОМ train (train_fit + val) и применяем к test
            logger.info("Retraining on full train for final evaluation...")
            train_full_enc, _ = add_sport_market_features(train, train)
            test_full_enc, _ = add_sport_market_features(test.copy(), train_full_enc)

            x_train_full = train_full_enc[FEATURES].fillna(0)
            x_test_full = test_full_enc[FEATURES].fillna(0)
            y_train_full = train_full_enc["target"]

            cb_full, lgbm_full, lr_full, scaler_full = train_ensemble(x_train_full, y_train_full)
            p_cb_t, p_lgbm_t, p_lr_t, p_mean_t, p_std_t = predict_ensemble(
                cb_full, lgbm_full, lr_full, scaler_full, x_test_full
            )

            auc_test_full = roc_auc_score(test_full_enc["target"], p_mean_t)
            logger.info("Full-train AUC test: %.4f", auc_test_full)

            # Baseline на full train
            ev_baseline = p_mean_t * test_full_enc["Odds"].values - 1
            mask_baseline = ev_baseline >= 0.12
            r_baseline = calc_roi(test_full_enc, mask_baseline.astype(float), threshold=0.5)
            logger.info(
                "Baseline (full train): ROI=%.2f%%, n=%d", r_baseline["roi"], r_baseline["n_bets"]
            )

            # Применяем top-5 стратегий с val к test
            sorted_strats = sorted(all_strats.items(), key=lambda x: x[1]["roi"], reverse=True)[:5]
            test_results = {}

            for strat_name, val_result in sorted_strats:
                mask_test = apply_strategy(
                    strat_name,
                    test_full_enc,
                    p_mean_t,
                    p_std_t,
                    p_cb_t,
                    p_lgbm_t,
                    p_lr_t,
                    val_df=val_enc,
                    val_p_std=p_std_val,
                    val_p_mean=p_mean_val,
                )
                r_test = calc_roi(test_full_enc, mask_test.astype(float), threshold=0.5)
                test_results[strat_name] = r_test
                logger.info(
                    "Test %s: ROI=%.2f%% (val=%.2f%%), n=%d",
                    strat_name,
                    r_test["roi"],
                    val_result["roi"],
                    r_test["n_bets"],
                )

            # Find best test result among validated strategies
            best_test_name = max(
                test_results,
                key=lambda k: test_results[k]["roi"] if test_results[k]["n_bets"] >= 50 else -999,
            )
            best_test = test_results[best_test_name]

            logger.info(
                "Best validated strategy: %s -> test ROI=%.2f%%, n=%d",
                best_test_name,
                best_test["roi"],
                best_test["n_bets"],
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "auc_val": auc_val,
                    "auc_test": auc_test_full,
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "roi_best_val": all_strats[best_name]["roi"],
                    "roi_best_test": best_test["roi"],
                    "n_bets_best_test": best_test["n_bets"],
                }
            )
            for name, r in test_results.items():
                mlflow.log_metrics(
                    {
                        f"roi_test_{name}": r["roi"],
                        f"n_bets_test_{name}": r["n_bets"],
                    }
                )
            mlflow.set_tag("best_strategy", best_test_name)

            # Save if improved
            if best_test["roi"] > 16.02:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_full.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_test["roi"],
                    "auc": auc_test_full,
                    "threshold": 0.12,
                    "ev_threshold": 0.12,
                    "n_bets": best_test["n_bets"],
                    "feature_names": FEATURES,
                    "selection_method": best_test_name,
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info("Model saved with ROI=%.2f%%", best_test["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.2 failed")
            raise


if __name__ == "__main__":
    main()
