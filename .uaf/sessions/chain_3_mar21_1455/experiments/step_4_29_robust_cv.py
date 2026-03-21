"""Step 4.29 — Max-min CV optimization: find most robust strategy.

Цель: вместо max mean ROI на CV, найти стратегию с max(min fold ROI).
Это даст самую устойчивую стратегию, которая работает во всех периодах.

Grid: combination of selection method + threshold + odds cap.
5-fold expanding window CV.
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


def train_ensemble(x: pd.DataFrame, y: pd.Series) -> tuple:
    """3-model ensemble."""
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y)
    return cb, lgbm, lr, scaler


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
    return p_mean, p_std


def evaluate_strategies(
    p: np.ndarray,
    s: np.ndarray,
    odds: np.ndarray,
    df: pd.DataFrame,
) -> dict[str, dict]:
    """Evaluate all strategy variants."""
    results: dict[str, dict] = {}

    # EV and confidence
    ev = p * odds - 1
    conf = 1 / (1 + s * 10)
    ev_conf = ev * conf

    # Edge
    p_implied = 1 / odds
    edge = p - p_implied

    # 1. conf_ev at various thresholds
    for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        results[f"confev_{thr:.2f}"] = calc_roi(df, mask.astype(float), threshold=0.5)

    # 2. conf_ev with odds cap
    for max_odds in [5, 10, 50]:
        for thr in [0.05, 0.08, 0.10, 0.12]:
            mask = (ev_conf >= thr) & (odds <= max_odds)
            results[f"confev_{thr:.2f}_cap{max_odds}"] = calc_roi(
                df, mask.astype(float), threshold=0.5
            )

    # 3. Edge-based
    for max_odds in [2, 3, 5]:
        for e_thr in [0.05, 0.08, 0.10]:
            mask = (edge >= e_thr) & (odds <= max_odds)
            results[f"edge_e{e_thr:.2f}_cap{max_odds}"] = calc_roi(
                df, mask.astype(float), threshold=0.5
            )

    # 4. Pure EV (no confidence)
    for thr in [0.05, 0.10, 0.12, 0.15]:
        mask = ev >= thr
        results[f"ev_{thr:.2f}"] = calc_roi(df, mask.astype(float), threshold=0.5)

    # 5. p_mean threshold
    for p_thr in [0.55, 0.60]:
        mask = p >= p_thr
        results[f"pmean_{p_thr:.2f}"] = calc_roi(df, mask.astype(float), threshold=0.5)

    return results


def main() -> None:
    """Robust CV optimization."""
    with mlflow.start_run(run_name="phase4/robust_cv") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            mlflow.log_params(
                {
                    "validation_scheme": "expanding_cv_5fold",
                    "seed": 42,
                    "method": "robust_cv_maxmin",
                }
            )

            # 5-fold expanding window CV
            n = len(df)
            fold_size = n // 6
            cv_fold_results: dict[str, list[float]] = {}

            for fold in range(5):
                train_end = (fold + 1) * fold_size + fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, n)
                if test_end <= test_start:
                    continue

                cv_train = df.iloc[:train_end].copy()
                cv_test = df.iloc[test_start:test_end].copy()

                cv_train_enc, _ = add_sport_market_features(cv_train, cv_train)
                cv_test_enc, _ = add_sport_market_features(cv_test, cv_train_enc)

                x_tr = cv_train_enc[FEATURES].fillna(0)
                x_te = cv_test_enc[FEATURES].fillna(0)

                cb, lgbm, lr, scaler = train_ensemble(x_tr, cv_train_enc["target"])
                p, s = predict_ensemble(cb, lgbm, lr, scaler, x_te)
                odds = cv_test_enc["Odds"].values

                fold_strats = evaluate_strategies(p, s, odds, cv_test_enc)

                for name, r in fold_strats.items():
                    if name not in cv_fold_results:
                        cv_fold_results[name] = []
                    cv_fold_results[name].append(r["roi"])

                logger.info("Fold %d: %d strategies evaluated", fold, len(fold_strats))

            # Compute metrics per strategy
            strategy_stats: list[dict] = []
            for name, rois in cv_fold_results.items():
                if len(rois) < 4:
                    continue
                strategy_stats.append(
                    {
                        "name": name,
                        "mean": np.mean(rois),
                        "std": np.std(rois),
                        "min": np.min(rois),
                        "max": np.max(rois),
                        "sharpe": np.mean(rois) / np.std(rois) if np.std(rois) > 0 else 0,
                        "n_folds": len(rois),
                    }
                )

            # Rank by max-min (most robust)
            by_min = sorted(strategy_stats, key=lambda x: x["min"], reverse=True)
            logger.info("=== Top-10 by MIN fold ROI (most robust) ===")
            for s in by_min[:10]:
                logger.info(
                    "  %s: min=%.2f%%, mean=%.2f%%, std=%.2f%%, sharpe=%.2f",
                    s["name"],
                    s["min"],
                    s["mean"],
                    s["std"],
                    s["sharpe"],
                )

            # Rank by Sharpe
            by_sharpe = sorted(strategy_stats, key=lambda x: x["sharpe"], reverse=True)
            logger.info("=== Top-10 by Sharpe ratio ===")
            for s in by_sharpe[:10]:
                logger.info(
                    "  %s: sharpe=%.2f, mean=%.2f%%, std=%.2f%%, min=%.2f%%",
                    s["name"],
                    s["sharpe"],
                    s["mean"],
                    s["std"],
                    s["min"],
                )

            # Rank by mean
            by_mean = sorted(strategy_stats, key=lambda x: x["mean"], reverse=True)
            logger.info("=== Top-10 by mean ROI ===")
            for s in by_mean[:10]:
                logger.info(
                    "  %s: mean=%.2f%%, std=%.2f%%, min=%.2f%%",
                    s["name"],
                    s["mean"],
                    s["std"],
                    s["min"],
                )

            # Test evaluation for top strategies
            logger.info("=== Test evaluation ===")
            train, test = time_series_split(df, test_size=0.2)
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_train, train_enc["target"])
            p_t, s_t = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_test)
            odds_test = test_enc["Odds"].values

            test_strats = evaluate_strategies(p_t, s_t, odds_test, test_enc)

            # Show top min-robust strategies on test
            logger.info("Top min-robust strategies → test:")
            for s in by_min[:10]:
                name = s["name"]
                if name in test_strats:
                    t = test_strats[name]
                    logger.info(
                        "  %s: CV min=%.2f%% → test=%.2f%% (n=%d)",
                        name,
                        s["min"],
                        t["roi"],
                        t["n_bets"],
                    )

            # Best min-robust
            best_robust = by_min[0] if by_min else None
            best_sharpe = by_sharpe[0] if by_sharpe else None

            if best_robust and best_robust["name"] in test_strats:
                br = test_strats[best_robust["name"]]
                mlflow.log_metrics(
                    {
                        "roi_best_robust": br["roi"],
                        "cv_min_best_robust": best_robust["min"],
                    }
                )

            if best_sharpe and best_sharpe["name"] in test_strats:
                bs = test_strats[best_sharpe["name"]]
                mlflow.log_metrics(
                    {
                        "roi_best_sharpe": bs["roi"],
                        "cv_sharpe_best": best_sharpe["sharpe"],
                    }
                )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.29 failed")
            raise


if __name__ == "__main__":
    main()
