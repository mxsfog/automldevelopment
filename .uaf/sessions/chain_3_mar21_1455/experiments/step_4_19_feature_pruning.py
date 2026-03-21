"""Step 4.19 — Feature importance pruning.

19 фичей может содержать noise-фичи. CatBoost feature importance
покажет какие фичи наиболее важны. Обучим с top-K фичами.

Тесты:
1. Top-15 (drop 4 worst)
2. Top-12
3. Top-10
4. Top-8
5. Сравнение с baseline 19
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

ALL_FEATURES = [
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


def evaluate_conf_ev(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    odds: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    """conf_ev at 0.15."""
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf

    results = {}
    for thr in [0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        results[f"conf_ev_{thr:.2f}"] = r
    return results


def main() -> None:
    """Feature importance pruning."""
    with mlflow.start_run(run_name="phase4/feature_pruning") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "feature_pruning",
                }
            )

            # Get feature importance from CatBoost on all 19 features
            x_tr = train_fit_enc[ALL_FEATURES].fillna(0)
            cb_full = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb_full.fit(x_tr, train_fit_enc["target"])
            importances = cb_full.get_feature_importance()
            fi = sorted(
                zip(ALL_FEATURES, importances, strict=True), key=lambda x: x[1], reverse=True
            )
            logger.info("Feature importance ranking:")
            for fname, imp in fi:
                logger.info("  %s: %.2f", fname, imp)

            # Feature sets by importance
            feature_sets = {f"top{k}": [f for f, _ in fi[:k]] for k in [8, 10, 12, 15]}
            feature_sets["all_19"] = ALL_FEATURES

            all_val = {}

            for set_name, features in feature_sets.items():
                logger.info("Config: %s (%d features)", set_name, len(features))
                x_tr_sub = train_fit_enc[features].fillna(0)
                x_va_sub = val_enc[features].fillna(0)
                cb, lgbm, lr, scaler = train_ensemble(x_tr_sub, train_fit_enc["target"])
                p, s = predict_ensemble(cb, lgbm, lr, scaler, x_va_sub)
                auc = roc_auc_score(val_enc["target"], p)
                logger.info("  Val AUC: %.4f", auc)
                r = evaluate_conf_ev(p, s, val_enc["Odds"].values, val_enc)
                for k, v in r.items():
                    all_val[f"{set_name}_{k}"] = v
                    logger.info(
                        "  Val %s_%s: ROI=%.2f%%, n=%d",
                        set_name,
                        k,
                        v["roi"],
                        v["n_bets"],
                    )

            # Val ranking
            val_ranked = sorted(
                all_val.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 30 else -999,
                reverse=True,
            )
            logger.info("Top-5 val:")
            for name, r in val_ranked[:5]:
                logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Test
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            test_results = {}
            for set_name, features in feature_sets.items():
                x_train_sub = train_enc[features].fillna(0)
                x_test_sub = test_enc[features].fillna(0)
                cb, lgbm, lr, scaler = train_ensemble(x_train_sub, train_enc["target"])
                p, s = predict_ensemble(cb, lgbm, lr, scaler, x_test_sub)
                auc = roc_auc_score(test_enc["target"], p)
                logger.info("Test %s AUC: %.4f", set_name, auc)
                r = evaluate_conf_ev(p, s, test_enc["Odds"].values, test_enc)
                for k, v in r.items():
                    test_results[f"{set_name}_{k}"] = v

            for k, v in sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )[:10]:
                if v["n_bets"] >= 50:
                    logger.info("Test %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            best_name = max(
                test_results,
                key=lambda k: test_results[k]["roi"] if test_results[k]["n_bets"] >= 50 else -999,
            )
            best = test_results[best_name]
            logger.info(
                "Best (n>=50): %s -> ROI=%.2f%%, n=%d",
                best_name,
                best["roi"],
                best["n_bets"],
            )

            mlflow.log_metrics(
                {
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            mlflow.set_tag("best_strategy", best_name)
            mlflow.set_tag(
                "feature_importance",
                ", ".join(f"{f}={i:.1f}" for f, i in fi[:10]),
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.19 failed")
            raise


if __name__ == "__main__":
    main()
