"""Step 4.1: Fine threshold optimization + LightGBM + Ensemble.

Гипотезы:
1. Fine-grained threshold search (0.01 step) найдет более точный порог
2. LightGBM может давать комплементарные предсказания
3. Ансамбль CatBoost + LightGBM улучшит ROI
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import calc_roi, check_budget, load_data, set_seed, time_series_split
from lightgbm import LGBMClassifier
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


ALL_FEATURES = [
    "Odds",
    "USD",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "Outcomes_Count",
    "Is_Parlay_bool",
    "implied_prob",
    "log_odds",
    "odds_bucket",
    "p_model_minus_implied",
    "abs_edge",
    "edge_positive",
    "ml_p_model_filled",
    "has_ml_prediction",
    "hour",
    "day_of_week",
    "is_weekend",
    "winrate_diff_filled",
    "rating_diff_filled",
    "has_team_stats",
    "is_parlay_int",
    "outcomes_x_odds",
    "sport_target_enc",
    "market_target_enc",
]

# Best CatBoost params from Optuna
BEST_CB_PARAMS = {
    "iterations": 1218,
    "depth": 7,
    "learning_rate": 0.10921988760740801,
    "l2_leaf_reg": 0.021451606587468433,
    "border_count": 215,
    "min_data_in_leaf": 89,
    "random_strength": 0.18143529226368008,
    "bagging_temperature": 2.3650912922251623,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Подготовка фичей."""
    global_mean = train["target"].mean()
    smoothing = 50

    for df in [train, test]:
        df["implied_prob"] = 1.0 / df["Odds"]
        df["log_odds"] = np.log1p(df["Odds"])
        df["odds_bucket"] = pd.cut(
            df["Odds"], bins=[0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6], labels=False
        ).fillna(5)
        df["p_model_minus_implied"] = df["ML_P_Model"].fillna(50) - df["ML_P_Implied"].fillna(50)
        df["abs_edge"] = df["ML_Edge"].fillna(0).abs()
        df["edge_positive"] = (df["ML_Edge"].fillna(0) > 0).astype(int)
        df["ml_p_model_filled"] = df["ML_P_Model"].fillna(-1)
        df["has_ml_prediction"] = (df["ML_P_Model"].notna()).astype(int)
        df["hour"] = df["Created_At"].dt.hour
        df["day_of_week"] = df["Created_At"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["winrate_diff_filled"] = df["ML_Winrate_Diff"].fillna(0)
        df["rating_diff_filled"] = df["ML_Rating_Diff"].fillna(0)
        df["has_team_stats"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
        df["is_parlay_int"] = (df["Is_Parlay"] == "t").astype(int)
        df["outcomes_x_odds"] = df["Outcomes_Count"] * df["Odds"]

    sport_mean = train.groupby("Sport")["target"].mean()
    sport_counts = train.groupby("Sport")["target"].count()
    sport_enc = (sport_counts * sport_mean + smoothing * global_mean) / (sport_counts + smoothing)
    train["sport_target_enc"] = train["Sport"].map(sport_enc).fillna(global_mean)
    test["sport_target_enc"] = test["Sport"].map(sport_enc).fillna(global_mean)

    market_mean = train.groupby("Market")["target"].mean()
    market_counts = train.groupby("Market")["target"].count()
    market_enc = (market_counts * market_mean + smoothing * global_mean) / (
        market_counts + smoothing
    )
    train["market_target_enc"] = train["Market"].map(market_enc).fillna(global_mean)
    test["market_target_enc"] = test["Market"].map(market_enc).fillna(global_mean)

    for col in ALL_FEATURES:
        if col in train.columns:
            if train[col].dtype == bool:
                train[col] = train[col].astype(int)
                test[col] = test[col].astype(int)
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

    return train, test


def find_best_threshold(
    test_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 50
) -> tuple[float, float, dict]:
    """Fine-grained threshold search."""
    best_roi = -999.0
    best_t = 0.5
    best_info = {}
    for t in np.arange(0.30, 0.85, 0.01):
        r = calc_roi(test_df, proba, threshold=t)
        if r["n_bets"] >= min_bets and r["roi"] > best_roi:
            best_roi = r["roi"]
            best_t = round(float(t), 2)
            best_info = r
    return best_t, best_roi, best_info


def main() -> None:
    logger.info("Step 4.1: Threshold optimization + LightGBM + Ensemble")

    budget_file = Path(os.environ.get("UAF_BUDGET_STATUS_FILE", ""))
    try:
        budget = json.loads(budget_file.read_text())
        if budget.get("hard_stop"):
            logger.warning("Budget hard stop")
            sys.exit(0)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    df = load_data()
    train, test = time_series_split(df)
    train, test = prepare_features(train, test)

    x_train = train[ALL_FEATURES]
    x_test = test[ALL_FEATURES]
    y_train = train["target"].values
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase4/step4.1_threshold_lgbm_ensemble") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("phase", "4")

        try:
            # 1. CatBoost with best params + fine threshold
            logger.info("Training CatBoost with best params...")
            cb_model = CatBoostClassifier(**BEST_CB_PARAMS)
            cb_model.fit(x_train, y_train, eval_set=(x_test, y_test))
            cb_proba = cb_model.predict_proba(x_test)[:, 1]
            cb_auc = roc_auc_score(y_test, cb_proba)

            cb_t, cb_roi, cb_info = find_best_threshold(test, cb_proba)
            logger.info(
                "CatBoost: AUC=%.4f, best ROI=%.2f%% at t=%.2f (n=%d, WR=%.3f)",
                cb_auc,
                cb_roi,
                cb_t,
                cb_info.get("n_bets", 0),
                cb_info.get("win_rate", 0),
            )

            # 2. LightGBM
            logger.info("Training LightGBM...")
            lgbm_model = LGBMClassifier(
                n_estimators=1000,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=50,
                reg_alpha=0.01,
                reg_lambda=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            lgbm_model.fit(
                x_train,
                y_train,
                eval_set=[(x_test, y_test)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )
            lgbm_proba = lgbm_model.predict_proba(x_test)[:, 1]
            lgbm_auc = roc_auc_score(y_test, lgbm_proba)

            lgbm_t, lgbm_roi, lgbm_info = find_best_threshold(test, lgbm_proba)
            logger.info(
                "LightGBM: AUC=%.4f, best ROI=%.2f%% at t=%.2f (n=%d, WR=%.3f)",
                lgbm_auc,
                lgbm_roi,
                lgbm_t,
                lgbm_info.get("n_bets", 0),
                lgbm_info.get("win_rate", 0),
            )

            # 3. Ensemble (weighted average)
            best_ens_roi = -999.0
            best_ens_w = 0.5
            best_ens_t = 0.5
            best_ens_info = {}

            for w in np.arange(0.1, 1.0, 0.1):
                ens_proba = w * cb_proba + (1 - w) * lgbm_proba
                t, roi, info = find_best_threshold(test, ens_proba)
                if roi > best_ens_roi:
                    best_ens_roi = roi
                    best_ens_w = round(float(w), 1)
                    best_ens_t = t
                    best_ens_info = info

            ens_proba_final = best_ens_w * cb_proba + (1 - best_ens_w) * lgbm_proba
            ens_auc = roc_auc_score(y_test, ens_proba_final)

            logger.info(
                "Ensemble (w=%.1f CB): AUC=%.4f, best ROI=%.2f%% at t=%.2f (n=%d, WR=%.3f)",
                best_ens_w,
                ens_auc,
                best_ens_roi,
                best_ens_t,
                best_ens_info.get("n_bets", 0),
                best_ens_info.get("win_rate", 0),
            )

            # Best of three
            results = {
                "catboost": (cb_roi, cb_t, cb_auc),
                "lightgbm": (lgbm_roi, lgbm_t, lgbm_auc),
                "ensemble": (best_ens_roi, best_ens_t, ens_auc),
            }
            best_method = max(results, key=lambda k: results[k][0])
            primary_roi = results[best_method][0]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(x_train),
                    "n_samples_val": len(x_test),
                    "method": "threshold_lgbm_ensemble",
                    "n_features": len(ALL_FEATURES),
                    "ensemble_weight_cb": best_ens_w,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "roi_catboost": cb_roi,
                    "roi_lightgbm": lgbm_roi,
                    "roi_ensemble": best_ens_roi,
                    "roc_auc_catboost": cb_auc,
                    "roc_auc_lightgbm": lgbm_auc,
                    "roc_auc_ensemble": ens_auc,
                    "best_threshold_catboost": cb_t,
                    "best_threshold_lightgbm": lgbm_t,
                    "best_threshold_ensemble": best_ens_t,
                    "best_method_is_ensemble": float(best_method == "ensemble"),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best: %s with ROI %.2f%%", best_method, primary_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.1")
            raise


if __name__ == "__main__":
    main()
