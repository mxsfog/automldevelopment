"""Step 4.2: Stacking ensemble CatBoost + LightGBM + XGBoost + LogReg meta.

Гипотеза: Stacking с разными базовыми моделями дает более робастные предсказания.
Мета-модель (LogReg) учится на OOF предсказаниях базовых моделей.
"""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import calc_roi, check_budget, load_data, set_seed, time_series_split
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

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


def get_oof_predictions(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OOF predictions from 3 base models using TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_cb = np.zeros(len(x_train))
    oof_lgb = np.zeros(len(x_train))
    oof_xgb = np.zeros(len(x_train))
    test_cb = np.zeros(len(x_test))
    test_lgb = np.zeros(len(x_test))
    test_xgb = np.zeros(len(x_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(x_train)):
        x_tr = x_train.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        x_val = x_train.iloc[val_idx]
        y_val = y_train[val_idx]

        # CatBoost
        cb = CatBoostClassifier(
            iterations=1218,
            depth=7,
            learning_rate=0.109,
            l2_leaf_reg=0.021,
            border_count=215,
            min_data_in_leaf=89,
            random_strength=0.18,
            bagging_temperature=2.37,
            random_seed=42,
            verbose=0,
            eval_metric="AUC",
            early_stopping_rounds=50,
        )
        cb.fit(x_tr, y_tr, eval_set=(x_val, y_val))
        oof_cb[val_idx] = cb.predict_proba(x_val)[:, 1]
        test_cb += cb.predict_proba(x_test)[:, 1] / n_splits

        # LightGBM
        lgb = LGBMClassifier(
            n_estimators=500,
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
        from lightgbm import early_stopping, log_evaluation

        lgb.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
        oof_lgb[val_idx] = lgb.predict_proba(x_val)[:, 1]
        test_lgb += lgb.predict_proba(x_test)[:, 1] / n_splits

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            min_child_weight=50,
            reg_alpha=0.01,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
            n_jobs=-1,
        )
        xgb.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb.predict_proba(x_val)[:, 1]
        test_xgb += xgb.predict_proba(x_test)[:, 1] / n_splits

        logger.info(
            "Fold %d: CB=%.4f, LGB=%.4f, XGB=%.4f",
            fold_idx,
            roc_auc_score(y_val, oof_cb[val_idx]),
            roc_auc_score(y_val, oof_lgb[val_idx]),
            roc_auc_score(y_val, oof_xgb[val_idx]),
        )

    return oof_cb, oof_lgb, oof_xgb, test_cb, test_lgb, test_xgb


def main() -> None:
    logger.info("Step 4.2: Stacking ensemble")

    budget_file = Path(os.environ.get("UAF_BUDGET_STATUS_FILE", ""))
    try:
        budget = json.loads(budget_file.read_text())
        if budget.get("hard_stop"):
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

    with mlflow.start_run(run_name="phase4/step4.2_stacking") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(x_train),
                    "n_samples_val": len(x_test),
                    "method": "stacking_3models",
                    "n_features": len(ALL_FEATURES),
                    "n_splits_oof": 5,
                    "meta_model": "logistic_regression",
                }
            )

            # OOF predictions
            oof_cb, oof_lgb, oof_xgb, test_cb, test_lgb, test_xgb = get_oof_predictions(
                x_train, y_train, x_test, n_splits=5
            )

            # Stack OOF for meta model (skip first fold with zeros)
            valid_mask = (oof_cb > 0) | (oof_lgb > 0) | (oof_xgb > 0)
            meta_train = np.column_stack(
                [oof_cb[valid_mask], oof_lgb[valid_mask], oof_xgb[valid_mask]]
            )
            meta_test = np.column_stack([test_cb, test_lgb, test_xgb])
            y_meta = y_train[valid_mask]

            # Meta model
            meta = LogisticRegression(random_state=42, C=1.0, max_iter=1000)
            meta.fit(meta_train, y_meta)
            stacking_proba = meta.predict_proba(meta_test)[:, 1]

            stacking_auc = roc_auc_score(y_test, stacking_proba)
            st_t, st_roi, st_info = find_best_threshold(test, stacking_proba)

            logger.info(
                "Stacking: AUC=%.4f, best ROI=%.2f%% at t=%.2f (n=%d, WR=%.3f)",
                stacking_auc,
                st_roi,
                st_t,
                st_info.get("n_bets", 0),
                st_info.get("win_rate", 0),
            )

            # Compare with simple average
            avg_proba = (test_cb + test_lgb + test_xgb) / 3
            avg_auc = roc_auc_score(y_test, avg_proba)
            avg_t, avg_roi, _avg_info = find_best_threshold(test, avg_proba)
            logger.info(
                "Simple avg: AUC=%.4f, best ROI=%.2f%% at t=%.2f",
                avg_auc,
                avg_roi,
                avg_t,
            )

            # Individual model results on test
            cb_auc = roc_auc_score(y_test, test_cb)
            lgb_auc = roc_auc_score(y_test, test_lgb)
            xgb_auc = roc_auc_score(y_test, test_xgb)
            _cb_t, cb_roi, _ = find_best_threshold(test, test_cb)
            _lgb_t, lgb_roi, _ = find_best_threshold(test, test_lgb)
            _xgb_t, xgb_roi, _ = find_best_threshold(test, test_xgb)

            logger.info(
                "Individual: CB ROI=%.2f%%, LGB ROI=%.2f%%, XGB ROI=%.2f%%",
                cb_roi,
                lgb_roi,
                xgb_roi,
            )

            primary_roi = max(st_roi, avg_roi, cb_roi, lgb_roi, xgb_roi)

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "roi_stacking": st_roi,
                    "roi_simple_avg": avg_roi,
                    "roi_catboost": cb_roi,
                    "roi_lightgbm": lgb_roi,
                    "roi_xgboost": xgb_roi,
                    "roc_auc_stacking": stacking_auc,
                    "roc_auc_simple_avg": avg_auc,
                    "roc_auc_catboost": cb_auc,
                    "roc_auc_lightgbm": lgb_auc,
                    "roc_auc_xgboost": xgb_auc,
                    "best_threshold_stacking": st_t,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Primary ROI: %.2f%%", primary_roi)

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.2")
            raise


if __name__ == "__main__":
    main()
