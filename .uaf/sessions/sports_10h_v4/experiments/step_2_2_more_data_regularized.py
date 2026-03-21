"""Step 2.2: CatBoost с расширенными фичами + сильная регуляризация + использование всех данных train."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    calc_roi,
    check_budget,
    find_best_threshold,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uaf/sports_10h_v4")
SESSION_ID = os.environ.get("UAF_SESSION_ID", "sports_10h_v4")
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

CAT_COLS = ["Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket", "sport_market"]


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Все фичи из step 2.1 + дополнительные."""
    df = df.copy()

    # implied probability
    df["implied_prob"] = 1.0 / df["Odds"]

    # ML vs market
    df["ml_vs_market"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]

    # Edge нормализованный
    df["edge_normalized"] = df["ML_Edge"] / (df["ML_P_Implied"].replace(0, np.nan) + 1e-6)

    # Value indicator
    df["is_value_bet"] = (df["ML_P_Model"] / 100.0 > df["implied_prob"] * 1.05).astype(int)

    # EV ratio
    df["ev_ratio"] = df["ML_EV"] / 100.0

    # Odds buckets
    df["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 5.0, 100, 10000],
        labels=[
            "1.0-1.3",
            "1.3-1.5",
            "1.5-1.8",
            "1.8-2.0",
            "2.0-2.5",
            "2.5-3.0",
            "3.0-5.0",
            "5.0-100",
            "100+",
        ],
    ).astype(str)

    # Kelly
    p = df["ML_P_Model"].fillna(50) / 100.0
    b = df["Odds"] - 1
    q = 1 - p
    df["kelly_fraction"] = ((b * p - q) / (b + 1e-6)).clip(-1, 1)

    # Confidence
    df["ml_confidence"] = (df["ML_P_Model"].fillna(50) - 50).abs()

    # Log odds
    df["log_odds"] = np.log(df["Odds"].clip(1.001))

    # Time
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Sport x Market interaction
    df["sport_market"] = (
        df["Sport"].fillna("_").astype(str) + "__" + df["Market"].fillna("_").astype(str)
    )

    # Stake size relative (log)
    df["log_usd"] = np.log1p(df["USD"])

    # Parlay x odds interaction
    df["parlay_odds"] = df["Is_Parlay"].map({"t": 1, "f": 0}).fillna(0) * df["Odds"]

    # ML model availability
    df["has_ml_prediction"] = (~df["ML_P_Model"].isna()).astype(int)

    return df


def prepare_data(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS]

    x_train = train[features].copy()
    x_test = test[features].copy()

    for idx in cat_indices:
        col = features[idx]
        x_train[col] = x_train[col].astype(str).replace("nan", "_missing_")
        x_test[col] = x_test[col].astype(str).replace("nan", "_missing_")

    return x_train, x_test, cat_indices


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase2/step_2_2_regularized") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "2.2")
            mlflow.set_tag("phase", "2")

            df = load_data()
            df = add_all_features(df)

            train, test = time_series_split(df)

            # Val = последние 20% train
            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            features = [
                "Odds",
                "USD",
                "Outcomes_Count",
                "ML_P_Model",
                "ML_P_Implied",
                "ML_Edge",
                "ML_EV",
                "ML_Winrate_Diff",
                "ML_Rating_Diff",
                "Odds_outcome",
                "n_outcomes",
                "Is_Parlay",
                "Sport",
                "Market",
                "ML_Team_Stats_Found",
                "implied_prob",
                "ml_vs_market",
                "edge_normalized",
                "is_value_bet",
                "ev_ratio",
                "kelly_fraction",
                "ml_confidence",
                "log_odds",
                "hour",
                "day_of_week",
                "is_weekend",
                "odds_bucket",
                "sport_market",
                "log_usd",
                "parlay_odds",
                "has_ml_prediction",
            ]
            features = [f for f in features if f in df.columns]

            x_train, x_test, cat_indices = prepare_data(train_inner, test, features)
            x_val = val[features].copy()
            for idx in cat_indices:
                col = features[idx]
                x_val[col] = x_val[col].astype(str).replace("nan", "_missing_")

            params = {
                "iterations": 2000,
                "learning_rate": 0.01,
                "depth": 4,
                "l2_leaf_reg": 10,
                "min_data_in_leaf": 50,
                "random_strength": 2,
                "bagging_temperature": 1,
                "random_seed": 42,
                "verbose": 100,
                "eval_metric": "AUC",
                "cat_features": cat_indices,
                "early_stopping_rounds": 100,
                "task_type": "CPU",
            }

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "catboost_regularized",
                    "n_features": len(features),
                    "gap_days": 7,
                    "iterations": 2000,
                    "learning_rate": 0.01,
                    "depth": 4,
                    "l2_leaf_reg": 10,
                    "min_data_in_leaf": 50,
                }
            )

            model = CatBoostClassifier(**params)
            model.fit(
                x_train,
                train_inner["target"].values,
                eval_set=(x_val, val["target"].values),
                use_best_model=True,
            )

            # Val threshold
            val_probas = model.predict_proba(x_val)[:, 1]
            best_thr = find_best_threshold(val, val_probas)

            # Test
            test_probas = model.predict_proba(x_test)[:, 1]
            result = calc_roi(test, test_probas, threshold=best_thr)
            result_50 = calc_roi(test, test_probas, threshold=0.5)
            auc = roc_auc_score(test["target"].values, test_probas)

            logger.info(
                "Test (thr=%.2f): ROI=%.2f%%, bets=%d/%d, prec=%.3f, sel=%.3f",
                best_thr,
                result["roi"],
                result["n_bets"],
                len(test),
                result["precision"],
                result["selectivity"],
            )
            logger.info(
                "Test (thr=0.50): ROI=%.2f%%, bets=%d/%d, prec=%.3f",
                result_50["roi"],
                result_50["n_bets"],
                len(test),
                result_50["precision"],
            )
            logger.info("AUC: %.4f, best iter: %d", auc, model.best_iteration_)

            # Scan thresholds on test with val-derived threshold vicinity
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                r = calc_roi(test, test_probas, threshold=thr)
                logger.info(
                    "  thr=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )

            mlflow.log_metrics(
                {
                    "roi": result["roi"],
                    "roi_thr_50": result_50["roi"],
                    "best_threshold": best_thr,
                    "n_bets": result["n_bets"],
                    "precision": result["precision"],
                    "selectivity": result["selectivity"],
                    "roc_auc": auc,
                    "best_iteration": model.best_iteration_,
                }
            )

            # Feature importance
            fi = model.get_feature_importance()
            fi_sorted = sorted(zip(features, fi), key=lambda x: x[1], reverse=True)
            for fname, fimp in fi_sorted[:15]:
                mlflow.log_metric(f"fi_{fname}", fimp)
                logger.info("  fi %s: %.2f", fname, fimp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.2")
            logger.exception("Step 2.2 failed")
            raise


if __name__ == "__main__":
    main()
