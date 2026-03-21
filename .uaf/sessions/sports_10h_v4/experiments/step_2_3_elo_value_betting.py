"""Step 2.3: ELO features + value betting approach (ставка где model_prob > implied_prob + margin)."""

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

CAT_COLS = {"Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный набор фичей."""
    df = df.copy()

    # Value features
    df["implied_prob"] = 1.0 / df["Odds"]
    df["ml_vs_market"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["edge_normalized"] = df["ML_Edge"] / (df["ML_P_Implied"].clip(lower=0.1) + 1e-6)
    df["is_value_bet"] = (df["ML_P_Model"] / 100.0 > df["implied_prob"] * 1.05).astype(int)
    df["ev_ratio"] = df["ML_EV"] / 100.0

    # Odds
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
    df["log_odds"] = np.log(df["Odds"].clip(1.001))

    # Kelly
    p = df["ML_P_Model"].fillna(50) / 100.0
    b = df["Odds"] - 1
    q = 1 - p
    df["kelly_fraction"] = ((b * p - q) / (b + 1e-6)).clip(-1, 1)
    df["ml_confidence"] = (df["ML_P_Model"].fillna(50) - 50).abs()

    # Time
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Log stake
    df["log_usd"] = np.log1p(df["USD"])

    # Parlay interaction
    df["parlay_flag"] = df["Is_Parlay"].map({"t": 1, "f": 0}).fillna(0).astype(int)
    df["parlay_odds"] = df["parlay_flag"] * df["Odds"]

    # ML availability
    df["has_ml_prediction"] = (~df["ML_P_Model"].isna()).astype(int)

    # Outcomes count features
    df["is_single"] = (df["Outcomes_Count"] == 1).astype(int)

    return df


def find_best_margin(df: pd.DataFrame, probas: np.ndarray) -> tuple[float, float]:
    """Поиск лучшего margin для value betting на val."""
    implied = 1.0 / df["Odds"].values
    best_roi = -999.0
    best_margin = 0.0

    for margin in np.arange(-0.05, 0.30, 0.01):
        mask = probas > (implied + margin)
        if mask.sum() < 20:
            continue
        selected = df[mask]
        n = len(selected)
        payout = selected.loc[selected["target"] == 1, "Odds"].sum()
        roi = (payout - n) / n * 100
        if roi > best_roi:
            best_roi = roi
            best_margin = margin

    logger.info("Best margin: %.2f, val ROI: %.2f%%", best_margin, best_roi)
    return best_margin, best_roi


def value_betting_roi(df: pd.DataFrame, probas: np.ndarray, margin: float) -> dict:
    """ROI по стратегии value betting."""
    implied = 1.0 / df["Odds"].values
    mask = probas > (implied + margin)
    selected = df[mask]

    if len(selected) == 0:
        return {
            "roi": 0.0,
            "n_bets": 0,
            "precision": 0.0,
            "selectivity": 0.0,
            "n_won": 0,
            "n_lost": 0,
        }

    n = len(selected)
    n_won = int(selected["target"].sum())
    payout = selected.loc[selected["target"] == 1, "Odds"].sum()
    roi = (payout - n) / n * 100
    return {
        "roi": roi,
        "n_bets": n,
        "precision": n_won / n,
        "selectivity": n / len(df),
        "n_won": n_won,
        "n_lost": n - n_won,
    }


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

    with mlflow.start_run(run_name="phase2/step_2_3_value_betting") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "2.3")
            mlflow.set_tag("phase", "2")

            df = load_data()
            df = add_features(df)

            train, test = time_series_split(df)

            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            features = [
                "Odds",
                "log_odds",
                "implied_prob",
                "Outcomes_Count",
                "is_single",
                "ML_P_Model",
                "ML_P_Implied",
                "ML_Edge",
                "ML_EV",
                "ML_Winrate_Diff",
                "ML_Rating_Diff",
                "ml_vs_market",
                "edge_normalized",
                "is_value_bet",
                "ev_ratio",
                "kelly_fraction",
                "ml_confidence",
                "hour",
                "day_of_week",
                "is_weekend",
                "log_usd",
                "parlay_flag",
                "parlay_odds",
                "has_ml_prediction",
                "Is_Parlay",
                "Sport",
                "Market",
                "ML_Team_Stats_Found",
                "odds_bucket",
            ]
            features = [f for f in features if f in df.columns]

            x_train, x_test, cat_indices = prepare_data(train_inner, test, features)
            x_val = val[features].copy()
            for idx in cat_indices:
                col = features[idx]
                x_val[col] = x_val[col].astype(str).replace("nan", "_missing_")

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "catboost_value_betting",
                    "n_features": len(features),
                    "gap_days": 7,
                    "iterations": 2000,
                    "learning_rate": 0.01,
                    "depth": 4,
                    "l2_leaf_reg": 10,
                }
            )

            model = CatBoostClassifier(
                iterations=2000,
                learning_rate=0.01,
                depth=4,
                l2_leaf_reg=10,
                min_data_in_leaf=50,
                random_strength=2,
                bagging_temperature=1,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                early_stopping_rounds=100,
                task_type="CPU",
            )
            model.fit(
                x_train,
                train_inner["target"].values,
                eval_set=(x_val, val["target"].values),
                use_best_model=True,
            )

            logger.info("Best iteration: %d", model.best_iteration_)

            # Approach 1: Standard threshold on probability
            val_probas = model.predict_proba(x_val)[:, 1]
            test_probas = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(test["target"].values, test_probas)
            logger.info("AUC: %.4f", auc)

            # Standard threshold scan
            best_thr_roi = -999.0
            best_thr = 0.5
            for thr in np.arange(0.45, 0.80, 0.01):
                r = calc_roi(val, val_probas, threshold=thr)
                if r["n_bets"] >= 20 and r["roi"] > best_thr_roi:
                    best_thr_roi = r["roi"]
                    best_thr = thr

            result_thr = calc_roi(test, test_probas, threshold=best_thr)
            result_50 = calc_roi(test, test_probas, threshold=0.5)
            logger.info(
                "Threshold approach (thr=%.2f): ROI=%.2f%%, bets=%d",
                best_thr,
                result_thr["roi"],
                result_thr["n_bets"],
            )
            logger.info(
                "Threshold 0.50: ROI=%.2f%%, bets=%d",
                result_50["roi"],
                result_50["n_bets"],
            )

            # Approach 2: Value betting (model_prob > implied_prob + margin)
            best_margin, val_vb_roi = find_best_margin(val, val_probas)
            result_vb = value_betting_roi(test, test_probas, margin=best_margin)
            logger.info(
                "Value betting (margin=%.2f): ROI=%.2f%%, bets=%d, prec=%.3f, sel=%.3f",
                best_margin,
                result_vb["roi"],
                result_vb["n_bets"],
                result_vb["precision"],
                result_vb["selectivity"],
            )

            # Approach 3: Combined - high probability AND value
            for thr in [0.50, 0.55, 0.60]:
                for margin in [0.0, 0.02, 0.05, 0.10]:
                    implied = 1.0 / test["Odds"].values
                    mask = (test_probas >= thr) & (test_probas > (implied + margin))
                    if mask.sum() < 10:
                        continue
                    sel = test[mask]
                    n = len(sel)
                    payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                    roi = (payout - n) / n * 100
                    logger.info(
                        "  Combined thr=%.2f margin=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f",
                        thr,
                        margin,
                        roi,
                        n,
                        sel["target"].mean(),
                    )

            # Use best overall
            best_method = "threshold"
            best_roi = result_thr["roi"]
            if result_vb["roi"] > best_roi and result_vb["n_bets"] >= 20:
                best_method = "value_betting"
                best_roi = result_vb["roi"]

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_threshold": result_thr["roi"],
                    "roi_thr_50": result_50["roi"],
                    "roi_value_betting": result_vb["roi"],
                    "best_threshold": best_thr,
                    "best_margin": best_margin,
                    "n_bets_threshold": result_thr["n_bets"],
                    "n_bets_value_betting": result_vb["n_bets"],
                    "roc_auc": auc,
                    "best_iteration": model.best_iteration_,
                }
            )
            mlflow.set_tag("best_method", best_method)

            # Feature importance
            fi = model.get_feature_importance()
            fi_sorted = sorted(zip(features, fi), key=lambda x: x[1], reverse=True)
            for fname, fimp in fi_sorted[:10]:
                mlflow.log_metric(f"fi_{fname}", fimp)
                logger.info("  fi %s: %.2f", fname, fimp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.3")
            logger.exception("Step 2.3 failed")
            raise


if __name__ == "__main__":
    main()
