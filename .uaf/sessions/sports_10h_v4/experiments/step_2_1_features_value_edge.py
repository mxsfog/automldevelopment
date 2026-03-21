"""Step 2.1: Feature Engineering - Value/Edge фичи + ELO + Teams data (shadow feature trick)."""

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

BASELINE_FEATURES = [
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
]

CAT_COLS = ["Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found"]


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Фичи на основе расхождения модели и рынка."""
    df = df.copy()

    # implied probability from odds
    df["implied_prob"] = 1.0 / df["Odds"]

    # ML model vs market
    df["ml_vs_market"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]

    # Edge нормализованный
    df["edge_normalized"] = df["ML_Edge"] / (df["ML_P_Implied"] + 1e-6)

    # Value indicator: модель видит значительно больше рыночных
    df["is_value_bet"] = (df["ML_P_Model"] / 100.0 > df["implied_prob"] * 1.05).astype(int)

    # Expected return (EV as ratio)
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
    )

    # Kelly criterion proxy: f = (bp - q) / b where b = odds-1, p = ML_P_Model/100
    p = df["ML_P_Model"].fillna(50) / 100.0
    b = df["Odds"] - 1
    q = 1 - p
    df["kelly_fraction"] = (b * p - q) / (b + 1e-6)
    df["kelly_fraction"] = df["kelly_fraction"].clip(-1, 1)

    # Confidence: как далеко ML_P_Model от 50%
    df["ml_confidence"] = (df["ML_P_Model"].fillna(50) - 50).abs()

    # Log odds (for linearity)
    df["log_odds"] = np.log(df["Odds"].clip(1.001))

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Временные фичи."""
    df = df.copy()
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_sport_market_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Взаимодействие Sport x Market."""
    df = df.copy()
    df["sport_market"] = df["Sport"].fillna("_") + "__" + df["Market"].fillna("_")
    return df


def prepare_catboost(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Подготовка данных для CatBoost."""
    cat_indices = []
    for i, f in enumerate(features):
        if f in cat_cols:
            cat_indices.append(i)

    X_train = train[features].copy()
    X_test = test[features].copy()

    for idx in cat_indices:
        col = features[idx]
        X_train[col] = X_train[col].astype(str).replace("nan", "_missing_")
        X_test[col] = X_test[col].astype(str).replace("nan", "_missing_")

    return X_train, X_test, cat_indices


def run_catboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
    run_suffix: str,
) -> dict:
    """Обучение и оценка CatBoost."""
    X_train, X_test, cat_indices = prepare_catboost(train, test, features, cat_cols)
    X_val = val[features].copy()
    for idx in cat_indices:
        col = features[idx]
        X_val[col] = X_val[col].astype(str).replace("nan", "_missing_")

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        cat_features=cat_indices,
        early_stopping_rounds=50,
        task_type="CPU",
    )
    model.fit(
        X_train,
        train["target"].values,
        eval_set=(X_val, val["target"].values),
        use_best_model=True,
    )

    val_probas = model.predict_proba(X_val)[:, 1]
    best_thr = find_best_threshold(val, val_probas)

    test_probas = model.predict_proba(X_test)[:, 1]
    result = calc_roi(test, test_probas, threshold=best_thr)
    result_50 = calc_roi(test, test_probas, threshold=0.5)
    auc = roc_auc_score(test["target"].values, test_probas)

    logger.info(
        "%s - thr=%.2f: ROI=%.2f%%, bets=%d | thr=0.50: ROI=%.2f%%, bets=%d | AUC=%.4f | iters=%d",
        run_suffix,
        best_thr,
        result["roi"],
        result["n_bets"],
        result_50["roi"],
        result_50["n_bets"],
        auc,
        model.best_iteration_,
    )

    fi = model.get_feature_importance()
    fi_dict = dict(zip(features, fi))

    return {
        "roi": result["roi"],
        "roi_50": result_50["roi"],
        "auc": auc,
        "best_thr": best_thr,
        "n_bets": result["n_bets"],
        "precision": result["precision"],
        "selectivity": result["selectivity"],
        "n_won": result["n_won"],
        "n_lost": result["n_lost"],
        "best_iteration": model.best_iteration_,
        "fi": fi_dict,
        "model": model,
    }


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase2/step_2_1_value_edge_features") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "2.1")
            mlflow.set_tag("phase", "2")

            df = load_data()

            # Добавляем новые фичи
            df = add_value_features(df)
            df = add_time_features(df)
            df = add_sport_market_interaction(df)

            train, test = time_series_split(df)

            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            # Baseline features (те же что в step 1.4)
            baseline_feats = [f for f in BASELINE_FEATURES if f in df.columns]
            baseline_cats = [c for c in CAT_COLS if c in df.columns]

            # Candidate features = baseline + new
            new_features = [
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
            ]
            candidate_feats = baseline_feats + [f for f in new_features if f in df.columns]
            candidate_cats = baseline_cats + ["odds_bucket", "sport_market"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "shadow_feature_trick",
                    "n_baseline_features": len(baseline_feats),
                    "n_candidate_features": len(candidate_feats),
                    "new_features": str(new_features),
                    "gap_days": 7,
                }
            )

            # Run baseline
            logger.info("Running baseline with %d features", len(baseline_feats))
            res_baseline = run_catboost(
                train_inner, val, test, baseline_feats, baseline_cats, "baseline"
            )

            # Run candidate
            logger.info("Running candidate with %d features", len(candidate_feats))
            res_candidate = run_catboost(
                train_inner, val, test, candidate_feats, candidate_cats, "candidate"
            )

            # Delta
            delta_roi = res_candidate["roi"] - res_baseline["roi"]
            delta_roi_50 = res_candidate["roi_50"] - res_baseline["roi_50"]
            delta_auc = res_candidate["auc"] - res_baseline["auc"]

            logger.info(
                "Delta ROI (opt thr): %.2f%%, Delta ROI (0.5): %.2f%%, Delta AUC: %.4f",
                delta_roi,
                delta_roi_50,
                delta_auc,
            )

            # Decision
            if delta_roi_50 > 0.002:
                decision = "accepted"
            elif delta_roi_50 <= 0:
                decision = "rejected"
            else:
                decision = "marginal"

            logger.info("Decision: %s", decision)

            mlflow.log_metrics(
                {
                    "roi_baseline": res_baseline["roi"],
                    "roi_candidate": res_candidate["roi"],
                    "roi_50_baseline": res_baseline["roi_50"],
                    "roi_50_candidate": res_candidate["roi_50"],
                    "auc_baseline": res_baseline["auc"],
                    "auc_candidate": res_candidate["auc"],
                    "delta_roi": delta_roi,
                    "delta_roi_50": delta_roi_50,
                    "delta_auc": delta_auc,
                    "roi": res_candidate["roi_50"],
                    "roc_auc": res_candidate["auc"],
                    "n_bets": res_candidate["n_bets"],
                    "best_threshold": res_candidate["best_thr"],
                }
            )
            mlflow.set_tag("decision", decision)

            # Log new feature importances
            for fname, fimp in sorted(
                res_candidate["fi"].items(), key=lambda x: x[1], reverse=True
            )[:20]:
                mlflow.log_metric(f"fi_{fname}", fimp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.1")
            logger.exception("Step 2.1 failed")
            raise


if __name__ == "__main__":
    main()
