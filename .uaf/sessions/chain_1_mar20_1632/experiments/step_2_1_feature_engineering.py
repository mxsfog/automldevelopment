"""Step 2.1: Feature Engineering -- Shadow Feature Trick.

Новые группы фичей:
1. Odds-derived: log_odds, implied_prob, odds_bin, value_ratio
2. Temporal: hour, day_of_week, is_weekend
3. Sport encoding: sport winrate, is_esports
4. Market encoding: market winrate
5. Bet sizing: log_usd, usd_per_outcome
"""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
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

ESPORTS = {"CS2", "Dota 2", "League of Legends", "Valorant", "FIFA", "NBA 2K"}


def add_features(
    df: pd.DataFrame, sport_wr: dict | None = None, market_wr: dict | None = None
) -> tuple[pd.DataFrame, dict, dict]:
    """Добавить новые фичи. sport_wr/market_wr передаются для test."""
    df = df.copy()

    # Odds-derived
    df["log_odds"] = np.log1p(df["Odds"])
    df["implied_prob"] = 1.0 / df["Odds"]
    df["value_ratio"] = df["ML_P_Model"] / 100.0 / df["implied_prob"]
    df["value_ratio"] = df["value_ratio"].clip(0, 10).fillna(1.0)
    df["odds_bucket"] = (
        pd.cut(df["Odds"], bins=[0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 5.0, 100, 10000], labels=False)
        .fillna(0)
        .astype(float)
    )
    df["edge_x_ev"] = df["ML_Edge"] * df["ML_EV"]

    # Temporal
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)

    # Sport-based
    df["is_esports"] = df["Sport"].isin(ESPORTS).astype(float)

    # Sport winrate (target encoding on train only)
    if sport_wr is None:
        sport_wr = df.groupby("Sport")["target"].mean().to_dict()
    df["sport_winrate"] = df["Sport"].map(sport_wr).fillna(0.5)

    # Market winrate (target encoding on train only)
    if market_wr is None:
        market_wr = df.groupby("Market")["target"].mean().to_dict()
    df["market_winrate"] = df["Market"].map(market_wr).fillna(0.5)

    # Bet sizing
    df["log_usd"] = np.log1p(df["USD"])
    df["usd_per_outcome"] = df["USD"] / df["Outcomes_Count"].clip(lower=1)
    df["log_usd_per_outcome"] = np.log1p(df["usd_per_outcome"])

    return df, sport_wr, market_wr


def get_extended_feature_columns() -> list[str]:
    """Расширенный набор фичей."""
    base = get_feature_columns()
    new = [
        "log_odds",
        "implied_prob",
        "value_ratio",
        "odds_bucket",
        "edge_x_ev",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_esports",
        "sport_winrate",
        "market_winrate",
        "log_usd",
        "log_usd_per_outcome",
    ]
    return base + new


def run_catboost(
    train_fit: pd.DataFrame,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    run_suffix: str,
) -> dict:
    """Обучение CatBoost и возврат метрик."""
    x_fit = train_fit[feature_cols].values.astype(float)
    y_fit = train_fit["target"].values
    x_val = train_val[feature_cols].values.astype(float)
    y_val = train_val["target"].values
    x_test = test[feature_cols].values.astype(float)
    y_test = test["target"].values

    x_fit = np.nan_to_num(x_fit, nan=0.0)
    x_val = np.nan_to_num(x_val, nan=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0)

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        random_seed=42,
        verbose=100,
        eval_metric="AUC",
        auto_class_weights="Balanced",
    )
    model.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=100)

    proba_val = model.predict_proba(x_val)[:, 1]
    best_threshold, val_roi = find_best_threshold_on_val(train_val, proba_val)

    proba_test = model.predict_proba(x_test)[:, 1]
    roi_result = calc_roi(test, proba_test, threshold=best_threshold)
    auc = roc_auc_score(y_test, proba_test)

    logger.info(
        "[%s] ROI=%.2f%%, AUC=%.4f, threshold=%.2f, n=%d",
        run_suffix,
        roi_result["roi"],
        auc,
        best_threshold,
        roi_result["n_bets"],
    )

    importances = model.get_feature_importance()
    ranked = sorted(zip(feature_cols, importances, strict=True), key=lambda x: -x[1])
    for fname, imp in ranked[:10]:
        logger.info("  [%s] %s: %.2f", run_suffix, fname, imp)

    return {
        "roi": roi_result["roi"],
        "auc": auc,
        "threshold": best_threshold,
        "val_roi": val_roi,
        "n_bets": roi_result["n_bets"],
        "win_rate": roi_result["win_rate"],
        "pct_selected": roi_result["pct_selected"],
        "best_iteration": model.get_best_iteration(),
        "importances": dict(zip(feature_cols, importances, strict=True)),
    }


def main() -> None:
    logger.info("Step 2.1: Feature Engineering (Shadow Feature Trick)")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    with mlflow.start_run(run_name="phase2/step2.1_feature_engineering") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "2.1")
        mlflow.set_tag("phase", "2")

        try:
            # Baseline: original features
            baseline_cols = get_feature_columns()
            logger.info("Running baseline with %d features", len(baseline_cols))
            baseline_result = run_catboost(train_fit, train_val, test, baseline_cols, "baseline")

            # Candidate: extended features
            # Target encoding fit only on train_fit (anti-leakage)
            train_fit_fe, sport_wr, market_wr = add_features(train_fit)
            train_val_fe, _, _ = add_features(train_val, sport_wr, market_wr)
            test_fe, _, _ = add_features(test, sport_wr, market_wr)

            candidate_cols = get_extended_feature_columns()
            logger.info("Running candidate with %d features", len(candidate_cols))
            candidate_result = run_catboost(
                train_fit_fe, train_val_fe, test_fe, candidate_cols, "candidate"
            )

            delta_roi = candidate_result["roi"] - baseline_result["roi"]
            delta_auc = candidate_result["auc"] - baseline_result["auc"]

            if delta_roi > 0.002:
                decision = "accepted"
            elif delta_roi <= 0:
                decision = "rejected"
            else:
                decision = "marginal"

            logger.info(
                "Delta ROI=%.4f%%, Delta AUC=%.4f, Decision=%s",
                delta_roi,
                delta_auc,
                decision,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "shadow_feature_trick",
                    "baseline_n_features": len(baseline_cols),
                    "candidate_n_features": len(candidate_cols),
                    "new_features": ",".join(
                        [f for f in candidate_cols if f not in baseline_cols]
                    ),
                }
            )

            mlflow.log_metrics(
                {
                    "roi": candidate_result["roi"],
                    "roi_baseline": baseline_result["roi"],
                    "roi_candidate": candidate_result["roi"],
                    "delta_roi": delta_roi,
                    "auc_baseline": baseline_result["auc"],
                    "auc_candidate": candidate_result["auc"],
                    "delta_auc": delta_auc,
                    "threshold_baseline": baseline_result["threshold"],
                    "threshold_candidate": candidate_result["threshold"],
                    "n_bets_baseline": baseline_result["n_bets"],
                    "n_bets_candidate": candidate_result["n_bets"],
                }
            )

            for fname, imp in candidate_result["importances"].items():
                mlflow.log_metric(f"importance_{fname}", imp)

            mlflow.set_tag("fe_decision", decision)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.1")
            raise


if __name__ == "__main__":
    main()
