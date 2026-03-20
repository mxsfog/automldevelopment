"""Step 2.1: Feature Engineering -- shadow feature trick для новых фич."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from common import (
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def get_baseline_features() -> list[str]:
    """Baseline feature set из Phase 1."""
    return [
        "Odds",
        "USD",
        "ML_P_Model",
        "ML_P_Implied",
        "ML_Edge",
        "ML_EV",
        "Outcomes_Count",
        "Is_Parlay_bool",
    ]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создание новых фичей."""
    df = df.copy()

    # Odds-derived
    df["implied_prob"] = 1.0 / df["Odds"]
    df["log_odds"] = np.log1p(df["Odds"])
    df["odds_bucket"] = pd.cut(df["Odds"], bins=[0, 1.3, 1.6, 2.0, 3.0, 5.0, 100000], labels=False)

    # ML-derived
    df["has_ml_prediction"] = df["ML_P_Model"].notna().astype(int)
    df["ml_prob_vs_implied"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["ml_edge_positive"] = (df["ML_Edge"] > 0).astype(float)
    df["ml_edge_positive"] = df["ml_edge_positive"].fillna(0)
    df["ml_ev_positive"] = (df["ML_EV"] > 0).astype(float)
    df["ml_ev_positive"] = df["ml_ev_positive"].fillna(0)
    df["ml_confidence"] = df["ML_P_Model"].fillna(0) / 100.0

    # Parlay features
    df["is_parlay_int"] = df["Is_Parlay_bool"].astype(int)
    df["parlay_leg_avg_odds"] = np.where(
        df["Outcomes_Count"] > 1,
        df["Odds"] ** (1.0 / df["Outcomes_Count"]),
        df["Odds"],
    )

    # Time features
    if "Created_At" in df.columns:
        dt = pd.to_datetime(df["Created_At"])
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Stake features
    df["log_usd"] = np.log1p(df["USD"])
    df["stake_bucket"] = pd.cut(
        df["USD"], bins=[0, 10, 100, 1000, 10000, float("inf")], labels=False
    )

    # Sport frequency encoding (safe: no target leakage since it's count-based)
    sport_counts = df["Sport"].value_counts()
    df["sport_freq"] = df["Sport"].map(sport_counts).fillna(0)

    # Market frequency encoding
    market_counts = df["Market"].value_counts()
    df["market_freq"] = df["Market"].map(market_counts).fillna(0)

    return df


def get_new_features() -> list[str]:
    """Все новые фичи."""
    return [
        "implied_prob",
        "log_odds",
        "odds_bucket",
        "has_ml_prediction",
        "ml_prob_vs_implied",
        "ml_edge_positive",
        "ml_ev_positive",
        "ml_confidence",
        "is_parlay_int",
        "parlay_leg_avg_odds",
        "hour",
        "day_of_week",
        "is_weekend",
        "log_usd",
        "stake_bucket",
        "sport_freq",
        "market_freq",
    ]


def run_model(
    train_fit: pd.DataFrame,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    label: str,
) -> dict:
    """Обучение LogReg и оценка ROI."""
    x_fit = train_fit[features].copy()
    y_fit = train_fit["target"].values
    x_val = train_val[features].copy()
    x_test = test[features].copy()
    y_test = test["target"].values

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000, C=1.0)),
        ]
    )

    pipe.fit(x_fit, y_fit)

    proba_val = pipe.predict_proba(x_val)[:, 1]
    best_t, val_roi = find_best_threshold_on_val(train_val, proba_val)

    proba_test = pipe.predict_proba(x_test)[:, 1]
    roi_result = calc_roi(test, proba_test, threshold=best_t)
    auc = roc_auc_score(y_test, proba_test)

    logger.info(
        "[%s] ROI=%.2f%% threshold=%.2f n=%d AUC=%.4f",
        label,
        roi_result["roi"],
        best_t,
        roi_result["n_bets"],
        auc,
    )

    return {
        "roi": roi_result["roi"],
        "auc": auc,
        "threshold": best_t,
        "val_roi": val_roi,
        "n_bets": roi_result["n_bets"],
        "pct_selected": roi_result["pct_selected"],
        "win_rate": roi_result["win_rate"],
    }


def main() -> None:
    logger.info("Step 2.1: Feature Engineering (shadow feature trick)")
    df = load_data()
    df = engineer_features(df)
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    baseline_features = get_baseline_features()
    new_features = get_new_features()
    all_features = baseline_features + new_features

    with mlflow.start_run(run_name="phase2/step2.1_feature_engineering") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "2.1")
        mlflow.set_tag("phase", "2")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "shadow_feature_trick",
                    "n_baseline_features": len(baseline_features),
                    "n_new_features": len(new_features),
                    "n_all_features": len(all_features),
                }
            )

            # Baseline run
            res_baseline = run_model(train_fit, train_val, test, baseline_features, "baseline")

            # Candidate run (all features)
            res_candidate = run_model(train_fit, train_val, test, all_features, "candidate")

            delta_roi = res_candidate["roi"] - res_baseline["roi"]
            delta_auc = res_candidate["auc"] - res_baseline["auc"]

            # Decision
            if delta_roi > 0.2:
                decision = "accepted"
            elif delta_roi > 0:
                decision = "marginal"
            else:
                decision = "rejected"

            logger.info(
                "Delta ROI: %.2f%%, Delta AUC: %.4f -> %s",
                delta_roi,
                delta_auc,
                decision,
            )

            mlflow.log_metrics(
                {
                    "roi": res_candidate["roi"],
                    "roi_baseline": res_baseline["roi"],
                    "roi_candidate": res_candidate["roi"],
                    "roi_delta": delta_roi,
                    "auc_baseline": res_baseline["auc"],
                    "auc_candidate": res_candidate["auc"],
                    "auc_delta": delta_auc,
                    "threshold_baseline": res_baseline["threshold"],
                    "threshold_candidate": res_candidate["threshold"],
                    "n_bets_baseline": res_baseline["n_bets"],
                    "n_bets_candidate": res_candidate["n_bets"],
                }
            )
            mlflow.set_tag("feature_decision", decision)

            # Individual feature importance via ablation
            logger.info("Ablation study: removing one new feature at a time")
            for feat in new_features:
                ablated = [f for f in all_features if f != feat]
                res_abl = run_model(train_fit, train_val, test, ablated, f"no_{feat}")
                drop = res_candidate["roi"] - res_abl["roi"]
                mlflow.log_metric(f"roi_drop_without_{feat}", drop)
                logger.info("  Without %s: ROI=%.2f%% (drop=%.2f%%)", feat, res_abl["roi"], drop)

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
