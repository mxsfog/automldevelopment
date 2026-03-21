"""Step 2: Feature Engineering с shadow feature trick.

Гипотезы:
1. Sport/Market target encoding — разные виды спорта имеют разный ROI
2. Odds-derived: implied probability, log_odds, odds_bucket
3. ML-derived: P_Model - P_Implied, abs(Edge), Edge_sign
4. Time features: hour, day_of_week, is_weekend
5. Winrate/Rating diff (ML_Winrate_Diff, ML_Rating_Diff)
"""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    calc_roi_at_thresholds,
    check_budget,
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


def add_shadow_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Добавить новые фичи (shadow) к train и test."""
    shadow_features = []

    for df in [train, test]:
        # Odds-derived
        df["implied_prob"] = 1.0 / df["Odds"]
        df["log_odds"] = np.log1p(df["Odds"])
        df["odds_bucket"] = pd.cut(
            df["Odds"], bins=[0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6], labels=False
        ).fillna(5)

        # ML-derived
        df["p_model_minus_implied"] = df["ML_P_Model"].fillna(50) - df["ML_P_Implied"].fillna(50)
        df["abs_edge"] = df["ML_Edge"].fillna(0).abs()
        df["edge_positive"] = (df["ML_Edge"].fillna(0) > 0).astype(int)
        df["ml_p_model_filled"] = df["ML_P_Model"].fillna(-1)
        df["has_ml_prediction"] = (df["ML_P_Model"].notna()).astype(int)

        # Time features
        df["hour"] = df["Created_At"].dt.hour
        df["day_of_week"] = df["Created_At"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Winrate/Rating diff
        df["winrate_diff_filled"] = df["ML_Winrate_Diff"].fillna(0)
        df["rating_diff_filled"] = df["ML_Rating_Diff"].fillna(0)
        df["has_team_stats"] = (df["ML_Team_Stats_Found"] == "t").astype(int)

        # Parlay-related
        df["is_parlay_int"] = (df["Is_Parlay"] == "t").astype(int)
        df["outcomes_x_odds"] = df["Outcomes_Count"] * df["Odds"]

    shadow_features = [
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
    ]

    return train, test, shadow_features


def add_sport_target_encoding(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Target encoding для Sport (fit only on train)."""
    sport_mean = train.groupby("Sport")["target"].mean()
    global_mean = train["target"].mean()

    # Smoothing to avoid overfitting on rare sports
    sport_counts = train.groupby("Sport")["target"].count()
    smoothing = 50
    sport_enc = (sport_counts * sport_mean + smoothing * global_mean) / (sport_counts + smoothing)

    train["sport_target_enc"] = train["Sport"].map(sport_enc).fillna(global_mean)
    test["sport_target_enc"] = test["Sport"].map(sport_enc).fillna(global_mean)

    # Market target encoding
    market_mean = train.groupby("Market")["target"].mean()
    market_counts = train.groupby("Market")["target"].count()
    market_enc = (market_counts * market_mean + smoothing * global_mean) / (
        market_counts + smoothing
    )

    train["market_target_enc"] = train["Market"].map(market_enc).fillna(global_mean)
    test["market_target_enc"] = test["Market"].map(market_enc).fillna(global_mean)

    return train, test, ["sport_target_enc", "market_target_enc"]


def run_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    features: list[str],
    run_suffix: str,
) -> tuple[float, float, float, dict]:
    """Обучение CatBoost и расчет метрик."""
    params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.1,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
    }

    model = CatBoostClassifier(**params)
    model.fit(x_train[features], y_train, eval_set=(x_test[features], y_test))

    proba = model.predict_proba(x_test[features])[:, 1]
    auc = roc_auc_score(y_test, proba)

    importances = dict(zip(features, model.get_feature_importance(), strict=True))
    logger.info("%s: AUC=%.4f, best_iter=%s", run_suffix, auc, model.get_best_iteration())

    return auc, proba, model.get_best_iteration() or 0, importances


def main() -> None:
    logger.info("Step 2: Feature Engineering with Shadow Feature Trick")
    df = load_data()
    train, test = time_series_split(df)

    base_features = get_feature_columns()
    y_train = train["target"].values
    y_test = test["target"].values

    # Fill base features
    for col in base_features:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)
    for col in base_features:
        if train[col].dtype == bool:
            train[col] = train[col].astype(int)
            test[col] = test[col].astype(int)

    with mlflow.start_run(run_name="phase2/step2_feature_engineering") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "2")
        mlflow.set_tag("phase", "2")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "shadow_feature_trick",
                    "test_size": 0.2,
                    "base_features": ",".join(base_features),
                }
            )

            # Baseline run with original features
            logger.info("Running baseline model...")
            auc_base, proba_base, _iter_base, _imp_base = run_model(
                train, y_train, test, y_test, base_features, "baseline"
            )
            roi_base = calc_roi_at_thresholds(test, proba_base)

            best_roi_base = -999.0
            best_t_base = 0.5
            for t, r in roi_base.items():
                if r["n_bets"] >= 50 and r["roi"] > best_roi_base:
                    best_roi_base = r["roi"]
                    best_t_base = t

            logger.info(
                "Baseline: AUC=%.4f, best ROI=%.2f%% at t=%.2f",
                auc_base,
                best_roi_base,
                best_t_base,
            )

            # Add shadow features
            train, test, shadow_feats = add_shadow_features(train, test)
            train, test, te_feats = add_sport_target_encoding(train, test)
            all_shadow = shadow_feats + te_feats
            candidate_features = base_features + all_shadow

            # Fill NaN in shadow features
            for col in all_shadow:
                train[col] = train[col].fillna(-999)
                test[col] = test[col].fillna(-999)

            # Candidate run with all features
            logger.info("Running candidate model with %d features...", len(candidate_features))
            auc_cand, proba_cand, _iter_cand, imp_cand = run_model(
                train, y_train, test, y_test, candidate_features, "candidate"
            )
            roi_cand = calc_roi_at_thresholds(test, proba_cand)

            best_roi_cand = -999.0
            best_t_cand = 0.5
            for t, r in roi_cand.items():
                if r["n_bets"] >= 50 and r["roi"] > best_roi_cand:
                    best_roi_cand = r["roi"]
                    best_t_cand = t

            logger.info(
                "Candidate: AUC=%.4f, best ROI=%.2f%% at t=%.2f",
                auc_cand,
                best_roi_cand,
                best_t_cand,
            )

            # Delta analysis
            delta_auc = auc_cand - auc_base
            delta_roi = best_roi_cand - best_roi_base

            logger.info("Delta AUC: %.4f, Delta ROI: %.2f%%", delta_auc, delta_roi)

            # Feature importance of shadow features
            logger.info("Shadow feature importances:")
            accepted = []
            for feat in all_shadow:
                imp = imp_cand.get(feat, 0)
                logger.info("  %s: %.2f", feat, imp)
                if imp > 0.5:
                    accepted.append(feat)

            # Decision
            if delta_roi > 0.2:
                decision = "accepted"
                final_features = candidate_features
                primary_roi = best_roi_cand
                primary_threshold = best_t_cand
            elif delta_roi > 0:
                decision = "marginal_accepted"
                final_features = base_features + accepted
                primary_roi = best_roi_cand
                primary_threshold = best_t_cand
            else:
                decision = "rejected"
                final_features = base_features
                primary_roi = best_roi_base
                primary_threshold = best_t_base

            logger.info("Decision: %s", decision)
            logger.info("Final features (%d): %s", len(final_features), final_features)

            # Log ROI at all thresholds for the best model
            best_roi_results = roi_cand if "accepted" in decision else roi_base
            for t, r in best_roi_results.items():
                logger.info(
                    "  t=%.2f: ROI=%.2f%%, n=%d, WR=%.3f",
                    t,
                    r["roi"],
                    r["n_bets"],
                    r["win_rate"],
                )

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "roc_auc_baseline": auc_base,
                    "roc_auc_candidate": auc_cand,
                    "roi_baseline": best_roi_base,
                    "roi_candidate": best_roi_cand,
                    "delta_auc": delta_auc,
                    "delta_roi": delta_roi,
                    "best_threshold": primary_threshold,
                    "n_features_base": len(base_features),
                    "n_features_candidate": len(candidate_features),
                    "n_features_accepted": len(final_features),
                    "n_shadow_accepted": len(accepted),
                }
            )

            mlflow.log_param("decision", decision)
            mlflow.log_param("accepted_shadows", ",".join(accepted))
            mlflow.log_param("final_features", ",".join(final_features))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Primary ROI: %.2f%%", primary_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2")
            raise


if __name__ == "__main__":
    main()
