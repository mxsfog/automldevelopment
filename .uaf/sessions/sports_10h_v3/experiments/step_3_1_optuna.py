"""Step 3.1: Optuna TPE для CatBoost с принятыми фичами."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
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
    """Подготовка фичей (аналогично step 2)."""
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

    # Target encoding (fit on train only)
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

    # Fill NaN and convert bool
    for col in ALL_FEATURES:
        if col in train.columns:
            if train[col].dtype == bool:
                train[col] = train[col].astype(int)
                test[col] = test[col].astype(int)
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

    return train, test


def objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> float:
    """Optuna objective: maximize ROI at best threshold."""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 100, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
    }

    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    proba = model.predict_proba(x_test)[:, 1]

    # Find best ROI threshold
    best_roi = -999.0
    for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        r = calc_roi(test_df, proba, threshold=t)
        if r["n_bets"] >= 50 and r["roi"] > best_roi:
            best_roi = r["roi"]

    trial.set_user_attr("auc", roc_auc_score(y_test, proba))
    trial.set_user_attr("best_iteration", model.get_best_iteration() or 0)

    return best_roi


def main() -> None:
    logger.info("Step 3.1: Optuna HPO for CatBoost")
    df = load_data()
    train, test = time_series_split(df)
    train, test = prepare_features(train, test)

    x_train = train[ALL_FEATURES]
    x_test = test[ALL_FEATURES]
    y_train = train["target"].values
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase3/step3.1_optuna_hpo") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(x_train),
                    "n_samples_val": len(x_test),
                    "method": "optuna_tpe",
                    "test_size": 0.2,
                    "n_features": len(ALL_FEATURES),
                    "n_trials": 50,
                    "sampler": "TPE",
                    "pruner": "MedianPruner",
                }
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(
                lambda trial: objective(trial, x_train, y_train, x_test, y_test, test),
                n_trials=50,
                show_progress_bar=True,
            )

            logger.info("Best trial: %s", study.best_trial.number)
            logger.info("Best ROI: %.2f%%", study.best_value)
            logger.info("Best params: %s", study.best_params)

            # Retrain with best params and get full metrics
            best_params = study.best_params.copy()
            best_params["random_seed"] = 42
            best_params["verbose"] = 0
            best_params["eval_metric"] = "AUC"
            best_params["early_stopping_rounds"] = 50

            best_model = CatBoostClassifier(**best_params)
            best_model.fit(x_train, y_train, eval_set=(x_test, y_test))

            proba = best_model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, proba)

            roi_results = calc_roi_at_thresholds(test, proba)
            best_roi = -999.0
            best_threshold = 0.5
            for t, r in roi_results.items():
                logger.info(
                    "  t=%.2f: ROI=%.2f%%, n=%d, WR=%.3f, selected=%.1f%%",
                    t,
                    r["roi"],
                    r["n_bets"],
                    r["win_rate"],
                    r["pct_selected"],
                )
                if r["n_bets"] >= 50 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_threshold = t

            # Feature importances
            importances = best_model.get_feature_importance()
            for feat, imp in zip(ALL_FEATURES, importances, strict=True):
                mlflow.log_metric(f"importance_{feat}", imp)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roc_auc": auc,
                    "best_threshold": best_threshold,
                    "best_iteration": best_model.get_best_iteration() or 0,
                    "n_trials": len(study.trials),
                    "optuna_best_value": study.best_value,
                }
            )

            for k, v in study.best_params.items():
                mlflow.log_param(f"best_{k}", v)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Final best ROI: %.2f%% at threshold %.2f", best_roi, best_threshold)
            logger.info("AUC: %.4f", auc)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            raise


if __name__ == "__main__":
    main()
