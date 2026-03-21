"""Step 3.1: Optuna HPO для CatBoost + value betting стратегия."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import optuna
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

CAT_COLS = {"Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный набор фичей."""
    df = df.copy()
    df["implied_prob"] = 1.0 / df["Odds"]
    df["ml_vs_market"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["edge_normalized"] = df["ML_Edge"] / (df["ML_P_Implied"].clip(lower=0.1) + 1e-6)
    df["is_value_bet"] = (df["ML_P_Model"] / 100.0 > df["implied_prob"] * 1.05).astype(int)
    df["ev_ratio"] = df["ML_EV"] / 100.0
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
    p = df["ML_P_Model"].fillna(50) / 100.0
    b = df["Odds"] - 1
    q = 1 - p
    df["kelly_fraction"] = ((b * p - q) / (b + 1e-6)).clip(-1, 1)
    df["ml_confidence"] = (df["ML_P_Model"].fillna(50) - 50).abs()
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["log_usd"] = np.log1p(df["USD"])
    df["parlay_flag"] = df["Is_Parlay"].map({"t": 1, "f": 0}).fillna(0).astype(int)
    df["parlay_odds"] = df["parlay_flag"] * df["Odds"]
    df["has_ml_prediction"] = (~df["ML_P_Model"].isna()).astype(int)
    df["is_single"] = (df["Outcomes_Count"] == 1).astype(int)
    return df


def prepare_data(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS]
    x = df[features].copy()
    for idx in cat_indices:
        col = features[idx]
        x[col] = x[col].astype(str).replace("nan", "_missing_")
    return x, cat_indices


def combined_roi(df: pd.DataFrame, probas: np.ndarray, threshold: float, margin: float) -> dict:
    """ROI с комбинированным подходом: prob >= threshold AND prob > implied + margin."""
    implied = 1.0 / df["Odds"].values
    mask = (probas >= threshold) & (probas > (implied + margin))
    selected = df[mask]
    if len(selected) < 10:
        return {"roi": -100.0, "n_bets": len(selected)}
    n = len(selected)
    payout = selected.loc[selected["target"] == 1, "Odds"].sum()
    roi = (payout - n) / n * 100
    return {
        "roi": roi,
        "n_bets": n,
        "precision": selected["target"].mean(),
        "selectivity": n / len(df),
    }


def objective(
    trial: optuna.Trial,
    train_inner: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    cat_indices: list[int],
) -> float:
    """Optuna objective: maximize val ROI."""
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 3),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "cat_features": cat_indices,
        "early_stopping_rounds": 100,
        "task_type": "CPU",
    }

    x_train, _ = prepare_data(train_inner, features)
    x_val, _ = prepare_data(val, features)

    model = CatBoostClassifier(**params)
    model.fit(
        x_train,
        train_inner["target"].values,
        eval_set=(x_val, val["target"].values),
        use_best_model=True,
    )

    val_probas = model.predict_proba(x_val)[:, 1]

    # Optimize threshold + margin on val
    best_val_roi = -999.0
    best_combo = (0.5, 0.0)
    for thr in np.arange(0.40, 0.70, 0.02):
        for margin in np.arange(-0.02, 0.10, 0.01):
            r = combined_roi(val, val_probas, thr, margin)
            if r["n_bets"] >= 15 and r["roi"] > best_val_roi:
                best_val_roi = r["roi"]
                best_combo = (thr, margin)

    trial.set_user_attr("best_threshold", best_combo[0])
    trial.set_user_attr("best_margin", best_combo[1])
    trial.set_user_attr("best_iteration", model.best_iteration_)

    return best_val_roi


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase3/step_3_1_optuna_hpo") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "3.1")
            mlflow.set_tag("phase", "3")

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
            _, cat_indices = prepare_data(train_inner, features)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "optuna_tpe",
                    "n_features": len(features),
                    "n_trials": 50,
                    "gap_days": 7,
                }
            )

            # Optuna study
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(
                lambda trial: objective(trial, train_inner, val, features, cat_indices),
                n_trials=50,
                show_progress_bar=False,
            )

            best_trial = study.best_trial
            logger.info("Best trial: %d, val ROI: %.2f%%", best_trial.number, best_trial.value)
            logger.info("Best params: %s", best_trial.params)

            # Retrain with best params on full train
            best_params = {
                "iterations": 3000,
                **best_trial.params,
                "random_seed": 42,
                "verbose": 0,
                "eval_metric": "AUC",
                "cat_features": cat_indices,
                "early_stopping_rounds": 100,
                "task_type": "CPU",
            }
            x_train_full, _ = prepare_data(train, features)
            x_test, _ = prepare_data(test, features)

            # Use val as eval_set even for full train
            x_val, _ = prepare_data(val, features)
            model = CatBoostClassifier(**best_params)
            model.fit(
                x_train_full,
                train["target"].values,
                eval_set=(x_val, val["target"].values),
                use_best_model=True,
            )

            test_probas = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(test["target"].values, test_probas)

            best_thr = best_trial.user_attrs["best_threshold"]
            best_margin = best_trial.user_attrs["best_margin"]

            result = combined_roi(test, test_probas, best_thr, best_margin)
            result_50 = calc_roi(test, test_probas, threshold=0.5)

            logger.info(
                "Test (thr=%.2f, margin=%.2f): ROI=%.2f%%, bets=%d, prec=%.3f",
                best_thr,
                best_margin,
                result["roi"],
                result["n_bets"],
                result.get("precision", 0),
            )
            logger.info(
                "Test (thr=0.50): ROI=%.2f%%, bets=%d", result_50["roi"], result_50["n_bets"]
            )
            logger.info("AUC: %.4f", auc)

            # Scan more combinations on test
            for thr in [0.45, 0.50, 0.55]:
                for margin in [0.0, 0.01, 0.02, 0.03, 0.05]:
                    r = combined_roi(test, test_probas, thr, margin)
                    if r["n_bets"] >= 10:
                        logger.info(
                            "  thr=%.2f margin=%.2f: ROI=%.2f%%, bets=%d",
                            thr,
                            margin,
                            r["roi"],
                            r["n_bets"],
                        )

            mlflow.log_metrics(
                {
                    "roi": result["roi"],
                    "roi_thr_50": result_50["roi"],
                    "best_threshold": best_thr,
                    "best_margin": best_margin,
                    "n_bets": result["n_bets"],
                    "roc_auc": auc,
                    "best_iteration": model.best_iteration_,
                    "n_trials": len(study.trials),
                    "best_trial_val_roi": best_trial.value,
                }
            )

            for k, v in best_trial.params.items():
                mlflow.log_param(f"best_{k}", v)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            logger.exception("Step 3.1 failed")
            raise


if __name__ == "__main__":
    main()
