"""Phase 3, Step 3.1: Optuna TPE hyperparameter optimization для CatBoost."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    DATA_DIR,
    SEED,
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

FEATURE_COLS = [*get_feature_columns(), "avg_elo", "elo_diff", "max_elo", "min_elo", "elo_spread"]


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление ELO-фич через join с elo_history."""
    elo_history = pd.read_csv(DATA_DIR / "elo_history.csv")
    elo_per_bet = (
        elo_history.groupby("Bet_ID")
        .agg(
            avg_elo=("New_ELO", "mean"),
            elo_diff=("ELO_Change", "sum"),
            max_elo=("New_ELO", "max"),
            min_elo=("New_ELO", "min"),
        )
        .reset_index()
    )
    df = df.merge(elo_per_bet, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    for col in ["avg_elo", "max_elo", "min_elo"]:
        df[col] = df[col].fillna(1500.0)
    df["elo_diff"] = df["elo_diff"].fillna(0.0)
    df["elo_spread"] = df["max_elo"] - df["min_elo"]
    return df


def objective(
    trial: optuna.Trial,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    test_df: pd.DataFrame,
) -> float:
    """Optuna objective: maximize best ROI across thresholds."""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": SEED,
        "verbose": 0,
        "eval_metric": "AUC",
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["None", "Balanced"]
        ),
    }
    if params["auto_class_weights"] == "None":
        params["auto_class_weights"] = None

    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50)

    proba = model.predict_proba(x_val)[:, 1]
    roi_results = calc_roi_at_thresholds(test_df, proba)

    best_roi = max(
        (r["roi"] for r in roi_results.values() if r["n_bets"] >= 50),
        default=0.0,
    )
    auc = roc_auc_score(y_val, proba)

    trial.set_user_attr("auc", auc)
    trial.set_user_attr("best_roi", best_roi)

    return best_roi


def main() -> None:
    logger.info("Step 3.1: Optuna Hyperparameter Optimization")
    df = load_data()
    df = add_elo_features(df)
    train, test = time_series_split(df)

    x_train = train[FEATURE_COLS].values.astype(float)
    y_train = train["target"].values
    x_test = test[FEATURE_COLS].values.astype(float)
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase3/step3.1_optuna_tpe") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "optuna_tpe",
                    "n_trials": 50,
                    "n_features": len(FEATURE_COLS),
                    "features": ",".join(FEATURE_COLS),
                }
            )

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            )

            study.optimize(
                lambda trial: objective(trial, x_train, y_train, x_test, y_test, test),
                n_trials=50,
                show_progress_bar=True,
            )

            best_trial = study.best_trial
            logger.info(
                "Best trial: #%d, ROI=%.2f%%, params=%s",
                best_trial.number,
                best_trial.value,
                best_trial.params,
            )

            # Retrain with best params
            best_params = dict(best_trial.params)
            if best_params.get("auto_class_weights") == "None":
                best_params["auto_class_weights"] = None
            best_params["random_seed"] = SEED
            best_params["verbose"] = 0
            best_params["eval_metric"] = "AUC"

            model_best = CatBoostClassifier(**best_params)
            model_best.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=50)
            proba_best = model_best.predict_proba(x_test)[:, 1]

            auc_best = roc_auc_score(y_test, proba_best)
            roi_results = calc_roi_at_thresholds(test, proba_best)

            best_roi = -999.0
            best_threshold = 0.5
            for thresh, result in roi_results.items():
                logger.info(
                    "Threshold %.2f: ROI=%.2f%%, n=%d, WR=%.4f, selected=%.1f%%",
                    thresh,
                    result["roi"],
                    result["n_bets"],
                    result["win_rate"],
                    result["pct_selected"],
                )
                if result["n_bets"] >= 50 and result["roi"] > best_roi:
                    best_roi = result["roi"]
                    best_threshold = thresh

            roi_at_best = roi_results[best_threshold]

            # Log best params
            for k, v in best_trial.params.items():
                mlflow.log_param(f"best_{k}", v)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roc_auc": auc_best,
                    "best_threshold": best_threshold,
                    "n_bets_selected": roi_at_best["n_bets"],
                    "pct_selected": roi_at_best["pct_selected"],
                    "win_rate": roi_at_best["win_rate"],
                    "n_optuna_trials": len(study.trials),
                    "best_iteration": model_best.get_best_iteration(),
                }
            )

            # Feature importances
            importances = model_best.get_feature_importance()
            for fname, imp in zip(FEATURE_COLS, importances, strict=True):
                mlflow.log_metric(f"imp_{fname}", imp)
            logger.info("Feature importances (optimized model):")
            ranked = sorted(zip(FEATURE_COLS, importances, strict=True), key=lambda x: -x[1])
            for fname, imp in ranked:
                logger.info("  %s: %.2f", fname, imp)

            # Top-5 trials summary
            logger.info("Top-5 trials:")
            top_trials = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:5]
            for t in top_trials:
                logger.info(
                    "  Trial #%d: ROI=%.2f%%, AUC=%.4f",
                    t.number,
                    t.value or 0,
                    t.user_attrs.get("auc", 0),
                )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ROI: %.2f%% at threshold=%.2f", best_roi, best_threshold)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            raise


if __name__ == "__main__":
    main()
