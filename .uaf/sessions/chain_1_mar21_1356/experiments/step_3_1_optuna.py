"""Phase 3 — Hyperparameter Optimization с Optuna.

Оптимизация CatBoost гиперпараметров через Optuna TPE.
Метрика: ROI на валидации (порог выбирается на val).
"""

import json
import logging
import os
import random
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
from data_loader import (
    calc_roi,
    find_best_threshold,
    get_base_features,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import (
    add_sport_market_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

ACCEPTED_FEATURES = [
    *get_base_features(),
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def prepare_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Подготовка данных с accepted features."""
    bets, outcomes, _, _ = load_raw_data()
    df = prepare_dataset(bets, outcomes)
    train, test = time_series_split(df, test_size=0.2)

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split].copy()
    val = train.iloc[val_split:].copy()

    # Sport/Market encoding (fit on train_fit only)
    train_fit, _ = add_sport_market_features(train_fit, train_fit)
    val, _ = add_sport_market_features(val, train_fit)
    test, _ = add_sport_market_features(test, train_fit)

    return train_fit, val, test, ACCEPTED_FEATURES


def objective(
    trial: optuna.Trial,
    train_fit: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
) -> float:
    """Optuna objective: maximize ROI on val."""
    params = {
        "iterations": trial.suggest_int("iterations", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "bootstrap_type": "Bernoulli",
    }

    x_train = train_fit[features].fillna(0)
    x_val = val[features].fillna(0)
    y_train = train_fit["target"]
    y_val = val["target"]

    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)

    probas_val = model.predict_proba(x_val)[:, 1]
    _thr, val_result = find_best_threshold(val, probas_val, min_bets=50)

    roi = val_result.get("roi", -100)
    auc = roc_auc_score(y_val, probas_val)

    trial.set_user_attr("auc", auc)
    trial.set_user_attr("n_bets", val_result.get("n_bets", 0))
    trial.set_user_attr("threshold", _thr)

    return roi


def main() -> None:
    with mlflow.start_run(run_name="phase3/step_3_1_optuna") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            train_fit, val, test, features = prepare_splits()

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "optuna_tpe",
                    "n_features": len(features),
                    "n_trials": 50,
                }
            )

            # Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(
                lambda trial: objective(trial, train_fit, val, features),
                n_trials=50,
                show_progress_bar=True,
            )

            best_trial = study.best_trial
            logger.info("Best trial: %s", best_trial.params)
            logger.info("Best val ROI: %.2f%%", best_trial.value)
            logger.info("Best AUC: %.4f", best_trial.user_attrs["auc"])

            # Retrain best model
            best_params = {
                **best_trial.params,
                "random_seed": 42,
                "verbose": 0,
                "eval_metric": "AUC",
                "early_stopping_rounds": 50,
                "bootstrap_type": "Bernoulli",
            }

            x_train = train_fit[features].fillna(0)
            x_val = val[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train_fit["target"]
            y_val = val["target"]
            y_test = test["target"]

            model = CatBoostClassifier(**best_params)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)

            probas_val = model.predict_proba(x_val)[:, 1]
            probas_test = model.predict_proba(x_test)[:, 1]

            auc_test = roc_auc_score(y_test, probas_test)
            best_thr, _val_result = find_best_threshold(val, probas_val, min_bets=50)
            test_result = calc_roi(test, probas_test, threshold=best_thr)

            logger.info("Test AUC: %.4f", auc_test)
            logger.info("Test ROI (thr=%.2f): %s", best_thr, test_result)

            for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, probas_test, threshold=thr)
                logger.info("Test thr=%.2f: ROI=%.2f%%, n=%d", thr, r["roi"], r["n_bets"])

            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test": test_result["roi"],
                    "n_bets_test": test_result["n_bets"],
                    "threshold": best_thr,
                    "roi_val_best": best_trial.value,
                    "n_optuna_trials": len(study.trials),
                    "win_rate_test": test_result.get("win_rate", 0),
                    "avg_odds_test": test_result.get("avg_odds", 0),
                }
            )

            # Save model if ROI > 0
            if test_result["roi"] > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": test_result["roi"],
                    "auc": auc_test,
                    "threshold": best_thr,
                    "n_bets": test_result["n_bets"],
                    "feature_names": features,
                    "params": best_trial.params,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                mlflow.log_artifact(str(model_dir / "model.cbm"))
                mlflow.log_artifact(str(model_dir / "metadata.json"))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Phase 3 failed")
            raise


if __name__ == "__main__":
    main()
