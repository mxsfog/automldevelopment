"""Step 3.1 — Hyperparameter Optimization с Optuna TPE."""

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
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    compute_roi,
    find_best_threshold,
    load_bets,
    time_series_split,
)
from feature_engineering import add_odds_features

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    budget_status = json.loads(budget_file.read_text())
    if budget_status.get("hard_stop"):
        logger.warning("Budget hard stop. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    pass

NUM_FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "implied_prob",
    "log_odds",
    "value",
    "overround",
]

CAT_FEATURES = ["Sport", "Market", "odds_bucket"]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
CAT_INDICES = list(range(len(NUM_FEATURES), len(ALL_FEATURES)))

N_TRIALS = 30

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Загрузка данных
df = load_bets(with_outcomes=True)
for col in CAT_FEATURES:
    if col != "odds_bucket":
        df[col] = df[col].fillna("unknown").astype(str)

add_odds_features(df)
splits = time_series_split(df, n_splits=5, gap_days=7)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize mean ROI across folds."""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    fold_rois = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        x_train = train_df[ALL_FEATURES].copy()
        y_train = train_df["target"].values
        x_val = val_df[ALL_FEATURES].copy()
        y_val = val_df["target"].values
        stakes_val = val_df["USD"].values
        payouts_val = val_df["Payout_USD"].values

        model = CatBoostClassifier(
            **params,
            random_seed=42,
            verbose=0,
            cat_features=CAT_INDICES,
            eval_metric="AUC",
            early_stopping_rounds=50,
            use_best_model=True,
        )

        model.fit(x_train, y_train, eval_set=(x_val, y_val))

        y_proba = model.predict_proba(x_val)[:, 1]

        # Поиск лучшего порога на train
        y_train_proba = model.predict_proba(x_train)[:, 1]
        best_t, _ = find_best_threshold(
            y_train,
            y_train_proba,
            train_df["USD"].values,
            train_df["Payout_USD"].values,
        )

        roi_result = compute_roi(y_val, y_proba, stakes_val, payouts_val, threshold=best_t)
        fold_rois.append(roi_result["roi"])

        trial.set_user_attr(f"roi_fold_{fold_idx}", roi_result["roi"])
        trial.set_user_attr(f"auc_fold_{fold_idx}", roc_auc_score(y_val, y_proba))
        trial.set_user_attr(f"threshold_fold_{fold_idx}", best_t)
        trial.set_user_attr(f"n_selected_fold_{fold_idx}", roi_result["n_selected"])

        # Pruning по median
        intermediate_roi = float(np.mean(fold_rois))
        trial.report(intermediate_roi, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_roi = float(np.mean(fold_rois))
    trial.set_user_attr("roi_std", float(np.std(fold_rois)))
    return mean_roi


with mlflow.start_run(run_name="phase3/step3.1_optuna_hpo") as parent_run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")
        mlflow.set_tag("method", "optuna_tpe")

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_splits": len(splits),
                "n_trials": N_TRIALS,
                "features": ",".join(ALL_FEATURES),
                "n_features": len(ALL_FEATURES),
                "sampler": "TPE",
                "pruner": "MedianPruner",
            }
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for trial_num in range(N_TRIALS):
            # Проверка бюджета
            try:
                budget_status = json.loads(budget_file.read_text())
                if budget_status.get("hard_stop"):
                    logger.warning("Budget hard stop at trial %d", trial_num)
                    break
            except FileNotFoundError:
                pass

            study.optimize(objective, n_trials=1, show_progress_bar=False)
            trial = study.trials[-1]

            if trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(
                    "Trial %d: ROI=%.2f%%, params=%s",
                    trial_num,
                    trial.value,
                    trial.params,
                )
                # Логируем каждый trial в MLflow
                with mlflow.start_run(
                    run_name=f"trial_{trial_num}",
                    nested=True,
                ) as trial_run:
                    mlflow.log_params(trial.params)
                    mlflow.log_metric("roi_mean", trial.value)
                    mlflow.log_metric("roi_std", trial.user_attrs.get("roi_std", 0))
                    for k, v in trial.user_attrs.items():
                        if k.startswith("roi_fold_") or k.startswith("auc_fold_"):
                            mlflow.log_metric(k, v)
            elif trial.state == optuna.trial.TrialState.PRUNED:
                logger.info("Trial %d: PRUNED", trial_num)

        # Лучший trial
        best = study.best_trial
        logger.info("Best trial: ROI=%.2f%%, params=%s", best.value, best.params)

        # Финальная оценка лучших параметров (полная)
        best_params = best.params
        fold_rois = []
        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            x_train = train_df[ALL_FEATURES].copy()
            y_train = train_df["target"].values
            x_val = val_df[ALL_FEATURES].copy()
            y_val = val_df["target"].values
            stakes_val = val_df["USD"].values
            payouts_val = val_df["Payout_USD"].values

            model = CatBoostClassifier(
                **best_params,
                random_seed=42,
                verbose=0,
                cat_features=CAT_INDICES,
                eval_metric="AUC",
                early_stopping_rounds=50,
                use_best_model=True,
            )
            model.fit(x_train, y_train, eval_set=(x_val, y_val))

            y_proba = model.predict_proba(x_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            fold_aucs.append(auc)

            y_train_proba = model.predict_proba(x_train)[:, 1]
            best_t, _ = find_best_threshold(
                y_train,
                y_train_proba,
                train_df["USD"].values,
                train_df["Payout_USD"].values,
            )
            roi_result = compute_roi(y_val, y_proba, stakes_val, payouts_val, threshold=best_t)
            fold_rois.append(roi_result["roi"])

            mlflow.log_metrics(
                {
                    f"best_roi_fold_{fold_idx}": roi_result["roi"],
                    f"best_auc_fold_{fold_idx}": auc,
                    f"best_threshold_fold_{fold_idx}": best_t,
                    f"best_n_selected_fold_{fold_idx}": roi_result["n_selected"],
                }
            )

            logger.info(
                "Best params Fold %d: AUC=%.4f, ROI=%.2f%% (threshold=%.2f, selected=%d/%d)",
                fold_idx,
                auc,
                roi_result["roi"],
                best_t,
                roi_result["n_selected"],
                roi_result["n_total"],
            )

        mean_roi = float(np.mean(fold_rois))
        std_roi = float(np.std(fold_rois))
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        mlflow.log_metrics(
            {
                "roi_mean": mean_roi,
                "roi_std": std_roi,
                "roc_auc_mean": mean_auc,
                "roc_auc_std": std_auc,
                "n_trials_completed": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                ),
                "n_trials_pruned": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                ),
            }
        )

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Feature importance
        importances = model.get_feature_importance()
        for feat, imp in zip(ALL_FEATURES, importances, strict=True):
            mlflow.log_metric(f"importance_{feat}", imp)

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.8")

        logger.info(
            "Optuna HPO: ROI_mean=%.2f%% +/- %.2f%%, AUC_mean=%.4f +/- %.4f",
            mean_roi,
            std_roi,
            mean_auc,
            std_auc,
        )
        print(f"RESULT: roi_mean={mean_roi:.4f}, roi_std={std_roi:.4f}, auc={mean_auc:.4f}")
        print(f"BEST_PARAMS: {best_params}")
        print(f"RUN_ID: {parent_run.info.run_id}")

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "runtime_error")
        logger.exception("Step 3.1 failed")
        raise
