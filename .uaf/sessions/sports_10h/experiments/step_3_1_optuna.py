"""Step 3.1: Hyperparameter Optimization (Optuna TPE).

Оптимизация CatBoost гиперпараметров по ROI метрике.
Feature set зафиксирован из Phase 1/2.
Optuna TPE sampler с MedianPruner.
Двухуровневая оптимизация: CatBoost params + prediction threshold.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.warning("Budget hard stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

NUM_FEATURES = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "Outcomes_Count",
    "USD",
]
CAT_FEATURES = ["Sport", "Market", "is_parlay_str"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
CAT_INDICES = [ALL_FEATURES.index(c) for c in CAT_FEATURES]

N_OPTUNA_TRIALS = 40
N_SPLITS = 5


def compute_roi_at_threshold(df_val: pd.DataFrame, proba: np.ndarray, threshold: float) -> float:
    """ROI на ставках где P(won) > threshold."""
    selected = proba >= threshold
    if selected.sum() == 0:
        return -100.0
    sel = df_val[selected]
    total_staked = sel["USD"].sum()
    total_returned = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    return (total_returned - total_staked) / total_staked * 100


def prepare_data() -> pd.DataFrame:
    """Загрузка и подготовка данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", low_memory=False)
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv", low_memory=False)

    exclude = ["pending", "cancelled", "error", "cashout"]
    df = bets[~bets["Status"].isin(exclude)].copy()
    df["Created_At"] = pd.to_datetime(df["Created_At"])
    df = df.sort_values("Created_At").reset_index(drop=True)

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(Sport=("Sport", "first"), Market=("Market", "first"))
        .reset_index()
    )
    df = df.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")

    df["is_parlay_str"] = df["Is_Parlay"].astype(str)
    for col in ["ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV"]:
        df[col] = df[col].fillna(0.0)
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)

    return df


def objective(trial: optuna.Trial, df: pd.DataFrame, y_binary: pd.Series) -> float:
    """Optuna objective: средний ROI по time series CV."""
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
    }
    threshold = trial.suggest_float("threshold", 0.50, 0.80)
    use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])

    n = len(df)
    fold_size = n // (N_SPLITS + 1)
    fold_rois = []

    for fold_idx in range(N_SPLITS):
        train_end = fold_size * (fold_idx + 1)
        val_start = train_end
        val_end = train_end + fold_size
        if fold_idx == N_SPLITS - 1:
            val_end = n

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        y_train = y_binary.iloc[:train_end].values
        y_val = y_binary.iloc[val_start:val_end].values

        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            cat_features=CAT_INDICES,
            verbose=0,
            auto_class_weights="Balanced" if use_class_weights else None,
            early_stopping_rounds=50,
        )
        model.fit(
            train_df[ALL_FEATURES], y_train, eval_set=(val_df[ALL_FEATURES], y_val), verbose=0
        )
        proba = model.predict_proba(val_df[ALL_FEATURES])[:, 1]

        roi = compute_roi_at_threshold(val_df, proba, threshold)
        fold_rois.append(roi)

        # Pruning: после каждого fold
        trial.report(np.mean(fold_rois), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_rois)


def main() -> None:
    logger.info("Загрузка данных")
    df = prepare_data()
    y_binary = (df["Status"] == "won").astype(int)
    logger.info("Подготовленный датасет: %d строк", len(df))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase3/step3.1_optuna") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")

        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            )
            study.optimize(
                lambda trial: objective(trial, df, y_binary),
                n_trials=N_OPTUNA_TRIALS,
                show_progress_bar=True,
            )

            best = study.best_trial
            logger.info("Best trial %d: roi=%.4f", best.number, best.value)
            logger.info("Best params: %s", best.params)

            # Теперь запустим финальную модель с лучшими параметрами
            # и соберём подробные метрики по фолдам
            best_params = {
                k: v
                for k, v in best.params.items()
                if k != "threshold" and k != "use_class_weights"
            }
            best_threshold = best.params["threshold"]
            best_use_weights = best.params["use_class_weights"]

            n = len(df)
            fold_size = n // (N_SPLITS + 1)
            final_rois = []
            final_aucs = []

            for fold_idx in range(N_SPLITS):
                train_end = fold_size * (fold_idx + 1)
                val_start = train_end
                val_end = train_end + fold_size
                if fold_idx == N_SPLITS - 1:
                    val_end = n

                train_df = df.iloc[:train_end]
                val_df = df.iloc[val_start:val_end]
                y_train = y_binary.iloc[:train_end].values
                y_val = y_binary.iloc[val_start:val_end].values

                model = CatBoostClassifier(
                    **best_params,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=SEED,
                    cat_features=CAT_INDICES,
                    verbose=0,
                    auto_class_weights="Balanced" if best_use_weights else None,
                    early_stopping_rounds=50,
                )
                model.fit(
                    train_df[ALL_FEATURES],
                    y_train,
                    eval_set=(val_df[ALL_FEATURES], y_val),
                    verbose=0,
                )
                proba = model.predict_proba(val_df[ALL_FEATURES])[:, 1]

                roi = compute_roi_at_threshold(val_df, proba, best_threshold)
                auc = roc_auc_score(y_val, proba)
                final_rois.append(roi)
                final_aucs.append(auc)

                mlflow.log_metric(f"roi_fold_{fold_idx}", round(roi, 4))
                mlflow.log_metric(f"auc_fold_{fold_idx}", round(auc, 4))

                # Coverage
                selected = proba >= best_threshold
                coverage = selected.mean()
                mlflow.log_metric(f"coverage_fold_{fold_idx}", round(coverage, 4))

                logger.info(
                    "Final fold %d: roi=%.4f, auc=%.4f, coverage=%.2f%%",
                    fold_idx,
                    roi,
                    auc,
                    coverage * 100,
                )

            roi_mean = np.mean(final_rois)
            roi_std = np.std(final_rois)
            auc_mean = np.mean(final_aucs)

            # Feature importance
            fi = model.get_feature_importance()
            fi_dict = dict(zip(ALL_FEATURES, fi, strict=True))
            fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted:
                mlflow.log_metric(f"fi_{fname}", round(fval, 4))

            mlflow.log_params(
                {
                    **best_params,
                    "threshold": best_threshold,
                    "use_class_weights": best_use_weights,
                    "model": "CatBoost_optimized",
                    "features": ",".join(ALL_FEATURES),
                    "n_optuna_trials": N_OPTUNA_TRIALS,
                    "seed": SEED,
                    "validation_scheme": "time_series",
                    "n_splits": N_SPLITS,
                    "n_samples_total": n,
                }
            )

            mlflow.log_metrics(
                {
                    "roi_mean": round(roi_mean, 4),
                    "roi_std": round(roi_std, 4),
                    "auc_mean": round(auc_mean, 4),
                    "optuna_best_value": round(best.value, 4),
                    "optuna_n_trials": N_OPTUNA_TRIALS,
                    "optuna_n_complete": len(
                        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    ),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info(
                "Final: roi=%.4f +/- %.4f, auc=%.4f, threshold=%.2f",
                roi_mean,
                roi_std,
                auc_mean,
                best_threshold,
            )
            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT: roi_mean={roi_mean:.4f}, roi_std={roi_std:.4f}, "
                f"auc_mean={auc_mean:.4f}, threshold={best_threshold:.2f}"
            )
            print(f"BEST_PARAMS: {best.params}")
            print(f"RUN_ID: {run.info.run_id}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "optuna optimization failed")
            logger.exception("Step 3.1 failed")
            raise


if __name__ == "__main__":
    main()
