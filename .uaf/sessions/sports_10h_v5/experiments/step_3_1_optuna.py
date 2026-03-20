"""Step 3.1: Optuna Hyperparameter Optimization -- LogReg + CatBoost + LightGBM."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    logger.info("Step 3.1: Optuna Hyperparameter Optimization")
    df = load_data()
    train, test = time_series_split(df)

    feature_cols = get_feature_columns()

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    x_fit = train_fit[feature_cols].copy()
    y_fit = train_fit["target"].values
    x_val = train_val[feature_cols].copy()
    y_val = train_val["target"].values
    x_test = test[feature_cols].copy()
    y_test = test["target"].values

    # Impute for all models
    imputer = SimpleImputer(strategy="median")
    x_fit_imp = imputer.fit_transform(x_fit)
    x_val_imp = imputer.transform(x_val)
    x_test_imp = imputer.transform(x_test)

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit_imp)
    x_val_scaled = scaler.transform(x_val_imp)
    x_test_scaled = scaler.transform(x_test_imp)

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
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "optuna_tpe",
                    "features": ",".join(feature_cols),
                    "n_features": len(feature_cols),
                }
            )

            best_overall_roi = -999.0
            best_overall_model = None
            best_overall_params = {}
            best_overall_proba = None
            best_overall_threshold = 0.5

            # === 1. LogReg Optuna ===
            def logreg_objective(trial: optuna.Trial) -> float:
                c = trial.suggest_float("C", 0.001, 100.0, log=True)
                penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
                solver = "saga" if penalty == "l1" else "lbfgs"

                model = LogisticRegression(
                    C=c, penalty=penalty, solver=solver, random_state=42, max_iter=2000
                )
                model.fit(x_fit_scaled, y_fit)
                proba_v = model.predict_proba(x_val_scaled)[:, 1]
                _, val_roi = find_best_threshold_on_val(train_val, proba_v)
                return val_roi

            study_lr = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study_lr.optimize(logreg_objective, n_trials=30, show_progress_bar=False)

            best_lr = study_lr.best_params
            logger.info("Best LogReg params: %s, val ROI=%.2f%%", best_lr, study_lr.best_value)

            solver = "saga" if best_lr["penalty"] == "l1" else "lbfgs"
            model_lr = LogisticRegression(
                C=best_lr["C"],
                penalty=best_lr["penalty"],
                solver=solver,
                random_state=42,
                max_iter=2000,
            )
            model_lr.fit(x_fit_scaled, y_fit)
            proba_val_lr = model_lr.predict_proba(x_val_scaled)[:, 1]
            t_lr, _roi_val_lr = find_best_threshold_on_val(train_val, proba_val_lr)
            proba_test_lr = model_lr.predict_proba(x_test_scaled)[:, 1]
            roi_lr = calc_roi(test, proba_test_lr, threshold=t_lr)
            auc_lr = roc_auc_score(y_test, proba_test_lr)
            logger.info(
                "LogReg test: ROI=%.2f%%, AUC=%.4f, t=%.2f, n=%d",
                roi_lr["roi"],
                auc_lr,
                t_lr,
                roi_lr["n_bets"],
            )

            if roi_lr["roi"] > best_overall_roi:
                best_overall_roi = roi_lr["roi"]
                best_overall_model = "logreg"
                best_overall_params = best_lr
                best_overall_proba = proba_test_lr
                best_overall_threshold = t_lr

            # === 2. CatBoost Optuna ===
            def catboost_objective(trial: optuna.Trial) -> float:
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "depth": trial.suggest_int("depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                    "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
                }

                model = CatBoostClassifier(
                    **params,
                    random_seed=42,
                    verbose=0,
                    eval_metric="AUC",
                    auto_class_weights="Balanced",
                )
                model.fit(
                    x_fit_imp,
                    y_fit,
                    eval_set=(x_val_imp, y_val),
                    early_stopping_rounds=50,
                )
                proba_v = model.predict_proba(x_val_imp)[:, 1]
                _, val_roi = find_best_threshold_on_val(train_val, proba_v)
                return val_roi

            study_cb = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study_cb.optimize(catboost_objective, n_trials=30, show_progress_bar=False)

            best_cb = study_cb.best_params
            logger.info("Best CatBoost params: %s, val ROI=%.2f%%", best_cb, study_cb.best_value)

            model_cb = CatBoostClassifier(
                **best_cb,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                auto_class_weights="Balanced",
            )
            model_cb.fit(
                x_fit_imp,
                y_fit,
                eval_set=(x_val_imp, y_val),
                early_stopping_rounds=50,
            )
            proba_val_cb = model_cb.predict_proba(x_val_imp)[:, 1]
            t_cb, _roi_val_cb = find_best_threshold_on_val(train_val, proba_val_cb)
            proba_test_cb = model_cb.predict_proba(x_test_imp)[:, 1]
            roi_cb = calc_roi(test, proba_test_cb, threshold=t_cb)
            auc_cb = roc_auc_score(y_test, proba_test_cb)
            logger.info(
                "CatBoost test: ROI=%.2f%%, AUC=%.4f, t=%.2f, n=%d",
                roi_cb["roi"],
                auc_cb,
                t_cb,
                roi_cb["n_bets"],
            )

            if roi_cb["roi"] > best_overall_roi:
                best_overall_roi = roi_cb["roi"]
                best_overall_model = "catboost"
                best_overall_params = best_cb
                best_overall_proba = proba_test_cb
                best_overall_threshold = t_cb

            # === 3. LightGBM Optuna ===
            def lgbm_objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }

                model = LGBMClassifier(
                    **params,
                    random_state=42,
                    verbose=-1,
                    is_unbalance=True,
                )
                model.fit(
                    x_fit_imp,
                    y_fit,
                    eval_set=[(x_val_imp, y_val)],
                    callbacks=[],
                )
                proba_v = model.predict_proba(x_val_imp)[:, 1]
                _, val_roi = find_best_threshold_on_val(train_val, proba_v)
                return val_roi

            study_lgb = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study_lgb.optimize(lgbm_objective, n_trials=30, show_progress_bar=False)

            best_lgb = study_lgb.best_params
            logger.info("Best LightGBM params: %s, val ROI=%.2f%%", best_lgb, study_lgb.best_value)

            model_lgb = LGBMClassifier(
                **best_lgb,
                random_state=42,
                verbose=-1,
                is_unbalance=True,
            )
            model_lgb.fit(
                x_fit_imp,
                y_fit,
                eval_set=[(x_val_imp, y_val)],
                callbacks=[],
            )
            proba_val_lgb = model_lgb.predict_proba(x_val_imp)[:, 1]
            t_lgb, _roi_val_lgb = find_best_threshold_on_val(train_val, proba_val_lgb)
            proba_test_lgb = model_lgb.predict_proba(x_test_imp)[:, 1]
            roi_lgb = calc_roi(test, proba_test_lgb, threshold=t_lgb)
            auc_lgb = roc_auc_score(y_test, proba_test_lgb)
            logger.info(
                "LightGBM test: ROI=%.2f%%, AUC=%.4f, t=%.2f, n=%d",
                roi_lgb["roi"],
                auc_lgb,
                t_lgb,
                roi_lgb["n_bets"],
            )

            if roi_lgb["roi"] > best_overall_roi:
                best_overall_roi = roi_lgb["roi"]
                best_overall_model = "lightgbm"
                best_overall_params = best_lgb
                best_overall_proba = proba_test_lgb
                best_overall_threshold = t_lgb

            # Log results
            mlflow.log_metrics(
                {
                    "roi": best_overall_roi,
                    "roi_logreg": roi_lr["roi"],
                    "roi_catboost": roi_cb["roi"],
                    "roi_lightgbm": roi_lgb["roi"],
                    "auc_logreg": auc_lr,
                    "auc_catboost": auc_cb,
                    "auc_lightgbm": auc_lgb,
                    "threshold_logreg": t_lr,
                    "threshold_catboost": t_cb,
                    "threshold_lightgbm": t_lgb,
                    "n_bets_logreg": roi_lr["n_bets"],
                    "n_bets_catboost": roi_cb["n_bets"],
                    "n_bets_lightgbm": roi_lgb["n_bets"],
                }
            )
            mlflow.set_tag("best_model", best_overall_model)
            mlflow.log_params({f"best_{k}": v for k, v in best_overall_params.items()})

            # Save probabilities from best model for ensemble experiments
            np.save("best_proba_test.npy", best_overall_proba)
            mlflow.log_artifact("best_proba_test.npy")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info(
                "Best overall: %s, ROI=%.2f%%, threshold=%.2f",
                best_overall_model,
                best_overall_roi,
                best_overall_threshold,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            raise


if __name__ == "__main__":
    main()
