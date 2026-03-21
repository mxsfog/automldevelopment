"""Phase 3: Optuna HPO on ELO-only subset."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
from catboost import CatBoostClassifier
from common import (
    add_elo_features,
    add_engineered_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_elo_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
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


def main() -> None:
    """Optuna HPO для CatBoost на ELO-only subset."""
    logger.info("Phase 3: Optuna HPO on ELO-only subset")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

    # ELO-only subset
    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test[feat_list])

    def objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "random_strength": trial.suggest_float("random_strength", 0.5, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

        proba_val = model.predict_proba(x_val)[:, 1]
        _best_t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
        return val_roi

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    best_params = study.best_params
    best_val_roi = study.best_value
    logger.info("Best val ROI: %.2f%%, params: %s", best_val_roi, best_params)

    # Retrain with best params
    with mlflow.start_run(run_name="phase3/step3.1_optuna_cb") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")

        try:
            final_params = {
                "iterations": 1000,
                "depth": best_params["depth"],
                "learning_rate": best_params["learning_rate"],
                "l2_leaf_reg": best_params["l2_leaf_reg"],
                "min_data_in_leaf": best_params["min_data_in_leaf"],
                "random_strength": best_params["random_strength"],
                "bagging_temperature": best_params["bagging_temperature"],
                "border_count": best_params["border_count"],
                "random_seed": 42,
                "verbose": 0,
                "eval_metric": "AUC",
                "early_stopping_rounds": 50,
            }
            model = CatBoostClassifier(**final_params)
            model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(test["target"], proba_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "optuna_catboost_elo_only",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "best_iteration": model.best_iteration_,
                    "n_optuna_trials": 40,
                    "best_val_roi": best_val_roi,
                    **{f"hp_{k}": v for k, v in best_params.items()},
                }
            )

            # Threshold scan on test
            for t_scan in np.arange(0.50, 0.90, 0.01):
                r = calc_roi(test, proba_test, threshold=t_scan)
                if r["n_bets"] >= 20:
                    mlflow.log_metric(f"roi_t{int(t_scan * 100):03d}", r["roi"])

            fi = dict(zip(feat_list, model.feature_importances_, strict=False))
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted[:15]:
                logger.info("  FI: %s = %.3f", fname, fval)
                mlflow.log_metric(f"fi_{fname}", fval)

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                    "best_threshold": best_t,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info(
                "Step 3.1: ROI=%.2f%% AUC=%.4f t=%.2f n=%d iter=%d run=%s",
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                model.best_iteration_,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
