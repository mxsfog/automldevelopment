"""Step 3.1: Optuna CatBoost optimization on ELO-only subset."""

import logging
import os
import traceback

import mlflow
import optuna
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from step_2_5_safe_elo import build_safe_elo_features, get_safe_elo_features

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
    """Optuna TPE optimization on ELO-only subset."""
    logger.info("Step 3.1: Optuna CatBoost on ELO-only subset")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)

    train_all, test_all = time_series_split(df)

    # ELO-only
    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    feature_cols = get_base_features() + get_engineered_features() + get_safe_elo_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feature_cols])
    x_val = imp.transform(val_df[feature_cols])
    x_test = imp.transform(test[feature_cols])
    y_fit = train_fit["target"].values
    y_val = val_df["target"].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 50, 500),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 30,
        }

        model = CatBoostClassifier(**params)
        model.fit(x_fit, y_fit, eval_set=(x_val, y_val))

        proba_val = model.predict_proba(x_val)[:, 1]
        _best_t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
        return val_roi

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=60, show_progress_bar=False)

    logger.info("Best trial: value=%.2f%%, params=%s", study.best_value, study.best_params)

    # Retrain with best params
    best_params = study.best_params.copy()
    best_params["random_seed"] = 42
    best_params["verbose"] = 0
    best_params["eval_metric"] = "AUC"
    best_params["early_stopping_rounds"] = 30

    with mlflow.start_run(run_name="phase3/step3.1_optuna_elo") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")

        try:
            model = CatBoostClassifier(**best_params)
            model.fit(x_fit, y_fit, eval_set=(x_val, y_val))

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(test["target"], proba_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "optuna_catboost_elo",
                    "n_features": len(feature_cols),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "best_iteration": model.best_iteration_,
                    "optuna_n_trials": 60,
                    "optuna_best_val_roi": study.best_value,
                    "subset": "elo_only",
                    "leakage_free": "true",
                    **{f"hp_{k}": v for k, v in study.best_params.items()},
                }
            )

            roi_thresholds = calc_roi_at_thresholds(test, proba_test)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            fi = dict(zip(feature_cols, model.feature_importances_, strict=False))
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted[:10]:
                logger.info("  FI: %s = %.3f", fname, fval)

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
                "Optuna CatBoost ELO: ROI=%.2f%% AUC=%.4f t=%.2f n=%d iter=%d run=%s",
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
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            raise


if __name__ == "__main__":
    main()
