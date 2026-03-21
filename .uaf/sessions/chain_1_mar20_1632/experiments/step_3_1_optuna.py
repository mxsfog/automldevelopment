"""Step 3.1: Optuna hyperparameter optimization for CatBoost."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
from catboost import CatBoostClassifier
from common import (
    add_safe_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_extended_feature_columns,
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


def main() -> None:
    logger.info("Step 3.1: Optuna CatBoost optimization")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    # Add features
    train_fit = add_safe_features(train_fit)
    train_val = add_safe_features(train_val)
    test = add_safe_features(test)

    feature_cols = get_extended_feature_columns()

    x_fit = np.nan_to_num(train_fit[feature_cols].values.astype(float), nan=0.0)
    y_fit = train_fit["target"].values
    x_val = np.nan_to_num(train_val[feature_cols].values.astype(float), nan=0.0)
    y_val = train_val["target"].values
    x_test = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)
    y_test = test["target"].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])

        model = CatBoostClassifier(
            **params,
            random_seed=42,
            verbose=0,
            eval_metric="AUC",
            auto_class_weights="Balanced" if use_class_weights else None,
        )
        model.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=50)

        proba_val = model.predict_proba(x_val)[:, 1]
        _thr, val_roi = find_best_threshold_on_val(train_val, proba_val)
        return val_roi

    with mlflow.start_run(run_name="phase3/step3.1_optuna") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")

        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(objective, n_trials=50, timeout=600)

            best_params = study.best_params
            logger.info("Best params: %s", best_params)
            logger.info("Best val ROI: %.2f%%", study.best_value)

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_roi", study.best_value)
            mlflow.log_metric("n_trials", len(study.trials))

            # Retrain best model on full data
            use_cw = best_params.pop("use_class_weights")
            best_model = CatBoostClassifier(
                **best_params,
                random_seed=42,
                verbose=100,
                eval_metric="AUC",
                auto_class_weights="Balanced" if use_cw else None,
            )
            best_model.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=50)

            proba_val = best_model.predict_proba(x_val)[:, 1]
            best_threshold, val_roi = find_best_threshold_on_val(train_val, proba_val)

            proba_test = best_model.predict_proba(x_test)[:, 1]
            roi_result = calc_roi(test, proba_test, threshold=best_threshold)
            auc = roc_auc_score(y_test, proba_test)

            logger.info(
                "Test ROI=%.2f%%, AUC=%.4f, threshold=%.2f, n=%d, WR=%.4f",
                roi_result["roi"],
                auc,
                best_threshold,
                roi_result["n_bets"],
                roi_result["win_rate"],
            )

            # Log ROI at various thresholds
            for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                r = calc_roi(test, proba_test, threshold=t)
                mlflow.log_metric(f"roi_t{int(t * 100)}", r["roi"])
                mlflow.log_metric(f"nbets_t{int(t * 100)}", r["n_bets"])
                logger.info("  t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "best_threshold": best_threshold,
                    "val_roi_at_threshold": val_roi,
                    "n_bets_selected": roi_result["n_bets"],
                    "pct_selected": roi_result["pct_selected"],
                    "win_rate_selected": roi_result["win_rate"],
                    "best_iteration": best_model.get_best_iteration(),
                }
            )

            importances = best_model.get_feature_importance()
            ranked = sorted(zip(feature_cols, importances, strict=True), key=lambda x: -x[1])
            for fname, imp in ranked[:10]:
                logger.info("  %s: %.2f", fname, imp)
                mlflow.log_metric(f"importance_{fname}", imp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            raise


if __name__ == "__main__":
    main()
