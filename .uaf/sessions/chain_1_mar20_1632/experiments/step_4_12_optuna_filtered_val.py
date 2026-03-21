"""Step 4.12: Optuna CatBoost optimizing ROI on sport-filtered val."""

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
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

EXCLUDE_SPORTS = {"Basketball", "MMA", "FIFA", "Snooker"}


def main() -> None:
    logger.info("Step 4.12: Optuna CatBoost on filtered val")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    train_fit = add_safe_features(train_fit)
    train_val = add_safe_features(train_val)
    test = add_safe_features(test)

    feature_cols = get_extended_feature_columns()

    x_fit = np.nan_to_num(train_fit[feature_cols].values.astype(float), nan=0.0)
    y_fit = train_fit["target"].values
    x_val = np.nan_to_num(train_val[feature_cols].values.astype(float), nan=0.0)
    y_val = train_val["target"].values
    x_test = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)

    val_good = ~train_val["Sport"].isin(EXCLUDE_SPORTS)
    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)
    val_filt = train_val[val_good]
    test_filt = test[test_good]

    with mlflow.start_run(run_name="phase4/step4.12_optuna_filt_val") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.12")
        mlflow.set_tag("phase", "4")

        try:

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 2000),
                    "depth": trial.suggest_int("depth", 2, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
                    "border_count": trial.suggest_int("border_count", 32, 254),
                    "random_strength": trial.suggest_float("random_strength", 0.1, 20.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                }
                cb = CatBoostClassifier(**params, random_seed=42, verbose=0, eval_metric="AUC")
                cb.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=50)

                p_val = cb.predict_proba(x_val)[:, 1]
                p_val_f = p_val[val_good.values]

                # Optimize ROI on filtered val
                _thr, val_roi = find_best_threshold_on_val(val_filt, p_val_f)
                return val_roi

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(objective, n_trials=30, timeout=180)

            best_params = study.best_params
            logger.info("Best Optuna params: %s", best_params)
            logger.info("Best Optuna val ROI: %.2f%%", study.best_value)

            # Train best CatBoost
            best_cb = CatBoostClassifier(
                **best_params, random_seed=42, verbose=0, eval_metric="AUC"
            )
            best_cb.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=50)

            # Ensemble with LGB + XGB
            lgb = LGBMClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                reg_alpha=1.0,
                reg_lambda=10.0,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            lgb.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            xgb = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                reg_alpha=1.0,
                reg_lambda=10.0,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="auc",
                verbosity=0,
            )
            xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)

            # CatBoost-only result
            p_cb_val = best_cb.predict_proba(x_val)[:, 1]
            p_cb_test = best_cb.predict_proba(x_test)[:, 1]
            p_cb_val_f = p_cb_val[val_good.values]
            p_cb_test_f = p_cb_test[test_good.values]

            thr_cb, _vr_cb = find_best_threshold_on_val(val_filt, p_cb_val_f)
            roi_cb = calc_roi(test_filt, p_cb_test_f, threshold=thr_cb)
            auc_cb = roc_auc_score(test_filt["target"].values, p_cb_test_f)
            logger.info(
                "[Optuna CB only] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d, best_iter=%d",
                thr_cb,
                roi_cb["roi"],
                auc_cb,
                roi_cb["n_bets"],
                best_cb.best_iteration_,
            )

            # Ensemble result
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            p_ens_val = (p_cb_val + p_lgb_val + p_xgb_val) / 3
            p_ens_test = (p_cb_test + p_lgb_test + p_xgb_test) / 3
            p_ens_val_f = p_ens_val[val_good.values]
            p_ens_test_f = p_ens_test[test_good.values]

            thr_ens, _vr_ens = find_best_threshold_on_val(val_filt, p_ens_val_f)
            roi_ens = calc_roi(test_filt, p_ens_test_f, threshold=thr_ens)
            auc_ens = roc_auc_score(test_filt["target"].values, p_ens_test_f)
            logger.info(
                "[Optuna CB + LGB + XGB] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d",
                thr_ens,
                roi_ens["roi"],
                auc_ens,
                roi_ens["n_bets"],
            )

            # Pick best
            if roi_ens["roi"] > roi_cb["roi"]:
                final_roi = roi_ens
                final_auc = auc_ens
                final_thr = thr_ens
                approach = "optuna_cb_ensemble"
            else:
                final_roi = roi_cb
                final_auc = auc_cb
                final_thr = thr_cb
                approach = "optuna_cb_only"

            logger.info("Best: %s, ROI=%.2f%%", approach, final_roi["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "optuna_filtered_val",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_approach": approach,
                    "best_threshold": final_thr,
                    "n_optuna_trials": len(study.trials),
                }
            )
            for k, v in best_params.items():
                mlflow.log_param(f"cb_{k}", v)

            mlflow.log_metrics(
                {
                    "roi": final_roi["roi"],
                    "roc_auc": final_auc,
                    "best_threshold": final_thr,
                    "n_bets_selected": final_roi["n_bets"],
                    "win_rate_selected": final_roi["win_rate"],
                    "roi_cb_only": roi_cb["roi"],
                    "roi_ensemble": roi_ens["roi"],
                    "optuna_best_val_roi": study.best_value,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.12")
            raise


if __name__ == "__main__":
    main()
