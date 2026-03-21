"""Step 4.5: Optuna LightGBM + ensemble + segment filter."""

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
    logger.info("Step 4.5: Optuna LGB + ensemble + segment filter")
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

    # Masks for filtering
    val_good = ~train_val["Sport"].isin(EXCLUDE_SPORTS)
    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)

    with mlflow.start_run(run_name="phase4/step4.5_optuna_lgb_segment") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("phase", "4")

        try:
            # Optuna for LightGBM
            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }
                model = LGBMClassifier(**params, random_state=42, verbose=-1)
                model.fit(
                    x_fit,
                    y_fit,
                    eval_set=[(x_val, y_val)],
                    callbacks=[
                        __import__("lightgbm").early_stopping(50, verbose=False),
                        __import__("lightgbm").log_evaluation(0),
                    ],
                )
                p_val = model.predict_proba(x_val)[:, 1]
                # Optimize ROI on val with segment filter
                p_val_filt = p_val[val_good.values]
                val_filt = train_val[val_good]
                _thr, val_roi = find_best_threshold_on_val(val_filt, p_val_filt)
                return val_roi

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(objective, n_trials=40, timeout=300)

            best_lgb_params = study.best_params
            logger.info("Best LGB params: %s", best_lgb_params)
            logger.info("Best LGB val ROI: %.2f%%", study.best_value)

            # Train best LGB
            best_lgb = LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1)
            best_lgb.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            # CatBoost (same best params)
            cb = CatBoostClassifier(
                iterations=855,
                depth=3,
                learning_rate=0.059,
                l2_leaf_reg=21.0,
                border_count=254,
                random_strength=9.26,
                bagging_temperature=4.82,
                min_data_in_leaf=77,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
            )
            cb.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=50)

            # XGBoost
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

            # Ensemble predictions
            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgb_test = best_lgb.predict_proba(x_test)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]
            p_ensemble_test = (p_cb_test + p_lgb_test + p_xgb_test) / 3

            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_lgb_val = best_lgb.predict_proba(x_val)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]
            p_ensemble_val = (p_cb_val + p_lgb_val + p_xgb_val) / 3

            # Filtered results
            p_test_filt = p_ensemble_test[test_good.values]
            test_filt = test[test_good]
            p_val_filt = p_ensemble_val[val_good.values]
            val_filt = train_val[val_good]

            thr, val_roi = find_best_threshold_on_val(val_filt, p_val_filt)
            roi_filt = calc_roi(test_filt, p_test_filt, threshold=thr)
            auc_filt = roc_auc_score(test_filt["target"].values, p_test_filt)

            logger.info(
                "[filtered ensemble] ROI=%.2f%%, AUC=%.4f, thr=%.2f, n=%d, WR=%.4f",
                roi_filt["roi"],
                auc_filt,
                thr,
                roi_filt["n_bets"],
                roi_filt["win_rate"],
            )

            # Also try individual models with filter
            for name, p_t, p_v in [
                ("catboost", p_cb_test, p_cb_val),
                ("lightgbm", p_lgb_test, p_lgb_val),
                ("xgboost", p_xgb_test, p_xgb_val),
            ]:
                pt_f = p_t[test_good.values]
                pv_f = p_v[val_good.values]
                t_i, _ = find_best_threshold_on_val(val_filt, pv_f)
                r_i = calc_roi(test_filt, pt_f, threshold=t_i)
                logger.info(
                    "  [%s filtered] ROI=%.2f%%, thr=%.2f, n=%d",
                    name,
                    r_i["roi"],
                    t_i,
                    r_i["n_bets"],
                )
                mlflow.log_metric(f"roi_{name}_filtered", r_i["roi"])

            # Test specific thresholds on filtered ensemble
            for t in np.arange(0.45, 0.75, 0.05):
                r = calc_roi(test_filt, p_test_filt, threshold=t)
                logger.info("  ens_filt t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "optuna_lgb_ensemble_segment",
                    "n_optuna_trials": len(study.trials),
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_threshold": thr,
                }
            )
            for k, v in best_lgb_params.items():
                mlflow.log_param(f"lgb_{k}", v)

            mlflow.log_metrics(
                {
                    "roi": roi_filt["roi"],
                    "roc_auc": auc_filt,
                    "best_threshold": thr,
                    "val_roi_at_threshold": val_roi,
                    "n_bets_selected": roi_filt["n_bets"],
                    "pct_selected": roi_filt["pct_selected"],
                    "win_rate_selected": roi_filt["win_rate"],
                    "best_lgb_val_roi": study.best_value,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.5")
            raise


if __name__ == "__main__":
    main()
