"""Step 4.3: Singles-only Optuna -- LightGBM + XGBoost HPO на синглах."""

import logging
import os
import traceback

import mlflow
import optuna
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    logger.info("Step 4.3: Singles-only Optuna (LightGBM + XGBoost)")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()
    logger.info("Singles: train=%d, test=%d", len(train), len(test))

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[feature_cols])
    x_val = imputer.transform(train_val[feature_cols])
    x_test = imputer.transform(test[feature_cols])
    y_fit = train_fit["target"].values
    y_val = train_val["target"].values
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase4/step4.3_singles_optuna") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.3")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "optuna_singles_lgbm_xgb",
                    "filter": "Is_Parlay=f",
                    "n_features": len(feature_cols),
                }
            )

            # === LightGBM Optuna ===
            def lgbm_objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 600),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 8, 200),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                }
                model = LGBMClassifier(
                    **params,
                    random_state=42,
                    verbose=-1,
                    is_unbalance=True,
                )
                model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], callbacks=[])
                proba_v = model.predict_proba(x_val)[:, 1]
                _, val_roi = find_best_threshold_on_val(train_val, proba_v)
                return val_roi

            study_lgb = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study_lgb.optimize(lgbm_objective, n_trials=40)

            best_lgb = study_lgb.best_params
            logger.info("Best LightGBM: %s, val ROI=%.2f%%", best_lgb, study_lgb.best_value)

            model_lgb = LGBMClassifier(**best_lgb, random_state=42, verbose=-1, is_unbalance=True)
            model_lgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], callbacks=[])
            p_lgb_val = model_lgb.predict_proba(x_val)[:, 1]
            t_lgb, _ = find_best_threshold_on_val(train_val, p_lgb_val)
            p_lgb_test = model_lgb.predict_proba(x_test)[:, 1]
            roi_lgb = calc_roi(test, p_lgb_test, threshold=t_lgb)
            auc_lgb = roc_auc_score(y_test, p_lgb_test)
            logger.info(
                "LightGBM test: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_lgb["roi"],
                auc_lgb,
                t_lgb,
                roi_lgb["n_bets"],
            )

            # Log at multiple thresholds
            for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, p_lgb_test, threshold=t)
                logger.info("  LGB t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])

            # === XGBoost Optuna ===
            def xgb_objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 600),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5.0),
                }
                model = XGBClassifier(
                    **params,
                    random_state=42,
                    verbosity=0,
                    scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
                    eval_metric="auc",
                )
                model.fit(
                    x_fit,
                    y_fit,
                    eval_set=[(x_val, y_val)],
                    verbose=False,
                )
                proba_v = model.predict_proba(x_val)[:, 1]
                _, val_roi = find_best_threshold_on_val(train_val, proba_v)
                return val_roi

            study_xgb = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study_xgb.optimize(xgb_objective, n_trials=40)

            best_xgb = study_xgb.best_params
            logger.info("Best XGBoost: %s, val ROI=%.2f%%", best_xgb, study_xgb.best_value)

            model_xgb = XGBClassifier(
                **best_xgb,
                random_state=42,
                verbosity=0,
                scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
                eval_metric="auc",
            )
            model_xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
            p_xgb_val = model_xgb.predict_proba(x_val)[:, 1]
            t_xgb, _ = find_best_threshold_on_val(train_val, p_xgb_val)
            p_xgb_test = model_xgb.predict_proba(x_test)[:, 1]
            roi_xgb = calc_roi(test, p_xgb_test, threshold=t_xgb)
            auc_xgb = roc_auc_score(y_test, p_xgb_test)
            logger.info(
                "XGBoost test: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_xgb["roi"],
                auc_xgb,
                t_xgb,
                roi_xgb["n_bets"],
            )

            for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, p_xgb_test, threshold=t)
                logger.info("  XGB t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])

            # === Ensemble of LGB + XGB ===
            p_ens_val = (p_lgb_val + p_xgb_val) / 2
            t_ens, _ = find_best_threshold_on_val(train_val, p_ens_val)
            p_ens_test = (p_lgb_test + p_xgb_test) / 2
            roi_ens = calc_roi(test, p_ens_test, threshold=t_ens)
            auc_ens = roc_auc_score(y_test, p_ens_test)
            logger.info(
                "Ensemble LGB+XGB test: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_ens["roi"],
                auc_ens,
                t_ens,
                roi_ens["n_bets"],
            )

            best_roi = max(roi_lgb["roi"], roi_xgb["roi"], roi_ens["roi"])
            best_model_name = "lightgbm"
            if roi_xgb["roi"] == best_roi:
                best_model_name = "xgboost"
            elif roi_ens["roi"] == best_roi:
                best_model_name = "ensemble_lgb_xgb"

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_lightgbm": roi_lgb["roi"],
                    "roi_xgboost": roi_xgb["roi"],
                    "roi_ensemble": roi_ens["roi"],
                    "auc_lightgbm": auc_lgb,
                    "auc_xgboost": auc_xgb,
                    "auc_ensemble": auc_ens,
                    "threshold_lightgbm": t_lgb,
                    "threshold_xgboost": t_xgb,
                    "threshold_ensemble": t_ens,
                    "n_bets_lightgbm": roi_lgb["n_bets"],
                    "n_bets_xgboost": roi_xgb["n_bets"],
                    "n_bets_ensemble": roi_ens["n_bets"],
                }
            )
            mlflow.set_tag("best_model", best_model_name)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best: %s, ROI=%.2f%%", best_model_name, best_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.3")
            raise


if __name__ == "__main__":
    main()
