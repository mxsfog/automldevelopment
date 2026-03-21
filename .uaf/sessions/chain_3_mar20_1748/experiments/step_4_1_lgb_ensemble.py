"""Step 4.1: Optuna LightGBM + CB50 Ensemble on ELO-only subset."""

import logging
import os
import traceback

import lightgbm as lgb
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


def main() -> None:
    """Optuna LightGBM + Ensemble."""
    logger.info("Step 4.1: Optuna LightGBM + CB50 Ensemble")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

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

    # 1. Optuna LightGBM
    logger.info("Optimizing LightGBM with Optuna (30 trials)")

    def lgb_objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_fit,
            train_fit["target"],
            eval_set=[(x_val, val_df["target"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        proba_val = model.predict_proba(x_val)[:, 1]
        _t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
        return val_roi

    study_lgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=True)
    best_lgb_params = study_lgb.best_params
    logger.info("LGB best val ROI: %.2f%%, params: %s", study_lgb.best_value, best_lgb_params)

    # Train final LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=best_lgb_params["max_depth"],
        learning_rate=best_lgb_params["learning_rate"],
        reg_lambda=best_lgb_params["reg_lambda"],
        min_child_samples=best_lgb_params["min_child_samples"],
        subsample=best_lgb_params["subsample"],
        colsample_bytree=best_lgb_params["colsample_bytree"],
        num_leaves=best_lgb_params["num_leaves"],
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    proba_lgb_val = lgb_model.predict_proba(x_val)[:, 1]
    proba_lgb_test = lgb_model.predict_proba(x_test)[:, 1]

    # 2. Train Optuna CatBoost (best from step 3.1)
    logger.info("Training Optuna CatBoost")
    cb_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.08,
        l2_leaf_reg=21.1,
        min_data_in_leaf=20,
        random_strength=1.0,
        bagging_temperature=0.06,
        border_count=102,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=50,
    )
    cb_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    proba_cb_val = cb_model.predict_proba(x_val)[:, 1]
    proba_cb_test = cb_model.predict_proba(x_test)[:, 1]

    # 3. Train XGBoost
    logger.info("Training XGBoost")
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_lambda=10.0,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(x_fit, train_fit["target"], eval_set=[(x_val, val_df["target"])], verbose=False)
    proba_xgb_val = xgb_model.predict_proba(x_val)[:, 1]
    proba_xgb_test = xgb_model.predict_proba(x_test)[:, 1]

    # 4. CB50 Ensemble (50% CB + 25% LGB + 25% XGB)
    logger.info("Building CB50 Ensemble")
    weights = [0.50, 0.25, 0.25]
    proba_ens_val = (
        weights[0] * proba_cb_val + weights[1] * proba_lgb_val + weights[2] * proba_xgb_val
    )
    proba_ens_test = (
        weights[0] * proba_cb_test + weights[1] * proba_lgb_test + weights[2] * proba_xgb_test
    )

    best_t, _val_roi = find_best_threshold_on_val(val_df, proba_ens_val, min_bets=20)

    # Log individual model results
    for name, p_test in [("cb", proba_cb_test), ("lgb", proba_lgb_test), ("xgb", proba_xgb_test)]:
        r = calc_roi(test, p_test, threshold=best_t)
        auc_m = roc_auc_score(test["target"], p_test)
        logger.info("  %s solo: ROI=%.2f%% AUC=%.4f n=%d", name, r["roi"], auc_m, r["n_bets"])

    roi_result = calc_roi(test, proba_ens_test, threshold=best_t)
    auc = roc_auc_score(test["target"], proba_ens_test)

    # Log to MLflow
    with mlflow.start_run(run_name="phase4/step4.1_lgb_ens_cb50") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "cb50_ensemble_optuna",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "ensemble_weights": "CB50_LGB25_XGB25",
                    "lgb_optuna_trials": 30,
                    **{f"lgb_hp_{k}": v for k, v in best_lgb_params.items()},
                }
            )

            # Threshold scan
            for t_scan in np.arange(0.50, 0.90, 0.01):
                r = calc_roi(test, proba_ens_test, threshold=t_scan)
                if r["n_bets"] >= 20:
                    mlflow.log_metric(f"roi_t{int(t_scan * 100):03d}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                    "best_threshold": best_t,
                    "auc_cb": float(roc_auc_score(test["target"], proba_cb_test)),
                    "auc_lgb": float(roc_auc_score(test["target"], proba_lgb_test)),
                    "auc_xgb": float(roc_auc_score(test["target"], proba_xgb_test)),
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info(
                "Step 4.1: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
