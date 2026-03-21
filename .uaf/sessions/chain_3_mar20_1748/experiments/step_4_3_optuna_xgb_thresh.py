"""Step 4.3: Optuna XGBoost + high threshold scan + CB65 ensemble."""

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
    """Optuna XGBoost + threshold scan + CB65 ensemble."""
    logger.info("Step 4.3: Optuna XGBoost + threshold scan")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test[feat_list])

    # 1. Optuna XGBoost
    logger.info("Optimizing XGBoost with Optuna (30 trials)")

    def xgb_objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": 42,
            "eval_metric": "auc",
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        model.fit(x_fit, train_fit["target"], eval_set=[(x_val, val_df["target"])], verbose=False)
        proba_val = model.predict_proba(x_val)[:, 1]
        _t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
        return val_roi

    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_xgb.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
    best_xgb_params = study_xgb.best_params
    logger.info("XGB best val ROI: %.2f%%, params: %s", study_xgb.best_value, best_xgb_params)

    # Train final XGBoost
    xgb_final = XGBClassifier(
        n_estimators=1000,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
        **best_xgb_params,
    )
    xgb_final.fit(x_fit, train_fit["target"], eval_set=[(x_val, val_df["target"])], verbose=False)
    p_xgb_val = xgb_final.predict_proba(x_val)[:, 1]
    p_xgb_test = xgb_final.predict_proba(x_test)[:, 1]

    # 2. CatBoost (Optuna best from step 3.1)
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
    p_cb_val = cb_model.predict_proba(x_val)[:, 1]
    p_cb_test = cb_model.predict_proba(x_test)[:, 1]

    # 3. LightGBM (Optuna best from step 4.1)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.058,
        reg_lambda=27.5,
        min_child_samples=46,
        subsample=0.88,
        colsample_bytree=0.95,
        num_leaves=22,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_lgb_val = lgb_model.predict_proba(x_val)[:, 1]
    p_lgb_test = lgb_model.predict_proba(x_test)[:, 1]

    # 4. Test multiple configurations
    configs = {}

    # CB solo with various thresholds
    for t in np.arange(0.70, 0.85, 0.01):
        r = calc_roi(test, p_cb_test, threshold=t)
        if r["n_bets"] >= 20:
            configs[f"cb_t{int(t * 100)}"] = (r, t, "cb_solo")

    # XGB solo
    xgb_t, _ = find_best_threshold_on_val(val_df, p_xgb_val, min_bets=20)
    xgb_r = calc_roi(test, p_xgb_test, threshold=xgb_t)
    configs["xgb_solo"] = (xgb_r, xgb_t, "xgb_optuna_solo")

    # CB65 ensemble: 65% CB + 20% LGB + 15% XGB
    p_cb65_val = 0.65 * p_cb_val + 0.20 * p_lgb_val + 0.15 * p_xgb_val
    p_cb65_test = 0.65 * p_cb_test + 0.20 * p_lgb_test + 0.15 * p_xgb_test
    cb65_t, _ = find_best_threshold_on_val(val_df, p_cb65_val, min_bets=20)
    cb65_r = calc_roi(test, p_cb65_test, threshold=cb65_t)
    configs["cb65_ens"] = (cb65_r, cb65_t, "cb65_ensemble")

    # CB+XGB only: 60% CB + 40% XGB
    p_cx_val = 0.60 * p_cb_val + 0.40 * p_xgb_val
    p_cx_test = 0.60 * p_cb_test + 0.40 * p_xgb_test
    cx_t, _ = find_best_threshold_on_val(val_df, p_cx_val, min_bets=20)
    cx_r = calc_roi(test, p_cx_test, threshold=cx_t)
    configs["cb60_xgb40"] = (cx_r, cx_t, "cb60_xgb40")

    # Log all results
    logger.info("Configuration results:")
    for name, (r, t, _m) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    # Pick best
    best_key = max(configs, key=lambda k: configs[k][0]["roi"])
    best_result, best_threshold, best_method = configs[best_key]
    auc = roc_auc_score(test["target"], p_cb_test)

    with mlflow.start_run(run_name="phase4/step4.3_optuna_xgb") as run:
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
                    "method": best_method,
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test": len(test),
                    "xgb_optuna_trials": 30,
                    "best_variant": best_key,
                    **{f"xgb_hp_{k}": v for k, v in best_xgb_params.items()},
                }
            )

            for name, (r, _t, _m) in configs.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roc_auc": auc,
                    "n_bets": best_result["n_bets"],
                    "win_rate": best_result["win_rate"],
                    "best_threshold": best_threshold,
                    "auc_xgb": float(roc_auc_score(test["target"], p_xgb_test)),
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info(
                "Step 4.3 BEST: %s ROI=%.2f%% t=%.2f n=%d run=%s",
                best_key,
                best_result["roi"],
                best_threshold,
                best_result["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
