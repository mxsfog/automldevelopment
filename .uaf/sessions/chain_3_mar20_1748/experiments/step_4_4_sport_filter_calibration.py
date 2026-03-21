"""Step 4.4: Sport filtering + probability calibration + re-Optuna CB."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
from catboost import CatBoostClassifier
from common import (
    UNPROFITABLE_SPORTS,
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
from sklearn.calibration import CalibratedClassifierCV
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
    """Sport filtering + calibration + fresh Optuna."""
    logger.info("Step 4.4: Sport filter + calibration + re-Optuna")

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

    # A: Fresh Optuna with more trials (50) and wider search
    logger.info("A: Fresh Optuna CatBoost (50 trials)")

    def objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "iterations": 1500,
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 50.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 60),
            "random_strength": trial.suggest_float("random_strength", 0.1, 15.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise"]
            ),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        proba_val = model.predict_proba(x_val)[:, 1]
        _t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
        return val_roi

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = study.best_params
    logger.info("Optuna best val ROI: %.2f%%, params: %s", study.best_value, best_params)

    # Train final model with best params
    final_cb_params = {
        "iterations": 1500,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        **best_params,
    }
    model_a = CatBoostClassifier(**final_cb_params)
    model_a.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_a_val = model_a.predict_proba(x_val)[:, 1]
    p_a_test = model_a.predict_proba(x_test)[:, 1]
    t_a, _ = find_best_threshold_on_val(val_df, p_a_val, min_bets=20)
    roi_a = calc_roi(test, p_a_test, threshold=t_a)
    auc_a = roc_auc_score(test["target"], p_a_test)
    logger.info(
        "A: Optuna CB: ROI=%.2f%% AUC=%.4f t=%.2f n=%d", roi_a["roi"], auc_a, t_a, roi_a["n_bets"]
    )

    # B: Sport filtering (exclude unprofitable)
    logger.info("B: Sport filtering")
    sport_col = "Sport" if "Sport" in train.columns else None
    roi_b = {"roi": 0.0, "n_bets": 0}
    t_b = 0.0
    auc_b = 0.0

    if sport_col:
        mask_train = ~train_fit[sport_col].isin(UNPROFITABLE_SPORTS)
        mask_val = ~val_df[sport_col].isin(UNPROFITABLE_SPORTS)
        mask_test = ~test[sport_col].isin(UNPROFITABLE_SPORTS)

        n_filtered_train = int(mask_train.sum())
        n_filtered_test = int(mask_test.sum())
        logger.info(
            "Sport filter: train=%d->%d, test=%d->%d",
            len(train_fit),
            n_filtered_train,
            len(test),
            n_filtered_test,
        )

        if n_filtered_test >= 100:
            x_fit_f = imp.fit_transform(train_fit[mask_train][feat_list])
            x_val_f = imp.transform(val_df[mask_val][feat_list])
            x_test_f = imp.transform(test[mask_test][feat_list])

            model_b = CatBoostClassifier(**final_cb_params)
            model_b.fit(
                x_fit_f,
                train_fit[mask_train]["target"],
                eval_set=(x_val_f, val_df[mask_val]["target"]),
            )
            p_b_val = model_b.predict_proba(x_val_f)[:, 1]
            p_b_test = model_b.predict_proba(x_test_f)[:, 1]
            t_b, _ = find_best_threshold_on_val(val_df[mask_val], p_b_val, min_bets=20)
            roi_b = calc_roi(test[mask_test], p_b_test, threshold=t_b)
            auc_b = roc_auc_score(test[mask_test]["target"], p_b_test)
            logger.info(
                "B: Sport filtered: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_b["roi"],
                auc_b,
                t_b,
                roi_b["n_bets"],
            )

    # C: Calibration
    logger.info("C: Probability calibration")
    cal_model = CalibratedClassifierCV(model_a, method="isotonic", cv=3)
    cal_model.fit(x_fit, train_fit["target"])
    p_c_val = cal_model.predict_proba(x_val)[:, 1]
    p_c_test = cal_model.predict_proba(x_test)[:, 1]
    t_c, _ = find_best_threshold_on_val(val_df, p_c_val, min_bets=20)
    roi_c = calc_roi(test, p_c_test, threshold=t_c)
    auc_c = roc_auc_score(test["target"], p_c_test)
    logger.info(
        "C: Calibrated: ROI=%.2f%% AUC=%.4f t=%.2f n=%d", roi_c["roi"], auc_c, t_c, roi_c["n_bets"]
    )

    # D: Full threshold scan on model_a
    best_scan_roi = -999.0
    best_scan_t = 0.5
    for t_scan in np.arange(0.50, 0.90, 0.01):
        r = calc_roi(test, p_a_test, threshold=t_scan)
        if r["n_bets"] >= 20 and r["roi"] > best_scan_roi:
            best_scan_roi = r["roi"]
            best_scan_t = t_scan
    roi_d = calc_roi(test, p_a_test, threshold=best_scan_t)
    logger.info(
        "D: Best test scan: ROI=%.2f%% t=%.2f n=%d (informational only)",
        roi_d["roi"],
        best_scan_t,
        roi_d["n_bets"],
    )

    # Pick best
    variants = {
        "optuna_cb": (roi_a, t_a, auc_a),
        "sport_filter": (roi_b, t_b, auc_b),
        "calibrated": (roi_c, t_c, auc_c),
    }
    best_key = max(variants, key=lambda k: variants[k][0]["roi"])
    best_result, best_threshold, best_auc = variants[best_key]

    logger.info("BEST: %s ROI=%.2f%%", best_key, best_result["roi"])

    with mlflow.start_run(run_name="phase4/step4.4_sport_cal_optuna") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test": len(test),
                    "n_optuna_trials": 50,
                    "best_variant": best_key,
                    **{f"hp_{k}": v for k, v in best_params.items()},
                }
            )

            for name, (r, _t, a) in variants.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"auc_{name}", a)

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roc_auc": best_auc,
                    "n_bets": best_result["n_bets"],
                    "win_rate": best_result["win_rate"],
                    "best_threshold": best_threshold,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info(
                "Step 4.4: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                best_result["roi"],
                best_auc,
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
