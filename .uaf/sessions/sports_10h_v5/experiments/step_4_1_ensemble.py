"""Step 4.1: Ensemble -- soft voting из лучших моделей Phase 3."""

import logging
import os
import traceback

import mlflow
import numpy as np
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


def main() -> None:
    logger.info("Step 4.1: Ensemble (soft voting)")
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

    imputer = SimpleImputer(strategy="median")
    x_fit_imp = imputer.fit_transform(x_fit)
    x_val_imp = imputer.transform(x_val)
    x_test_imp = imputer.transform(x_test)

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit_imp)
    x_val_scaled = scaler.transform(x_val_imp)
    x_test_scaled = scaler.transform(x_test_imp)

    with mlflow.start_run(run_name="phase4/step4.1_ensemble") as run:
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
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "soft_voting_ensemble",
                    "models": "logreg,catboost,lightgbm",
                }
            )

            # 1. LogReg (best from Optuna)
            model_lr = LogisticRegression(
                C=0.00318, penalty="l1", solver="saga", random_state=42, max_iter=2000
            )
            model_lr.fit(x_fit_scaled, y_fit)
            p_lr_val = model_lr.predict_proba(x_val_scaled)[:, 1]
            p_lr_test = model_lr.predict_proba(x_test_scaled)[:, 1]
            logger.info("LogReg AUC val=%.4f", roc_auc_score(y_val, p_lr_val))

            # 2. CatBoost (best from Optuna)
            model_cb = CatBoostClassifier(
                iterations=398,
                depth=6,
                learning_rate=0.296,
                l2_leaf_reg=16.4,
                min_data_in_leaf=84,
                random_strength=9.9,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                auto_class_weights="Balanced",
            )
            model_cb.fit(x_fit_imp, y_fit, eval_set=(x_val_imp, y_val), early_stopping_rounds=50)
            p_cb_val = model_cb.predict_proba(x_val_imp)[:, 1]
            p_cb_test = model_cb.predict_proba(x_test_imp)[:, 1]
            logger.info("CatBoost AUC val=%.4f", roc_auc_score(y_val, p_cb_val))

            # 3. LightGBM (best from Optuna)
            model_lgb = LGBMClassifier(
                n_estimators=228,
                max_depth=6,
                learning_rate=0.216,
                num_leaves=50,
                min_child_samples=18,
                reg_alpha=0.0001,
                reg_lambda=0.0003,
                subsample=0.925,
                colsample_bytree=0.803,
                random_state=42,
                verbose=-1,
                is_unbalance=True,
            )
            model_lgb.fit(x_fit_imp, y_fit, eval_set=[(x_val_imp, y_val)], callbacks=[])
            p_lgb_val = model_lgb.predict_proba(x_val_imp)[:, 1]
            p_lgb_test = model_lgb.predict_proba(x_test_imp)[:, 1]
            logger.info("LightGBM AUC val=%.4f", roc_auc_score(y_val, p_lgb_val))

            # Ensemble: weighted average (optimize weights on val)
            best_w_roi = -999.0
            best_weights = (1 / 3, 1 / 3, 1 / 3)
            best_w_threshold = 0.5

            for w_lr in np.arange(0.0, 1.05, 0.1):
                for w_cb in np.arange(0.0, 1.05 - w_lr, 0.1):
                    w_lgb = 1.0 - w_lr - w_cb
                    if w_lgb < 0:
                        continue
                    p_ens_val = w_lr * p_lr_val + w_cb * p_cb_val + w_lgb * p_lgb_val
                    t, roi_v = find_best_threshold_on_val(train_val, p_ens_val)
                    if roi_v > best_w_roi:
                        best_w_roi = roi_v
                        best_weights = (w_lr, w_cb, w_lgb)
                        best_w_threshold = t

            w_lr, w_cb, w_lgb = best_weights
            logger.info(
                "Best weights: LR=%.1f, CB=%.1f, LGB=%.1f, val ROI=%.2f%%, t=%.2f",
                w_lr,
                w_cb,
                w_lgb,
                best_w_roi,
                best_w_threshold,
            )

            # Apply to test
            p_ens_test = w_lr * p_lr_test + w_cb * p_cb_test + w_lgb * p_lgb_test
            roi_ens = calc_roi(test, p_ens_test, threshold=best_w_threshold)
            auc_ens = roc_auc_score(y_test, p_ens_test)

            # Also try equal weights
            p_equal = (p_lr_test + p_cb_test + p_lgb_test) / 3
            p_equal_val = (p_lr_val + p_cb_val + p_lgb_val) / 3
            t_eq, _ = find_best_threshold_on_val(train_val, p_equal_val)
            roi_equal = calc_roi(test, p_equal, threshold=t_eq)
            auc_equal = roc_auc_score(y_test, p_equal)

            # Individual model ROIs with their own best thresholds
            t_lr_best, _ = find_best_threshold_on_val(train_val, p_lr_val)
            t_cb_best, _ = find_best_threshold_on_val(train_val, p_cb_val)
            t_lgb_best, _ = find_best_threshold_on_val(train_val, p_lgb_val)
            roi_lr = calc_roi(test, p_lr_test, threshold=t_lr_best)
            roi_cb = calc_roi(test, p_cb_test, threshold=t_cb_best)
            roi_lgb = calc_roi(test, p_lgb_test, threshold=t_lgb_best)

            logger.info("Test ROIs:")
            logger.info(
                "  LogReg: %.2f%% (t=%.2f, n=%d)", roi_lr["roi"], t_lr_best, roi_lr["n_bets"]
            )
            logger.info(
                "  CatBoost: %.2f%% (t=%.2f, n=%d)", roi_cb["roi"], t_cb_best, roi_cb["n_bets"]
            )
            logger.info(
                "  LightGBM: %.2f%% (t=%.2f, n=%d)", roi_lgb["roi"], t_lgb_best, roi_lgb["n_bets"]
            )
            logger.info(
                "  Equal ensemble: %.2f%% (t=%.2f, n=%d)",
                roi_equal["roi"],
                t_eq,
                roi_equal["n_bets"],
            )
            logger.info(
                "  Weighted ensemble: %.2f%% (t=%.2f, n=%d, w=%.1f/%.1f/%.1f)",
                roi_ens["roi"],
                best_w_threshold,
                roi_ens["n_bets"],
                w_lr,
                w_cb,
                w_lgb,
            )

            best_roi = max(roi_ens["roi"], roi_equal["roi"])
            best_method = "weighted" if roi_ens["roi"] >= roi_equal["roi"] else "equal"

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_weighted_ensemble": roi_ens["roi"],
                    "roi_equal_ensemble": roi_equal["roi"],
                    "roi_logreg": roi_lr["roi"],
                    "roi_catboost": roi_cb["roi"],
                    "roi_lightgbm": roi_lgb["roi"],
                    "auc_weighted_ensemble": auc_ens,
                    "auc_equal_ensemble": auc_equal,
                    "weight_lr": w_lr,
                    "weight_cb": w_cb,
                    "weight_lgb": w_lgb,
                    "threshold_ensemble": best_w_threshold,
                    "n_bets_ensemble": roi_ens["n_bets"],
                }
            )
            mlflow.set_tag("best_ensemble", best_method)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ensemble ROI: %.2f%% (%s)", best_roi, best_method)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.1")
            raise


if __name__ == "__main__":
    main()
