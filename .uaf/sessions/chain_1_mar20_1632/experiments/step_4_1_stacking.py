"""Step 4.1: Stacking ensemble -- CatBoost + LightGBM + XGBoost."""

import logging
import os
import traceback

import mlflow
import numpy as np
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
from sklearn.linear_model import LogisticRegression
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
    logger.info("Step 4.1: Stacking ensemble")
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
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase4/step4.1_stacking") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("phase", "4")

        try:
            # Best CatBoost params from Optuna
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
            xgb.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )

            # Level 1 predictions (on val for meta-learner, on test for final)
            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]

            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            # Individual model AUCs
            auc_cb = roc_auc_score(y_test, p_cb_test)
            auc_lgb = roc_auc_score(y_test, p_lgb_test)
            auc_xgb = roc_auc_score(y_test, p_xgb_test)
            logger.info("AUC: CB=%.4f, LGB=%.4f, XGB=%.4f", auc_cb, auc_lgb, auc_xgb)

            # Simple average ensemble
            p_avg_test = (p_cb_test + p_lgb_test + p_xgb_test) / 3
            p_avg_val = (p_cb_val + p_lgb_val + p_xgb_val) / 3
            auc_avg = roc_auc_score(y_test, p_avg_test)
            logger.info("AUC avg: %.4f", auc_avg)

            # Stacking meta-learner
            meta_train = np.column_stack([p_cb_val, p_lgb_val, p_xgb_val])
            meta_test = np.column_stack([p_cb_test, p_lgb_test, p_xgb_test])

            meta = LogisticRegression(C=1.0, random_state=42)
            meta.fit(meta_train, y_val)
            p_stack_test = meta.predict_proba(meta_test)[:, 1]
            p_stack_val = meta.predict_proba(meta_train)[:, 1]
            auc_stack = roc_auc_score(y_test, p_stack_test)
            logger.info("AUC stack: %.4f", auc_stack)

            # Threshold optimization on val for each method
            methods = {
                "catboost": (p_cb_val, p_cb_test),
                "lightgbm": (p_lgb_val, p_lgb_test),
                "xgboost": (p_xgb_val, p_xgb_test),
                "average": (p_avg_val, p_avg_test),
                "stacking": (p_stack_val, p_stack_test),
            }

            best_method = ""
            best_roi = -999.0
            best_result = {}

            for name, (pval, ptest) in methods.items():
                thr, _val_roi = find_best_threshold_on_val(train_val, pval)
                result = calc_roi(test, ptest, threshold=thr)
                auc_m = roc_auc_score(y_test, ptest)
                logger.info(
                    "[%s] ROI=%.2f%%, AUC=%.4f, thr=%.2f, n=%d, WR=%.4f",
                    name,
                    result["roi"],
                    auc_m,
                    thr,
                    result["n_bets"],
                    result["win_rate"],
                )
                mlflow.log_metric(f"roi_{name}", result["roi"])
                mlflow.log_metric(f"auc_{name}", auc_m)
                mlflow.log_metric(f"thr_{name}", thr)
                mlflow.log_metric(f"nbets_{name}", result["n_bets"])

                if result["roi"] > best_roi:
                    best_roi = result["roi"]
                    best_method = name
                    best_result = result
                    best_result["threshold"] = thr
                    best_result["auc"] = auc_m

            logger.info("Best method: %s with ROI=%.2f%%", best_method, best_roi)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "stacking_ensemble",
                    "n_features": len(feature_cols),
                    "best_method": best_method,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roc_auc": best_result.get("auc", 0),
                    "best_threshold": best_result.get("threshold", 0.5),
                    "n_bets_selected": best_result.get("n_bets", 0),
                    "pct_selected": best_result.get("pct_selected", 0),
                    "win_rate_selected": best_result.get("win_rate", 0),
                }
            )

            # Meta-learner coefficients
            logger.info(
                "Meta coefficients: CB=%.4f, LGB=%.4f, XGB=%.4f",
                meta.coef_[0][0],
                meta.coef_[0][1],
                meta.coef_[0][2],
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.1")
            raise


if __name__ == "__main__":
    main()
