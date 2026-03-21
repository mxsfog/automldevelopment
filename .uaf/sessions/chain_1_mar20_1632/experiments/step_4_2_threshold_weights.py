"""Step 4.2: Fine-grained threshold + weighted average optimization."""

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


def main() -> None:
    logger.info("Step 4.2: Fine threshold + weighted average")
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

    with mlflow.start_run(run_name="phase4/step4.2_threshold_weights") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")

        try:
            # Train models
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
            xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)

            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]
            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            # Grid search over weights and thresholds on val
            weight_options = [
                (0.5, 0.3, 0.2),
                (0.5, 0.25, 0.25),
                (0.4, 0.3, 0.3),
                (0.4, 0.4, 0.2),
                (1 / 3, 1 / 3, 1 / 3),
                (0.6, 0.2, 0.2),
                (0.6, 0.3, 0.1),
                (0.7, 0.2, 0.1),
                (0.7, 0.15, 0.15),
                (0.8, 0.1, 0.1),
            ]
            thresholds = np.arange(0.40, 0.75, 0.02)

            best_val_roi = -999.0
            best_w = (1 / 3, 1 / 3, 1 / 3)
            best_t = 0.5

            for w in weight_options:
                p_val = w[0] * p_cb_val + w[1] * p_lgb_val + w[2] * p_xgb_val
                for t in thresholds:
                    result = calc_roi(train_val, p_val, threshold=t)
                    if result["n_bets"] >= 30 and result["roi"] > best_val_roi:
                        best_val_roi = result["roi"]
                        best_w = w
                        best_t = float(t)

            logger.info(
                "Best val: w=(%.2f,%.2f,%.2f), thr=%.2f, val_roi=%.2f%%",
                best_w[0],
                best_w[1],
                best_w[2],
                best_t,
                best_val_roi,
            )

            # Apply to test
            p_test = best_w[0] * p_cb_test + best_w[1] * p_lgb_test + best_w[2] * p_xgb_test
            roi_result = calc_roi(test, p_test, threshold=best_t)
            auc = roc_auc_score(y_test, p_test)

            logger.info(
                "Test ROI=%.2f%%, AUC=%.4f, thr=%.2f, n=%d, WR=%.4f",
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                roi_result["win_rate"],
            )

            # Also test equal weights at different thresholds
            p_eq = (p_cb_test + p_lgb_test + p_xgb_test) / 3
            for t in np.arange(0.40, 0.75, 0.05):
                r = calc_roi(test, p_eq, threshold=t)
                logger.info("  eq_avg t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "weighted_avg_threshold",
                    "best_w_cb": best_w[0],
                    "best_w_lgb": best_w[1],
                    "best_w_xgb": best_w[2],
                    "best_threshold": best_t,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "best_threshold": best_t,
                    "val_roi_at_threshold": best_val_roi,
                    "n_bets_selected": roi_result["n_bets"],
                    "pct_selected": roi_result["pct_selected"],
                    "win_rate_selected": roi_result["win_rate"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.2")
            raise


if __name__ == "__main__":
    main()
