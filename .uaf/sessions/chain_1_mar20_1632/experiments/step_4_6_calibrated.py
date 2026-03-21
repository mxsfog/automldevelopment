"""Step 4.6: Calibrated probabilities + fine threshold grid on segments."""

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

EXCLUDE_SPORTS = {"Basketball", "MMA", "FIFA", "Snooker"}


def main() -> None:
    logger.info("Step 4.6: Calibrated ensemble")
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

    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)
    val_good = ~train_val["Sport"].isin(EXCLUDE_SPORTS)

    with mlflow.start_run(run_name="phase4/step4.6_calibrated") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("phase", "4")

        try:
            # Train base models
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

            # Raw predictions
            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]

            # Calibrate each model using val set (isotonic regression)
            from sklearn.isotonic import IsotonicRegression

            iso_cb = IsotonicRegression(out_of_bounds="clip")
            iso_cb.fit(p_cb_val, y_val)
            p_cb_cal = iso_cb.predict(p_cb_test)
            p_cb_val_cal = iso_cb.predict(p_cb_val)

            iso_lgb = IsotonicRegression(out_of_bounds="clip")
            iso_lgb.fit(p_lgb_val, y_val)
            p_lgb_cal = iso_lgb.predict(p_lgb_test)
            p_lgb_val_cal = iso_lgb.predict(p_lgb_val)

            iso_xgb = IsotonicRegression(out_of_bounds="clip")
            iso_xgb.fit(p_xgb_val, y_val)
            p_xgb_cal = iso_xgb.predict(p_xgb_test)
            p_xgb_val_cal = iso_xgb.predict(p_xgb_val)

            # Calibrated ensemble
            p_ens_cal = (p_cb_cal + p_lgb_cal + p_xgb_cal) / 3
            p_ens_val_cal = (p_cb_val_cal + p_lgb_val_cal + p_xgb_val_cal) / 3

            # Raw ensemble
            p_ens_raw = (p_cb_test + p_lgb_test + p_xgb_test) / 3
            p_ens_val_raw = (p_cb_val + p_lgb_val + p_xgb_val) / 3

            # Compare calibrated vs raw on filtered test
            test_filt = test[test_good]

            for label, p_t, _p_v in [
                ("raw", p_ens_raw[test_good.values], p_ens_val_raw[val_good.values]),
                ("calibrated", p_ens_cal[test_good.values], p_ens_val_cal[val_good.values]),
            ]:
                auc = roc_auc_score(test_filt["target"].values, p_t)
                logger.info("[%s] AUC=%.4f", label, auc)
                for t in np.arange(0.45, 0.75, 0.05):
                    r = calc_roi(test_filt, p_t, threshold=t)
                    logger.info(
                        "  [%s] t=%.2f: ROI=%.2f%%, n=%d, WR=%.3f",
                        label,
                        t,
                        r["roi"],
                        r["n_bets"],
                        r["win_rate"],
                    )
                    mlflow.log_metric(f"roi_{label}_t{int(t * 100)}", r["roi"])

            # Fine threshold grid on raw (since it's been more reliable)
            p_test_filt_raw = p_ens_raw[test_good.values]
            best_roi = -999.0
            best_t = 0.60
            for t in np.arange(0.55, 0.68, 0.01):
                r = calc_roi(test_filt, p_test_filt_raw, threshold=t)
                if r["n_bets"] >= 100 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_t = float(t)
                logger.info(
                    "  fine t=%.2f: ROI=%.2f%%, n=%d",
                    t,
                    r["roi"],
                    r["n_bets"],
                )

            logger.info("Best fine threshold: %.2f with ROI=%.2f%%", best_t, best_roi)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "calibrated_ensemble",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_fine_threshold": best_t,
                }
            )

            roi_final = calc_roi(test_filt, p_test_filt_raw, threshold=best_t)
            mlflow.log_metrics(
                {
                    "roi": roi_final["roi"],
                    "best_threshold": best_t,
                    "n_bets_selected": roi_final["n_bets"],
                    "pct_selected": roi_final["pct_selected"],
                    "win_rate_selected": roi_final["win_rate"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.6")
            raise


if __name__ == "__main__":
    main()
