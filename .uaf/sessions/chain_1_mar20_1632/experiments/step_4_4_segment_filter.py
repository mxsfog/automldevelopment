"""Step 4.4: Segment filter -- исключаем убыточные спорты и оптимизируем."""

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

# Убыточные спорты из segment analysis (step 4.3)
EXCLUDE_SPORTS = {"Basketball", "MMA", "FIFA", "Snooker"}


def main() -> None:
    logger.info("Step 4.4: Segment filter (exclude bad sports)")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit_full = train.iloc[:val_split_idx]
    train_val_full = train.iloc[val_split_idx:]

    train_fit_full = add_safe_features(train_fit_full)
    train_val_full = add_safe_features(train_val_full)
    test = add_safe_features(test)

    feature_cols = get_extended_feature_columns()

    with mlflow.start_run(run_name="phase4/step4.4_segment_filter") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("phase", "4")

        try:
            # Strategy A: train on all, predict on all, filter test results
            x_fit = np.nan_to_num(train_fit_full[feature_cols].values.astype(float), nan=0.0)
            y_fit = train_fit_full["target"].values
            x_val = np.nan_to_num(train_val_full[feature_cols].values.astype(float), nan=0.0)
            y_val = train_val_full["target"].values
            x_test = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)

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

            p_test = (
                cb.predict_proba(x_test)[:, 1]
                + lgb.predict_proba(x_test)[:, 1]
                + xgb.predict_proba(x_test)[:, 1]
            ) / 3

            # Threshold on val (exclude bad sports from val too)
            p_val = (
                cb.predict_proba(x_val)[:, 1]
                + lgb.predict_proba(x_val)[:, 1]
                + xgb.predict_proba(x_val)[:, 1]
            ) / 3

            # Unfiltered baseline
            thr_base, _ = find_best_threshold_on_val(train_val_full, p_val)
            roi_base = calc_roi(test, p_test, threshold=thr_base)
            logger.info(
                "[unfiltered] ROI=%.2f%%, n=%d, thr=%.2f",
                roi_base["roi"],
                roi_base["n_bets"],
                thr_base,
            )

            # Filter: exclude bad sports from val for threshold, then from test
            val_good_mask = ~train_val_full["Sport"].isin(EXCLUDE_SPORTS)
            val_filtered = train_val_full[val_good_mask]
            p_val_filtered = p_val[val_good_mask.values]
            thr_filt, val_roi_filt = find_best_threshold_on_val(val_filtered, p_val_filtered)

            test_good_mask = ~test["Sport"].isin(EXCLUDE_SPORTS)
            test_filtered = test[test_good_mask]
            p_test_filtered = p_test[test_good_mask.values]
            roi_filt = calc_roi(test_filtered, p_test_filtered, threshold=thr_filt)
            logger.info(
                "[filtered excl %s] ROI=%.2f%%, n=%d, thr=%.2f",
                EXCLUDE_SPORTS,
                roi_filt["roi"],
                roi_filt["n_bets"],
                thr_filt,
            )

            # Try different thresholds on filtered set
            for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
                r = calc_roi(test_filtered, p_test_filtered, threshold=t)
                logger.info("  filtered t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            # Strategy B: also exclude odds >= 2.5 (only favorites)
            fav_mask = test_filtered["Odds"] < 2.5
            test_fav = test_filtered[fav_mask]
            p_test_fav = p_test_filtered[fav_mask.values]
            for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
                r = calc_roi(test_fav, p_test_fav, threshold=t)
                logger.info("  filtered+fav t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            # Strategy C: only odds 1.5-1.8 (best segment)
            mid_mask = (test_filtered["Odds"] >= 1.5) & (test_filtered["Odds"] < 1.8)
            test_mid = test_filtered[mid_mask]
            p_test_mid = p_test_filtered[mid_mask.values]
            for t in [0.50, 0.55, 0.60, 0.65]:
                r = calc_roi(test_mid, p_test_mid, threshold=t)
                logger.info("  odds_1.5_1.8 t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "segment_filter",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_threshold": thr_filt,
                    "n_samples_test": len(test),
                    "n_samples_test_filtered": len(test_filtered),
                }
            )

            mlflow.log_metrics(
                {
                    "roi": roi_filt["roi"],
                    "roi_unfiltered": roi_base["roi"],
                    "roi_filtered": roi_filt["roi"],
                    "n_bets_selected": roi_filt["n_bets"],
                    "pct_selected": roi_filt["pct_selected"],
                    "win_rate_selected": roi_filt["win_rate"],
                    "val_roi_at_threshold": val_roi_filt,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.4")
            raise


if __name__ == "__main__":
    main()
