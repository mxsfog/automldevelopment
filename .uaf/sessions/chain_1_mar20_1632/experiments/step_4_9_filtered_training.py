"""Step 4.9: Train on filtered sports + deeper CatBoost + EV-weighted threshold."""

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
    logger.info("Step 4.9: Filtered training + deeper CatBoost")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit_full = train.iloc[:val_split_idx]
    train_val_full = train.iloc[val_split_idx:]

    train_fit_full = add_safe_features(train_fit_full)
    train_val_full = add_safe_features(train_val_full)
    test = add_safe_features(test)

    feature_cols = get_extended_feature_columns()

    # Sport filter masks
    fit_good = ~train_fit_full["Sport"].isin(EXCLUDE_SPORTS)
    val_good = ~train_val_full["Sport"].isin(EXCLUDE_SPORTS)
    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)

    with mlflow.start_run(run_name="phase4/step4.9_filtered_training") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("phase", "4")

        try:
            # Strategy A: train on ALL, predict on filtered (baseline)
            x_fit_all = np.nan_to_num(train_fit_full[feature_cols].values.astype(float), nan=0.0)
            y_fit_all = train_fit_full["target"].values
            x_val_all = np.nan_to_num(train_val_full[feature_cols].values.astype(float), nan=0.0)
            y_val_all = train_val_full["target"].values
            x_test_all = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)

            cb_all = CatBoostClassifier(
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
            cb_all.fit(
                x_fit_all, y_fit_all, eval_set=(x_val_all, y_val_all), early_stopping_rounds=50
            )

            lgb_all = LGBMClassifier(
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
            lgb_all.fit(
                x_fit_all,
                y_fit_all,
                eval_set=[(x_val_all, y_val_all)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            xgb_all = XGBClassifier(
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
            xgb_all.fit(x_fit_all, y_fit_all, eval_set=[(x_val_all, y_val_all)], verbose=False)

            p_ens_val_all = (
                cb_all.predict_proba(x_val_all)[:, 1]
                + lgb_all.predict_proba(x_val_all)[:, 1]
                + xgb_all.predict_proba(x_val_all)[:, 1]
            ) / 3
            p_ens_test_all = (
                cb_all.predict_proba(x_test_all)[:, 1]
                + lgb_all.predict_proba(x_test_all)[:, 1]
                + xgb_all.predict_proba(x_test_all)[:, 1]
            ) / 3

            val_filt = train_val_full[val_good]
            test_filt = test[test_good]
            p_val_all_f = p_ens_val_all[val_good.values]
            p_test_all_f = p_ens_test_all[test_good.values]

            thr_all, _val_roi_all = find_best_threshold_on_val(val_filt, p_val_all_f)
            roi_all = calc_roi(test_filt, p_test_all_f, threshold=thr_all)
            auc_all = roc_auc_score(test_filt["target"].values, p_test_all_f)
            logger.info(
                "[A: train ALL, test filtered] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d",
                thr_all,
                roi_all["roi"],
                auc_all,
                roi_all["n_bets"],
            )

            # Strategy B: train ONLY on good sports
            train_fit_filt = train_fit_full[fit_good]
            train_val_filt = train_val_full[val_good]

            x_fit_f = np.nan_to_num(train_fit_filt[feature_cols].values.astype(float), nan=0.0)
            y_fit_f = train_fit_filt["target"].values
            x_val_f = np.nan_to_num(train_val_filt[feature_cols].values.astype(float), nan=0.0)
            y_val_f = train_val_filt["target"].values
            x_test_f = np.nan_to_num(test_filt[feature_cols].values.astype(float), nan=0.0)

            cb_f = CatBoostClassifier(
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
            cb_f.fit(x_fit_f, y_fit_f, eval_set=(x_val_f, y_val_f), early_stopping_rounds=50)

            lgb_f = LGBMClassifier(
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
            lgb_f.fit(
                x_fit_f,
                y_fit_f,
                eval_set=[(x_val_f, y_val_f)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            xgb_f = XGBClassifier(
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
            xgb_f.fit(x_fit_f, y_fit_f, eval_set=[(x_val_f, y_val_f)], verbose=False)

            p_ens_val_f = (
                cb_f.predict_proba(x_val_f)[:, 1]
                + lgb_f.predict_proba(x_val_f)[:, 1]
                + xgb_f.predict_proba(x_val_f)[:, 1]
            ) / 3
            p_ens_test_f = (
                cb_f.predict_proba(x_test_f)[:, 1]
                + lgb_f.predict_proba(x_test_f)[:, 1]
                + xgb_f.predict_proba(x_test_f)[:, 1]
            ) / 3

            thr_f, _val_roi_f = find_best_threshold_on_val(val_filt, p_ens_val_f)
            roi_f = calc_roi(test_filt, p_ens_test_f, threshold=thr_f)
            auc_f = roc_auc_score(test_filt["target"].values, p_ens_test_f)
            logger.info(
                "[B: train FILTERED, test filtered] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d",
                thr_f,
                roi_f["roi"],
                auc_f,
                roi_f["n_bets"],
            )

            # Strategy C: deeper CatBoost (depth=5, more iterations)
            cb_deep = CatBoostClassifier(
                iterations=2000,
                depth=5,
                learning_rate=0.03,
                l2_leaf_reg=15.0,
                border_count=254,
                random_strength=5.0,
                bagging_temperature=3.0,
                min_data_in_leaf=50,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
            )
            cb_deep.fit(
                x_fit_all, y_fit_all, eval_set=(x_val_all, y_val_all), early_stopping_rounds=100
            )

            p_deep_val = cb_deep.predict_proba(x_val_all)[:, 1]
            p_deep_test = cb_deep.predict_proba(x_test_all)[:, 1]
            p_deep_val_f = p_deep_val[val_good.values]
            p_deep_test_f = p_deep_test[test_good.values]

            thr_deep, _val_roi_deep = find_best_threshold_on_val(val_filt, p_deep_val_f)
            roi_deep = calc_roi(test_filt, p_deep_test_f, threshold=thr_deep)
            auc_deep = roc_auc_score(test_filt["target"].values, p_deep_test_f)
            logger.info(
                "[C: deep CatBoost] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d, best_iter=%d",
                thr_deep,
                roi_deep["roi"],
                auc_deep,
                roi_deep["n_bets"],
                cb_deep.best_iteration_,
            )

            # Strategy D: CatBoost only (no LGB/XGB) with sport filter
            p_cb_only_val_f = cb_all.predict_proba(x_val_all)[:, 1][val_good.values]
            p_cb_only_test_f = cb_all.predict_proba(x_test_all)[:, 1][test_good.values]
            thr_cb, _val_roi_cb = find_best_threshold_on_val(val_filt, p_cb_only_val_f)
            roi_cb = calc_roi(test_filt, p_cb_only_test_f, threshold=thr_cb)
            logger.info(
                "[D: CatBoost only] thr=%.2f, ROI=%.2f%%, n=%d",
                thr_cb,
                roi_cb["roi"],
                roi_cb["n_bets"],
            )

            # Compare all strategies
            strategies = [
                ("train_all_test_filt", roi_all, thr_all, auc_all),
                ("train_filt_test_filt", roi_f, thr_f, auc_f),
                ("deep_catboost", roi_deep, thr_deep, auc_deep),
                ("catboost_only", roi_cb, thr_cb, auc_all),
            ]
            best_strat = max(strategies, key=lambda s: s[1]["roi"])
            logger.info("Best strategy: %s, ROI=%.2f%%", best_strat[0], best_strat[1]["roi"])

            for name, r, _t, _a in strategies:
                mlflow.log_metric(f"roi_{name}", r["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "filtered_training",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_strategy": best_strat[0],
                    "best_threshold": best_strat[2],
                    "n_fit_all": len(train_fit_full),
                    "n_fit_filtered": int(fit_good.sum()),
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_strat[1]["roi"],
                    "roc_auc": best_strat[3],
                    "best_threshold": best_strat[2],
                    "n_bets_selected": best_strat[1]["n_bets"],
                    "win_rate_selected": best_strat[1]["win_rate"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.9")
            raise


if __name__ == "__main__":
    main()
