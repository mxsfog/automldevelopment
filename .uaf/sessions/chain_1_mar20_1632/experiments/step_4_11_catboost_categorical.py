"""Step 4.11: CatBoost with native categorical Sport feature + ensemble."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
    logger.info("Step 4.11: CatBoost with categorical Sport")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    train_fit = add_safe_features(train_fit)
    train_val = add_safe_features(train_val)
    test = add_safe_features(test)

    feature_cols = get_extended_feature_columns()
    cat_indices = [len(feature_cols)]  # Sport is the last column

    val_good = ~train_val["Sport"].isin(EXCLUDE_SPORTS)
    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)

    with mlflow.start_run(run_name="phase4/step4.11_catboost_cat") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("phase", "4")

        try:
            # Prepare data with categorical Sport
            def prepare_cat_data(df_part: pd.DataFrame) -> pd.DataFrame:
                data = df_part[feature_cols].copy()
                data = data.astype(float)
                data = data.fillna(0.0)
                data["Sport"] = df_part["Sport"].fillna("Unknown").astype(str).values
                return data

            x_fit_cat = prepare_cat_data(train_fit)
            x_val_cat = prepare_cat_data(train_val)
            x_test_cat = prepare_cat_data(test)
            y_fit = train_fit["target"].values
            y_val = train_val["target"].values

            # CatBoost with categorical features
            pool_fit = Pool(x_fit_cat, y_fit, cat_features=cat_indices)
            pool_val = Pool(x_val_cat, y_val, cat_features=cat_indices)

            cb_cat = CatBoostClassifier(
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
            cb_cat.fit(pool_fit, eval_set=pool_val, early_stopping_rounds=50)

            pool_test = Pool(x_test_cat, cat_features=cat_indices)
            p_cb_cat_test = cb_cat.predict_proba(pool_test)[:, 1]
            p_cb_cat_val = cb_cat.predict_proba(pool_val)[:, 1]

            # A: CatBoost-cat only, with sport filter
            val_filt = train_val[val_good]
            test_filt = test[test_good]
            p_cat_val_f = p_cb_cat_val[val_good.values]
            p_cat_test_f = p_cb_cat_test[test_good.values]

            thr_cat, _vr_cat = find_best_threshold_on_val(val_filt, p_cat_val_f)
            roi_cat = calc_roi(test_filt, p_cat_test_f, threshold=thr_cat)
            auc_cat = roc_auc_score(test_filt["target"].values, p_cat_test_f)
            logger.info(
                "[A: CatBoost-cat only] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d, best_iter=%d",
                thr_cat,
                roi_cat["roi"],
                auc_cat,
                roi_cat["n_bets"],
                cb_cat.best_iteration_,
            )

            # B: Ensemble with CatBoost-cat + LGB + XGB (LGB/XGB without cat feature)
            x_fit_num = np.nan_to_num(train_fit[feature_cols].values.astype(float), nan=0.0)
            x_val_num = np.nan_to_num(train_val[feature_cols].values.astype(float), nan=0.0)
            x_test_num = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)

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
                x_fit_num,
                y_fit,
                eval_set=[(x_val_num, y_val)],
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
            xgb.fit(x_fit_num, y_fit, eval_set=[(x_val_num, y_val)], verbose=False)

            p_lgb_val = lgb.predict_proba(x_val_num)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test_num)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val_num)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test_num)[:, 1]

            # Ensemble: CatBoost-cat + LGB + XGB
            p_ens_val = (p_cb_cat_val + p_lgb_val + p_xgb_val) / 3
            p_ens_test = (p_cb_cat_test + p_lgb_test + p_xgb_test) / 3
            p_ens_val_f = p_ens_val[val_good.values]
            p_ens_test_f = p_ens_test[test_good.values]

            thr_ens, _vr_ens = find_best_threshold_on_val(val_filt, p_ens_val_f)
            roi_ens = calc_roi(test_filt, p_ens_test_f, threshold=thr_ens)
            auc_ens = roc_auc_score(test_filt["target"].values, p_ens_test_f)
            logger.info(
                "[B: CatBoost-cat+LGB+XGB] thr=%.2f, ROI=%.2f%%, AUC=%.4f, n=%d",
                thr_ens,
                roi_ens["roi"],
                auc_ens,
                roi_ens["n_bets"],
            )

            # C: Baseline ensemble (CatBoost without cat features)
            cb_nocat = CatBoostClassifier(
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
            cb_nocat.fit(x_fit_num, y_fit, eval_set=(x_val_num, y_val), early_stopping_rounds=50)

            p_nocat_val = cb_nocat.predict_proba(x_val_num)[:, 1]
            p_nocat_test = cb_nocat.predict_proba(x_test_num)[:, 1]
            p_ens_nocat_val = (p_nocat_val + p_lgb_val + p_xgb_val) / 3
            p_ens_nocat_test = (p_nocat_test + p_lgb_test + p_xgb_test) / 3
            p_ens_nocat_val_f = p_ens_nocat_val[val_good.values]
            p_ens_nocat_test_f = p_ens_nocat_test[test_good.values]

            thr_nocat, _vr_nocat = find_best_threshold_on_val(val_filt, p_ens_nocat_val_f)
            roi_nocat = calc_roi(test_filt, p_ens_nocat_test_f, threshold=thr_nocat)
            logger.info(
                "[C: baseline ens no-cat] thr=%.2f, ROI=%.2f%%, n=%d",
                thr_nocat,
                roi_nocat["roi"],
                roi_nocat["n_bets"],
            )

            # Compare
            delta_cat = roi_ens["roi"] - roi_nocat["roi"]
            logger.info("Delta (cat vs no-cat ensemble): %.2f%%", delta_cat)

            # Feature importance from CatBoost-cat
            importances = cb_cat.feature_importances_
            col_names = list(x_fit_cat.columns)
            for idx in np.argsort(importances)[::-1][:10]:
                logger.info(
                    "  Feature %s: importance=%.2f%%",
                    col_names[idx],
                    importances[idx],
                )

            # Pick best
            strategies = [
                ("catboost_cat_only", roi_cat, thr_cat, auc_cat),
                ("catboost_cat_ensemble", roi_ens, thr_ens, auc_ens),
                ("baseline_ensemble", roi_nocat, thr_nocat, auc_ens),
            ]
            best_strat = max(strategies, key=lambda s: s[1]["roi"])
            logger.info("Best: %s, ROI=%.2f%%", best_strat[0], best_strat[1]["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "catboost_categorical",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_strategy": best_strat[0],
                    "best_threshold": best_strat[2],
                    "delta_cat_vs_nocat": round(delta_cat, 4),
                }
            )

            for name, r, _t, _a in strategies:
                mlflow.log_metric(f"roi_{name}", r["roi"])

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
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.11")
            raise


if __name__ == "__main__":
    main()
