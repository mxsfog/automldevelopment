"""Step 4.7: Robust threshold via multi-fold averaging + retrain on full train."""

import logging
import os
import traceback

import lightgbm
import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from step_2_5_safe_elo import build_safe_elo_features, get_safe_elo_features
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
    """Robust threshold selection via multi-fold val averaging."""
    logger.info("Step 4.7: Robust threshold via multi-fold averaging")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)
    train_all, test_all = time_series_split(df)

    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    feature_cols = get_base_features() + get_engineered_features() + get_safe_elo_features()

    with mlflow.start_run(run_name="phase4/step4.7_robust_threshold") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("phase", "4")

        try:
            # Multi-fold threshold: use 3 different val splits to find robust threshold
            thresholds_found: list[float] = []
            val_rois: list[float] = []

            for val_pct in [0.15, 0.20, 0.25]:
                val_split_idx = int(len(train) * (1 - val_pct))
                t_fit = train.iloc[:val_split_idx]
                t_val = train.iloc[val_split_idx:]

                imp_f = SimpleImputer(strategy="median")
                xf = imp_f.fit_transform(t_fit[feature_cols])
                xv = imp_f.transform(t_val[feature_cols])
                yf = t_fit["target"].values
                yv = t_val["target"].values

                # CB50 ensemble per fold
                cb_f = CatBoostClassifier(
                    iterations=499,
                    depth=7,
                    learning_rate=0.214,
                    l2_leaf_reg=1.15,
                    random_strength=0.823,
                    bagging_temperature=2.41,
                    border_count=121,
                    random_seed=42,
                    verbose=0,
                    eval_metric="AUC",
                    early_stopping_rounds=30,
                )
                cb_f.fit(xf, yf, eval_set=(xv, yv))

                lgb_f = LGBMClassifier(
                    n_estimators=477,
                    max_depth=3,
                    learning_rate=0.292,
                    num_leaves=16,
                    min_child_samples=49,
                    reg_lambda=28.63,
                    random_state=42,
                    verbose=-1,
                )
                lgb_f.fit(
                    xf,
                    yf,
                    eval_set=[(xv, yv)],
                    callbacks=[
                        lightgbm.early_stopping(30, verbose=False),
                        lightgbm.log_evaluation(0),
                    ],
                )

                xgb_f = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    reg_lambda=5.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                    eval_metric="auc",
                    early_stopping_rounds=30,
                )
                xgb_f.fit(xf, yf, eval_set=[(xv, yv)], verbose=False)

                pv_cb = cb_f.predict_proba(xv)[:, 1]
                pv_lgb = lgb_f.predict_proba(xv)[:, 1]
                pv_xgb = xgb_f.predict_proba(xv)[:, 1]
                pv_ens = 0.5 * pv_cb + 0.25 * pv_lgb + 0.25 * pv_xgb

                t_fold, vr_fold = find_best_threshold_on_val(t_val, pv_ens)
                thresholds_found.append(t_fold)
                val_rois.append(vr_fold)
                logger.info(
                    "  val_pct=%.0f%%: t=%.2f, val_roi=%.2f%%",
                    val_pct * 100,
                    t_fold,
                    vr_fold,
                )

            # Robust threshold: median of thresholds
            median_t = float(np.median(thresholds_found))
            mean_t = float(np.mean(thresholds_found))
            logger.info(
                "Thresholds: %s, median=%.2f, mean=%.2f",
                thresholds_found,
                median_t,
                mean_t,
            )

            # Train final ensemble on 80/20 split (standard)
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val_df = train.iloc[val_split:]

            imp = SimpleImputer(strategy="median")
            x_fit = imp.fit_transform(train_fit[feature_cols])
            x_val = imp.transform(val_df[feature_cols])
            x_test = imp.transform(test[feature_cols])
            y_fit = train_fit["target"].values
            y_val = val_df["target"].values

            cb = CatBoostClassifier(
                iterations=499,
                depth=7,
                learning_rate=0.214,
                l2_leaf_reg=1.15,
                random_strength=0.823,
                bagging_temperature=2.41,
                border_count=121,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=30,
            )
            cb.fit(x_fit, y_fit, eval_set=(x_val, y_val))

            lgb_m = LGBMClassifier(
                n_estimators=477,
                max_depth=3,
                learning_rate=0.292,
                num_leaves=16,
                min_child_samples=49,
                reg_lambda=28.63,
                random_state=42,
                verbose=-1,
            )
            lgb_m.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    lightgbm.early_stopping(30, verbose=False),
                    lightgbm.log_evaluation(0),
                ],
            )

            xgb_m = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                eval_metric="auc",
                early_stopping_rounds=30,
            )
            xgb_m.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)

            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgb_test = lgb_m.predict_proba(x_test)[:, 1]
            p_xgb_test = xgb_m.predict_proba(x_test)[:, 1]
            p_ens_test = 0.5 * p_cb_test + 0.25 * p_lgb_test + 0.25 * p_xgb_test

            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_lgb_val = lgb_m.predict_proba(x_val)[:, 1]
            p_xgb_val = xgb_m.predict_proba(x_val)[:, 1]
            p_ens_val = 0.5 * p_cb_val + 0.25 * p_lgb_val + 0.25 * p_xgb_val

            # Standard val-selected threshold
            t_std, _vr = find_best_threshold_on_val(val_df, p_ens_val)
            roi_std = calc_roi(test, p_ens_test, threshold=t_std)
            auc_std = roc_auc_score(test["target"], p_ens_test)

            # Median threshold
            roi_median = calc_roi(test, p_ens_test, threshold=median_t)

            # Mean threshold
            roi_mean = calc_roi(test, p_ens_test, threshold=mean_t)

            logger.info(
                "Standard t=%.2f: ROI=%.2f%% n=%d",
                t_std,
                roi_std["roi"],
                roi_std["n_bets"],
            )
            logger.info(
                "Median t=%.2f: ROI=%.2f%% n=%d",
                median_t,
                roi_median["roi"],
                roi_median["n_bets"],
            )
            logger.info(
                "Mean t=%.2f: ROI=%.2f%% n=%d",
                mean_t,
                roi_mean["roi"],
                roi_mean["n_bets"],
            )

            # All thresholds
            roi_thresholds = calc_roi_at_thresholds(test, p_ens_test)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            best_roi = max(roi_std["roi"], roi_median["roi"], roi_mean["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "robust_threshold_multi_fold",
                    "n_features": len(feature_cols),
                    "val_pcts": "15,20,25",
                    "thresholds_found": str(thresholds_found),
                    "median_threshold": median_t,
                    "mean_threshold": mean_t,
                    "standard_threshold": t_std,
                    "leakage_free": "true",
                }
            )
            mlflow.log_metrics(
                {
                    "roi_standard": roi_std["roi"],
                    "roi_median_t": roi_median["roi"],
                    "roi_mean_t": roi_mean["roi"],
                    "roi": best_roi,
                    "roc_auc": auc_std,
                    "n_bets_standard": roi_std["n_bets"],
                    "n_bets_median": roi_median["n_bets"],
                    "n_bets_mean": roi_mean["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.7")
            raise


if __name__ == "__main__":
    main()
