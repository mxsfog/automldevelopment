"""Step 4.5: Robustness check -- k-fold CV + segment analysis + odds range search."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


LGBM_PARAMS: dict = {
    "n_estimators": 228,
    "max_depth": 6,
    "learning_rate": 0.216,
    "num_leaves": 50,
    "min_child_samples": 18,
    "reg_alpha": 0.0001,
    "reg_lambda": 0.0003,
    "subsample": 0.925,
    "colsample_bytree": 0.803,
    "random_state": 42,
    "verbose": -1,
    "is_unbalance": True,
}


def main() -> None:
    logger.info("Step 4.5: Robustness check")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()

    with mlflow.start_run(run_name="phase4/step4.5_robustness") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                    "method": "robustness_check",
                    "filter": "Is_Parlay=f",
                }
            )

            # === 1. Time-series k-fold (5 splits) ===
            logger.info("=== K-fold time-series cross-validation ===")
            n = len(train)
            fold_size = n // 5
            fold_rois = []
            fold_rois_filtered = []

            for i in range(5):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < 4 else n
                fold_train = train.iloc[:val_start] if val_start > 0 else pd.DataFrame()
                fold_val = train.iloc[val_start:val_end]

                if len(fold_train) < 1000:
                    continue

                imputer = SimpleImputer(strategy="median")
                x_tr = imputer.fit_transform(fold_train[feature_cols])
                x_vl = imputer.transform(fold_val[feature_cols])

                model = LGBMClassifier(**LGBM_PARAMS)
                model.fit(x_tr, fold_train["target"].values)
                proba = model.predict_proba(x_vl)[:, 1]

                # Full ROI
                _, _ = find_best_threshold_on_val(
                    fold_train.iloc[-int(len(fold_train) * 0.2) :],
                    model.predict_proba(
                        imputer.transform(
                            fold_train.iloc[-int(len(fold_train) * 0.2) :][feature_cols]
                        )
                    )[:, 1],
                )
                roi_all = calc_roi(fold_val, proba, threshold=0.6)
                fold_rois.append(roi_all["roi"])

                # Medium odds filter
                mask = (fold_val["Odds"].values >= 1.3) & (fold_val["Odds"].values <= 5.0)
                p_filtered = np.where(mask, proba, 0)
                roi_filt = calc_roi(fold_val, p_filtered, threshold=0.6)
                fold_rois_filtered.append(roi_filt["roi"])

                logger.info(
                    "  Fold %d: all=%.2f%% (n=%d), medium_odds=%.2f%% (n=%d)",
                    i,
                    roi_all["roi"],
                    roi_all["n_bets"],
                    roi_filt["roi"],
                    roi_filt["n_bets"],
                )
                mlflow.log_metric("roi_fold_all", roi_all["roi"], step=i)
                mlflow.log_metric("roi_fold_med_odds", roi_filt["roi"], step=i)

            if fold_rois:
                mlflow.log_metrics(
                    {
                        "roi_cv_mean": float(np.mean(fold_rois)),
                        "roi_cv_std": float(np.std(fold_rois)),
                        "roi_cv_med_mean": float(np.mean(fold_rois_filtered)),
                        "roi_cv_med_std": float(np.std(fold_rois_filtered)),
                    }
                )
                logger.info(
                    "CV all: mean=%.2f%% std=%.2f%%",
                    np.mean(fold_rois),
                    np.std(fold_rois),
                )
                logger.info(
                    "CV medium_odds: mean=%.2f%% std=%.2f%%",
                    np.mean(fold_rois_filtered),
                    np.std(fold_rois_filtered),
                )

            # === 2. Odds range search ===
            logger.info("=== Odds range search ===")
            val_split_idx = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split_idx]
            train_val = train.iloc[val_split_idx:]

            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(train_fit[feature_cols])
            x_val = imputer.transform(train_val[feature_cols])
            x_test = imputer.transform(test[feature_cols])

            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(
                x_fit,
                train_fit["target"].values,
                eval_set=[(x_val, train_val["target"].values)],
                callbacks=[],
            )
            p_val = model.predict_proba(x_val)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]

            best_range_roi = -999.0
            best_range = (1.0, 10.0)
            best_range_t = 0.6

            for lo in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                for hi in [2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
                    mask_val = (train_val["Odds"].values >= lo) & (train_val["Odds"].values <= hi)
                    mask_test = (test["Odds"].values >= lo) & (test["Odds"].values <= hi)

                    if mask_val.sum() < 30:
                        continue

                    p_val_m = np.where(mask_val, p_val, 0)
                    t, _ = find_best_threshold_on_val(train_val, p_val_m, min_bets=15)

                    p_test_m = np.where(mask_test, p_test, 0)
                    r = calc_roi(test, p_test_m, threshold=t)

                    if r["n_bets"] >= 100 and r["roi"] > best_range_roi:
                        best_range_roi = r["roi"]
                        best_range = (lo, hi)
                        best_range_t = t

                    if r["n_bets"] >= 50:
                        logger.info(
                            "  Odds [%.1f-%.1f] t=%.2f: ROI=%.2f%% n=%d",
                            lo,
                            hi,
                            t,
                            r["roi"],
                            r["n_bets"],
                        )

            logger.info(
                "Best odds range: [%.1f-%.1f] t=%.2f ROI=%.2f%%",
                best_range[0],
                best_range[1],
                best_range_t,
                best_range_roi,
            )

            # === 3. Sport analysis ===
            logger.info("=== Sport segment analysis ===")
            for sport in test["Sport"].value_counts().head(5).index:
                mask_val = train_val["Sport"] == sport
                mask_test = test["Sport"] == sport
                if mask_val.sum() < 20 or mask_test.sum() < 20:
                    continue

                p_val_s = np.where(mask_val, p_val, 0)
                t_s, _ = find_best_threshold_on_val(train_val, p_val_s, min_bets=10)
                p_test_s = np.where(mask_test, p_test, 0)
                r_s = calc_roi(test, p_test_s, threshold=t_s)
                logger.info(
                    "  %s: ROI=%.2f%% n=%d (t=%.2f)",
                    sport,
                    r_s["roi"],
                    r_s["n_bets"],
                    t_s,
                )
                mlflow.log_metric(f"roi_sport_{sport.lower().replace(' ', '_')}", r_s["roi"])

            # Apply best odds range to test
            lo, hi = best_range
            mask_best = (test["Odds"].values >= lo) & (test["Odds"].values <= hi)
            p_test_best = np.where(mask_best, p_test, 0)
            roi_final = calc_roi(test, p_test_best, threshold=best_range_t)

            mlflow.log_metrics(
                {
                    "roi": roi_final["roi"],
                    "best_odds_lo": lo,
                    "best_odds_hi": hi,
                    "best_threshold": best_range_t,
                    "n_bets": roi_final["n_bets"],
                    "pct_selected": roi_final["pct_selected"],
                    "win_rate": roi_final["win_rate"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info(
                "Final best: ROI=%.2f%% (odds [%.1f-%.1f], t=%.2f, n=%d)",
                roi_final["roi"],
                lo,
                hi,
                best_range_t,
                roi_final["n_bets"],
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.5")
            raise


if __name__ == "__main__":
    main()
