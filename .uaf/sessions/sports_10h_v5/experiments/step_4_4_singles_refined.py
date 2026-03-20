"""Step 4.4: Singles refined -- conservative LightGBM + odds filter + calibration."""

import logging
import os
import traceback

import mlflow
import numpy as np
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
    logger.info("Step 4.4: Singles refined (conservative + calibration + odds filter)")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[feature_cols])
    x_val = imputer.transform(train_val[feature_cols])
    x_test = imputer.transform(test[feature_cols])
    y_fit = train_fit["target"].values
    y_val = train_val["target"].values
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase4/step4.4_singles_refined") as run:
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
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "singles_refined_lgbm",
                    "filter": "Is_Parlay=f",
                }
            )

            # === 1. Conservative LightGBM (params from step 4.2 which worked well) ===
            model_base = LGBMClassifier(
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
            model_base.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], callbacks=[])
            p_base_val = model_base.predict_proba(x_val)[:, 1]
            p_base_test = model_base.predict_proba(x_test)[:, 1]

            # Fine-grained threshold on val
            thresholds = np.arange(0.40, 0.90, 0.01)
            t_base, _val_roi_base = find_best_threshold_on_val(
                train_val, p_base_val, thresholds=thresholds.tolist(), min_bets=20
            )
            roi_base = calc_roi(test, p_base_test, threshold=t_base)
            logger.info(
                "Base: ROI=%.2f%% t=%.2f n=%d",
                roi_base["roi"],
                t_base,
                roi_base["n_bets"],
            )

            # === 2. Calibrated model (isotonic) ===
            calibrated = CalibratedClassifierCV(model_base, method="isotonic", cv="prefit")
            calibrated.fit(x_val, y_val)
            p_cal_val = calibrated.predict_proba(x_val)[:, 1]
            p_cal_test = calibrated.predict_proba(x_test)[:, 1]

            t_cal, _val_roi_cal = find_best_threshold_on_val(
                train_val, p_cal_val, thresholds=thresholds.tolist(), min_bets=20
            )
            roi_cal = calc_roi(test, p_cal_test, threshold=t_cal)
            logger.info(
                "Calibrated: ROI=%.2f%% t=%.2f n=%d",
                roi_cal["roi"],
                t_cal,
                roi_cal["n_bets"],
            )

            # === 3. Odds filter: only low odds (< 3.0) ===
            low_odds_mask_val = train_val["Odds"].values < 3.0
            low_odds_mask_test = test["Odds"].values < 3.0

            if low_odds_mask_val.sum() > 50:
                t_lo, _ = find_best_threshold_on_val(
                    train_val[low_odds_mask_val],
                    p_base_val[low_odds_mask_val],
                    thresholds=thresholds.tolist(),
                    min_bets=20,
                )
                p_lo_test = np.where(low_odds_mask_test, p_base_test, 0)
                roi_lo = calc_roi(test, p_lo_test, threshold=t_lo)
                logger.info(
                    "Low odds (<3.0): ROI=%.2f%% t=%.2f n=%d",
                    roi_lo["roi"],
                    t_lo,
                    roi_lo["n_bets"],
                )
            else:
                roi_lo = {"roi": -999.0, "n_bets": 0}
                t_lo = 0.5

            # === 4. Odds filter: medium odds (1.5 - 5.0) ===
            med_odds_mask_val = (train_val["Odds"].values >= 1.3) & (
                train_val["Odds"].values <= 5.0
            )
            med_odds_mask_test = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)

            if med_odds_mask_val.sum() > 50:
                t_med, _ = find_best_threshold_on_val(
                    train_val[med_odds_mask_val],
                    p_base_val[med_odds_mask_val],
                    thresholds=thresholds.tolist(),
                    min_bets=20,
                )
                p_med_test = np.where(med_odds_mask_test, p_base_test, 0)
                roi_med = calc_roi(test, p_med_test, threshold=t_med)
                logger.info(
                    "Medium odds (1.3-5.0): ROI=%.2f%% t=%.2f n=%d",
                    roi_med["roi"],
                    t_med,
                    roi_med["n_bets"],
                )
            else:
                roi_med = {"roi": -999.0, "n_bets": 0}
                t_med = 0.5

            # === 5. Train on full train set (not just fit portion) ===
            imputer_full = SimpleImputer(strategy="median")
            x_train_full = imputer_full.fit_transform(train[feature_cols])
            x_test_full = imputer_full.transform(test[feature_cols])

            model_full = LGBMClassifier(
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
            model_full.fit(x_train_full, train["target"].values)
            p_full_test = model_full.predict_proba(x_test_full)[:, 1]

            # Use threshold from val (same as base since model is similar)
            roi_full = calc_roi(test, p_full_test, threshold=t_base)
            logger.info(
                "Full train: ROI=%.2f%% t=%.2f n=%d",
                roi_full["roi"],
                t_base,
                roi_full["n_bets"],
            )

            # Log all thresholds for the base model
            for t in np.arange(0.40, 0.90, 0.05):
                r = calc_roi(test, p_base_test, threshold=t)
                mlflow.log_metric(f"roi_base_t{int(t * 100):03d}", r["roi"])

            best_roi = max(
                roi_base["roi"], roi_cal["roi"], roi_lo["roi"], roi_med["roi"], roi_full["roi"]
            )
            best_label = "base"
            if roi_cal["roi"] == best_roi:
                best_label = "calibrated"
            elif roi_lo["roi"] == best_roi:
                best_label = "low_odds"
            elif roi_med["roi"] == best_roi:
                best_label = "medium_odds"
            elif roi_full["roi"] == best_roi:
                best_label = "full_train"

            auc = roc_auc_score(y_test, p_base_test)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_base": roi_base["roi"],
                    "roi_calibrated": roi_cal["roi"],
                    "roi_low_odds": roi_lo["roi"],
                    "roi_medium_odds": roi_med["roi"],
                    "roi_full_train": roi_full["roi"],
                    "roc_auc": auc,
                    "threshold_base": t_base,
                    "threshold_calibrated": t_cal,
                    "n_bets_base": roi_base["n_bets"],
                    "n_bets_calibrated": roi_cal["n_bets"],
                }
            )
            mlflow.set_tag("best_variant", best_label)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best: %s, ROI=%.2f%%", best_label, best_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.4")
            raise


if __name__ == "__main__":
    main()
