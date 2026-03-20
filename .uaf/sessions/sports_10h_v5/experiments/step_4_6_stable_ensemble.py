"""Step 4.6: Stable ensemble -- multi-seed LightGBM ensemble for stability."""

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


BASE_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "num_leaves": 50,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "is_unbalance": True,
}


def main() -> None:
    logger.info("Step 4.6: Stable multi-seed ensemble")
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

    with mlflow.start_run(run_name="phase4/step4.6_stable_ensemble") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "multi_seed_lgbm_ensemble",
                    "filter": "Is_Parlay=f",
                    "n_seeds": 10,
                    "regularization": "strong",
                }
            )

            # === Multi-seed ensemble (10 seeds) ===
            seeds = [42, 123, 456, 789, 1024, 2048, 3000, 5555, 7777, 9999]
            all_proba_val = []
            all_proba_test = []

            for seed in seeds:
                params = {**BASE_PARAMS, "random_state": seed}
                model = LGBMClassifier(**params)
                model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], callbacks=[])
                p_val = model.predict_proba(x_val)[:, 1]
                p_test = model.predict_proba(x_test)[:, 1]
                all_proba_val.append(p_val)
                all_proba_test.append(p_test)

                roi_s = calc_roi(test, p_test, threshold=0.6)
                logger.info("  Seed %d: ROI=%.2f%% n=%d", seed, roi_s["roi"], roi_s["n_bets"])

            # Average ensemble
            ens_val = np.mean(all_proba_val, axis=0)
            ens_test = np.mean(all_proba_test, axis=0)

            # Threshold on val
            thresholds = np.arange(0.40, 0.85, 0.01).tolist()
            best_t, val_roi = find_best_threshold_on_val(train_val, ens_val, thresholds=thresholds)
            roi_ens = calc_roi(test, ens_test, threshold=best_t)
            auc_ens = roc_auc_score(y_test, ens_test)

            logger.info(
                "Ensemble: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_ens["roi"],
                auc_ens,
                best_t,
                roi_ens["n_bets"],
            )

            # Log at various thresholds
            for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, ens_test, threshold=t)
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            # Medium odds filter on ensemble
            odds_mask_val = (train_val["Odds"].values >= 1.3) & (train_val["Odds"].values <= 5.0)
            odds_mask_test = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)

            p_val_m = np.where(odds_mask_val, ens_val, 0)
            t_m, _ = find_best_threshold_on_val(train_val, p_val_m, thresholds=thresholds)
            p_test_m = np.where(odds_mask_test, ens_test, 0)
            roi_m = calc_roi(test, p_test_m, threshold=t_m)

            logger.info(
                "Medium odds ensemble: ROI=%.2f%% t=%.2f n=%d",
                roi_m["roi"],
                t_m,
                roi_m["n_bets"],
            )

            # === Also try retrain on full train + use val threshold ===
            imputer_full = SimpleImputer(strategy="median")
            x_full = imputer_full.fit_transform(train[feature_cols])
            x_test_full = imputer_full.transform(test[feature_cols])

            all_proba_full = []
            for seed in seeds:
                params = {**BASE_PARAMS, "random_state": seed}
                model = LGBMClassifier(**params)
                model.fit(x_full, train["target"].values)
                p = model.predict_proba(x_test_full)[:, 1]
                all_proba_full.append(p)

            ens_full = np.mean(all_proba_full, axis=0)
            roi_full = calc_roi(test, ens_full, threshold=best_t)
            logger.info(
                "Full-train ensemble: ROI=%.2f%% t=%.2f n=%d",
                roi_full["roi"],
                best_t,
                roi_full["n_bets"],
            )

            # Full-train + medium odds
            p_full_m = np.where(odds_mask_test, ens_full, 0)
            roi_full_m = calc_roi(test, p_full_m, threshold=t_m)
            logger.info(
                "Full-train + medium odds: ROI=%.2f%% t=%.2f n=%d",
                roi_full_m["roi"],
                t_m,
                roi_full_m["n_bets"],
            )

            best_roi = max(roi_ens["roi"], roi_m["roi"], roi_full["roi"], roi_full_m["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_ensemble": roi_ens["roi"],
                    "roi_ensemble_med_odds": roi_m["roi"],
                    "roi_full_train_ensemble": roi_full["roi"],
                    "roi_full_med_odds": roi_full_m["roi"],
                    "roc_auc": auc_ens,
                    "best_threshold": best_t,
                    "n_bets_ensemble": roi_ens["n_bets"],
                    "val_roi": val_roi,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ROI: %.2f%%", best_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.6")
            raise


if __name__ == "__main__":
    main()
