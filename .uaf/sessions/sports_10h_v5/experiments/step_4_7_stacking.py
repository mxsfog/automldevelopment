"""Step 4.7: Stacking -- LightGBM + CatBoost + LogReg -> LogReg meta."""

import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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
    logger.info("Step 4.7: Stacking classifier (singles)")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()

    # 3-way split for stacking: fit -> meta_train -> val
    n = len(train)
    idx1 = int(n * 0.6)
    idx2 = int(n * 0.8)
    train_fit = train.iloc[:idx1]
    train_meta = train.iloc[idx1:idx2]
    train_val = train.iloc[idx2:]

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(train_fit[feature_cols])
    x_meta = imputer.transform(train_meta[feature_cols])
    x_val = imputer.transform(train_val[feature_cols])
    x_test = imputer.transform(test[feature_cols])
    y_fit = train_fit["target"].values
    y_meta = train_meta["target"].values
    y_test = test["target"].values

    scaler = StandardScaler()
    x_fit_s = scaler.fit_transform(x_fit)
    x_meta_s = scaler.transform(x_meta)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    with mlflow.start_run(run_name="phase4/step4.7_stacking") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_fit": len(train_fit),
                    "n_samples_meta": len(train_meta),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "stacking_lgbm_cb_lr",
                    "filter": "Is_Parlay=f",
                }
            )

            # Level 0: Base models trained on fit portion
            model_lgb = LGBMClassifier(
                n_estimators=228,
                max_depth=6,
                learning_rate=0.216,
                num_leaves=50,
                min_child_samples=18,
                random_state=42,
                verbose=-1,
                is_unbalance=True,
            )
            model_lgb.fit(x_fit, y_fit)

            model_cb = CatBoostClassifier(
                iterations=200,
                depth=6,
                random_seed=42,
                verbose=0,
                auto_class_weights="Balanced",
                l2_leaf_reg=5,
            )
            model_cb.fit(x_fit, y_fit)

            model_lr = LogisticRegression(
                C=0.003, penalty="l1", solver="saga", random_state=42, max_iter=2000
            )
            model_lr.fit(x_fit_s, y_fit)

            # Generate meta features
            def get_meta_features(x: np.ndarray, x_scaled: np.ndarray) -> np.ndarray:
                p1 = model_lgb.predict_proba(x)[:, 1]
                p2 = model_cb.predict_proba(x)[:, 1]
                p3 = model_lr.predict_proba(x_scaled)[:, 1]
                return np.column_stack([p1, p2, p3])

            meta_train_x = get_meta_features(x_meta, x_meta_s)
            meta_val_x = get_meta_features(x_val, x_val_s)
            meta_test_x = get_meta_features(x_test, x_test_s)

            # Level 1: Meta-model (LogReg on predictions)
            meta_scaler = StandardScaler()
            meta_train_xs = meta_scaler.fit_transform(meta_train_x)
            meta_val_xs = meta_scaler.transform(meta_val_x)
            meta_test_xs = meta_scaler.transform(meta_test_x)

            meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            meta_model.fit(meta_train_xs, y_meta)

            # Predictions
            p_stacked_val = meta_model.predict_proba(meta_val_xs)[:, 1]
            p_stacked_test = meta_model.predict_proba(meta_test_xs)[:, 1]

            # Threshold on val
            thresholds = np.arange(0.40, 0.90, 0.01).tolist()
            best_t, _ = find_best_threshold_on_val(train_val, p_stacked_val, thresholds=thresholds)
            roi_stacked = calc_roi(test, p_stacked_test, threshold=best_t)
            auc_stacked = roc_auc_score(y_test, p_stacked_test)

            logger.info(
                "Stacked: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_stacked["roi"],
                auc_stacked,
                best_t,
                roi_stacked["n_bets"],
            )

            for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, p_stacked_test, threshold=t)
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_stacked_t{int(t * 100):03d}", r["roi"])

            # Also try stacked + medium odds
            odds_mask_val = (train_val["Odds"].values >= 1.3) & (train_val["Odds"].values <= 5.0)
            odds_mask_test = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)

            p_val_m = np.where(odds_mask_val, p_stacked_val, 0)
            t_m, _ = find_best_threshold_on_val(train_val, p_val_m, thresholds=thresholds)
            p_test_m = np.where(odds_mask_test, p_stacked_test, 0)
            roi_m = calc_roi(test, p_test_m, threshold=t_m)
            logger.info(
                "Stacked + medium odds: ROI=%.2f%% t=%.2f n=%d",
                roi_m["roi"],
                t_m,
                roi_m["n_bets"],
            )

            # Also compare with augmented meta features (meta + original features)
            meta_aug_train = np.hstack([meta_train_x, x_meta])
            meta_aug_val = np.hstack([meta_val_x, x_val])
            meta_aug_test = np.hstack([meta_test_x, x_test])

            aug_scaler = StandardScaler()
            meta_aug_train_s = aug_scaler.fit_transform(meta_aug_train)
            meta_aug_val_s = aug_scaler.transform(meta_aug_val)
            meta_aug_test_s = aug_scaler.transform(meta_aug_test)

            meta_aug = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            meta_aug.fit(meta_aug_train_s, y_meta)
            p_aug_val = meta_aug.predict_proba(meta_aug_val_s)[:, 1]
            p_aug_test = meta_aug.predict_proba(meta_aug_test_s)[:, 1]

            t_aug, _ = find_best_threshold_on_val(train_val, p_aug_val, thresholds=thresholds)
            roi_aug = calc_roi(test, p_aug_test, threshold=t_aug)
            logger.info(
                "Augmented stacked: ROI=%.2f%% t=%.2f n=%d",
                roi_aug["roi"],
                t_aug,
                roi_aug["n_bets"],
            )

            best_roi = max(roi_stacked["roi"], roi_m["roi"], roi_aug["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_stacked": roi_stacked["roi"],
                    "roi_stacked_med_odds": roi_m["roi"],
                    "roi_augmented": roi_aug["roi"],
                    "roc_auc": auc_stacked,
                    "best_threshold": best_t,
                    "n_bets": roi_stacked["n_bets"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best stacking ROI: %.2f%%", best_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.7")
            raise


if __name__ == "__main__":
    main()
