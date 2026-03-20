"""Step 4.2: Singles-only model -- фильтрация парлаев для повышения ROI."""

import logging
import os
import traceback

import mlflow
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


def main() -> None:
    logger.info("Step 4.2: Singles-only model")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = get_feature_columns()

    # Фильтр: только синглы
    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()
    logger.info(
        "Singles only: train=%d (was %d), test=%d (was %d)",
        len(train),
        len(train_all),
        len(test),
        len(test_all),
    )

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    with mlflow.start_run(run_name="phase4/step4.2_singles_only") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")

        try:
            # Remove Is_Parlay_bool since all are singles
            features = [f for f in feature_cols if f != "Is_Parlay_bool"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "lightgbm_singles_only",
                    "features": ",".join(features),
                    "filter": "Is_Parlay=f",
                }
            )

            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(train_fit[features])
            x_val = imputer.transform(train_val[features])
            x_test = imputer.transform(test[features])
            y_fit = train_fit["target"].values
            y_val = train_val["target"].values
            y_test = test["target"].values

            model = LGBMClassifier(
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
            model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], callbacks=[])

            proba_val = model.predict_proba(x_val)[:, 1]
            best_t, val_roi = find_best_threshold_on_val(train_val, proba_val)
            logger.info("Best threshold from val: %.2f, val ROI=%.2f%%", best_t, val_roi)

            proba_test = model.predict_proba(x_test)[:, 1]
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(y_test, proba_test)

            # Also test multiple thresholds
            for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                r = calc_roi(test, proba_test, threshold=t)
                logger.info(
                    "  t=%.2f: ROI=%.2f%% n=%d WR=%.4f",
                    t,
                    r["roi"],
                    r["n_bets"],
                    r["win_rate"],
                )
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            # Comparison: LightGBM on all data (same hparams)
            imputer2 = SimpleImputer(strategy="median")
            val_idx_all = int(len(train_all) * 0.8)
            tf_all = train_all.iloc[:val_idx_all]
            tv_all = train_all.iloc[val_idx_all:]

            x_fit_all = imputer2.fit_transform(tf_all[feature_cols])
            x_val_all = imputer2.transform(tv_all[feature_cols])
            x_test_all = imputer2.transform(test_all[feature_cols])

            model_all = LGBMClassifier(
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
            model_all.fit(
                x_fit_all,
                tf_all["target"].values,
                eval_set=[(x_val_all, tv_all["target"].values)],
                callbacks=[],
            )
            p_all_val = model_all.predict_proba(x_val_all)[:, 1]
            t_all, _ = find_best_threshold_on_val(tv_all, p_all_val)
            p_all_test = model_all.predict_proba(x_test_all)[:, 1]
            roi_all = calc_roi(test_all, p_all_test, threshold=t_all)

            logger.info(
                "All data model: ROI=%.2f%% (t=%.2f, n=%d)",
                roi_all["roi"],
                t_all,
                roi_all["n_bets"],
            )

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "best_threshold": best_t,
                    "n_bets_selected": roi_result["n_bets"],
                    "pct_selected": roi_result["pct_selected"],
                    "win_rate_selected": roi_result["win_rate"],
                    "roi_all_data_model": roi_all["roi"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info(
                "Singles-only ROI: %.2f%% (t=%.2f, n=%d)",
                roi_result["roi"],
                best_t,
                roi_result["n_bets"],
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.2")
            raise


if __name__ == "__main__":
    main()
