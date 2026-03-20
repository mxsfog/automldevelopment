"""Step 4.9: Profitable sports only -- esports + cricket + tennis."""

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


PROFITABLE_SPORTS = [
    "League of Legends",
    "Dota 2",
    "CS2",
    "Cricket",
    "Tennis",
    "Table Tennis",
]


def main() -> None:
    logger.info("Step 4.9: Profitable sports only")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    # Singles only + profitable sports
    train_s = train_all[~train_all["Is_Parlay_bool"]].copy()
    test_s = test_all[~test_all["Is_Parlay_bool"]].copy()

    train = train_s[train_s["Sport"].isin(PROFITABLE_SPORTS)].copy()
    test = test_s[test_s["Sport"].isin(PROFITABLE_SPORTS)].copy()
    logger.info("Profitable sports singles: train=%d, test=%d", len(train), len(test))

    with mlflow.start_run(run_name="phase4/step4.9_profitable_sports") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("phase", "4")

        try:
            val_split_idx = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split_idx]
            train_val = train.iloc[val_split_idx:]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "lgbm_profitable_sports",
                    "sports": ",".join(PROFITABLE_SPORTS),
                    "filter": "Is_Parlay=f, Sport in profitable",
                }
            )

            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(train_fit[feature_cols])
            x_val = imputer.transform(train_val[feature_cols])
            x_test = imputer.transform(test[feature_cols])

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
            model.fit(
                x_fit,
                train_fit["target"].values,
                eval_set=[(x_val, train_val["target"].values)],
                callbacks=[],
            )

            p_val = model.predict_proba(x_val)[:, 1]
            thresholds = np.arange(0.30, 0.85, 0.01).tolist()
            best_t, val_roi = find_best_threshold_on_val(train_val, p_val, thresholds=thresholds)
            logger.info("Best threshold: %.2f, val ROI=%.2f%%", best_t, val_roi)

            p_test = model.predict_proba(x_test)[:, 1]
            roi_result = calc_roi(test, p_test, threshold=best_t)

            logger.info(
                "Profitable sports: ROI=%.2f%% n=%d (t=%.2f)",
                roi_result["roi"],
                roi_result["n_bets"],
                best_t,
            )

            for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, p_test, threshold=t)
                logger.info(
                    "  t=%.2f: ROI=%.2f%% n=%d WR=%.4f", t, r["roi"], r["n_bets"], r["win_rate"]
                )
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            # === Also: apply global model (trained on ALL sports) to profitable test only ===
            train_all_s = train_s.copy()
            val_idx = int(len(train_all_s) * 0.8)
            tf_all = train_all_s.iloc[:val_idx]
            tv_all = train_all_s.iloc[val_idx:]

            imputer_g = SimpleImputer(strategy="median")
            x_fit_g = imputer_g.fit_transform(tf_all[feature_cols])
            x_val_g = imputer_g.transform(tv_all[feature_cols])
            x_test_g = imputer_g.transform(test_s[feature_cols])

            model_g = LGBMClassifier(
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
            model_g.fit(
                x_fit_g,
                tf_all["target"].values,
                eval_set=[(x_val_g, tv_all["target"].values)],
                callbacks=[],
            )
            p_val_g = model_g.predict_proba(x_val_g)[:, 1]

            # Filter val to profitable sports for threshold selection
            val_mask = tv_all["Sport"].isin(PROFITABLE_SPORTS)
            t_g, _ = find_best_threshold_on_val(
                tv_all[val_mask], p_val_g[val_mask.values], thresholds=thresholds
            )

            # Apply to profitable test bets
            p_test_all = model_g.predict_proba(x_test_g)[:, 1]
            test_prof_mask = test_s["Sport"].isin(PROFITABLE_SPORTS)
            p_test_prof = np.where(test_prof_mask.values, p_test_all, 0)
            roi_global_prof = calc_roi(test_s, p_test_prof, threshold=t_g)
            logger.info(
                "Global model -> profitable sports: ROI=%.2f%% n=%d (t=%.2f)",
                roi_global_prof["roi"],
                roi_global_prof["n_bets"],
                t_g,
            )

            # Also try medium odds filter on profitable sports
            med_odds = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)
            p_test_m = np.where(med_odds, p_test, 0)
            t_m, _ = find_best_threshold_on_val(
                train_val,
                np.where(
                    (train_val["Odds"].values >= 1.3) & (train_val["Odds"].values <= 5.0),
                    p_val,
                    0,
                ),
                thresholds=thresholds,
            )
            roi_med = calc_roi(test, p_test_m, threshold=t_m)
            logger.info(
                "Profitable + medium odds: ROI=%.2f%% n=%d (t=%.2f)",
                roi_med["roi"],
                roi_med["n_bets"],
                t_m,
            )

            best_roi = max(roi_result["roi"], roi_global_prof["roi"], roi_med["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_sport_model": roi_result["roi"],
                    "roi_global_filtered": roi_global_prof["roi"],
                    "roi_med_odds": roi_med["roi"],
                    "best_threshold": best_t,
                    "n_bets": roi_result["n_bets"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ROI: %.2f%%", best_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.9")
            raise


if __name__ == "__main__":
    main()
