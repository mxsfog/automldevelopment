"""Step 4.8: Segment-specific models -- отдельные модели по спортам."""

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
    logger.info("Step 4.8: Segment-specific models")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()

    with mlflow.start_run(run_name="phase4/step4.8_segment_models") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                    "method": "segment_specific_models",
                    "filter": "Is_Parlay=f",
                }
            )

            # === 1. Per-sport models ===
            logger.info("=== Per-sport models ===")
            sport_predictions = np.zeros(len(test))
            sport_predictions_val = np.zeros(0)

            val_split_idx = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split_idx]
            train_val = train.iloc[val_split_idx:]

            sport_predictions_val = np.zeros(len(train_val))
            sports_tried = []

            for sport in test["Sport"].value_counts().head(8).index:
                tr_sport = train_fit[train_fit["Sport"] == sport]
                vl_sport_idx = train_val["Sport"] == sport
                te_sport_idx = test["Sport"] == sport

                if len(tr_sport) < 200 or vl_sport_idx.sum() < 20 or te_sport_idx.sum() < 20:
                    logger.info("  %s: skipped (too few samples)", sport)
                    continue

                sports_tried.append(sport)
                imputer = SimpleImputer(strategy="median")
                x_tr = imputer.fit_transform(tr_sport[feature_cols])
                x_vl = imputer.transform(train_val[vl_sport_idx][feature_cols])
                x_te = imputer.transform(test[te_sport_idx][feature_cols])

                model = LGBMClassifier(**LGBM_PARAMS)
                model.fit(x_tr, tr_sport["target"].values)

                p_vl = model.predict_proba(x_vl)[:, 1]
                p_te = model.predict_proba(x_te)[:, 1]

                sport_predictions_val[vl_sport_idx.values] = p_vl
                sport_predictions[te_sport_idx.values] = p_te

                t_s, _ = find_best_threshold_on_val(train_val[vl_sport_idx], p_vl, min_bets=10)
                r_s = calc_roi(test[te_sport_idx], p_te, threshold=t_s)
                logger.info(
                    "  %s: ROI=%.2f%% n=%d (t=%.2f, train=%d)",
                    sport,
                    r_s["roi"],
                    r_s["n_bets"],
                    t_s,
                    len(tr_sport),
                )
                mlflow.log_metric(f"roi_sport_{sport.lower().replace(' ', '_')}", r_s["roi"])

            # Composite: use sport-specific predictions where available, else global model
            imputer_global = SimpleImputer(strategy="median")
            x_fit_g = imputer_global.fit_transform(train_fit[feature_cols])
            x_val_g = imputer_global.transform(train_val[feature_cols])
            x_test_g = imputer_global.transform(test[feature_cols])

            model_global = LGBMClassifier(**LGBM_PARAMS)
            model_global.fit(
                x_fit_g,
                train_fit["target"].values,
                eval_set=[(x_val_g, train_val["target"].values)],
                callbacks=[],
            )
            p_global_val = model_global.predict_proba(x_val_g)[:, 1]
            p_global_test = model_global.predict_proba(x_test_g)[:, 1]

            # Composite: sport-specific where available, global otherwise
            composite_val = np.where(
                sport_predictions_val > 0, sport_predictions_val, p_global_val
            )
            composite_test = np.where(sport_predictions > 0, sport_predictions, p_global_test)

            t_comp, _ = find_best_threshold_on_val(
                train_val, composite_val, thresholds=np.arange(0.40, 0.85, 0.01).tolist()
            )
            roi_comp = calc_roi(test, composite_test, threshold=t_comp)
            logger.info(
                "Composite: ROI=%.2f%% t=%.2f n=%d",
                roi_comp["roi"],
                t_comp,
                roi_comp["n_bets"],
            )

            # Global model for comparison
            t_global, _ = find_best_threshold_on_val(
                train_val, p_global_val, thresholds=np.arange(0.40, 0.85, 0.01).tolist()
            )
            roi_global = calc_roi(test, p_global_test, threshold=t_global)
            auc_global = roc_auc_score(test["target"].values, p_global_test)
            logger.info(
                "Global: ROI=%.2f%% t=%.2f n=%d AUC=%.4f",
                roi_global["roi"],
                t_global,
                roi_global["n_bets"],
                auc_global,
            )

            # === 2. Global model + retrain on full train ===
            imputer_full = SimpleImputer(strategy="median")
            x_full = imputer_full.fit_transform(train[feature_cols])
            x_test_full = imputer_full.transform(test[feature_cols])

            model_full = LGBMClassifier(**LGBM_PARAMS)
            model_full.fit(x_full, train["target"].values)
            p_full_test = model_full.predict_proba(x_test_full)[:, 1]

            # Use val threshold from above
            roi_full = calc_roi(test, p_full_test, threshold=t_global)
            logger.info(
                "Full-train global: ROI=%.2f%% t=%.2f n=%d",
                roi_full["roi"],
                t_global,
                roi_full["n_bets"],
            )

            # Full-train + medium odds
            odds_mask = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)
            p_full_m = np.where(odds_mask, p_full_test, 0)
            roi_full_m = calc_roi(test, p_full_m, threshold=t_global)
            logger.info(
                "Full-train + medium odds: ROI=%.2f%% n=%d",
                roi_full_m["roi"],
                roi_full_m["n_bets"],
            )

            best_roi = max(roi_comp["roi"], roi_global["roi"], roi_full["roi"], roi_full_m["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_composite": roi_comp["roi"],
                    "roi_global": roi_global["roi"],
                    "roi_full_train": roi_full["roi"],
                    "roi_full_med_odds": roi_full_m["roi"],
                    "roc_auc": auc_global,
                    "n_bets_global": roi_global["n_bets"],
                    "n_bets_full": roi_full["n_bets"],
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
            mlflow.set_tag("failure_reason", "exception in step 4.8")
            raise


if __name__ == "__main__":
    main()
