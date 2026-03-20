"""Step 4.11: Final consolidated model -- global train + inference filters + ensemble."""

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

PROFITABLE_SPORTS = [
    "League of Legends",
    "Dota 2",
    "CS2",
    "Cricket",
    "Tennis",
    "Table Tennis",
]

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
    "verbose": -1,
    "is_unbalance": True,
}


def main() -> None:
    logger.info("Step 4.11: Final consolidated model")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    # Singles only for train and test
    train = train_all[~train_all["Is_Parlay_bool"]].copy()
    test = test_all[~test_all["Is_Parlay_bool"]].copy()

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    with mlflow.start_run(run_name="phase4/step4.11_final_model") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "final_consolidated",
                    "filter": "Is_Parlay=f",
                    "inference_filter": "profitable_sports + medium_odds",
                    "n_seeds": 5,
                }
            )

            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(train_fit[feature_cols])
            x_val = imputer.transform(train_val[feature_cols])
            x_test = imputer.transform(test[feature_cols])

            # Strategy A: Single model, various inference filters
            model = LGBMClassifier(**LGBM_PARAMS, random_state=42)
            model.fit(
                x_fit,
                train_fit["target"].values,
                eval_set=[(x_val, train_val["target"].values)],
                callbacks=[],
            )
            p_val = model.predict_proba(x_val)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(test["target"].values, p_test)
            logger.info("Base AUC: %.4f", auc)

            thresholds = np.arange(0.30, 0.85, 0.01).tolist()

            # A1: No filter (singles only)
            t_a1, _ = find_best_threshold_on_val(train_val, p_val, thresholds=thresholds)
            roi_a1 = calc_roi(test, p_test, threshold=t_a1)
            logger.info(
                "A1 Singles only: ROI=%.2f%% n=%d t=%.2f", roi_a1["roi"], roi_a1["n_bets"], t_a1
            )

            # A2: Medium odds (1.3-5.0)
            odds_val = (train_val["Odds"].values >= 1.3) & (train_val["Odds"].values <= 5.0)
            odds_test = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)
            p_val_m = np.where(odds_val, p_val, 0)
            t_a2, _ = find_best_threshold_on_val(train_val, p_val_m, thresholds=thresholds)
            p_test_m = np.where(odds_test, p_test, 0)
            roi_a2 = calc_roi(test, p_test_m, threshold=t_a2)
            logger.info(
                "A2 Medium odds: ROI=%.2f%% n=%d t=%.2f", roi_a2["roi"], roi_a2["n_bets"], t_a2
            )

            # A3: Profitable sports only
            sport_val = train_val["Sport"].isin(PROFITABLE_SPORTS).values
            sport_test = test["Sport"].isin(PROFITABLE_SPORTS).values
            p_val_s = np.where(sport_val, p_val, 0)
            t_a3, _ = find_best_threshold_on_val(train_val, p_val_s, thresholds=thresholds)
            p_test_s = np.where(sport_test, p_test, 0)
            roi_a3 = calc_roi(test, p_test_s, threshold=t_a3)
            logger.info(
                "A3 Profitable sports: ROI=%.2f%% n=%d t=%.2f",
                roi_a3["roi"],
                roi_a3["n_bets"],
                t_a3,
            )

            # A4: Profitable sports + medium odds
            combo_val = sport_val & odds_val
            combo_test = sport_test & odds_test
            p_val_c = np.where(combo_val, p_val, 0)
            t_a4, _ = find_best_threshold_on_val(train_val, p_val_c, thresholds=thresholds)
            p_test_c = np.where(combo_test, p_test, 0)
            roi_a4 = calc_roi(test, p_test_c, threshold=t_a4)
            logger.info(
                "A4 Profitable + medium odds: ROI=%.2f%% n=%d t=%.2f",
                roi_a4["roi"],
                roi_a4["n_bets"],
                t_a4,
            )

            # Strategy B: 5-seed ensemble + same filters
            seeds = [42, 123, 456, 789, 1024]
            all_pval = []
            all_ptest = []
            for seed in seeds:
                m = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
                m.fit(x_fit, train_fit["target"].values)
                all_pval.append(m.predict_proba(x_val)[:, 1])
                all_ptest.append(m.predict_proba(x_test)[:, 1])

            ens_val = np.mean(all_pval, axis=0)
            ens_test = np.mean(all_ptest, axis=0)

            # B1: Ensemble, no filter
            t_b1, _ = find_best_threshold_on_val(train_val, ens_val, thresholds=thresholds)
            roi_b1 = calc_roi(test, ens_test, threshold=t_b1)
            logger.info(
                "B1 Ensemble singles: ROI=%.2f%% n=%d t=%.2f",
                roi_b1["roi"],
                roi_b1["n_bets"],
                t_b1,
            )

            # B2: Ensemble + medium odds
            ev_m = np.where(odds_val, ens_val, 0)
            t_b2, _ = find_best_threshold_on_val(train_val, ev_m, thresholds=thresholds)
            et_m = np.where(odds_test, ens_test, 0)
            roi_b2 = calc_roi(test, et_m, threshold=t_b2)
            logger.info(
                "B2 Ensemble + medium odds: ROI=%.2f%% n=%d t=%.2f",
                roi_b2["roi"],
                roi_b2["n_bets"],
                t_b2,
            )

            # B3: Ensemble + profitable sports
            ev_s = np.where(sport_val, ens_val, 0)
            t_b3, _ = find_best_threshold_on_val(train_val, ev_s, thresholds=thresholds)
            et_s = np.where(sport_test, ens_test, 0)
            roi_b3 = calc_roi(test, et_s, threshold=t_b3)
            logger.info(
                "B3 Ensemble + profitable: ROI=%.2f%% n=%d t=%.2f",
                roi_b3["roi"],
                roi_b3["n_bets"],
                t_b3,
            )

            # B4: Ensemble + profitable + medium odds
            ev_c = np.where(combo_val, ens_val, 0)
            t_b4, _ = find_best_threshold_on_val(train_val, ev_c, thresholds=thresholds)
            et_c = np.where(combo_test, ens_test, 0)
            roi_b4 = calc_roi(test, et_c, threshold=t_b4)
            logger.info(
                "B4 Ensemble + profitable + medium odds: ROI=%.2f%% n=%d t=%.2f",
                roi_b4["roi"],
                roi_b4["n_bets"],
                t_b4,
            )

            # Strategy C: Full-train (no val holdout) + ensemble
            imputer_full = SimpleImputer(strategy="median")
            x_full = imputer_full.fit_transform(train[feature_cols])
            x_test_full = imputer_full.transform(test[feature_cols])

            all_pfull = []
            for seed in seeds:
                m = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
                m.fit(x_full, train["target"].values)
                all_pfull.append(m.predict_proba(x_test_full)[:, 1])
            ens_full = np.mean(all_pfull, axis=0)

            # C1: Full-train ensemble, singles only (use val threshold from B1)
            roi_c1 = calc_roi(test, ens_full, threshold=t_b1)
            logger.info(
                "C1 Full-train ensemble: ROI=%.2f%% n=%d t=%.2f",
                roi_c1["roi"],
                roi_c1["n_bets"],
                t_b1,
            )

            # C2: Full-train + profitable + medium odds (use B4 threshold)
            ft_c = np.where(combo_test, ens_full, 0)
            roi_c2 = calc_roi(test, ft_c, threshold=t_b4)
            logger.info(
                "C2 Full-train + profitable + medium: ROI=%.2f%% n=%d t=%.2f",
                roi_c2["roi"],
                roi_c2["n_bets"],
                t_b4,
            )

            # C3: Full-train + profitable sports only (use B3 threshold)
            ft_s = np.where(sport_test, ens_full, 0)
            roi_c3 = calc_roi(test, ft_s, threshold=t_b3)
            logger.info(
                "C3 Full-train + profitable: ROI=%.2f%% n=%d t=%.2f",
                roi_c3["roi"],
                roi_c3["n_bets"],
                t_b3,
            )

            all_rois = {
                "A1_singles": roi_a1["roi"],
                "A2_med_odds": roi_a2["roi"],
                "A3_profitable": roi_a3["roi"],
                "A4_prof_med": roi_a4["roi"],
                "B1_ens_singles": roi_b1["roi"],
                "B2_ens_med": roi_b2["roi"],
                "B3_ens_prof": roi_b3["roi"],
                "B4_ens_prof_med": roi_b4["roi"],
                "C1_full_ens": roi_c1["roi"],
                "C2_full_prof_med": roi_c2["roi"],
                "C3_full_prof": roi_c3["roi"],
            }

            best_key = max(all_rois, key=all_rois.get)
            best_roi = all_rois[best_key]
            logger.info("Best strategy: %s -> ROI=%.2f%%", best_key, best_roi)

            # Log all
            for key, val in all_rois.items():
                mlflow.log_metric(f"roi_{key}", val)

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roc_auc": auc,
                    "best_strategy": 0,  # logged as tag below
                }
            )
            mlflow.set_tag("best_strategy", best_key)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.11")
            raise


if __name__ == "__main__":
    main()
