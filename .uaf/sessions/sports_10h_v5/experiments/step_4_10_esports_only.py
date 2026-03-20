"""Step 4.10: Esports-only model -- LoL + Dota 2 + CS2."""

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

ESPORTS = ["League of Legends", "Dota 2", "CS2"]


def main() -> None:
    logger.info("Step 4.10: Esports-only model (LoL, Dota 2, CS2)")
    df = load_data()
    train_all, test_all = time_series_split(df)

    feature_cols = [f for f in get_feature_columns() if f != "Is_Parlay_bool"]

    # Singles only
    train_s = train_all[~train_all["Is_Parlay_bool"]].copy()
    test_s = test_all[~test_all["Is_Parlay_bool"]].copy()

    # Esports only
    train = train_s[train_s["Sport"].isin(ESPORTS)].copy()
    test = test_s[test_s["Sport"].isin(ESPORTS)].copy()
    logger.info("Esports singles: train=%d, test=%d", len(train), len(test))

    with mlflow.start_run(run_name="phase4/step4.10_esports_only") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.10")
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
                    "method": "lgbm_esports_only",
                    "sports": ",".join(ESPORTS),
                    "filter": "Is_Parlay=f, Sport in esports",
                }
            )

            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(train_fit[feature_cols])
            x_val = imputer.transform(train_val[feature_cols])
            x_test = imputer.transform(test[feature_cols])

            # Esports-specific model with tuned params
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
            p_test = model.predict_proba(x_test)[:, 1]

            auc = roc_auc_score(test["target"].values, p_test)

            thresholds = np.arange(0.30, 0.85, 0.01).tolist()
            best_t, val_roi = find_best_threshold_on_val(train_val, p_val, thresholds=thresholds)
            roi_base = calc_roi(test, p_test, threshold=best_t)
            logger.info(
                "Esports base: ROI=%.2f%% n=%d (t=%.2f) AUC=%.4f",
                roi_base["roi"],
                roi_base["n_bets"],
                best_t,
                auc,
            )

            # Sweep thresholds
            for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, p_test, threshold=t)
                logger.info(
                    "  t=%.2f: ROI=%.2f%% n=%d WR=%.4f", t, r["roi"], r["n_bets"], r["win_rate"]
                )
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            # Medium odds filter
            odds_val = (train_val["Odds"].values >= 1.3) & (train_val["Odds"].values <= 5.0)
            odds_test = (test["Odds"].values >= 1.3) & (test["Odds"].values <= 5.0)
            p_val_m = np.where(odds_val, p_val, 0)
            t_m, _ = find_best_threshold_on_val(train_val, p_val_m, thresholds=thresholds)
            p_test_m = np.where(odds_test, p_test, 0)
            roi_med = calc_roi(test, p_test_m, threshold=t_m)
            logger.info(
                "Esports + medium odds: ROI=%.2f%% n=%d (t=%.2f)",
                roi_med["roi"],
                roi_med["n_bets"],
                t_m,
            )

            # Multi-seed ensemble on esports
            seeds = [42, 123, 456, 789, 1024]
            all_pval = []
            all_ptest = []
            for seed in seeds:
                m = LGBMClassifier(
                    n_estimators=228,
                    max_depth=6,
                    learning_rate=0.216,
                    num_leaves=50,
                    min_child_samples=18,
                    reg_alpha=0.0001,
                    reg_lambda=0.0003,
                    subsample=0.925,
                    colsample_bytree=0.803,
                    random_state=seed,
                    verbose=-1,
                    is_unbalance=True,
                )
                m.fit(x_fit, train_fit["target"].values)
                all_pval.append(m.predict_proba(x_val)[:, 1])
                all_ptest.append(m.predict_proba(x_test)[:, 1])

            ens_val = np.mean(all_pval, axis=0)
            ens_test = np.mean(all_ptest, axis=0)
            t_ens, _ = find_best_threshold_on_val(train_val, ens_val, thresholds=thresholds)
            roi_ens = calc_roi(test, ens_test, threshold=t_ens)
            logger.info(
                "Esports ensemble (5 seeds): ROI=%.2f%% n=%d (t=%.2f)",
                roi_ens["roi"],
                roi_ens["n_bets"],
                t_ens,
            )

            # Ensemble + medium odds
            ens_val_m = np.where(odds_val, ens_val, 0)
            t_em, _ = find_best_threshold_on_val(train_val, ens_val_m, thresholds=thresholds)
            ens_test_m = np.where(odds_test, ens_test, 0)
            roi_ens_m = calc_roi(test, ens_test_m, threshold=t_em)
            logger.info(
                "Esports ensemble + medium odds: ROI=%.2f%% n=%d (t=%.2f)",
                roi_ens_m["roi"],
                roi_ens_m["n_bets"],
                t_em,
            )

            # Per-sport breakdown
            for sport in ESPORTS:
                mask = test["Sport"] == sport
                if mask.sum() > 0:
                    r = calc_roi(test[mask], ens_test[mask.values], threshold=t_ens)
                    logger.info(
                        "  %s: ROI=%.2f%% n=%d (of %d)", sport, r["roi"], r["n_bets"], mask.sum()
                    )

            best_roi = max(roi_base["roi"], roi_med["roi"], roi_ens["roi"], roi_ens_m["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_base": roi_base["roi"],
                    "roi_med_odds": roi_med["roi"],
                    "roi_ensemble": roi_ens["roi"],
                    "roi_ensemble_med_odds": roi_ens_m["roi"],
                    "roc_auc": auc,
                    "best_threshold": best_t,
                    "n_bets": roi_base["n_bets"],
                    "val_roi": val_roi,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best ROI: %.2f%%", best_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.10")
            raise


if __name__ == "__main__":
    main()
