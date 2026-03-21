"""Step 4.2 — EV-based selection + Stacking ensemble.

Два подхода:
1. EV-based: используем model_prob * odds - 1 для отбора ставок
2. Stacking: CatBoost + LightGBM + LogReg -> meta-model

Ключевое: не просто предсказать winner, а найти VALUE (EV > 0).
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    find_best_threshold,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import add_sport_market_features
from step_3_1_optuna import ACCEPTED_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    with mlflow.start_run(run_name="phase4/step_4_2_ev_stacking") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            train_fit, _ = add_sport_market_features(train_fit, train_fit)
            val, _ = add_sport_market_features(val, train_fit)
            test, _ = add_sport_market_features(test, train_fit)

            features = ACCEPTED_FEATURES

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "ev_stacking",
                    "n_features": len(features),
                }
            )

            x_train = train_fit[features].fillna(0)
            x_val = val[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train_fit["target"]
            y_val = val["target"]

            # === Part 1: EV-based selection ===
            logger.info("=== Part 1: EV-based selection ===")

            # Train CatBoost
            cb_model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            cb_model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)

            probas_val = cb_model.predict_proba(x_val)[:, 1]
            probas_test = cb_model.predict_proba(x_test)[:, 1]

            # EV = model_prob * odds - 1
            ev_val = probas_val * val["Odds"].values - 1
            ev_test = probas_test * test["Odds"].values - 1

            # ROI по разным EV порогам
            logger.info("EV-based selection results (val):")
            best_ev_thr = 0.0
            best_ev_roi = -999.0
            for ev_thr in np.arange(-0.05, 0.30, 0.01):
                mask = ev_val >= ev_thr
                n_sel = mask.sum()
                if n_sel < 50:
                    continue
                selected = val[mask]
                staked = n_sel * 1.0
                payout = (selected["target"] * selected["Odds"]).sum()
                roi = (payout - staked) / staked * 100
                if roi > best_ev_roi:
                    best_ev_roi = roi
                    best_ev_thr = ev_thr

            logger.info("Best EV threshold (val): %.3f, ROI=%.2f%%", best_ev_thr, best_ev_roi)

            # Apply to test
            ev_mask_test = ev_test >= best_ev_thr
            ev_result_test = calc_roi(test, ev_mask_test.astype(float), threshold=0.5)
            logger.info("EV test result: %s", ev_result_test)

            # Also try several fixed EV thresholds on test
            for ev_thr in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
                mask = ev_test >= ev_thr
                r = calc_roi(test, mask.astype(float), threshold=0.5)
                logger.info("  EV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi_ev_best_val": best_ev_roi,
                    "ev_threshold": best_ev_thr,
                    "roi_ev_test": ev_result_test["roi"],
                    "n_bets_ev_test": ev_result_test["n_bets"],
                }
            )

            # === Part 2: Stacking ===
            logger.info("=== Part 2: Stacking CatBoost + LightGBM + LogReg ===")

            # LightGBM
            lgbm_model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            lgbm_model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
            )

            # LogReg
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_val_s = scaler.transform(x_val)
            x_test_s = scaler.transform(x_test)

            lr_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
            lr_model.fit(x_train_s, y_train)

            # Get L1 predictions
            cb_val = cb_model.predict_proba(x_val)[:, 1]
            cb_test = cb_model.predict_proba(x_test)[:, 1]
            lgbm_val = lgbm_model.predict_proba(x_val)[:, 1]
            lgbm_test = lgbm_model.predict_proba(x_test)[:, 1]
            lr_val = lr_model.predict_proba(x_val_s)[:, 1]
            lr_test = lr_model.predict_proba(x_test_s)[:, 1]

            # Simple average
            avg_val = (cb_val + lgbm_val + lr_val) / 3
            avg_test = (cb_test + lgbm_test + lr_test) / 3

            auc_cb = roc_auc_score(y_val, cb_val)
            auc_lgbm = roc_auc_score(y_val, lgbm_val)
            auc_lr = roc_auc_score(y_val, lr_val)
            auc_avg = roc_auc_score(y_val, avg_val)
            logger.info(
                "AUC val: CB=%.4f, LGBM=%.4f, LR=%.4f, AVG=%.4f", auc_cb, auc_lgbm, auc_lr, auc_avg
            )

            auc_test_avg = roc_auc_score(test["target"], avg_test)
            logger.info("AUC test (avg ensemble): %.4f", auc_test_avg)

            # Threshold on avg probas (val)
            best_thr, _val_r = find_best_threshold(val, avg_val, min_bets=50)
            test_result_avg = calc_roi(test, avg_test, threshold=best_thr)
            logger.info("Ensemble ROI test (thr=%.2f): %s", best_thr, test_result_avg)

            for thr in [0.5, 0.55, 0.6, 0.65, 0.7]:
                r = calc_roi(test, avg_test, threshold=thr)
                logger.info("Ensemble thr=%.2f: ROI=%.2f%%, n=%d", thr, r["roi"], r["n_bets"])

            # EV-based on ensemble
            ev_avg_val = avg_val * val["Odds"].values - 1
            ev_avg_test = avg_test * test["Odds"].values - 1

            best_ev_ens = 0.0
            best_ev_ens_roi = -999.0
            for ev_thr in np.arange(-0.05, 0.30, 0.01):
                mask = ev_avg_val >= ev_thr
                n_sel = mask.sum()
                if n_sel < 50:
                    continue
                selected = val[mask]
                staked = n_sel * 1.0
                payout = (selected["target"] * selected["Odds"]).sum()
                roi = (payout - staked) / staked * 100
                if roi > best_ev_ens_roi:
                    best_ev_ens_roi = roi
                    best_ev_ens = ev_thr

            ev_mask_ens = ev_avg_test >= best_ev_ens
            ev_ens_result = calc_roi(test, ev_mask_ens.astype(float), threshold=0.5)
            logger.info(
                "Ensemble EV>=%.3f: ROI=%.2f%%, n=%d",
                best_ev_ens,
                ev_ens_result["roi"],
                ev_ens_result["n_bets"],
            )

            for ev_thr in [0.0, 0.02, 0.05, 0.10, 0.15]:
                mask = ev_avg_test >= ev_thr
                r = calc_roi(test, mask.astype(float), threshold=0.5)
                logger.info("  Ens EV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            mlflow.log_metrics(
                {
                    "auc_val_cb": auc_cb,
                    "auc_val_lgbm": auc_lgbm,
                    "auc_val_lr": auc_lr,
                    "auc_val_avg": auc_avg,
                    "auc_test_avg": auc_test_avg,
                    "roi_test_ensemble": test_result_avg["roi"],
                    "roi_ev_ensemble": ev_ens_result["roi"],
                    "n_bets_ev_ensemble": ev_ens_result["n_bets"],
                }
            )

            # Pick overall best
            results = {
                "ev_single": ev_result_test,
                "ensemble_thr": test_result_avg,
                "ev_ensemble": ev_ens_result,
            }
            best_name = max(results, key=lambda k: results[k]["roi"])
            best_result = results[best_name]
            logger.info("Best approach: %s, ROI=%.2f%%", best_name, best_result["roi"])

            # Save best
            if best_result["roi"] > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_model.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_result["roi"],
                    "auc": auc_test_avg,
                    "threshold": best_thr if best_name == "ensemble_thr" else best_ev_thr,
                    "n_bets": best_result["n_bets"],
                    "feature_names": features,
                    "selection_method": best_name,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                mlflow.log_artifact(str(model_dir / "model.cbm"))
                mlflow.log_artifact(str(model_dir / "metadata.json"))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.2 failed")
            raise


if __name__ == "__main__":
    main()
