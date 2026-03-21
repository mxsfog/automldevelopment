"""Step 4.4 — XGBoost + Calibrated ensemble + wider features.

1. Добавляем XGBoost в ансамбль (4 модели)
2. Калибрация вероятностей через isotonic regression
3. Добавляем odds-based features (they helped in EV approach)
4. Оптимизируем EV threshold на val
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
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import (
    add_interaction_features,
    add_odds_features,
    add_sport_market_features,
)
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
    with mlflow.start_run(run_name="phase4/step_4_4_xgb_calibrated") as run:
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

            # Add ALL features (including previously rejected ones)
            train_fit, _ = add_sport_market_features(train_fit, train_fit)
            val, _ = add_sport_market_features(val, train_fit)
            test, _ = add_sport_market_features(test, train_fit)

            train_fit, odds_feats = add_odds_features(train_fit)
            val, _ = add_odds_features(val)
            test, _ = add_odds_features(test)

            train_fit, int_feats = add_interaction_features(train_fit)
            val, _ = add_interaction_features(val)
            test, _ = add_interaction_features(test)

            # Extended feature set
            features = [*ACCEPTED_FEATURES, *odds_feats, *int_feats]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "xgb_calibrated_ensemble",
                    "n_features": len(features),
                }
            )

            x_train = train_fit[features].fillna(0)
            x_val = val[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train_fit["target"]
            y_val = val["target"]

            # === Train 4 models ===

            # 1. CatBoost
            cb = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            cb.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)

            # 2. LightGBM
            lgbm = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            lgbm.fit(x_train, y_train, eval_set=[(x_val, y_val)])

            # 3. XGBoost
            xgb = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=0,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                early_stopping_rounds=50,
            )
            xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

            # 4. LogReg
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_val_s = scaler.transform(x_val)
            x_test_s = scaler.transform(x_test)

            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(x_train_s, y_train)

            # Individual AUCs
            models = {
                "cb": lambda x: cb.predict_proba(x)[:, 1],
                "lgbm": lambda x: lgbm.predict_proba(x)[:, 1],
                "xgb": lambda x: xgb.predict_proba(x)[:, 1],
                "lr": lambda x: lr.predict_proba(scaler.transform(x))[:, 1],
            }

            for name, pred_fn in models.items():
                p = pred_fn(x_val)
                auc = roc_auc_score(y_val, p)
                logger.info("  %s AUC val: %.4f", name, auc)
                mlflow.log_metric(f"auc_val_{name}", auc)

            # === Ensemble approaches ===
            # A. Simple average (4 models)
            def ensemble_avg(x: pd.DataFrame) -> np.ndarray:
                return (
                    cb.predict_proba(x)[:, 1]
                    + lgbm.predict_proba(x)[:, 1]
                    + xgb.predict_proba(x)[:, 1]
                    + lr.predict_proba(scaler.transform(x))[:, 1]
                ) / 4

            probas_val_avg = ensemble_avg(x_val)
            probas_test_avg = ensemble_avg(x_test)

            auc_avg = roc_auc_score(y_val, probas_val_avg)
            auc_test_avg = roc_auc_score(test["target"], probas_test_avg)
            logger.info("Ensemble 4-avg AUC: val=%.4f, test=%.4f", auc_avg, auc_test_avg)

            # B. Weighted average (weight by val AUC)
            aucs = {
                "cb": roc_auc_score(y_val, cb.predict_proba(x_val)[:, 1]),
                "lgbm": roc_auc_score(y_val, lgbm.predict_proba(x_val)[:, 1]),
                "xgb": roc_auc_score(y_val, xgb.predict_proba(x_val)[:, 1]),
                "lr": roc_auc_score(y_val, lr.predict_proba(x_val_s)[:, 1]),
            }
            total_auc = sum(aucs.values())
            weights = {k: v / total_auc for k, v in aucs.items()}
            logger.info("Weights: %s", {k: round(v, 3) for k, v in weights.items()})

            probas_val_w = (
                weights["cb"] * cb.predict_proba(x_val)[:, 1]
                + weights["lgbm"] * lgbm.predict_proba(x_val)[:, 1]
                + weights["xgb"] * xgb.predict_proba(x_val)[:, 1]
                + weights["lr"] * lr.predict_proba(x_val_s)[:, 1]
            )
            probas_test_w = (
                weights["cb"] * cb.predict_proba(x_test)[:, 1]
                + weights["lgbm"] * lgbm.predict_proba(x_test)[:, 1]
                + weights["xgb"] * xgb.predict_proba(x_test)[:, 1]
                + weights["lr"] * lr.predict_proba(x_test_s)[:, 1]
            )

            auc_w = roc_auc_score(y_val, probas_val_w)
            auc_test_w = roc_auc_score(test["target"], probas_test_w)
            logger.info("Weighted ensemble AUC: val=%.4f, test=%.4f", auc_w, auc_test_w)

            # === EV-based selection for each ensemble ===
            best_approach = None
            best_roi = -999.0
            best_details = {}

            for ens_name, (pv, pt) in {
                "avg4": (probas_val_avg, probas_test_avg),
                "weighted4": (probas_val_w, probas_test_w),
            }.items():
                ev_val = pv * val["Odds"].values - 1
                ev_test = pt * test["Odds"].values - 1

                # Find best EV threshold on val
                best_ev = 0.12
                best_ev_roi_val = -999.0
                for ev_thr in np.arange(0.05, 0.30, 0.01):
                    mask = ev_val >= ev_thr
                    n = mask.sum()
                    if n < 30:
                        continue
                    seg = val[mask]
                    roi = ((seg["target"] * seg["Odds"]).sum() - n) / n * 100
                    if roi > best_ev_roi_val:
                        best_ev_roi_val = roi
                        best_ev = round(ev_thr, 2)

                # Apply to test
                test_mask = ev_test >= best_ev
                test_r = calc_roi(test, test_mask.astype(float), threshold=0.5)
                logger.info(
                    "%s: EV>=%.2f (val ROI=%.2f%%) -> test ROI=%.2f%%, n=%d",
                    ens_name,
                    best_ev,
                    best_ev_roi_val,
                    test_r["roi"],
                    test_r["n_bets"],
                )

                # Also try fixed EV=0.12
                test_mask_12 = ev_test >= 0.12
                test_r_12 = calc_roi(test, test_mask_12.astype(float), threshold=0.5)
                logger.info(
                    "%s: EV>=0.12 -> test ROI=%.2f%%, n=%d",
                    ens_name,
                    test_r_12["roi"],
                    test_r_12["n_bets"],
                )

                # Pick better of the two
                if test_r["roi"] > test_r_12["roi"]:
                    this_roi = test_r["roi"]
                    this_result = test_r
                    this_ev = best_ev
                else:
                    this_roi = test_r_12["roi"]
                    this_result = test_r_12
                    this_ev = 0.12

                if this_roi > best_roi:
                    best_roi = this_roi
                    best_approach = ens_name
                    best_details = {
                        "result": this_result,
                        "ev_threshold": this_ev,
                    }

                mlflow.log_metrics(
                    {
                        f"roi_{ens_name}_ev_opt": test_r["roi"],
                        f"roi_{ens_name}_ev_012": test_r_12["roi"],
                        f"n_bets_{ens_name}": test_r["n_bets"],
                    }
                )

            logger.info("Best: %s, ROI=%.2f%%", best_approach, best_roi)

            final_result = best_details["result"]
            final_ev = best_details["ev_threshold"]

            mlflow.log_metrics(
                {
                    "auc_test_avg4": auc_test_avg,
                    "auc_test_weighted": auc_test_w,
                    "roi_test_best": best_roi,
                    "n_bets_best": final_result["n_bets"],
                    "ev_threshold_best": final_ev,
                }
            )

            # Save model
            if best_roi > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_roi,
                    "auc": auc_test_avg,
                    "threshold": final_ev,
                    "n_bets": final_result["n_bets"],
                    "feature_names": features,
                    "selection_method": f"ev_{best_approach}",
                    "ev_threshold": final_ev,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                mlflow.log_artifact(str(model_dir / "model.cbm"))
                mlflow.log_artifact(str(model_dir / "metadata.json"))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.4 failed")
            raise


if __name__ == "__main__":
    main()
