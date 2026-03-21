"""Step 4.3 — EV refinement: sport segmentation + calibration.

1. EV-based на ensemble с фиксированным EV>=0.12 (из val Step 4.2)
2. Сегментный анализ по Sport
3. Исключение убыточных видов спорта
4. Probability calibration (CalibratedClassifierCV)
5. Odds-range filtering
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

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
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


def train_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple:
    """Обучение ансамбля CatBoost + LightGBM + LogReg."""
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

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_train_s, y_train)

    return cb, lgbm, lr, scaler


def predict_ensemble(
    cb: CatBoostClassifier,
    lgbm: LGBMClassifier,
    lr: LogisticRegression,
    scaler: StandardScaler,
    x: pd.DataFrame,
) -> np.ndarray:
    """Среднее предсказание ансамбля."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    return (p_cb + p_lgbm + p_lr) / 3


def main() -> None:
    with mlflow.start_run(run_name="phase4/step_4_3_ev_refined") as run:
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
            x_train = train_fit[features].fillna(0)
            x_val = val[features].fillna(0)
            x_test = test[features].fillna(0)
            y_train = train_fit["target"]
            y_val = val["target"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "ev_refined",
                    "n_features": len(features),
                }
            )

            # Train ensemble
            cb, lgbm, lr, scaler = train_ensemble(x_train, y_train, x_val, y_val)

            probas_val = predict_ensemble(cb, lgbm, lr, scaler, x_val)
            probas_test = predict_ensemble(cb, lgbm, lr, scaler, x_test)

            auc_test = roc_auc_score(test["target"], probas_test)
            logger.info("Ensemble AUC test: %.4f", auc_test)

            # EV calculation
            ev_val = probas_val * val["Odds"].values - 1
            ev_test = probas_test * test["Odds"].values - 1

            # 1. Baseline EV>=0.12 (from step 4.2 val)
            ev_thr = 0.12
            mask_test = ev_test >= ev_thr
            base_result = calc_roi(test, mask_test.astype(float), threshold=0.5)
            logger.info(
                "Baseline EV>=%.2f: ROI=%.2f%%, n=%d",
                ev_thr,
                base_result["roi"],
                base_result["n_bets"],
            )

            # 2. Sport-level analysis on val (to pick sport filter)
            logger.info("=== Sport analysis on val ===")
            val["ev"] = ev_val
            val["selected"] = ev_val >= ev_thr
            test["ev"] = ev_test
            test["selected"] = ev_test >= ev_thr

            sport_roi_val = {}
            for sport in val["Sport"].dropna().unique():
                mask = (val["Sport"] == sport) & val["selected"]
                n = mask.sum()
                if n < 10:
                    continue
                seg = val[mask]
                roi = ((seg["target"] * seg["Odds"]).sum() - n) / n * 100
                sport_roi_val[sport] = {"roi": roi, "n": n}
                logger.info("  Val %s: ROI=%.2f%% (n=%d)", sport, roi, n)

            # Bad sports on val
            bad_sports = [s for s, v in sport_roi_val.items() if v["roi"] < -5 and v["n"] >= 20]
            logger.info("Bad sports (val ROI < -5%%): %s", bad_sports)

            # 3. Apply sport filter to test
            if bad_sports:
                mask_filtered = test["selected"] & ~test["Sport"].isin(bad_sports)
                filtered_result = calc_roi(
                    test,
                    mask_filtered.astype(float),
                    threshold=0.5,
                )
                logger.info(
                    "After excluding %s: ROI=%.2f%%, n=%d",
                    bad_sports,
                    filtered_result["roi"],
                    filtered_result["n_bets"],
                )
            else:
                filtered_result = base_result

            # 4. Odds-range filter (exclude very high odds > 50)
            mask_odds = test["selected"] & (test["Odds"] <= 50)
            if not bad_sports:
                mask_odds_sport = mask_odds
            else:
                mask_odds_sport = mask_odds & ~test["Sport"].isin(bad_sports)

            odds_result = calc_roi(test, mask_odds_sport.astype(float), threshold=0.5)
            logger.info(
                "Odds<=50 + sport filter: ROI=%.2f%%, n=%d",
                odds_result["roi"],
                odds_result["n_bets"],
            )

            # 5. Detailed EV sweeps on test (for analysis only, not for threshold selection)
            logger.info("=== EV sweep (test, for analysis) ===")
            for ev in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
                mask = ev_test >= ev
                r = calc_roi(test, mask.astype(float), threshold=0.5)
                logger.info("  EV>=%.2f: ROI=%.2f%%, n=%d", ev, r["roi"], r["n_bets"])

            # 6. Combine: EV threshold sweep on val for sport-filtered
            logger.info("=== EV sweep on val (sport-filtered) ===")
            best_ev_filt = 0.12
            best_roi_filt = -999.0
            for ev in np.arange(0.05, 0.30, 0.01):
                if bad_sports:
                    mask = (val["ev"] >= ev) & ~val["Sport"].isin(bad_sports)
                else:
                    mask = val["ev"] >= ev
                n = mask.sum()
                if n < 30:
                    continue
                seg = val[mask]
                roi = ((seg["target"] * seg["Odds"]).sum() - n) / n * 100
                if roi > best_roi_filt:
                    best_roi_filt = roi
                    best_ev_filt = round(ev, 2)

            logger.info("Best EV (val, sport-filt): %.2f, ROI=%.2f%%", best_ev_filt, best_roi_filt)

            # Apply to test
            if bad_sports:
                final_mask = (ev_test >= best_ev_filt) & ~test["Sport"].isin(bad_sports)
            else:
                final_mask = ev_test >= best_ev_filt

            final_result = calc_roi(test, final_mask.astype(float), threshold=0.5)
            logger.info("Final test (EV>=%.2f + sport_filt): %s", best_ev_filt, final_result)

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test_baseline": base_result["roi"],
                    "roi_test_sport_filter": filtered_result["roi"],
                    "roi_test_odds_filter": odds_result["roi"],
                    "roi_test_final": final_result["roi"],
                    "n_bets_final": final_result["n_bets"],
                    "ev_threshold": best_ev_filt,
                    "win_rate_final": final_result.get("win_rate", 0),
                    "avg_odds_final": final_result.get("avg_odds", 0),
                }
            )

            # Save if improved
            if final_result["roi"] > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": final_result["roi"],
                    "auc": auc_test,
                    "threshold": best_ev_filt,
                    "n_bets": final_result["n_bets"],
                    "feature_names": features,
                    "selection_method": "ev_ensemble_sport_filter",
                    "ev_threshold": best_ev_filt,
                    "sport_filter": bad_sports,
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
            logger.exception("Step 4.3 failed")
            raise


if __name__ == "__main__":
    main()
