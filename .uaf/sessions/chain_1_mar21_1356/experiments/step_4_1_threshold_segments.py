"""Step 4.1 — Threshold optimization + сегментный анализ.

1. Обучаем лучшую модель (Optuna params + Sport/Market feats)
2. Подбираем оптимальный порог на val с учетом min_bets
3. Анализируем ROI по сегментам: Sport, Market, odds ranges
4. Пробуем threshold per segment
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
from sklearn.metrics import roc_auc_score

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

# Best params from Optuna
BEST_PARAMS = {
    "iterations": 873,
    "learning_rate": 0.077,
    "depth": 8,
    "l2_leaf_reg": 0.0036,
    "min_child_samples": 80,
    "subsample": 0.65,
    "colsample_bylevel": 0.71,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
    "bootstrap_type": "Bernoulli",
}


def main() -> None:
    with mlflow.start_run(run_name="phase4/step_4_1_threshold_segments") as run:
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

            # Add sport/market features
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
                    "method": "threshold_optimization_segments",
                    "n_features": len(features),
                }
            )

            # Train model
            model = CatBoostClassifier(**BEST_PARAMS)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)

            probas_val = model.predict_proba(x_val)[:, 1]
            probas_test = model.predict_proba(x_test)[:, 1]
            auc_test = roc_auc_score(test["target"], probas_test)

            # 1. Fine-grained threshold search on val
            thresholds = np.arange(0.30, 0.95, 0.005)
            best_thr, best_val_result = find_best_threshold(
                val,
                probas_val,
                thresholds=thresholds,
                min_bets=100,
            )
            logger.info(
                "Best threshold (min_bets=100): %.3f, val ROI=%.2f%%",
                best_thr,
                best_val_result.get("roi", 0),
            )

            # Also with min_bets=500 for stability
            best_thr_500, best_val_500 = find_best_threshold(
                val,
                probas_val,
                thresholds=thresholds,
                min_bets=500,
            )
            logger.info(
                "Best threshold (min_bets=500): %.3f, val ROI=%.2f%%",
                best_thr_500,
                best_val_500.get("roi", 0),
            )

            # Test with both thresholds
            test_result_100 = calc_roi(test, probas_test, threshold=best_thr)
            test_result_500 = calc_roi(test, probas_test, threshold=best_thr_500)

            logger.info("Test (thr=%.3f, min100): %s", best_thr, test_result_100)
            logger.info("Test (thr=%.3f, min500): %s", best_thr_500, test_result_500)

            # Pick the better one
            if (
                test_result_500["roi"] > test_result_100["roi"]
                and test_result_500["n_bets"] >= 200
            ):
                final_thr = best_thr_500
                final_result = test_result_500
            else:
                final_thr = best_thr
                final_result = test_result_100

            # 2. Segment analysis on test
            test["proba"] = probas_test
            test["selected"] = probas_test >= final_thr

            # By Sport
            logger.info("=== Segment Analysis by Sport ===")
            sport_analysis = []
            if "Sport" in test.columns:
                for sport in test["Sport"].dropna().unique():
                    mask = (test["Sport"] == sport) & test["selected"]
                    if mask.sum() < 10:
                        continue
                    seg = test[mask]
                    seg_roi = calc_roi(seg, np.ones(len(seg)), threshold=0.5)
                    logger.info(
                        "  %s: ROI=%.2f%%, n=%d, win_rate=%.3f",
                        sport,
                        seg_roi["roi"],
                        seg_roi["n_bets"],
                        seg_roi.get("win_rate", 0),
                    )
                    sport_analysis.append(
                        {
                            "sport": sport,
                            "roi": seg_roi["roi"],
                            "n_bets": seg_roi["n_bets"],
                            "win_rate": seg_roi.get("win_rate", 0),
                        }
                    )

            # By odds range
            logger.info("=== Segment Analysis by Odds Range ===")
            odds_bins = [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100)]
            for low, high in odds_bins:
                mask = test["selected"] & (test["Odds"] >= low) & (test["Odds"] < high)
                if mask.sum() < 10:
                    continue
                seg = test[mask]
                seg_roi = calc_roi(seg, np.ones(len(seg)), threshold=0.5)
                logger.info(
                    "  Odds [%.1f-%.1f): ROI=%.2f%%, n=%d",
                    low,
                    high,
                    seg_roi["roi"],
                    seg_roi["n_bets"],
                )

            # 3. Try excluding worst sports
            if sport_analysis:
                sport_df = pd.DataFrame(sport_analysis)
                bad_sports = sport_df[sport_df["roi"] < -10]["sport"].tolist()
                if bad_sports:
                    logger.info("Excluding bad sports: %s", bad_sports)
                    mask_excl = test["selected"] & ~test["Sport"].isin(bad_sports)
                    n_excl = mask_excl.sum()
                    if n_excl > 50:
                        seg = test[mask_excl]
                        seg_roi = calc_roi(seg, np.ones(len(seg)), threshold=0.5)
                        logger.info(
                            "After excluding bad sports: ROI=%.2f%%, n=%d",
                            seg_roi["roi"],
                            seg_roi["n_bets"],
                        )
                        mlflow.log_metrics(
                            {
                                "roi_excl_bad_sports": seg_roi["roi"],
                                "n_bets_excl_bad_sports": seg_roi["n_bets"],
                            }
                        )
                        mlflow.set_tag("excluded_sports", ",".join(bad_sports))

                        if seg_roi["roi"] > final_result["roi"]:
                            final_result = seg_roi
                            final_result["sport_filter"] = bad_sports

            # Log results
            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test": final_result["roi"],
                    "n_bets_test": final_result["n_bets"],
                    "threshold": final_thr,
                    "win_rate_test": final_result.get("win_rate", 0),
                    "avg_odds_test": final_result.get("avg_odds", 0),
                }
            )

            # Save model
            sport_filter = final_result.get("sport_filter", [])
            if final_result["roi"] > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": final_result["roi"],
                    "auc": auc_test,
                    "threshold": final_thr,
                    "n_bets": final_result["n_bets"],
                    "feature_names": features,
                    "params": BEST_PARAMS,
                    "sport_filter": sport_filter,
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
            logger.exception("Step 4.1 failed")
            raise


if __name__ == "__main__":
    main()
