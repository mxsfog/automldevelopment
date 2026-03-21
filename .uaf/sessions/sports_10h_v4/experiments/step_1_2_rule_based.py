"""Step 1.2: Rule-based baseline (пороговые правила по ML_Edge и Odds)."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    calc_roi,
    check_budget,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uaf/sports_10h_v4")
SESSION_ID = os.environ.get("UAF_SESSION_ID", "sports_10h_v4")


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step_1_2_rule_based") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.2")
            mlflow.set_tag("phase", "1")

            df = load_data()
            train, test = time_series_split(df)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "rule_based_threshold",
                    "gap_days": 7,
                }
            )

            # Правило 1: ML_Edge > threshold (ставим где модель видит edge)
            # Порог ищем на val (последние 20% train)
            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split]
            val = train.iloc[val_split:]

            # Для строк без ML_Edge -> не ставим (proba = 0)
            best_roi = -999.0
            best_rule = ""
            results = {}

            # Rule 1: ML_Edge > thr
            for thr in [0, 2, 5, 8, 10, 15, 20]:
                preds = np.where(val["ML_Edge"].fillna(-999) > thr, 1.0, 0.0)
                r = calc_roi(val, preds, threshold=0.5)
                key = f"ml_edge_gt_{thr}"
                results[key] = r
                logger.info(
                    "Val Rule ML_Edge>%d: ROI=%.2f%%, bets=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )
                if r["n_bets"] >= 20 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_rule = key

            # Rule 2: ML_P_Model > thr (модель уверена в победе)
            for thr in [55, 60, 65, 70, 75, 80, 85, 90]:
                preds = np.where(val["ML_P_Model"].fillna(0) > thr, 1.0, 0.0)
                r = calc_roi(val, preds, threshold=0.5)
                key = f"ml_p_model_gt_{thr}"
                results[key] = r
                logger.info(
                    "Val Rule ML_P_Model>%d: ROI=%.2f%%, bets=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )
                if r["n_bets"] >= 20 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_rule = key

            # Rule 3: ML_EV > thr (positive expected value)
            for thr in [0, 5, 10, 15, 20, 30, 50]:
                preds = np.where(val["ML_EV"].fillna(-999) > thr, 1.0, 0.0)
                r = calc_roi(val, preds, threshold=0.5)
                key = f"ml_ev_gt_{thr}"
                results[key] = r
                logger.info(
                    "Val Rule ML_EV>%d: ROI=%.2f%%, bets=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )
                if r["n_bets"] >= 20 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_rule = key

            # Rule 4: Odds в диапазоне (фавориты)
            for lo, hi in [(1.0, 1.5), (1.5, 2.0), (1.0, 2.0), (1.2, 1.8)]:
                preds = np.where((val["Odds"] >= lo) & (val["Odds"] <= hi), 1.0, 0.0)
                r = calc_roi(val, preds, threshold=0.5)
                key = f"odds_{lo}_{hi}"
                results[key] = r
                logger.info(
                    "Val Rule Odds[%.1f,%.1f]: ROI=%.2f%%, bets=%d", lo, hi, r["roi"], r["n_bets"]
                )
                if r["n_bets"] >= 20 and r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_rule = key

            logger.info("Best rule on val: %s, ROI=%.2f%%", best_rule, best_roi)

            # Применяем лучшее правило к test
            rule_params = best_rule.split("_")
            if best_rule.startswith("ml_edge_gt_"):
                thr = int(best_rule.split("_")[-1])
                test_preds = np.where(test["ML_Edge"].fillna(-999) > thr, 1.0, 0.0)
            elif best_rule.startswith("ml_p_model_gt_"):
                thr = int(best_rule.split("_")[-1])
                test_preds = np.where(test["ML_P_Model"].fillna(0) > thr, 1.0, 0.0)
            elif best_rule.startswith("ml_ev_gt_"):
                thr = int(best_rule.split("_")[-1])
                test_preds = np.where(test["ML_EV"].fillna(-999) > thr, 1.0, 0.0)
            elif best_rule.startswith("odds_"):
                parts = best_rule.replace("odds_", "").split("_")
                lo, hi = float(parts[0]), float(parts[1])
                test_preds = np.where((test["Odds"] >= lo) & (test["Odds"] <= hi), 1.0, 0.0)
            else:
                test_preds = np.zeros(len(test))

            test_result = calc_roi(test, test_preds, threshold=0.5)
            logger.info(
                "Test result: ROI=%.2f%%, bets=%d/%d, precision=%.3f, selectivity=%.3f",
                test_result["roi"],
                test_result["n_bets"],
                len(test),
                test_result["precision"],
                test_result["selectivity"],
            )

            mlflow.log_metrics(
                {
                    "roi": test_result["roi"],
                    "roi_val_best": best_roi,
                    "n_bets": test_result["n_bets"],
                    "precision": test_result["precision"],
                    "selectivity": test_result["selectivity"],
                    "n_won": test_result["n_won"],
                    "n_lost": test_result["n_lost"],
                }
            )
            mlflow.log_params({"best_rule": best_rule})

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            logger.exception("Step 1.2 failed")
            raise


if __name__ == "__main__":
    main()
