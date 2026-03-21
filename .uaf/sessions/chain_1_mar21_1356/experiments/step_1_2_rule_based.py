"""Step 1.2 — Rule-based baseline.

Гипотеза: простое пороговое правило по ML_Edge (edge модели) или ML_EV
дает лучший ROI чем ставки на всё.
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

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

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
    with mlflow.start_run(run_name="phase1/step_1_2_rule_based") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)
            logger.info("Train: %d, Test: %d", len(train), len(test))

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                    "method": "threshold_rule",
                }
            )

            # Стратегия 1: ставить только на ML_Edge > порог
            # Подбираем порог на последних 20% train (val)
            val_split = int(len(train) * 0.8)
            val = train.iloc[val_split:]

            best_rule = None
            best_roi = -999.0
            results = {}

            # Перебираем признаки и пороги
            for feature in ["ML_Edge", "ML_EV", "ML_P_Model"]:
                if feature not in val.columns:
                    continue
                vals = val[feature].dropna()
                if len(vals) == 0:
                    continue

                for threshold in np.arange(-20, 60, 2):
                    if feature == "ML_P_Model":
                        threshold_p = threshold + 40  # shift range for probability
                        mask = val[feature].fillna(0) >= threshold_p
                    else:
                        mask = val[feature].fillna(0) >= threshold

                    n_selected = mask.sum()
                    if n_selected < 50:
                        continue

                    selected = val[mask]
                    staked = n_selected * 1.0
                    payout = (selected["target"] * selected["Odds"]).sum()
                    roi = (payout - staked) / staked * 100

                    key = f"{feature}>={threshold}"
                    if feature == "ML_P_Model":
                        key = f"{feature}>={threshold_p}"

                    if roi > best_roi:
                        best_roi = roi
                        best_rule = key
                        results[key] = {
                            "roi": round(roi, 2),
                            "n_bets": int(n_selected),
                            "win_rate": round(selected["target"].mean(), 4),
                        }

            logger.info("Best rule on val: %s -> ROI=%.2f%%", best_rule, best_roi)

            # Применяем лучшее правило к test
            # Парсим правило
            if best_rule:
                parts = best_rule.split(">=")
                feat_name = parts[0]
                thresh_val = float(parts[1])

                # Val result
                mlflow.log_metrics(
                    {
                        "roi_val_best_rule": best_roi,
                    }
                )
                mlflow.log_param("best_rule", best_rule)

                # Test
                test_mask = test[feat_name].fillna(0) >= thresh_val
                test_result = calc_roi(test, test_mask.astype(float), threshold=0.5)
                logger.info("Test result for %s: %s", best_rule, test_result)

                mlflow.log_metrics(
                    {
                        "roi_test_best_rule": test_result["roi"],
                        "n_bets_test": test_result["n_bets"],
                        "win_rate_test": test_result.get("win_rate", 0),
                    }
                )

            # Также проверяем несколько фиксированных правил на test
            fixed_rules = {
                "ML_Edge>=0": ("ML_Edge", 0),
                "ML_Edge>=5": ("ML_Edge", 5),
                "ML_Edge>=10": ("ML_Edge", 10),
                "ML_EV>=0": ("ML_EV", 0),
                "ML_EV>=10": ("ML_EV", 10),
                "ML_P_Model>=55": ("ML_P_Model", 55),
                "ML_P_Model>=60": ("ML_P_Model", 60),
            }

            for rule_name, (feat, thr) in fixed_rules.items():
                mask = test[feat].fillna(0) >= thr
                result = calc_roi(test, mask.astype(float), threshold=0.5)
                logger.info(
                    "Test %s: ROI=%.2f%%, n=%d", rule_name, result["roi"], result["n_bets"]
                )
                safe_name = rule_name.replace(">=", "_gte_")
                mlflow.log_metric(f"roi_test_{safe_name}", result["roi"])
                mlflow.log_metric(f"n_bets_test_{safe_name}", result["n_bets"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 1.2 failed")
            raise


if __name__ == "__main__":
    main()
