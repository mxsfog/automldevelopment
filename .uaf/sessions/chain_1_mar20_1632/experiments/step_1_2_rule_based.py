"""Step 1.2: Rule-based baseline -- пороговое правило по ML_Edge."""

import logging
import os
import traceback

import mlflow
import numpy as np
from common import (
    calc_roi_at_thresholds,
    check_budget,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    logger.info("Step 1.2: Rule-based baseline (ML_Edge threshold)")
    df = load_data()
    train, test = time_series_split(df)

    with mlflow.start_run(run_name="phase1/step1.2_rule_based") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "threshold_rule",
                    "test_size": 0.2,
                    "rule_feature": "ML_Edge",
                }
            )

            # Правило: ставим только когда ML_Edge > порог
            # Нормализуем ML_Edge в [0,1] для совместимости с calc_roi
            ml_edge_test = test["ML_Edge"].values.astype(float)

            # Подбор порога на val (последние 20% train)
            val_split = int(len(train) * 0.8)
            val_df = train.iloc[val_split:]
            ml_edge_val = val_df["ML_Edge"].values.astype(float)

            # Пробуем разные пороги ML_Edge напрямую
            thresholds = [0, 2, 5, 8, 10, 15, 20, 25, 30]
            best_roi_val = -999.0
            best_t = 0

            for t in thresholds:
                mask = ml_edge_val >= t
                n_sel = mask.sum()
                if n_sel < 30:
                    continue
                sel = val_df.iloc[np.where(mask)[0]]
                staked = sel["USD"].sum()
                payout = sel["Payout_USD"].sum()
                roi_val = (payout - staked) / staked * 100 if staked > 0 else 0
                logger.info("Val: ML_Edge >= %d: ROI=%.2f%%, n=%d", t, roi_val, n_sel)
                if roi_val > best_roi_val:
                    best_roi_val = roi_val
                    best_t = t

            logger.info("Best val threshold: ML_Edge >= %d (val ROI=%.2f%%)", best_t, best_roi_val)

            # Apply to test
            test_mask = ml_edge_test >= best_t
            n_test_sel = int(test_mask.sum())
            if n_test_sel > 0:
                sel_test = test.iloc[np.where(test_mask)[0]]
                staked = sel_test["USD"].sum()
                payout = sel_test["Payout_USD"].sum()
                test_roi = (payout - staked) / staked * 100 if staked > 0 else 0
                test_wr = (sel_test["Status"] == "won").mean()
            else:
                test_roi = 0.0
                test_wr = 0.0

            logger.info(
                "Test: ML_Edge >= %d: ROI=%.2f%%, n=%d, WR=%.4f",
                best_t,
                test_roi,
                n_test_sel,
                test_wr,
            )

            # Также: ML_P_Model как вероятность для calc_roi
            proba_test = test["ML_P_Model"].values.astype(float) / 100.0
            roi_thresholds = calc_roi_at_thresholds(test, proba_test)
            for t_val, res in roi_thresholds.items():
                logger.info(
                    "ML_P_Model >= %.0f%%: ROI=%.2f%%, n=%d",
                    t_val * 100,
                    res["roi"],
                    res["n_bets"],
                )

            mlflow.log_metrics(
                {
                    "roi": test_roi,
                    "best_edge_threshold": best_t,
                    "val_roi_at_threshold": best_roi_val,
                    "n_bets_selected": n_test_sel,
                    "win_rate_selected": test_wr,
                    "pct_selected": n_test_sel / len(test) * 100,
                }
            )

            for t_val, res in roi_thresholds.items():
                mlflow.log_metric(f"roi_pmodel_{int(t_val * 100)}", res["roi"])
                mlflow.log_metric(f"nbets_pmodel_{int(t_val * 100)}", res["n_bets"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            raise


if __name__ == "__main__":
    main()
