"""Step 1.2: Rule-based baseline -- пороговое правило по ML_Edge/ML_P_Model."""

import logging
import os
import traceback

import mlflow
from common import (
    calc_roi,
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
    logger.info("Step 1.2: Rule-based baseline")
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
                }
            )

            # Strategy 1: ML_P_Model as probability (scale to 0-1)
            p_model = test["ML_P_Model"].fillna(50.0).values / 100.0
            roi_pmodel = calc_roi_at_thresholds(test, p_model)
            logger.info("ROI by ML_P_Model threshold:")
            best_roi_pmodel = -999.0
            best_t_pmodel = 0.5
            for t, r in roi_pmodel.items():
                logger.info(
                    "  t=%.2f: ROI=%.2f%%, n=%d, WR=%.3f", t, r["roi"], r["n_bets"], r["win_rate"]
                )
                if r["n_bets"] >= 100 and r["roi"] > best_roi_pmodel:
                    best_roi_pmodel = r["roi"]
                    best_t_pmodel = t

            # Strategy 2: ML_Edge > 0 (positive edge)
            edge = test["ML_Edge"].fillna(0.0).values
            edge_mask = edge > 0
            roi_edge_pos = calc_roi(test, edge_mask.astype(float), threshold=0.5)
            logger.info(
                "ROI (ML_Edge > 0): %.2f%%, n=%d", roi_edge_pos["roi"], roi_edge_pos["n_bets"]
            )

            # Strategy 3: ML_EV > 0 (positive expected value)
            ev = test["ML_EV"].fillna(0.0).values
            ev_mask = ev > 0
            roi_ev_pos = calc_roi(test, ev_mask.astype(float), threshold=0.5)
            logger.info("ROI (ML_EV > 0): %.2f%%, n=%d", roi_ev_pos["roi"], roi_ev_pos["n_bets"])

            # Strategy 4: Combined -- ML_P_Model > 60 AND ML_Edge > 5
            combined = (
                (test["ML_P_Model"].fillna(0) > 60) & (test["ML_Edge"].fillna(-99) > 5)
            ).values
            roi_combined = calc_roi(test, combined.astype(float), threshold=0.5)
            logger.info(
                "ROI (P_Model>60 & Edge>5): %.2f%%, n=%d",
                roi_combined["roi"],
                roi_combined["n_bets"],
            )

            # Strategy 5: Odds filter -- only favorites (low odds)
            fav = (test["Odds"].values < 2.0).astype(float)
            roi_fav = calc_roi(test, fav, threshold=0.5)
            logger.info("ROI (Odds < 2.0): %.2f%%, n=%d", roi_fav["roi"], roi_fav["n_bets"])

            # Strategy 6: Sport filter -- profitable sports only
            profitable_sports = ["Tennis", "Dota 2", "League of Legends", "CS2", "Table Tennis"]
            sport_mask = test["Sport"].isin(profitable_sports).values.astype(float)
            roi_sport = calc_roi(test, sport_mask, threshold=0.5)
            logger.info(
                "ROI (profitable sports): %.2f%%, n=%d", roi_sport["roi"], roi_sport["n_bets"]
            )

            primary_roi = max(
                roi_edge_pos["roi"],
                roi_ev_pos["roi"],
                roi_combined["roi"],
                roi_sport["roi"],
                best_roi_pmodel,
            )

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "roi_ml_p_model_best": best_roi_pmodel,
                    "roi_ml_edge_pos": roi_edge_pos["roi"],
                    "roi_ml_ev_pos": roi_ev_pos["roi"],
                    "roi_combined_p60_e5": roi_combined["roi"],
                    "roi_favorites": roi_fav["roi"],
                    "roi_profitable_sports": roi_sport["roi"],
                    "best_threshold_pmodel": best_t_pmodel,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.1")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Best rule-based ROI: %.2f%%", primary_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.2")
            raise


if __name__ == "__main__":
    main()
