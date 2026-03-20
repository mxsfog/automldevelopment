"""Step 1.2 — Rule-based baseline. Пороговые правила по ML_Edge и ML_P_Model."""

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
    compute_roi,
    find_best_threshold,
    load_bets,
    time_series_split,
)

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.warning("Budget hard stop. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="phase1/step1.2_rule_based") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("phase", "1")
        mlflow.set_tag("method", "threshold_rule")

        df = load_bets(with_outcomes=True)
        splits = time_series_split(df, n_splits=5, gap_days=7)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_splits": len(splits),
                "gap_days": 7,
                "method": "threshold_rule",
            }
        )

        # Стратегия 1: ML_P_Model > threshold (берем ставки с высокой модельной вероятностью)
        # Стратегия 2: ML_Edge > threshold (берем ставки с положительным edge)
        # Стратегия 3: Odds в определенном диапазоне

        strategies = {}

        # --- Strategy: ML_P_Model threshold ---
        fold_rois_pmodel = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            y_val = val_df["target"].values
            stakes_val = val_df["USD"].values
            payouts_val = val_df["Payout_USD"].values

            # Используем ML_P_Model / 100 как вероятность
            p_model = val_df["ML_P_Model"].fillna(50.0).values / 100.0

            # На train найдем лучший порог
            y_train = train_df["target"].values
            p_train = train_df["ML_P_Model"].fillna(50.0).values / 100.0
            stakes_train = train_df["USD"].values
            payouts_train = train_df["Payout_USD"].values

            best_t, _ = find_best_threshold(
                y_train,
                p_train,
                stakes_train,
                payouts_train,
                thresholds=np.arange(0.50, 0.90, 0.02),
            )

            result = compute_roi(y_val, p_model, stakes_val, payouts_val, threshold=best_t)
            fold_rois_pmodel.append(result["roi"])
            mlflow.log_metric(f"roi_pmodel_fold_{fold_idx}", result["roi"])
            mlflow.log_metric(f"threshold_pmodel_fold_{fold_idx}", best_t)
            mlflow.log_metric(f"n_selected_pmodel_fold_{fold_idx}", result["n_selected"])
            logger.info(
                "ML_P_Model fold %d: threshold=%.2f, ROI=%.2f%%, selected=%d/%d",
                fold_idx,
                best_t,
                result["roi"],
                result["n_selected"],
                result["n_total"],
            )

        strategies["ML_P_Model"] = {
            "roi_mean": float(np.mean(fold_rois_pmodel)),
            "roi_std": float(np.std(fold_rois_pmodel)),
        }

        # --- Strategy: ML_Edge threshold ---
        fold_rois_edge = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            y_val = val_df["target"].values
            stakes_val = val_df["USD"].values
            payouts_val = val_df["Payout_USD"].values

            # Нормализуем ML_Edge в [0, 1]: edge > 0 = "выгодная ставка"
            edge_val = val_df["ML_Edge"].fillna(0.0).values
            edge_train = train_df["ML_Edge"].fillna(0.0).values

            # Переводим в "вероятность": sigmoid(edge/10)
            def edge_to_prob(edge: np.ndarray) -> np.ndarray:
                return 1.0 / (1.0 + np.exp(-edge / 10.0))

            p_val = edge_to_prob(edge_val)
            p_train = edge_to_prob(edge_train)

            y_train = train_df["target"].values
            stakes_train = train_df["USD"].values
            payouts_train = train_df["Payout_USD"].values

            best_t, _ = find_best_threshold(
                y_train,
                p_train,
                stakes_train,
                payouts_train,
                thresholds=np.arange(0.45, 0.80, 0.02),
            )

            result = compute_roi(y_val, p_val, stakes_val, payouts_val, threshold=best_t)
            fold_rois_edge.append(result["roi"])
            mlflow.log_metric(f"roi_edge_fold_{fold_idx}", result["roi"])
            mlflow.log_metric(f"threshold_edge_fold_{fold_idx}", best_t)
            logger.info(
                "ML_Edge fold %d: threshold=%.2f, ROI=%.2f%%, selected=%d/%d",
                fold_idx,
                best_t,
                result["roi"],
                result["n_selected"],
                result["n_total"],
            )

        strategies["ML_Edge"] = {
            "roi_mean": float(np.mean(fold_rois_edge)),
            "roi_std": float(np.std(fold_rois_edge)),
        }

        # --- Strategy: Odds range (low odds = higher implied prob) ---
        fold_rois_odds = []
        for fold_idx, (_train_idx, val_idx) in enumerate(splits):
            val_df = df.iloc[val_idx]
            y_val = val_df["target"].values
            stakes_val = val_df["USD"].values
            payouts_val = val_df["Payout_USD"].values

            # Implied probability from odds
            p_implied = 1.0 / val_df["Odds"].values
            # Фильтр: odds 1.2-2.5 (сбалансированные)
            selected = (val_df["Odds"].values >= 1.2) & (val_df["Odds"].values <= 2.5)
            p_val = np.where(selected, 1.0, 0.0)

            result = compute_roi(y_val, p_val, stakes_val, payouts_val, threshold=0.5)
            fold_rois_odds.append(result["roi"])
            mlflow.log_metric(f"roi_odds_fold_{fold_idx}", result["roi"])
            logger.info(
                "Odds[1.2-2.5] fold %d: ROI=%.2f%%, selected=%d/%d",
                fold_idx,
                result["roi"],
                result["n_selected"],
                result["n_total"],
            )

        strategies["Odds_range"] = {
            "roi_mean": float(np.mean(fold_rois_odds)),
            "roi_std": float(np.std(fold_rois_odds)),
        }

        # Лучшая стратегия
        best_name = max(strategies, key=lambda k: strategies[k]["roi_mean"])
        best = strategies[best_name]

        for name, s in strategies.items():
            mlflow.log_metric(f"roi_mean_{name}", s["roi_mean"])
            mlflow.log_metric(f"roi_std_{name}", s["roi_std"])
            logger.info(
                "Strategy %s: ROI_mean=%.2f%% +/- %.2f%%", name, s["roi_mean"], s["roi_std"]
            )

        mlflow.log_metric("roi_mean", best["roi_mean"])
        mlflow.log_metric("roi_std", best["roi_std"])
        mlflow.set_tag("best_strategy", best_name)

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.1")

        logger.info("Best rule-based: %s, ROI=%.2f%%", best_name, best["roi_mean"])
        print(f"RESULT: best_strategy={best_name}, roi_mean={best['roi_mean']:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "runtime_error")
        logger.exception("Step 1.2 failed")
        raise
