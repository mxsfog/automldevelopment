"""Step 4.32 — Temporal block analysis for best strategies.

Split test into 4 weekly blocks and evaluate strategy stability.
Goal: understand temporal variance of different strategy families.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
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
mlflow.set_experiment("uaf/chain_3_mar21_1455")

FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcome_Odds",
    "n_outcomes",
    "mean_outcome_odds",
    "max_outcome_odds",
    "min_outcome_odds",
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def train_ensemble(x: pd.DataFrame, y: pd.Series) -> tuple:
    """3-model ensemble."""
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y)
    return cb, lgbm, lr, scaler


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
    return p_mean, p_std


def main() -> None:
    """Temporal block stability analysis."""
    with mlflow.start_run(run_name="phase4/temporal_blocks") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            mlflow.log_params(
                {
                    "validation_scheme": "temporal_blocks",
                    "seed": 42,
                    "method": "temporal_block_analysis",
                    "n_blocks": 4,
                }
            )

            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb, lgbm, lr, scaler = train_ensemble(x_train, train_enc["target"])
            p_t, s_t = predict_ensemble(cb, lgbm, lr, scaler, x_test)
            odds_test = test_enc["Odds"].values

            auc_test = roc_auc_score(test_enc["target"], p_t)

            ev_t = p_t * odds_test - 1
            conf_t = 1 / (1 + s_t * 10)
            ev_conf_t = ev_t * conf_t
            p_implied_t = 1.0 / odds_test
            edge_t = p_t - p_implied_t

            # Split test into 4 temporal blocks
            n_test = len(test_enc)
            block_size = n_test // 4
            blocks = []
            for i in range(4):
                start = i * block_size
                end = start + block_size if i < 3 else n_test
                blocks.append((start, end))

            strategies = {
                "confev_0.15": lambda: ev_conf_t >= 0.15,
                "confev_0.10": lambda: ev_conf_t >= 0.10,
                "ev_0.05": lambda: ev_t >= 0.05,
                "ev_0.10": lambda: ev_t >= 0.10,
                "pmean_0.55": lambda: p_t >= 0.55,
                "edge_cap3_e0.10": lambda: (edge_t >= 0.10) & (odds_test <= 3),
                "edge_cap5_e0.10": lambda: (edge_t >= 0.10) & (odds_test <= 5),
            }

            logger.info("=== Temporal block analysis (4 blocks) ===")
            block_results: dict[str, list[float]] = {}

            for name, mask_fn in strategies.items():
                mask = mask_fn()
                block_rois = []
                for _bi, (start, end) in enumerate(blocks):
                    block_mask = mask[start:end]
                    block_df = test_enc.iloc[start:end]
                    r = calc_roi(block_df, block_mask.astype(float), threshold=0.5)
                    block_rois.append(r["roi"])

                block_results[name] = block_rois
                mean_roi = np.mean(block_rois)
                std_roi = np.std(block_rois)
                min_roi = np.min(block_rois)
                n_positive = sum(1 for r in block_rois if r > 0)

                logger.info(
                    "  %s: blocks=[%.1f%%, %.1f%%, %.1f%%, %.1f%%] "
                    "mean=%.1f%% std=%.1f%% pos=%d/4",
                    name,
                    *block_rois,
                    mean_roi,
                    std_roi,
                    n_positive,
                )

                mlflow.log_metrics(
                    {
                        f"block_mean_{name}": round(mean_roi, 2),
                        f"block_std_{name}": round(std_roi, 2),
                        f"block_min_{name}": round(min_roi, 2),
                    }
                )

            # Most temporally stable strategy
            stability = {name: np.std(rois) for name, rois in block_results.items()}
            most_stable = min(stability, key=stability.get)
            logger.info(
                "Most temporally stable: %s (std=%.2f%%)",
                most_stable,
                stability[most_stable],
            )

            mlflow.log_metrics({"auc_test": auc_test})
            mlflow.set_tag("most_stable_strategy", most_stable)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.32 failed")
            raise


if __name__ == "__main__":
    main()
