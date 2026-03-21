"""Step 4.0 — Chain Verification.

Загрузка модели из предыдущей сессии chain_1_mar21_1356,
воспроизведение пайплайна данных, верификация ROI ~ 16.02%.
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

PREV_SESSION_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_1_mar21_1356")
PREV_MODEL_DIR = PREV_SESSION_DIR / "models" / "best"


def main() -> None:
    """Верификация модели предыдущей сессии."""
    with mlflow.start_run(run_name="chain/verify") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "verification")
            mlflow.set_tag("status", "running")

            # 1. Загрузка metadata
            meta_path = PREV_MODEL_DIR / "metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)
            logger.info("Loaded metadata: %s", metadata)

            features = metadata["feature_names"]
            ev_threshold = metadata["ev_threshold"]
            expected_roi = metadata["roi"]

            mlflow.log_params(
                {
                    "prev_session": metadata["session_id"],
                    "prev_roi": expected_roi,
                    "ev_threshold": ev_threshold,
                    "selection_method": metadata["selection_method"],
                    "n_features": len(features),
                    "validation_scheme": "time_series",
                    "seed": 42,
                }
            )

            # 2. Загрузка данных и подготовка
            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            mlflow.log_params(
                {
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                }
            )

            # 3. Feature engineering (fit на train)
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_enc["target"]

            # 4. Воспроизведение ансамбля (точно как в step_4_5)
            cb = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            )
            cb.fit(x_train, y_train)

            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm.fit(x_train, y_train)

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(x_train_s, y_train)

            # Ensemble
            p_test = (
                cb.predict_proba(x_test)[:, 1]
                + lgbm.predict_proba(x_test)[:, 1]
                + lr.predict_proba(x_test_s)[:, 1]
            ) / 3

            auc_test = roc_auc_score(test_enc["target"], p_test)
            logger.info("Verification ensemble AUC: %.4f", auc_test)

            # EV selection
            ev_test = p_test * test_enc["Odds"].values - 1
            ev_mask = ev_test >= ev_threshold
            result = calc_roi(test_enc, ev_mask.astype(float), threshold=0.5)
            logger.info("Verification result: %s", result)

            roi_verified = result["roi"]
            roi_delta = abs(roi_verified - expected_roi)
            verified_ok = roi_delta < 2.0  # допуск 2 п.п.

            logger.info(
                "Verification: ROI=%.2f%% (expected=%.2f%%, delta=%.2f%%), OK=%s",
                roi_verified,
                expected_roi,
                roi_delta,
                verified_ok,
            )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test": roi_verified,
                    "roi_expected": expected_roi,
                    "roi_delta": roi_delta,
                    "n_bets_test": result["n_bets"],
                    "win_rate_test": result.get("win_rate", 0),
                    "avg_odds_test": result.get("avg_odds", 0),
                    "verified": 1 if verified_ok else 0,
                }
            )

            mlflow.set_tag("verified", str(verified_ok))
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.0 verification failed")
            raise


if __name__ == "__main__":
    main()
