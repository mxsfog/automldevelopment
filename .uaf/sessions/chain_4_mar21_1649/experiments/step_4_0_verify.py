"""Step 4.0 -- Chain Verification.

Загрузка метаданных из chain_3_mar21_1455, воспроизведение 3-model ensemble
(CB + LGBM + LR) с conf_ev_0.15 selection. Ожидаемый ROI ~ 27.95%.
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
    add_sport_market_features,
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)

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

PREV_MODEL_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_3_mar21_1455/models/best")


def main() -> None:
    """Верификация модели предыдущей сессии."""
    with mlflow.start_run(run_name="chain/verify") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "verification")
            mlflow.set_tag("status", "running")

            meta_path = PREV_MODEL_DIR / "metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)
            logger.info("Loaded metadata: %s", metadata)

            features = metadata["feature_names"]
            ev_threshold = metadata.get("ev_threshold", 0.15)
            expected_roi = metadata["roi"]
            selection_method = metadata.get("selection_method", "conf_ev_0.15")

            mlflow.log_params(
                {
                    "prev_session": metadata["session_id"],
                    "prev_roi": expected_roi,
                    "ev_threshold": ev_threshold,
                    "selection_method": selection_method,
                    "n_features": len(features),
                    "validation_scheme": "time_series",
                    "seed": 42,
                }
            )

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            mlflow.log_params(
                {
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                }
            )

            # Feature engineering (fit on train)
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_enc["target"]

            # 3-model ensemble: CB + LGBM + LR
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

            # Ensemble average probabilities
            p_cb = cb.predict_proba(x_test)[:, 1]
            p_lgbm = lgbm.predict_proba(x_test)[:, 1]
            p_lr = lr.predict_proba(x_test_s)[:, 1]
            p_test = (p_cb + p_lgbm + p_lr) / 3

            auc_test = roc_auc_score(test_enc["target"], p_test)
            logger.info("Verification ensemble AUC: %.4f", auc_test)

            # conf_ev_0.15 selection strategy
            p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
            ev_test = p_test * test_enc["Odds"].values - 1
            conf = 1.0 / (1.0 + p_std * 10)
            conf_ev = ev_test * conf
            conf_ev_mask = conf_ev >= ev_threshold

            # ROI via conf_ev
            n_selected = conf_ev_mask.sum()
            selected = test_enc[conf_ev_mask]
            if n_selected > 0:
                total_staked = n_selected
                payouts = selected["target"] * selected["Odds"]
                roi_verified = (payouts.sum() - total_staked) / total_staked * 100
            else:
                roi_verified = -100.0

            # Also compute simple EV selection for comparison
            ev_simple_mask = ev_test >= 0.12
            result_ev = calc_roi(test_enc, ev_simple_mask.astype(float), threshold=0.5)

            roi_delta = abs(roi_verified - expected_roi)
            verified_ok = roi_delta < 2.0

            logger.info(
                "conf_ev_0.15: ROI=%.2f%% (n=%d), expected=%.2f%%, delta=%.2f%%, OK=%s",
                roi_verified,
                n_selected,
                expected_roi,
                roi_delta,
                verified_ok,
            )
            logger.info(
                "Simple EV>=0.12: ROI=%.2f%% (n=%d)", result_ev["roi"], result_ev["n_bets"]
            )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test_conf_ev": roi_verified,
                    "roi_test_ev_simple": result_ev["roi"],
                    "roi_expected": expected_roi,
                    "roi_delta": roi_delta,
                    "n_bets_conf_ev": n_selected,
                    "n_bets_ev_simple": result_ev["n_bets"],
                    "verified": 1 if verified_ok else 0,
                }
            )

            # Save as baseline for this session
            model_dir = SESSION_DIR / "models" / "best"
            model_dir.mkdir(parents=True, exist_ok=True)
            cb.save_model(str(model_dir / "model.cbm"))
            meta_save = {
                "framework": "ensemble_cb_lgbm_lr",
                "model_file": "model.cbm",
                "roi": roi_verified,
                "auc": auc_test,
                "threshold": 0.12,
                "ev_threshold": ev_threshold,
                "n_bets": int(n_selected),
                "feature_names": features,
                "selection_method": "conf_ev_0.15",
                "selection_formula": (
                    "EV = p_mean * odds - 1; conf = 1/(1+p_std*10); "
                    "select where EV*conf >= 0.15"
                ),
                "ensemble": "avg(CatBoost, LightGBM, LogisticRegression)",
                "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                "sport_filter": [],
                "session_id": SESSION_ID,
            }
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(meta_save, f, indent=2)

            mlflow.set_tag("verified", str(verified_ok))
            mlflow.set_tag("reproduced_roi", f"{roi_verified:.2f}")
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
