"""Step 4.5 — Full train ensemble + EV>=0.12.

Обучаем 3-model ensemble на полном train (без val split),
используем фиксированный EV порог 0.12.
Также пробуем k-fold CV для оценки стабильности.
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


def main() -> None:
    with mlflow.start_run(run_name="phase4/step_4_5_full_train") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Full train — no val split
            train_full = train.copy()
            train_full, _ = add_sport_market_features(train_full, train_full)
            test_enc, _ = add_sport_market_features(test.copy(), train_full)

            features = ACCEPTED_FEATURES

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_full),
                    "n_samples_test": len(test_enc),
                    "method": "full_train_ensemble",
                    "n_features": len(features),
                    "ev_threshold": 0.12,
                }
            )

            x_train = train_full[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_full["target"]

            # CatBoost (no eval_set)
            cb = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            )
            cb.fit(x_train, y_train)

            # LightGBM
            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm.fit(x_train, y_train)

            # LogReg
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
            logger.info("Full-train ensemble AUC test: %.4f", auc_test)

            # EV-based selection
            ev_test = p_test * test_enc["Odds"].values - 1

            for ev_thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
                mask = ev_test >= ev_thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  EV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            # Fixed threshold
            ev_mask = ev_test >= 0.12
            result_012 = calc_roi(test_enc, ev_mask.astype(float), threshold=0.5)
            logger.info("EV>=0.12: %s", result_012)

            # === K-fold time series CV for stability ===
            logger.info("=== Time series K-fold CV ===")
            n_folds = 5
            fold_size = len(train_full) // (n_folds + 1)
            fold_rois = []

            for fold_idx in range(n_folds):
                fold_end = fold_size * (fold_idx + 2)
                fold_train = train_full.iloc[: fold_size * (fold_idx + 1)]
                fold_val = train_full.iloc[fold_size * (fold_idx + 1) : fold_end]

                if len(fold_val) < 100:
                    continue

                ft_x = fold_train[features].fillna(0)
                fv_x = fold_val[features].fillna(0)
                ft_y = fold_train["target"]

                cb_f = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0,
                )
                cb_f.fit(ft_x, ft_y)

                lgbm_f = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm_f.fit(ft_x, ft_y)

                sc_f = StandardScaler()
                lr_f = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_f.fit(sc_f.fit_transform(ft_x), ft_y)

                p_fv = (
                    cb_f.predict_proba(fv_x)[:, 1]
                    + lgbm_f.predict_proba(fv_x)[:, 1]
                    + lr_f.predict_proba(sc_f.transform(fv_x))[:, 1]
                ) / 3

                ev_fv = p_fv * fold_val["Odds"].values - 1
                mask_fv = ev_fv >= 0.12
                r_fv = calc_roi(fold_val, mask_fv.astype(float), threshold=0.5)
                fold_rois.append(r_fv["roi"])
                logger.info(
                    "  Fold %d: ROI=%.2f%%, n=%d (train=%d, val=%d)",
                    fold_idx,
                    r_fv["roi"],
                    r_fv["n_bets"],
                    len(fold_train),
                    len(fold_val),
                )
                mlflow.log_metric(f"roi_fold_{fold_idx}", r_fv["roi"])

            if fold_rois:
                mean_roi = np.mean(fold_rois)
                std_roi = np.std(fold_rois)
                logger.info("CV ROI: mean=%.2f%%, std=%.2f%%", mean_roi, std_roi)
                mlflow.log_metrics(
                    {
                        "roi_cv_mean": mean_roi,
                        "roi_cv_std": std_roi,
                    }
                )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test": result_012["roi"],
                    "n_bets_test": result_012["n_bets"],
                    "ev_threshold": 0.12,
                    "win_rate_test": result_012.get("win_rate", 0),
                    "avg_odds_test": result_012.get("avg_odds", 0),
                }
            )

            # Save
            if result_012["roi"] > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": result_012["roi"],
                    "auc": auc_test,
                    "threshold": 0.12,
                    "n_bets": result_012["n_bets"],
                    "feature_names": features,
                    "selection_method": "ev_ensemble_full_train",
                    "ev_threshold": 0.12,
                    "sport_filter": [],
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
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
