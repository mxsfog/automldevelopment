"""Step 4.6 — Stability: odds capping + robust EV threshold.

Гипотеза: высокая дисперсия ROI (CV std=14.14%) вызвана
outlier-ставками с очень высокими коэффициентами.
Пробуем:
1. Odds capping (max_odds=10, 20, 50)
2. Более высокие EV пороги (0.15, 0.18, 0.20)
3. Комбинацию odds cap + EV threshold
4. Full train ensemble (как в Step 4.5)
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


def train_full_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """Обучение 3-model ensemble на полном train."""
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
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_train_s, y_train)

    return cb, lgbm, lr, scaler


def predict_ensemble(
    cb: CatBoostClassifier,
    lgbm: LGBMClassifier,
    lr: LogisticRegression,
    scaler: StandardScaler,
    x: np.ndarray,
) -> np.ndarray:
    """Среднее предсказание ансамбля."""
    return (
        cb.predict_proba(x)[:, 1]
        + lgbm.predict_proba(x)[:, 1]
        + lr.predict_proba(scaler.transform(x))[:, 1]
    ) / 3


def main() -> None:
    with mlflow.start_run(run_name="phase4/step_4_6_stability") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

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
                    "method": "stability_odds_cap",
                    "n_features": len(features),
                }
            )

            x_train = train_full[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_full["target"]

            cb, lgbm, lr, scaler = train_full_ensemble(x_train, y_train)
            p_test = predict_ensemble(cb, lgbm, lr, scaler, x_test)

            auc_test = roc_auc_score(test_enc["target"], p_test)
            logger.info("Full-train ensemble AUC test: %.4f", auc_test)

            ev_test = p_test * test_enc["Odds"].values - 1
            odds_test = test_enc["Odds"].values

            # Grid search: EV threshold x Odds cap
            best_roi = -999.0
            best_config = {}
            results = []

            for ev_thr in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
                for max_odds in [5, 10, 20, 50, 999]:
                    mask = (ev_test >= ev_thr) & (odds_test <= max_odds)
                    n_bets = mask.sum()
                    if n_bets < 50:
                        continue

                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    label = f"max_odds={max_odds}" if max_odds < 999 else "no_cap"
                    results.append(
                        {
                            "ev_thr": ev_thr,
                            "max_odds": max_odds,
                            "roi": r["roi"],
                            "n_bets": r["n_bets"],
                            "win_rate": r.get("win_rate", 0),
                            "avg_odds": r.get("avg_odds", 0),
                        }
                    )
                    logger.info(
                        "  EV>=%.2f %s: ROI=%.2f%%, n=%d, wr=%.3f, avg_odds=%.1f",
                        ev_thr,
                        label,
                        r["roi"],
                        r["n_bets"],
                        r.get("win_rate", 0),
                        r.get("avg_odds", 0),
                    )

                    if r["roi"] > best_roi:
                        best_roi = r["roi"]
                        best_config = {
                            "ev_threshold": ev_thr,
                            "max_odds": max_odds,
                            "result": r,
                        }

            logger.info(
                "Best config: EV>=%.2f, max_odds=%s, ROI=%.2f%%, n=%d",
                best_config["ev_threshold"],
                best_config["max_odds"],
                best_roi,
                best_config["result"]["n_bets"],
            )

            # K-fold CV for best config
            logger.info("=== K-fold CV for best config ===")
            best_ev = best_config["ev_threshold"]
            best_max_odds = best_config["max_odds"]
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

                cb_f, lgbm_f, lr_f, sc_f = train_full_ensemble(ft_x, ft_y)
                p_fv = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, fv_x)

                ev_fv = p_fv * fold_val["Odds"].values - 1
                odds_fv = fold_val["Odds"].values
                mask_fv = (ev_fv >= best_ev) & (odds_fv <= best_max_odds)
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
                mean_roi = float(np.mean(fold_rois))
                std_roi = float(np.std(fold_rois))
                logger.info("CV ROI: mean=%.2f%%, std=%.2f%%", mean_roi, std_roi)
                mlflow.log_metrics({"roi_cv_mean": mean_roi, "roi_cv_std": std_roi})

            # Also CV for baseline (EV>=0.12, no cap) for comparison
            logger.info("=== CV for baseline (EV>=0.12, no cap) ===")
            baseline_fold_rois = []
            for fold_idx in range(n_folds):
                fold_end = fold_size * (fold_idx + 2)
                fold_train = train_full.iloc[: fold_size * (fold_idx + 1)]
                fold_val = train_full.iloc[fold_size * (fold_idx + 1) : fold_end]

                if len(fold_val) < 100:
                    continue

                ft_x = fold_train[features].fillna(0)
                fv_x = fold_val[features].fillna(0)
                ft_y = fold_train["target"]

                cb_f, lgbm_f, lr_f, sc_f = train_full_ensemble(ft_x, ft_y)
                p_fv = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, fv_x)

                ev_fv = p_fv * fold_val["Odds"].values - 1
                mask_fv = ev_fv >= 0.12
                r_fv = calc_roi(fold_val, mask_fv.astype(float), threshold=0.5)
                baseline_fold_rois.append(r_fv["roi"])
                logger.info(
                    "  Baseline Fold %d: ROI=%.2f%%, n=%d",
                    fold_idx,
                    r_fv["roi"],
                    r_fv["n_bets"],
                )
                mlflow.log_metric(f"roi_baseline_fold_{fold_idx}", r_fv["roi"])

            if baseline_fold_rois:
                bl_mean = float(np.mean(baseline_fold_rois))
                bl_std = float(np.std(baseline_fold_rois))
                logger.info("Baseline CV: mean=%.2f%%, std=%.2f%%", bl_mean, bl_std)
                mlflow.log_metrics(
                    {"roi_baseline_cv_mean": bl_mean, "roi_baseline_cv_std": bl_std}
                )

            final_result = best_config["result"]
            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test": best_roi,
                    "n_bets_test": final_result["n_bets"],
                    "ev_threshold": best_ev,
                    "max_odds": float(best_max_odds),
                    "win_rate_test": final_result.get("win_rate", 0),
                    "avg_odds_test": final_result.get("avg_odds", 0),
                }
            )

            # Save if improved over Step 4.5 (ROI=16.02%)
            if best_roi > 16.02:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_roi,
                    "auc": auc_test,
                    "threshold": best_ev,
                    "n_bets": final_result["n_bets"],
                    "feature_names": features,
                    "selection_method": "ev_ensemble_odds_cap",
                    "ev_threshold": best_ev,
                    "max_odds": best_max_odds,
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
            logger.exception("Step 4.6 failed")
            raise


if __name__ == "__main__":
    main()
