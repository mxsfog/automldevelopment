"""Step 4.1 — Calibrated EV ensemble.

Гипотеза: калибровка вероятностей ансамбля через isotonic regression на val
улучшит точность EV-расчёта и повысит ROI.

Подходы:
1. Baseline: 3-model ensemble (CB+LGBM+LR) full train, EV>=0.12
2. Calibrated: тот же ensemble, но с CalibratedClassifierCV (isotonic)
3. 4-model: добавить XGBoost
4. EV grid: оптимизация EV threshold на val (последние 20% train)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def find_best_ev_threshold(
    df: pd.DataFrame,
    probas: np.ndarray,
    min_bets: int = 50,
) -> tuple[float, dict]:
    """Поиск лучшего EV порога на валидации."""
    ev = probas * df["Odds"].values - 1
    best_roi = -999.0
    best_thr = 0.12
    best_result = {}

    for thr in np.arange(0.02, 0.40, 0.01):
        mask = ev >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        if r["n_bets"] >= min_bets and r["roi"] > best_roi:
            best_roi = r["roi"]
            best_thr = round(float(thr), 2)
            best_result = r

    return best_thr, best_result


def main() -> None:
    """Калибровка + EV grid optimization."""
    with mlflow.start_run(run_name="phase4/step_4_1_calibrated_ev") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # val split для подбора EV threshold (последние 20% train)
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            # Feature engineering
            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val.copy(), train_fit_enc)
            # Full train для финальной модели
            train_full, _ = add_sport_market_features(train.copy(), train)
            test_enc, _ = add_sport_market_features(test.copy(), train_full)

            features = FEATURES

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit_enc),
                    "n_samples_val": len(val_enc),
                    "n_samples_train_full": len(train_full),
                    "n_samples_test": len(test_enc),
                    "method": "calibrated_ev_ensemble",
                    "n_features": len(features),
                }
            )

            x_fit = train_fit_enc[features].fillna(0)
            x_val = val_enc[features].fillna(0)
            y_fit = train_fit_enc["target"]

            scaler_fit = StandardScaler()
            x_fit_s = scaler_fit.fit_transform(x_fit)
            x_val_s = scaler_fit.transform(x_val)

            # --- Approach 1: Baseline 3-model on train_fit, eval on val ---
            cb1 = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb1.fit(x_fit, y_fit)

            lgbm1 = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm1.fit(x_fit, y_fit)

            lr1 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr1.fit(x_fit_s, y_fit)

            p_val_base = (
                cb1.predict_proba(x_val)[:, 1]
                + lgbm1.predict_proba(x_val)[:, 1]
                + lr1.predict_proba(x_val_s)[:, 1]
            ) / 3

            # Baseline EV on val
            ev_thr_base, result_base = find_best_ev_threshold(val_enc, p_val_base)
            logger.info("Baseline val: best_ev=%.2f, %s", ev_thr_base, result_base)

            # --- Approach 2: Calibrated ensemble ---
            cb_cal = CalibratedClassifierCV(cb1, method="isotonic", cv="prefit")
            cb_cal.fit(x_val, val_enc["target"])

            lgbm_cal = CalibratedClassifierCV(lgbm1, method="isotonic", cv="prefit")
            lgbm_cal.fit(x_val, val_enc["target"])

            lr_cal = CalibratedClassifierCV(lr1, method="isotonic", cv="prefit")
            lr_cal.fit(x_val_s, val_enc["target"])

            p_val_cal = (
                cb_cal.predict_proba(x_val)[:, 1]
                + lgbm_cal.predict_proba(x_val)[:, 1]
                + lr_cal.predict_proba(x_val_s)[:, 1]
            ) / 3

            ev_thr_cal, result_cal = find_best_ev_threshold(val_enc, p_val_cal)
            logger.info("Calibrated val: best_ev=%.2f, %s", ev_thr_cal, result_cal)

            # --- Approach 3: 4-model ensemble (+ XGBoost) ---
            xgb1 = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=0,
                min_child_weight=50,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            xgb1.fit(x_fit, y_fit)

            p_val_4m = (
                cb1.predict_proba(x_val)[:, 1]
                + lgbm1.predict_proba(x_val)[:, 1]
                + lr1.predict_proba(x_val_s)[:, 1]
                + xgb1.predict_proba(x_val)[:, 1]
            ) / 4

            ev_thr_4m, result_4m = find_best_ev_threshold(val_enc, p_val_4m)
            logger.info("4-model val: best_ev=%.2f, %s", ev_thr_4m, result_4m)

            # --- Pick best approach based on val ---
            approaches = {
                "baseline_3m": (ev_thr_base, result_base, 3),
                "calibrated_3m": (ev_thr_cal, result_cal, 3),
                "ensemble_4m": (ev_thr_4m, result_4m, 4),
            }
            best_approach = max(approaches, key=lambda k: approaches[k][1].get("roi", -999))
            best_ev_thr = approaches[best_approach][0]
            best_n_models = approaches[best_approach][2]
            logger.info(
                "Best approach on val: %s, EV>=%.2f, ROI=%.2f%%",
                best_approach,
                best_ev_thr,
                approaches[best_approach][1]["roi"],
            )

            mlflow.log_params(
                {
                    "best_approach": best_approach,
                    "best_ev_threshold": best_ev_thr,
                    "n_models": best_n_models,
                }
            )

            # --- Final: train on full train, apply to test ---
            x_full = train_full[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_full = train_full["target"]

            scaler_full = StandardScaler()
            x_full_s = scaler_full.fit_transform(x_full)
            x_test_s = scaler_full.transform(x_test)

            cb_f = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb_f.fit(x_full, y_full)

            lgbm_f = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm_f.fit(x_full, y_full)

            lr_f = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_f.fit(x_full_s, y_full)

            if best_n_models == 4:
                xgb_f = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbosity=0,
                    min_child_weight=50,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
                xgb_f.fit(x_full, y_full)
                p_test_final = (
                    cb_f.predict_proba(x_test)[:, 1]
                    + lgbm_f.predict_proba(x_test)[:, 1]
                    + lr_f.predict_proba(x_test_s)[:, 1]
                    + xgb_f.predict_proba(x_test)[:, 1]
                ) / 4
            else:
                p_test_final = (
                    cb_f.predict_proba(x_test)[:, 1]
                    + lgbm_f.predict_proba(x_test)[:, 1]
                    + lr_f.predict_proba(x_test_s)[:, 1]
                ) / 3

            auc_test = roc_auc_score(test_enc["target"], p_test_final)
            logger.info("Final test AUC: %.4f", auc_test)

            # EV selection with optimized threshold
            ev_test = p_test_final * test_enc["Odds"].values - 1

            # Log all EV thresholds for analysis
            for ev_t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
                mask = ev_test >= ev_t
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  EV>=%.2f: ROI=%.2f%%, n=%d", ev_t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_ev_{ev_t:.2f}", r["roi"])
                mlflow.log_metric(f"n_bets_ev_{ev_t:.2f}", r["n_bets"])

            # Apply best val EV threshold to test
            ev_mask = ev_test >= best_ev_thr
            result_final = calc_roi(test_enc, ev_mask.astype(float), threshold=0.5)
            logger.info("Final (EV>=%.2f): %s", best_ev_thr, result_final)

            # Also check fixed EV=0.12 (baseline comparison)
            ev_mask_012 = ev_test >= 0.12
            result_012 = calc_roi(test_enc, ev_mask_012.astype(float), threshold=0.5)
            logger.info("Baseline EV>=0.12: %s", result_012)

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test": result_final["roi"],
                    "roi_test_ev012": result_012["roi"],
                    "n_bets_test": result_final["n_bets"],
                    "ev_threshold": best_ev_thr,
                    "win_rate_test": result_final.get("win_rate", 0),
                    "avg_odds_test": result_final.get("avg_odds", 0),
                }
            )

            # CV stability check
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

                sc_f = StandardScaler()
                ft_x_s = sc_f.fit_transform(ft_x)
                fv_x_s = sc_f.transform(fv_x)

                cb_cv = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
                )
                cb_cv.fit(ft_x, ft_y)
                lgbm_cv = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm_cv.fit(ft_x, ft_y)
                lr_cv = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_cv.fit(ft_x_s, ft_y)

                if best_n_models == 4:
                    xgb_cv = XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                        verbosity=0,
                        min_child_weight=50,
                        use_label_encoder=False,
                        eval_metric="logloss",
                    )
                    xgb_cv.fit(ft_x, ft_y)
                    p_fv = (
                        cb_cv.predict_proba(fv_x)[:, 1]
                        + lgbm_cv.predict_proba(fv_x)[:, 1]
                        + lr_cv.predict_proba(fv_x_s)[:, 1]
                        + xgb_cv.predict_proba(fv_x)[:, 1]
                    ) / 4
                else:
                    p_fv = (
                        cb_cv.predict_proba(fv_x)[:, 1]
                        + lgbm_cv.predict_proba(fv_x)[:, 1]
                        + lr_cv.predict_proba(fv_x_s)[:, 1]
                    ) / 3

                ev_fv = p_fv * fold_val["Odds"].values - 1
                mask_fv = ev_fv >= best_ev_thr
                r_fv = calc_roi(fold_val, mask_fv.astype(float), threshold=0.5)
                fold_rois.append(r_fv["roi"])
                logger.info(
                    "  Fold %d: ROI=%.2f%%, n=%d",
                    fold_idx,
                    r_fv["roi"],
                    r_fv["n_bets"],
                )
                mlflow.log_metric(f"roi_fold_{fold_idx}", r_fv["roi"])

            if fold_rois:
                mean_roi = float(np.mean(fold_rois))
                std_roi = float(np.std(fold_rois))
                logger.info("CV ROI: mean=%.2f%%, std=%.2f%%", mean_roi, std_roi)
                mlflow.log_metrics({"roi_cv_mean": mean_roi, "roi_cv_std": std_roi})

            # Save if improved
            if result_final["roi"] > 16.02:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_f.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": result_final["roi"],
                    "auc": auc_test,
                    "threshold": best_ev_thr,
                    "n_bets": result_final["n_bets"],
                    "feature_names": features,
                    "selection_method": f"calibrated_ev_{best_approach}",
                    "ev_threshold": best_ev_thr,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "n_models": best_n_models,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                mlflow.log_artifact(str(model_dir / "model.cbm"))
                mlflow.log_artifact(str(model_dir / "metadata.json"))
                logger.info("New best model saved: ROI=%.2f%%", result_final["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.1 failed")
            raise


if __name__ == "__main__":
    main()
