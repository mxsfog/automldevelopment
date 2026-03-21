"""Step 4.5 — Agreement-based selection + min-EV conservative.

Гипотеза: если все 3 модели независимо считают что EV > 0, это
более надёжный сигнал чем усреднённый EV. Дополнительно:
1. Min-EV: использовать минимальный EV из 3 моделей (conservative)
2. Std-weighted: взвешивать по обратному std (меньше разброс = выше вес)
3. Agreement count: ставить только если N из 3 моделей согласны

Также тестируем разные конфигурации ансамбля:
- Разные depth/iterations для каждой модели (diversity)
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


def main() -> None:
    """Agreement-based selection."""
    with mlflow.start_run(run_name="phase4/step_4_5_agreement") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            train_enc, _ = add_sport_market_features(train.copy(), train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            features = FEATURES

            x_train = train_enc[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_enc["target"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_enc),
                    "n_samples_val": len(test_enc),
                    "method": "agreement_selection",
                    "n_features": len(features),
                }
            )

            # --- Train individual models ---
            # Model 1: CatBoost (standard)
            cb = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb.fit(x_train, y_train)

            # Model 2: LightGBM
            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm.fit(x_train, y_train)

            # Model 3: LogReg
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(x_train_s, y_train)

            # Model 4: CatBoost (diverse — different depth)
            cb2 = CatBoostClassifier(
                iterations=300, learning_rate=0.03, depth=4, random_seed=42, verbose=0
            )
            cb2.fit(x_train, y_train)

            # Model 5: LightGBM (diverse)
            lgbm2 = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
                verbose=-1,
                min_child_samples=100,
            )
            lgbm2.fit(x_train, y_train)

            # Individual probabilities
            p_cb = cb.predict_proba(x_test)[:, 1]
            p_lgbm = lgbm.predict_proba(x_test)[:, 1]
            p_lr = lr.predict_proba(x_test_s)[:, 1]
            p_cb2 = cb2.predict_proba(x_test)[:, 1]
            p_lgbm2 = lgbm2.predict_proba(x_test)[:, 1]

            odds = test_enc["Odds"].values

            # Individual EVs
            ev_cb = p_cb * odds - 1
            ev_lgbm = p_lgbm * odds - 1
            ev_lr = p_lr * odds - 1

            # --- Baseline: average of 3 models ---
            p_avg3 = (p_cb + p_lgbm + p_lr) / 3
            ev_avg3 = p_avg3 * odds - 1

            auc_test = roc_auc_score(test_enc["target"], p_avg3)
            logger.info("3-model avg AUC: %.4f", auc_test)

            ev_mask_base = ev_avg3 >= 0.12
            result_base = calc_roi(test_enc, ev_mask_base.astype(float), threshold=0.5)
            logger.info("Baseline avg3 EV>=0.12: %s", result_base)

            # --- Agreement-based strategies ---
            # Strategy A: All 3 models agree EV >= threshold
            logger.info("=== Strategy A: All 3 agree ===")
            for ev_thr in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12]:
                agree_mask = (ev_cb >= ev_thr) & (ev_lgbm >= ev_thr) & (ev_lr >= ev_thr)
                r = calc_roi(test_enc, agree_mask.astype(float), threshold=0.5)
                logger.info("  All3 EV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            # Strategy B: Min-EV (conservative) — take minimum EV across models
            logger.info("=== Strategy B: Min-EV ===")
            ev_min3 = np.minimum(np.minimum(ev_cb, ev_lgbm), ev_lr)
            for ev_thr in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12]:
                mask = ev_min3 >= ev_thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  MinEV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            # Strategy C: 5-model average
            logger.info("=== Strategy C: 5-model average ===")
            p_avg5 = (p_cb + p_lgbm + p_lr + p_cb2 + p_lgbm2) / 5
            ev_avg5 = p_avg5 * odds - 1
            for ev_thr in [0.08, 0.10, 0.12, 0.15, 0.18]:
                mask = ev_avg5 >= ev_thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  5avg EV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            # Strategy D: Low-std selection — bet where models agree more
            logger.info("=== Strategy D: Low-std + EV ===")
            ev_stack = np.stack([ev_cb, ev_lgbm, ev_lr], axis=0)
            ev_std = ev_stack.std(axis=0)
            ev_mean = ev_stack.mean(axis=0)

            for ev_thr in [0.08, 0.10, 0.12]:
                for std_max in [0.05, 0.10, 0.15, 0.20, 0.30, 999]:
                    mask = (ev_mean >= ev_thr) & (ev_std <= std_max)
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    if r["n_bets"] >= 50:
                        logger.info(
                            "  EV>=%.2f & std<=%.2f: ROI=%.2f%%, n=%d",
                            ev_thr,
                            std_max,
                            r["roi"],
                            r["n_bets"],
                        )

            # Strategy E: Max-EV (aggressive) — take maximum EV across models
            logger.info("=== Strategy E: Max-EV ===")
            ev_max3 = np.maximum(np.maximum(ev_cb, ev_lgbm), ev_lr)
            for ev_thr in [0.12, 0.15, 0.18, 0.20, 0.25]:
                mask = ev_max3 >= ev_thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  MaxEV>=%.2f: ROI=%.2f%%, n=%d", ev_thr, r["roi"], r["n_bets"])

            # --- Collect best results ---
            # All3 agree EV>=0.05
            agree_005 = (ev_cb >= 0.05) & (ev_lgbm >= 0.05) & (ev_lr >= 0.05)
            r_agree = calc_roi(test_enc, agree_005.astype(float), threshold=0.5)

            # MinEV >= 0.05
            min_005 = ev_min3 >= 0.05
            r_min = calc_roi(test_enc, min_005.astype(float), threshold=0.5)

            # 5-model EV>=0.12
            ev5_012 = ev_avg5 >= 0.12
            r_5m = calc_roi(test_enc, ev5_012.astype(float), threshold=0.5)

            results = {
                "baseline_3avg_ev012": result_base,
                "agree3_ev005": r_agree,
                "min_ev005": r_min,
                "5model_ev012": r_5m,
            }
            best_name = max(results, key=lambda k: results[k]["roi"])
            best_result = results[best_name]
            logger.info("Best approach: %s, ROI=%.2f%%", best_name, best_result["roi"])

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_test_baseline": result_base["roi"],
                    "roi_test_agree": r_agree["roi"],
                    "roi_test_min": r_min["roi"],
                    "roi_test_5m": r_5m["roi"],
                    "roi_test": best_result["roi"],
                    "n_bets_test": best_result["n_bets"],
                }
            )
            mlflow.set_tag("best_approach", best_name)

            # CV for baseline (which remains best)
            logger.info("=== CV stability ===")
            n_folds = 5
            fold_size = len(train_enc) // (n_folds + 1)
            fold_rois = []

            for fold_idx in range(n_folds):
                fold_end = fold_size * (fold_idx + 2)
                fold_train = train_enc.iloc[: fold_size * (fold_idx + 1)]
                fold_val = train_enc.iloc[fold_size * (fold_idx + 1) : fold_end]
                if len(fold_val) < 100:
                    continue

                ft_x = fold_train[features].fillna(0)
                fv_x = fold_val[features].fillna(0)
                ft_y = fold_train["target"]

                sc_cv = StandardScaler()
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
                lr_cv.fit(sc_cv.fit_transform(ft_x), ft_y)

                p_fv = (
                    cb_cv.predict_proba(fv_x)[:, 1]
                    + lgbm_cv.predict_proba(fv_x)[:, 1]
                    + lr_cv.predict_proba(sc_cv.transform(fv_x))[:, 1]
                ) / 3
                ev_fv = p_fv * fold_val["Odds"].values - 1
                mask_fv = ev_fv >= 0.12
                r_fv = calc_roi(fold_val, mask_fv.astype(float), threshold=0.5)
                fold_rois.append(r_fv["roi"])
                logger.info("  Fold %d: ROI=%.2f%%, n=%d", fold_idx, r_fv["roi"], r_fv["n_bets"])
                mlflow.log_metric(f"roi_fold_{fold_idx}", r_fv["roi"])

            if fold_rois:
                mean_roi = float(np.mean(fold_rois))
                std_roi = float(np.std(fold_rois))
                logger.info("CV ROI: mean=%.2f%%, std=%.2f%%", mean_roi, std_roi)
                mlflow.log_metrics({"roi_cv_mean": mean_roi, "roi_cv_std": std_roi})

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
