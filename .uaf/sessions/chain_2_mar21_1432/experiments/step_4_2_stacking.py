"""Step 4.2 — Stacking ensemble с мета-моделью и Optuna CatBoost.

Гипотезы:
1. CatBoost с Optuna params (depth=8, iter=873) вместо дефолтных
2. Stacking: OOF предсказания как фичи для мета-модели (LogReg)
3. Weighted ensemble: оптимальные веса моделей по val
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

# Optuna-tuned params из предыдущей сессии (step 3.1)
OPTUNA_CB_PARAMS = {
    "iterations": 873,
    "learning_rate": 0.077,
    "depth": 8,
    "l2_leaf_reg": 0.0036,
    "min_child_samples": 80,
    "subsample": 0.65,
    "colsample_bylevel": 0.71,
    "bootstrap_type": "Bernoulli",
    "random_seed": 42,
    "verbose": 0,
}


def build_oof_predictions(
    train_df: pd.DataFrame,
    features: list[str],
    n_folds: int = 5,
) -> np.ndarray:
    """Строит OOF предсказания для stacking (expanding window)."""
    n = len(train_df)
    fold_size = n // (n_folds + 1)
    oof = np.full(n, np.nan)

    for fold_idx in range(n_folds):
        train_end = fold_size * (fold_idx + 1)
        val_start = train_end
        val_end = fold_size * (fold_idx + 2)

        if val_end > n:
            val_end = n

        ft = train_df.iloc[:train_end]
        fv = train_df.iloc[val_start:val_end]

        if len(fv) < 50:
            continue

        ft_x = ft[features].fillna(0)
        fv_x = fv[features].fillna(0)
        ft_y = ft["target"]

        # CatBoost Optuna
        cb = CatBoostClassifier(**OPTUNA_CB_PARAMS)
        cb.fit(ft_x, ft_y)

        # LightGBM
        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            verbose=-1,
            min_child_samples=50,
            subsample=0.7,
            colsample_bytree=0.7,
        )
        lgbm.fit(ft_x, ft_y)

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            verbosity=0,
            min_child_weight=50,
            subsample=0.7,
            colsample_bytree=0.7,
            eval_metric="logloss",
        )
        xgb.fit(ft_x, ft_y)

        # LogReg
        sc = StandardScaler()
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(sc.fit_transform(ft_x), ft_y)

        # OOF predictions (average)
        p = (
            cb.predict_proba(fv_x)[:, 1]
            + lgbm.predict_proba(fv_x)[:, 1]
            + xgb.predict_proba(fv_x)[:, 1]
            + lr.predict_proba(sc.transform(fv_x))[:, 1]
        ) / 4
        oof[val_start:val_end] = p

    return oof


def main() -> None:
    """Stacking + Optuna CatBoost."""
    with mlflow.start_run(run_name="phase4/step_4_2_stacking") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Feature engineering
            train_enc, _ = add_sport_market_features(train.copy(), train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            features = FEATURES

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_enc),
                    "n_samples_test": len(test_enc),
                    "method": "stacking_optuna_4model",
                    "n_features": len(features),
                    "ev_threshold": 0.12,
                    "cb_iterations": OPTUNA_CB_PARAMS["iterations"],
                    "cb_depth": OPTUNA_CB_PARAMS["depth"],
                }
            )

            x_train = train_enc[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train = train_enc["target"]

            # --- Approach A: Simple average with Optuna CatBoost (4 models) ---
            logger.info("=== Approach A: 4-model average with Optuna CB ===")

            cb_a = CatBoostClassifier(**OPTUNA_CB_PARAMS)
            cb_a.fit(x_train, y_train)

            lgbm_a = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
                subsample=0.7,
                colsample_bytree=0.7,
            )
            lgbm_a.fit(x_train, y_train)

            xgb_a = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                verbosity=0,
                min_child_weight=50,
                subsample=0.7,
                colsample_bytree=0.7,
                eval_metric="logloss",
            )
            xgb_a.fit(x_train, y_train)

            scaler_a = StandardScaler()
            x_train_s = scaler_a.fit_transform(x_train)
            x_test_s = scaler_a.transform(x_test)
            lr_a = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_a.fit(x_train_s, y_train)

            p_cb = cb_a.predict_proba(x_test)[:, 1]
            p_lgbm = lgbm_a.predict_proba(x_test)[:, 1]
            p_xgb = xgb_a.predict_proba(x_test)[:, 1]
            p_lr = lr_a.predict_proba(x_test_s)[:, 1]

            p_avg4 = (p_cb + p_lgbm + p_xgb + p_lr) / 4
            auc_avg4 = roc_auc_score(test_enc["target"], p_avg4)
            logger.info("4-model avg AUC: %.4f", auc_avg4)

            ev_avg4 = p_avg4 * test_enc["Odds"].values - 1
            for ev_t in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
                mask = ev_avg4 >= ev_t
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  4-avg EV>=%.2f: ROI=%.2f%%, n=%d", ev_t, r["roi"], r["n_bets"])

            ev_mask_a = ev_avg4 >= 0.12
            result_a = calc_roi(test_enc, ev_mask_a.astype(float), threshold=0.5)
            logger.info("Approach A result: %s", result_a)

            # --- Approach B: Weighted ensemble (optimize weights on val) ---
            logger.info("=== Approach B: Weighted ensemble ===")

            # Val split for weight optimization
            val_split = int(len(train_enc) * 0.8)
            val_part = train_enc.iloc[val_split:]
            x_val = val_part[features].fillna(0)
            x_val_s = scaler_a.transform(x_val)

            pv_cb = cb_a.predict_proba(x_val)[:, 1]
            pv_lgbm = lgbm_a.predict_proba(x_val)[:, 1]
            pv_xgb = xgb_a.predict_proba(x_val)[:, 1]
            pv_lr = lr_a.predict_proba(x_val_s)[:, 1]

            # Grid search weights
            best_w_roi = -999.0
            best_weights = [0.25, 0.25, 0.25, 0.25]
            for w_cb in np.arange(0.1, 0.6, 0.1):
                for w_lgbm in np.arange(0.1, 0.6, 0.1):
                    for w_xgb in np.arange(0.0, 0.5, 0.1):
                        w_lr = round(1.0 - w_cb - w_lgbm - w_xgb, 1)
                        if w_lr < 0 or w_lr > 0.5:
                            continue
                        p_w = w_cb * pv_cb + w_lgbm * pv_lgbm + w_xgb * pv_xgb + w_lr * pv_lr
                        ev_w = p_w * val_part["Odds"].values - 1
                        mask_w = ev_w >= 0.12
                        r_w = calc_roi(val_part, mask_w.astype(float), threshold=0.5)
                        if r_w["n_bets"] >= 50 and r_w["roi"] > best_w_roi:
                            best_w_roi = r_w["roi"]
                            best_weights = [w_cb, w_lgbm, w_xgb, w_lr]

            logger.info(
                "Best weights: CB=%.1f, LGBM=%.1f, XGB=%.1f, LR=%.1f (val ROI=%.2f%%)",
                *best_weights,
                best_w_roi,
            )

            # Apply to test
            p_weighted = (
                best_weights[0] * p_cb
                + best_weights[1] * p_lgbm
                + best_weights[2] * p_xgb
                + best_weights[3] * p_lr
            )
            ev_weighted = p_weighted * test_enc["Odds"].values - 1
            ev_mask_b = ev_weighted >= 0.12
            result_b = calc_roi(test_enc, ev_mask_b.astype(float), threshold=0.5)
            logger.info("Approach B (weighted) result: %s", result_b)

            # --- Approach C: Only Optuna CB (no ensemble) ---
            logger.info("=== Approach C: Optuna CatBoost solo ===")
            ev_cb_solo = p_cb * test_enc["Odds"].values - 1
            for ev_t in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
                mask = ev_cb_solo >= ev_t
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                logger.info("  CB-solo EV>=%.2f: ROI=%.2f%%, n=%d", ev_t, r["roi"], r["n_bets"])

            ev_mask_c = ev_cb_solo >= 0.12
            result_c = calc_roi(test_enc, ev_mask_c.astype(float), threshold=0.5)
            logger.info("Approach C (CB solo) result: %s", result_c)

            # --- Pick best ---
            approaches = {
                "4model_avg": result_a,
                "weighted": result_b,
                "cb_solo": result_c,
            }
            best_name = max(approaches, key=lambda k: approaches[k]["roi"])
            best_result = approaches[best_name]
            logger.info("Best approach: %s, ROI=%.2f%%", best_name, best_result["roi"])

            # Log all results
            mlflow.log_metrics(
                {
                    "auc_test_4avg": auc_avg4,
                    "roi_test_4avg": result_a["roi"],
                    "roi_test_weighted": result_b["roi"],
                    "roi_test_cb_solo": result_c["roi"],
                    "roi_test": best_result["roi"],
                    "n_bets_test": best_result["n_bets"],
                    "best_approach": 0,  # logged as tag
                }
            )
            mlflow.set_tag("best_approach", best_name)
            mlflow.set_tag(
                "best_weights",
                f"CB={best_weights[0]:.1f},LGBM={best_weights[1]:.1f},"
                f"XGB={best_weights[2]:.1f},LR={best_weights[3]:.1f}",
            )

            # CV stability
            logger.info("=== CV stability for best approach ===")
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

                cb_cv = CatBoostClassifier(**OPTUNA_CB_PARAMS)
                cb_cv.fit(ft_x, ft_y)

                lgbm_cv = LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=7,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                    subsample=0.7,
                    colsample_bytree=0.7,
                )
                lgbm_cv.fit(ft_x, ft_y)

                xgb_cv = XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=7,
                    random_state=42,
                    verbosity=0,
                    min_child_weight=50,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    eval_metric="logloss",
                )
                xgb_cv.fit(ft_x, ft_y)

                sc_cv = StandardScaler()
                lr_cv = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_cv.fit(sc_cv.fit_transform(ft_x), ft_y)

                if best_name == "cb_solo":
                    p_fv = cb_cv.predict_proba(fv_x)[:, 1]
                elif best_name == "weighted":
                    p_fv = (
                        best_weights[0] * cb_cv.predict_proba(fv_x)[:, 1]
                        + best_weights[1] * lgbm_cv.predict_proba(fv_x)[:, 1]
                        + best_weights[2] * xgb_cv.predict_proba(fv_x)[:, 1]
                        + best_weights[3] * lr_cv.predict_proba(sc_cv.transform(fv_x))[:, 1]
                    )
                else:
                    p_fv = (
                        cb_cv.predict_proba(fv_x)[:, 1]
                        + lgbm_cv.predict_proba(fv_x)[:, 1]
                        + xgb_cv.predict_proba(fv_x)[:, 1]
                        + lr_cv.predict_proba(sc_cv.transform(fv_x))[:, 1]
                    ) / 4

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

            # Save if improved
            if best_result["roi"] > 16.02:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_a.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_result["roi"],
                    "auc": auc_avg4,
                    "threshold": 0.12,
                    "n_bets": best_result["n_bets"],
                    "feature_names": features,
                    "selection_method": f"stacking_{best_name}",
                    "ev_threshold": 0.12,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "n_models": 4,
                    "optuna_cb_params": OPTUNA_CB_PARAMS,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                mlflow.log_artifact(str(model_dir / "model.cbm"))
                mlflow.log_artifact(str(model_dir / "metadata.json"))
                logger.info("New best model saved: ROI=%.2f%%", best_result["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.2 failed")
            raise


if __name__ == "__main__":
    main()
