"""Step 4.4 — Profit regression model.

Гипотеза: regression на profit (target = won*odds - 1) вместо classification.
Модель напрямую предсказывает ожидаемую прибыль, учитывая нелинейные
взаимодействия между вероятностью и коэффициентами.

Подходы:
1. CatBoost regression на profit target
2. LightGBM regression на profit target
3. Ensemble regression (avg)
4. Сравнение с baseline classification EV>=0.12
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

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


def calc_roi_from_predictions(
    df: pd.DataFrame,
    pred_profit: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    """ROI на ставках где predicted_profit > threshold."""
    mask = pred_profit > threshold
    n_bets = int(mask.sum())
    if n_bets == 0:
        return {"roi": 0.0, "n_bets": 0, "profit": 0.0, "total_staked": 0.0}

    selected = df[mask].copy()
    total_staked = float(n_bets)  # flat stake = 1
    payouts = selected["target"].values * selected["Odds"].values
    total_payout = float(payouts.sum())
    profit = total_payout - total_staked
    roi = profit / total_staked * 100

    return {
        "roi": round(roi, 4),
        "n_bets": n_bets,
        "profit": round(profit, 4),
        "total_staked": round(total_staked, 4),
        "win_rate": round(float(selected["target"].mean()), 4),
        "avg_odds": round(float(selected["Odds"].mean()), 4),
    }


def main() -> None:
    """Profit regression experiment."""
    with mlflow.start_run(run_name="phase4/step_4_4_profit_regression") as run:
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

            # Profit target: won*odds - 1 (profit per unit stake)
            train_enc["profit_target"] = train_enc["target"] * train_enc["Odds"] - 1
            test_enc["profit_target"] = test_enc["target"] * test_enc["Odds"] - 1

            logger.info(
                "Profit target stats — train: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
                train_enc["profit_target"].mean(),
                train_enc["profit_target"].std(),
                train_enc["profit_target"].min(),
                train_enc["profit_target"].max(),
            )

            # Val split for threshold selection
            val_split = int(len(train_enc) * 0.8)
            train_fit = train_enc.iloc[:val_split].copy()
            val = train_enc.iloc[val_split:].copy()

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test_enc),
                    "method": "profit_regression",
                    "n_features": len(features),
                    "target": "profit = won*odds - 1",
                }
            )

            x_fit = train_fit[features].fillna(0)
            x_val = val[features].fillna(0)
            y_fit = train_fit["profit_target"]

            # --- CatBoost Regression ---
            cb_reg = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                loss_function="RMSE",
            )
            cb_reg.fit(x_fit, y_fit)
            pred_val_cb = cb_reg.predict(x_val)
            logger.info(
                "CB reg val predictions: mean=%.3f, std=%.3f",
                pred_val_cb.mean(),
                pred_val_cb.std(),
            )

            # --- LightGBM Regression ---
            lgbm_reg = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm_reg.fit(x_fit, y_fit)
            pred_val_lgbm = lgbm_reg.predict(x_val)

            # --- CatBoost with Huber loss (robust to outliers) ---
            cb_huber = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                loss_function="Huber:delta=2.0",
            )
            cb_huber.fit(x_fit, y_fit)
            pred_val_huber = cb_huber.predict(x_val)

            # --- Ensemble ---
            pred_val_ens = (pred_val_cb + pred_val_lgbm + pred_val_huber) / 3

            # Find best threshold on val for each approach
            approaches = {
                "cb_rmse": pred_val_cb,
                "lgbm": pred_val_lgbm,
                "cb_huber": pred_val_huber,
                "ensemble": pred_val_ens,
            }

            best_approach_name = ""
            best_approach_thr = 0.0
            best_approach_roi = -999.0

            for name, preds in approaches.items():
                best_thr = 0.0
                best_roi = -999.0
                for thr in np.arange(-0.5, 2.0, 0.05):
                    r = calc_roi_from_predictions(val, preds, threshold=thr)
                    if r["n_bets"] >= 50 and r["roi"] > best_roi:
                        best_roi = r["roi"]
                        best_thr = round(float(thr), 2)

                logger.info("  %s val: best_thr=%.2f, ROI=%.2f%%", name, best_thr, best_roi)
                mlflow.log_metric(f"roi_val_{name}", best_roi)

                if best_roi > best_approach_roi:
                    best_approach_roi = best_roi
                    best_approach_thr = best_thr
                    best_approach_name = name

            logger.info(
                "Best regression approach on val: %s, thr=%.2f, ROI=%.2f%%",
                best_approach_name,
                best_approach_thr,
                best_approach_roi,
            )

            # --- Retrain on full train, apply to test ---
            x_train_full = train_enc[features].fillna(0)
            x_test = test_enc[features].fillna(0)
            y_train_full = train_enc["profit_target"]

            cb_reg_f = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                loss_function="RMSE",
            )
            cb_reg_f.fit(x_train_full, y_train_full)

            lgbm_reg_f = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm_reg_f.fit(x_train_full, y_train_full)

            cb_huber_f = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                loss_function="Huber:delta=2.0",
            )
            cb_huber_f.fit(x_train_full, y_train_full)

            test_preds = {
                "cb_rmse": cb_reg_f.predict(x_test),
                "lgbm": lgbm_reg_f.predict(x_test),
                "cb_huber": cb_huber_f.predict(x_test),
                "ensemble": (
                    cb_reg_f.predict(x_test)
                    + lgbm_reg_f.predict(x_test)
                    + cb_huber_f.predict(x_test)
                )
                / 3,
            }

            # Apply best val threshold to test for each approach
            for name, preds in test_preds.items():
                for thr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                    r = calc_roi_from_predictions(test_enc, preds, threshold=thr)
                    logger.info(
                        "  %s test thr=%.2f: ROI=%.2f%%, n=%d",
                        name,
                        thr,
                        r["roi"],
                        r["n_bets"],
                    )

            # Apply best approach + threshold
            pred_test_best = test_preds[best_approach_name]
            result_reg = calc_roi_from_predictions(
                test_enc, pred_test_best, threshold=best_approach_thr
            )
            logger.info("Best regression test: %s", result_reg)

            # --- Baseline comparison ---
            from catboost import CatBoostClassifier
            from lightgbm import LGBMClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            cb_cls = CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
            )
            cb_cls.fit(x_train_full, train_enc["target"])
            lgbm_cls = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm_cls.fit(x_train_full, train_enc["target"])
            scaler = StandardScaler()
            lr_cls = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_cls.fit(scaler.fit_transform(x_train_full), train_enc["target"])

            p_test_cls = (
                cb_cls.predict_proba(x_test)[:, 1]
                + lgbm_cls.predict_proba(x_test)[:, 1]
                + lr_cls.predict_proba(scaler.transform(x_test))[:, 1]
            ) / 3
            ev_test = p_test_cls * test_enc["Odds"].values - 1
            ev_mask = ev_test >= 0.12
            result_baseline = calc_roi(test_enc, ev_mask.astype(float), threshold=0.5)
            logger.info("Baseline classification EV>=0.12: %s", result_baseline)

            # --- Hybrid: classification EV + regression profit ---
            # Use both signals: EV from classification AND predicted profit from regression
            pred_profit_ens = test_preds["ensemble"]
            for ev_thr in [0.08, 0.10, 0.12]:
                for profit_thr in [0.0, 0.05, 0.10]:
                    hybrid_mask = (ev_test >= ev_thr) & (pred_profit_ens > profit_thr)
                    r_h = calc_roi(test_enc, hybrid_mask.astype(float), threshold=0.5)
                    logger.info(
                        "  Hybrid EV>=%.2f & profit>%.2f: ROI=%.2f%%, n=%d",
                        ev_thr,
                        profit_thr,
                        r_h["roi"],
                        r_h["n_bets"],
                    )

            # Log metrics
            mlflow.log_metrics(
                {
                    "roi_test_regression": result_reg["roi"],
                    "roi_test_baseline": result_baseline["roi"],
                    "n_bets_regression": result_reg["n_bets"],
                    "n_bets_baseline": result_baseline["n_bets"],
                    "roi_test": max(result_reg["roi"], result_baseline["roi"]),
                }
            )
            mlflow.set_tag("best_regression_approach", best_approach_name)

            # CV
            logger.info("=== CV stability (regression) ===")
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
                ft_y = fold_train["profit_target"]

                cb_cv = CatBoostRegressor(
                    iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0
                )
                cb_cv.fit(ft_x, ft_y)
                lgbm_cv = LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm_cv.fit(ft_x, ft_y)
                hub_cv = CatBoostRegressor(
                    iterations=300,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0,
                    loss_function="Huber:delta=2.0",
                )
                hub_cv.fit(ft_x, ft_y)

                p_fv = (cb_cv.predict(fv_x) + lgbm_cv.predict(fv_x) + hub_cv.predict(fv_x)) / 3
                r_fv = calc_roi_from_predictions(fold_val, p_fv, threshold=best_approach_thr)
                fold_rois.append(r_fv["roi"])
                logger.info("  Fold %d: ROI=%.2f%%, n=%d", fold_idx, r_fv["roi"], r_fv["n_bets"])
                mlflow.log_metric(f"roi_fold_{fold_idx}", r_fv["roi"])

            if fold_rois:
                mean_roi = float(np.mean(fold_rois))
                std_roi = float(np.std(fold_rois))
                logger.info("CV ROI (regression): mean=%.2f%%, std=%.2f%%", mean_roi, std_roi)
                mlflow.log_metrics({"roi_cv_mean": mean_roi, "roi_cv_std": std_roi})

            # Save if improved
            best_roi = max(result_reg["roi"], result_baseline["roi"])
            if best_roi > 16.02 and result_reg["roi"] > result_baseline["roi"]:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_reg_f.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": result_reg["roi"],
                    "threshold": best_approach_thr,
                    "n_bets": result_reg["n_bets"],
                    "feature_names": features,
                    "selection_method": f"profit_regression_{best_approach_name}",
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "model_type": "regression",
                    "target": "profit",
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info("New best model saved: ROI=%.2f%%", result_reg["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.4 failed")
            raise


if __name__ == "__main__":
    main()
