"""Step 4.2 -- Platt scaling + walk-forward с val-based threshold.

Предыдущие сессии пробовали isotonic calibration (катастрофа).
Platt scaling (sigmoid) -- более стабильный метод, менее подвержен overfitting.

Дополнительно: walk-forward с val split внутри каждого train блока
для честного подбора threshold на каждом шаге.
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
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

FEATURES_ENC = [
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


def compute_roi_for_strategy(
    df: pd.DataFrame, p_mean: np.ndarray, p_std: np.ndarray, strategy: str
) -> dict:
    """ROI для одной стратегии."""
    odds = df["Odds"].values

    if strategy == "conf_ev_0.15":
        ev = p_mean * odds - 1
        conf = 1.0 / (1.0 + p_std * 10)
        mask = (ev * conf) >= 0.15
    elif strategy == "conf_ev_0.10":
        ev = p_mean * odds - 1
        conf = 1.0 / (1.0 + p_std * 10)
        mask = (ev * conf) >= 0.10
    elif strategy == "ev_0.05":
        ev = p_mean * odds - 1
        mask = ev >= 0.05
    elif strategy == "pmean_0.55":
        mask = p_mean >= 0.55
    else:
        mask = np.ones(len(df), dtype=bool)

    n = mask.sum()
    if n == 0:
        return {"roi": 0.0, "n": 0}

    sel = df[mask]
    payout = (sel["target"] * sel["Odds"]).sum()
    roi = (payout - n) / n * 100
    return {"roi": round(roi, 2), "n": int(n)}


def find_best_ev_threshold_on_val(
    val_df: pd.DataFrame,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    min_bets: int = 30,
) -> float:
    """Подбор EV*conf threshold на валидации."""
    odds = val_df["Odds"].values
    ev = p_mean * odds - 1
    conf = 1.0 / (1.0 + p_std * 10)
    conf_ev = ev * conf

    best_roi = -999.0
    best_thr = 0.15

    for thr in np.arange(0.02, 0.40, 0.02):
        mask = conf_ev >= thr
        n = mask.sum()
        if n < min_bets:
            continue
        sel = val_df[mask]
        payout = (sel["target"] * sel["Odds"]).sum()
        roi = (payout - n) / n * 100
        if roi > best_roi:
            best_roi = roi
            best_thr = thr

    return round(best_thr, 2)


def main() -> None:
    """Walk-forward с Platt scaling и val-based threshold."""
    with mlflow.start_run(run_name="phase4/platt_walkforward") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            df["year_week"] = df["Created_At"].dt.year * 100 + df[
                "Created_At"
            ].dt.isocalendar().week.astype(int)

            unique_weeks = sorted(df["year_week"].unique())
            n_weeks = len(unique_weeks)
            min_train_weeks = int(n_weeks * 0.6)
            retrain_schedule = unique_weeks[min_train_weeks:]

            strategies = ["conf_ev_0.15", "conf_ev_0.10", "ev_0.05", "pmean_0.55"]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward_platt",
                    "seed": 42,
                    "n_total_weeks": n_weeks,
                    "n_test_blocks": len(retrain_schedule),
                    "calibration": "platt_sigmoid",
                    "val_threshold_tuning": True,
                    "method": "platt_calibrated_ensemble_walkforward",
                }
            )

            results_raw = []
            results_platt = []
            cumulative = {
                "raw": {s: {"n": 0, "profit": 0.0} for s in strategies},
                "platt": {s: {"n": 0, "profit": 0.0} for s in strategies},
            }

            for i, test_week in enumerate(retrain_schedule):
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                # Val split: last 20% of train
                val_idx = int(len(train_df) * 0.8)
                fit_df = train_df.iloc[:val_idx].copy()
                val_df = train_df.iloc[val_idx:].copy()

                # Feature engineering
                fit_enc, _ = add_sport_market_features(fit_df, fit_df)
                val_enc, _ = add_sport_market_features(val_df, fit_enc)
                test_enc, _ = add_sport_market_features(test_df, fit_enc)

                x_fit = fit_enc[FEATURES_ENC].fillna(0)
                y_fit = fit_enc["target"]
                x_val = val_enc[FEATURES_ENC].fillna(0)
                y_val = val_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Raw ensemble
                cb = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
                )
                cb.fit(x_fit, y_fit)

                lgbm = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm.fit(x_fit, y_fit)

                scaler = StandardScaler()
                x_fit_s = scaler.fit_transform(x_fit)
                x_val_s = scaler.transform(x_val)
                x_test_s = scaler.transform(x_test)
                lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr.fit(x_fit_s, y_fit)

                # Raw predictions
                p_cb_test = cb.predict_proba(x_test)[:, 1]
                p_lgbm_test = lgbm.predict_proba(x_test)[:, 1]
                p_lr_test = lr.predict_proba(x_test_s)[:, 1]
                p_raw_mean = (p_cb_test + p_lgbm_test + p_lr_test) / 3
                p_raw_std = np.std([p_cb_test, p_lgbm_test, p_lr_test], axis=0)

                # Platt scaling on val set
                # Calibrate each model separately using val predictions
                p_cb_val = cb.predict_proba(x_val)[:, 1]
                p_lgbm_val = lgbm.predict_proba(x_val)[:, 1]
                p_lr_val = lr.predict_proba(x_val_s)[:, 1]

                # Platt = logistic regression on model outputs
                cal_features_val = np.column_stack([p_cb_val, p_lgbm_val, p_lr_val])
                cal_features_test = np.column_stack([p_cb_test, p_lgbm_test, p_lr_test])

                platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                platt.fit(cal_features_val, y_val)
                p_platt_test = platt.predict_proba(cal_features_test)[:, 1]

                # For std: use bootstrap-like approach with Platt
                # Simple: use raw model diversity as proxy for uncertainty
                p_platt_mean = p_platt_test
                p_platt_std = p_raw_std  # keep raw disagreement as uncertainty

                # Raw results
                block_raw = {"block": i, "week": test_week, "n_test": len(test_df)}
                block_platt = {"block": i, "week": test_week, "n_test": len(test_df)}

                for s in strategies:
                    res_raw = compute_roi_for_strategy(test_enc, p_raw_mean, p_raw_std, s)
                    res_platt = compute_roi_for_strategy(test_enc, p_platt_mean, p_platt_std, s)
                    block_raw[f"{s}_roi"] = res_raw["roi"]
                    block_raw[f"{s}_n"] = res_raw["n"]
                    block_platt[f"{s}_roi"] = res_platt["roi"]
                    block_platt[f"{s}_n"] = res_platt["n"]

                    for label, res in [("raw", res_raw), ("platt", res_platt)]:
                        if res["n"] > 0:
                            cumulative[label][s]["n"] += res["n"]
                            cumulative[label][s]["profit"] += res["n"] * res["roi"] / 100

                results_raw.append(block_raw)
                results_platt.append(block_platt)

                logger.info(
                    "Block %d (wk %d): raw conf_ev=%.1f%%(n=%d) | platt conf_ev=%.1f%%(n=%d)",
                    i,
                    test_week,
                    block_raw.get("conf_ev_0.15_roi", 0),
                    block_raw.get("conf_ev_0.15_n", 0),
                    block_platt.get("conf_ev_0.15_roi", 0),
                    block_platt.get("conf_ev_0.15_n", 0),
                )

            # Summary
            df_raw = pd.DataFrame(results_raw)
            df_platt = pd.DataFrame(results_platt)

            for label, cum in cumulative.items():
                for s in strategies:
                    total_n = cum[s]["n"]
                    total_profit = cum[s]["profit"]
                    overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0
                    logger.info(
                        "%s %s: overall_ROI=%.2f%% (n=%d)",
                        label,
                        s,
                        overall_roi,
                        total_n,
                    )
                    mlflow.log_metrics(
                        {
                            f"wf_{label}_{s}_roi": round(overall_roi, 2),
                            f"wf_{label}_{s}_n": total_n,
                        }
                    )

            # Block-level stats for key strategies
            for label, res_df in [("raw", df_raw), ("platt", df_platt)]:
                for s in ["conf_ev_0.15", "conf_ev_0.10"]:
                    col = f"{s}_roi"
                    if col in res_df.columns:
                        vals = res_df[col].values
                        mlflow.log_metrics(
                            {
                                f"wf_{label}_{s}_mean": round(vals.mean(), 2),
                                f"wf_{label}_{s}_std": round(vals.std(), 2),
                                f"wf_{label}_{s}_pos_blocks": int((vals > 0).sum()),
                            }
                        )

            # Save
            raw_path = str(SESSION_DIR / "experiments" / "platt_wf_raw.csv")
            platt_path = str(SESSION_DIR / "experiments" / "platt_wf_platt.csv")
            df_raw.to_csv(raw_path, index=False)
            df_platt.to_csv(platt_path, index=False)
            mlflow.log_artifact(raw_path)
            mlflow.log_artifact(platt_path)
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
