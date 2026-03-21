"""Step 4.29 -- Isotonic regression calibration.

Platt scaling (step 4.2) hurt ROI. Isotonic regression is more flexible and
non-parametric. Also test: calibrating per odds bracket, using validation fold.

Key insight from 4.17: 5.3% overconfidence gap. If we fix calibration,
EV selection should pick different (better calibrated) bets.
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
from sklearn.isotonic import IsotonicRegression
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
    "Odds", "USD", "Is_Parlay", "Outcomes_Count",
    "ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV",
    "ML_Winrate_Diff", "ML_Rating_Diff",
    "Outcome_Odds", "n_outcomes", "mean_outcome_odds",
    "max_outcome_odds", "min_outcome_odds",
    "Sport_target_enc", "Sport_count_enc",
    "Market_target_enc", "Market_count_enc",
]


def calibrate_isotonic(
    p_train: np.ndarray, y_train: np.ndarray, p_test: np.ndarray
) -> np.ndarray:
    """Fit isotonic regression on train predictions, apply to test."""
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(p_train, y_train)
    return iso.predict(p_test)


def main() -> None:
    """Isotonic calibration walk-forward."""
    with mlflow.start_run(run_name="phase4/isotonic_calibration") as run:
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

            strategies = ["raw", "isotonic_ensemble", "isotonic_per_model"]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "isotonic_calibration",
                }
            )

            cumulative: dict[str, dict] = {
                k: {"n": 0, "profit": 0.0} for k in strategies
            }
            all_blocks: list[dict] = []

            for i, test_week in enumerate(retrain_schedule):
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train = train_enc[FEATURES_ENC].fillna(0)
                y_train = train_enc["target"].values
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Split train into fit/calibration (80/20 temporal)
                n_train = len(x_train)
                cal_split = int(n_train * 0.8)
                x_fit = x_train.iloc[:cal_split]
                y_fit = y_train[:cal_split]
                x_cal = x_train.iloc[cal_split:]
                y_cal = y_train[cal_split:]

                cb = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6,
                    random_seed=42, verbose=0,
                )
                cb.fit(x_fit, y_fit)

                lgbm = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=42, verbose=-1, min_child_samples=50,
                )
                lgbm.fit(x_fit, y_fit)

                scaler = StandardScaler()
                x_fit_s = scaler.fit_transform(x_fit)
                x_cal_s = scaler.transform(x_cal)
                x_test_s = scaler.transform(x_test)
                lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr.fit(x_fit_s, y_fit)

                # Raw predictions on calibration set
                p_cb_cal = cb.predict_proba(x_cal)[:, 1]
                p_lgbm_cal = lgbm.predict_proba(x_cal)[:, 1]
                p_lr_cal = lr.predict_proba(x_cal_s)[:, 1]
                p_mean_cal = (p_cb_cal + p_lgbm_cal + p_lr_cal) / 3

                # Raw predictions on test set
                p_cb_test = cb.predict_proba(x_test)[:, 1]
                p_lgbm_test = lgbm.predict_proba(x_test)[:, 1]
                p_lr_test = lr.predict_proba(x_test_s)[:, 1]

                odds = test_enc["Odds"].values

                block_info: dict = {"block": i, "week": test_week, "n_test": len(test_df)}

                for strategy in strategies:
                    if strategy == "raw":
                        p_mean = (p_cb_test + p_lgbm_test + p_lr_test) / 3
                        p_std = np.std([p_cb_test, p_lgbm_test, p_lr_test], axis=0)
                    elif strategy == "isotonic_ensemble":
                        p_mean_raw = (p_cb_test + p_lgbm_test + p_lr_test) / 3
                        p_mean = calibrate_isotonic(p_mean_cal, y_cal, p_mean_raw)
                        p_std = np.std([p_cb_test, p_lgbm_test, p_lr_test], axis=0)
                    elif strategy == "isotonic_per_model":
                        p_cb_cal_iso = calibrate_isotonic(p_cb_cal, y_cal, p_cb_test)
                        p_lgbm_cal_iso = calibrate_isotonic(p_lgbm_cal, y_cal, p_lgbm_test)
                        p_lr_cal_iso = calibrate_isotonic(p_lr_cal, y_cal, p_lr_test)
                        p_mean = (p_cb_cal_iso + p_lgbm_cal_iso + p_lr_cal_iso) / 3
                        p_std = np.std(
                            [p_cb_cal_iso, p_lgbm_cal_iso, p_lr_cal_iso], axis=0
                        )
                    else:
                        continue

                    ev = p_mean * odds - 1
                    mask = (ev >= 0.05) & (p_std <= 0.02)

                    n = mask.sum()
                    roi = 0.0
                    if n > 0:
                        sel = test_enc[mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                        cumulative[strategy]["n"] += n
                        cumulative[strategy]["profit"] += n * roi / 100

                    block_info[f"{strategy}_roi"] = round(roi, 2)
                    block_info[f"{strategy}_n"] = int(n)

                    if i == 0:
                        logger.info(
                            "  Block 0 %s: cal_gap=%.4f",
                            strategy,
                            p_mean[mask].mean() - test_enc[mask]["target"].mean()
                            if n > 0 else 0.0,
                        )

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nIsotonic calibration results (ev005_agree_p02):")
            for strategy in strategies:
                total_n = cumulative[strategy]["n"]
                total_profit = cumulative[strategy]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"{strategy}_roi"
                n_pos = (results_df[col].values > 0).sum() if col in results_df.columns else 0

                logger.info(
                    "  %s: overall=%.2f%% (n=%d) pos=%d/%d",
                    strategy, overall_roi, total_n, n_pos, len(all_blocks),
                )
                mlflow.log_metrics(
                    {
                        f"wf_{strategy}_roi": round(overall_roi, 2),
                        f"wf_{strategy}_n": total_n,
                        f"wf_{strategy}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "isotonic_results.csv")
            results_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.29 failed")
            raise


if __name__ == "__main__":
    main()
