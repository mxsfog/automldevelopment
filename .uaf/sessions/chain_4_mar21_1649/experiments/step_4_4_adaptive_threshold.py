"""Step 4.4 -- Adaptive threshold walk-forward + stacking meta-learner.

Два новых подхода:
1. Adaptive threshold: на каждом walk-forward блоке подбираем conf_ev threshold
   на val split (последние 20% train). Гипотеза: оптимальный threshold
   может меняться со временем.

2. Stacking meta-learner: вместо простого average(CB, LGBM, LR) обучаем
   LogisticRegression на OOF predictions (на val). Это отличается от step 4.6
   предыдущей сессии: там OOF был на фиксированном сплите, здесь walk-forward.
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


def find_best_conf_ev_threshold(
    df: pd.DataFrame,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    min_bets: int = 20,
) -> tuple[float, float]:
    """Подбор conf_ev threshold на валидации."""
    odds = df["Odds"].values
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
        sel = df[mask]
        payout = (sel["target"] * sel["Odds"]).sum()
        roi = (payout - n) / n * 100
        if roi > best_roi:
            best_roi = roi
            best_thr = thr

    return best_thr, best_roi


def compute_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """ROI для набора ставок."""
    n = mask.sum()
    if n == 0:
        return {"roi": 0.0, "n": 0}
    sel = df[mask]
    payout = (sel["target"] * sel["Odds"]).sum()
    roi = (payout - n) / n * 100
    return {"roi": round(roi, 2), "n": int(n)}


def main() -> None:
    """Adaptive threshold + stacking walk-forward."""
    with mlflow.start_run(run_name="phase4/adaptive_thr_stacking") as run:
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

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "adaptive_threshold_plus_stacking",
                    "n_test_blocks": len(retrain_schedule),
                    "val_fraction": 0.2,
                }
            )

            strategies = [
                "fixed_0.15",
                "adaptive_thr",
                "stacking_fixed_0.15",
                "stacking_adaptive",
            ]
            cumulative = {s: {"n": 0, "profit": 0.0} for s in strategies}
            all_blocks = []

            for i, test_week in enumerate(retrain_schedule):
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                # Val split
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
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Train base models
                cb = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0,
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

                # Val predictions
                p_cb_val = cb.predict_proba(x_val)[:, 1]
                p_lgbm_val = lgbm.predict_proba(x_val)[:, 1]
                p_lr_val = lr.predict_proba(x_val_s)[:, 1]

                # Test predictions
                p_cb_test = cb.predict_proba(x_test)[:, 1]
                p_lgbm_test = lgbm.predict_proba(x_test)[:, 1]
                p_lr_test = lr.predict_proba(x_test_s)[:, 1]

                # Simple average
                p_avg_val = (p_cb_val + p_lgbm_val + p_lr_val) / 3
                p_std_val = np.std([p_cb_val, p_lgbm_val, p_lr_val], axis=0)
                p_avg_test = (p_cb_test + p_lgbm_test + p_lr_test) / 3
                p_std_test = np.std([p_cb_test, p_lgbm_test, p_lr_test], axis=0)

                # Stacking meta-learner
                stack_features_val = np.column_stack([p_cb_val, p_lgbm_val, p_lr_val])
                stack_features_test = np.column_stack([p_cb_test, p_lgbm_test, p_lr_test])
                y_val = val_enc["target"]

                meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                meta.fit(stack_features_val, y_val)
                p_stack_test = meta.predict_proba(stack_features_test)[:, 1]
                # For stacking, use individual model std as uncertainty
                p_stack_std = p_std_test

                # Adaptive threshold (fit on val)
                adaptive_thr, val_roi = find_best_conf_ev_threshold(
                    val_enc, p_avg_val, p_std_val, min_bets=20
                )

                # Stacking adaptive threshold
                stack_val = meta.predict_proba(stack_features_val)[:, 1]
                adaptive_stack_thr, _ = find_best_conf_ev_threshold(
                    val_enc, stack_val, p_std_val, min_bets=20
                )

                # Compute test ROI for each strategy
                odds_test = test_enc["Odds"].values

                # Fixed 0.15 (baseline)
                ev_avg = p_avg_test * odds_test - 1
                conf_avg = 1.0 / (1.0 + p_std_test * 10)
                mask_fixed = (ev_avg * conf_avg) >= 0.15

                # Adaptive threshold
                mask_adaptive = (ev_avg * conf_avg) >= adaptive_thr

                # Stacking fixed 0.15
                ev_stack = p_stack_test * odds_test - 1
                conf_stack = 1.0 / (1.0 + p_stack_std * 10)
                mask_stack_fixed = (ev_stack * conf_stack) >= 0.15

                # Stacking adaptive
                mask_stack_adaptive = (ev_stack * conf_stack) >= adaptive_stack_thr

                masks = {
                    "fixed_0.15": mask_fixed,
                    "adaptive_thr": mask_adaptive,
                    "stacking_fixed_0.15": mask_stack_fixed,
                    "stacking_adaptive": mask_stack_adaptive,
                }

                block_info = {
                    "block": i,
                    "week": test_week,
                    "n_test": len(test_df),
                    "adaptive_thr": adaptive_thr,
                    "adaptive_stack_thr": adaptive_stack_thr,
                    "val_roi_at_adaptive_thr": round(val_roi, 2),
                }

                for s_name, mask in masks.items():
                    res = compute_roi(test_enc, mask)
                    block_info[f"{s_name}_roi"] = res["roi"]
                    block_info[f"{s_name}_n"] = res["n"]
                    if res["n"] > 0:
                        cumulative[s_name]["n"] += res["n"]
                        cumulative[s_name]["profit"] += res["n"] * res["roi"] / 100

                all_blocks.append(block_info)
                logger.info(
                    "Block %d (wk %d): adaptive_thr=%.2f "
                    "fixed=%.1f%%(n=%d) adaptive=%.1f%%(n=%d) "
                    "stack_fixed=%.1f%%(n=%d) stack_adapt=%.1f%%(n=%d)",
                    i,
                    test_week,
                    adaptive_thr,
                    block_info["fixed_0.15_roi"],
                    block_info["fixed_0.15_n"],
                    block_info["adaptive_thr_roi"],
                    block_info["adaptive_thr_n"],
                    block_info["stacking_fixed_0.15_roi"],
                    block_info["stacking_fixed_0.15_n"],
                    block_info["stacking_adaptive_roi"],
                    block_info["stacking_adaptive_n"],
                )

            # Summary
            results_df = pd.DataFrame(all_blocks)
            for s_name in strategies:
                total_n = cumulative[s_name]["n"]
                total_profit = cumulative[s_name]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col_roi = f"{s_name}_roi"
                if col_roi in results_df.columns:
                    vals = results_df[col_roi].values
                    mean_roi = vals.mean()
                    std_roi = vals.std()
                    n_pos = (vals > 0).sum()
                else:
                    mean_roi = std_roi = 0.0
                    n_pos = 0

                logger.info(
                    "%s: overall=%.2f%% (n=%d), mean=%.2f%% std=%.2f%%, pos=%d/%d",
                    s_name,
                    overall_roi,
                    total_n,
                    mean_roi,
                    std_roi,
                    n_pos,
                    len(all_blocks),
                )
                mlflow.log_metrics(
                    {
                        f"wf_{s_name}_roi": round(overall_roi, 2),
                        f"wf_{s_name}_n": total_n,
                        f"wf_{s_name}_mean": round(mean_roi, 2),
                        f"wf_{s_name}_std": round(std_roi, 2),
                        f"wf_{s_name}_pos": n_pos,
                    }
                )

            # Log adaptive thresholds
            if "adaptive_thr" in results_df.columns:
                thrs = results_df["adaptive_thr"].values
                logger.info("Adaptive thresholds: %s", thrs)
                mlflow.log_metric("adaptive_thr_mean", round(thrs.mean(), 3))
                mlflow.log_metric("adaptive_thr_std", round(thrs.std(), 3))

            # Save
            res_path = str(SESSION_DIR / "experiments" / "adaptive_stacking_results.csv")
            results_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
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
