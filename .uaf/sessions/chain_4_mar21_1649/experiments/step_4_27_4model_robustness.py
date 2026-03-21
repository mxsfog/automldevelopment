"""Step 4.27 -- Robustness check for 4-model ensemble with p_std<=0.015.

4m_p015 gives 16.98% ROI. Check:
1. Seed stability (5 seeds)
2. Odds cap sensitivity
3. Comparison with 3-model baseline under same conditions
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

SEEDS = [42, 123, 777, 2024, 31337]
ODDS_CAPS = [None, 50, 10, 5]


def run_walk_forward(
    df: pd.DataFrame,
    retrain_schedule: list,
    seed: int,
    n_models: int,
    p_std_thresh: float,
    odds_cap: float | None,
) -> dict:
    """Run single WF pass, return cumulative metrics."""
    random.seed(seed)
    np.random.seed(seed)

    cum_n = 0
    cum_profit = 0.0
    n_pos = 0
    n_blocks = 0

    for test_week in retrain_schedule:
        train_mask = df["year_week"] < test_week
        test_mask = df["year_week"] == test_week

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        if len(test_df) == 0:
            continue

        train_enc, _ = add_sport_market_features(train_df, train_df)
        test_enc, _ = add_sport_market_features(test_df, train_enc)

        x_train = train_enc[FEATURES_ENC].fillna(0)
        y_train = train_enc["target"]
        x_test = test_enc[FEATURES_ENC].fillna(0)

        cb = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
        )
        cb.fit(x_train, y_train)

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=seed, verbose=-1, min_child_samples=50,
        )
        lgbm.fit(x_train, y_train)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        lr.fit(x_train_s, y_train)

        preds = [
            cb.predict_proba(x_test)[:, 1],
            lgbm.predict_proba(x_test)[:, 1],
            lr.predict_proba(x_test_s)[:, 1],
        ]

        if n_models == 4:
            xgb = XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                random_state=seed, verbosity=0, min_child_weight=50,
                use_label_encoder=False, eval_metric="logloss",
            )
            xgb.fit(x_train, y_train)
            preds.append(xgb.predict_proba(x_test)[:, 1])

        p_mean = np.mean(preds, axis=0)
        p_std = np.std(preds, axis=0)

        odds = test_enc["Odds"].values
        ev = p_mean * odds - 1
        mask = (ev >= 0.05) & (p_std <= p_std_thresh)

        if odds_cap is not None:
            mask = mask & (odds <= odds_cap)

        n = mask.sum()
        n_blocks += 1
        if n > 0:
            sel = test_enc[mask]
            payout = (sel["target"] * sel["Odds"]).sum()
            roi = (payout - n) / n * 100
            cum_n += n
            cum_profit += n * roi / 100
            if roi > 0:
                n_pos += 1

    overall_roi = (cum_profit / cum_n * 100) if cum_n > 0 else 0.0
    return {
        "roi": round(overall_roi, 2),
        "n": cum_n,
        "n_pos": n_pos,
        "n_blocks": n_blocks,
    }


def main() -> None:
    """4-model robustness check."""
    with mlflow.start_run(run_name="phase4/4model_robustness") as run:
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
                    "method": "4model_robustness",
                    "seeds": str(SEEDS),
                    "odds_caps": str(ODDS_CAPS),
                }
            )

            configs = [
                ("3m_p02", 3, 0.02),
                ("4m_p02", 4, 0.02),
                ("4m_p015", 4, 0.015),
            ]

            all_results: list[dict] = []

            # 1. Seed stability (no odds cap)
            logger.info("=== Seed Stability (no odds cap) ===")
            for name, n_m, p_th in configs:
                rois = []
                for seed in SEEDS:
                    res = run_walk_forward(df, retrain_schedule, seed, n_m, p_th, None)
                    rois.append(res["roi"])
                    all_results.append({
                        "config": name, "seed": seed, "odds_cap": "none",
                        **res,
                    })
                mean_roi = np.mean(rois)
                std_roi = np.std(rois)
                logger.info(
                    "  %s: mean=%.2f%% std=%.2f%% [%.1f - %.1f]",
                    name, mean_roi, std_roi, min(rois), max(rois),
                )
                mlflow.log_metrics({
                    f"seed_{name}_mean": round(mean_roi, 2),
                    f"seed_{name}_std": round(std_roi, 2),
                })

            # 2. Odds cap sensitivity (seed=42 only)
            logger.info("\n=== Odds Cap Sensitivity (seed=42) ===")
            for name, n_m, p_th in configs:
                for cap in ODDS_CAPS:
                    if cap is None:
                        continue
                    res = run_walk_forward(df, retrain_schedule, 42, n_m, p_th, cap)
                    all_results.append({
                        "config": name, "seed": 42, "odds_cap": str(cap),
                        **res,
                    })
                    logger.info(
                        "  %s cap=%s: roi=%.2f%% n=%d pos=%d/%d",
                        name, cap, res["roi"], res["n"], res["n_pos"], res["n_blocks"],
                    )
                    mlflow.log_metrics({
                        f"cap{cap}_{name}_roi": round(res["roi"], 2),
                        f"cap{cap}_{name}_n": res["n"],
                    })

            res_df = pd.DataFrame(all_results)
            res_path = str(SESSION_DIR / "experiments" / "4model_robustness_results.csv")
            res_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.27 failed")
            raise


if __name__ == "__main__":
    main()
