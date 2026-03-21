"""Step 4.13 -- Permutation test: является ли edge реальным?

Метод: перемешать target (Status) N раз, каждый раз пересчитать walk-forward ROI.
Если реальный ROI > 95% permutation ROIs → edge статистически значим.
Это золотой стандарт для проверки ML-стратегий в трейдинге.
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

N_PERMUTATIONS = 20


def run_walkforward(
    df: pd.DataFrame,
    retrain_schedule: list,
    unique_weeks: list,
    shuffle_test_target: bool = False,
    perm_seed: int = 0,
) -> dict[str, float]:
    """Run walk-forward, optionally shuffling test targets for permutation test."""
    cumulative: dict[str, dict] = {
        "ev005_agree_p02": {"n": 0, "profit": 0.0},
        "conf_ev_015": {"n": 0, "profit": 0.0},
    }

    for test_week in retrain_schedule:
        train_mask = df["year_week"] < test_week
        test_mask = df["year_week"] == test_week

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        if len(test_df) == 0:
            continue

        train_enc, _ = add_sport_market_features(train_df, train_df)
        test_enc, _ = add_sport_market_features(test_df, train_enc)

        if shuffle_test_target:
            rng = np.random.RandomState(perm_seed + hash(test_week) % 10000)
            test_enc["target"] = rng.permutation(test_enc["target"].values)

        x_train = train_enc[FEATURES_ENC].fillna(0)
        y_train = train_enc["target"]
        x_test = test_enc[FEATURES_ENC].fillna(0)

        cb = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
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
        x_test_s = scaler.transform(x_test)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(x_train_s, y_train)

        p_cb = cb.predict_proba(x_test)[:, 1]
        p_lgbm = lgbm.predict_proba(x_test)[:, 1]
        p_lr = lr.predict_proba(x_test_s)[:, 1]
        p_mean = (p_cb + p_lgbm + p_lr) / 3
        p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)

        odds = test_enc["Odds"].values
        ev = p_mean * odds - 1
        conf = 1.0 / (1.0 + p_std * 10)
        conf_ev = ev * conf

        masks = {
            "ev005_agree_p02": (ev >= 0.05) & (p_std <= 0.02),
            "conf_ev_015": conf_ev >= 0.15,
        }

        for s_name, mask in masks.items():
            n = mask.sum()
            if n > 0:
                sel = test_enc[mask]
                payout = (sel["target"] * sel["Odds"]).sum()
                roi = (payout - n) / n * 100
                cumulative[s_name]["n"] += n
                cumulative[s_name]["profit"] += n * roi / 100

    results = {}
    for s_name in cumulative:
        total_n = cumulative[s_name]["n"]
        total_profit = cumulative[s_name]["profit"]
        results[s_name] = (total_profit / total_n * 100) if total_n > 0 else 0.0
    return results


def main() -> None:
    """Permutation test for walk-forward strategies."""
    with mlflow.start_run(run_name="phase4/permutation_test") as run:
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
                    "method": "permutation_test",
                    "n_permutations": N_PERMUTATIONS,
                }
            )

            # Real results
            logger.info("Computing real walk-forward ROI...")
            real_results = run_walkforward(df, retrain_schedule, unique_weeks)
            logger.info("Real results: %s", real_results)

            # Permutation results
            perm_results: dict[str, list[float]] = {s: [] for s in real_results}

            for perm_i in range(N_PERMUTATIONS):
                logger.info("Permutation %d/%d...", perm_i + 1, N_PERMUTATIONS)
                perm_roi = run_walkforward(
                    df,
                    retrain_schedule,
                    unique_weeks,
                    shuffle_test_target=True,
                    perm_seed=perm_i * 1000,
                )
                for s_name in perm_results:
                    perm_results[s_name].append(perm_roi[s_name])

            # Compute p-values
            logger.info("\nPermutation test results:")
            for s_name in real_results:
                real_roi = real_results[s_name]
                perm_rois = np.array(perm_results[s_name])
                p_value = (np.sum(perm_rois >= real_roi) + 1) / (N_PERMUTATIONS + 1)
                perm_mean = perm_rois.mean()
                perm_std = perm_rois.std()

                logger.info(
                    "  %s: real=%.2f%% perm_mean=%.2f%% perm_std=%.2f%% p=%.3f %s",
                    s_name,
                    real_roi,
                    perm_mean,
                    perm_std,
                    p_value,
                    "SIGNIFICANT" if p_value < 0.05 else "NOT significant",
                )
                mlflow.log_metrics(
                    {
                        f"perm_{s_name}_real_roi": round(real_roi, 2),
                        f"perm_{s_name}_perm_mean": round(perm_mean, 2),
                        f"perm_{s_name}_perm_std": round(perm_std, 2),
                        f"perm_{s_name}_p_value": round(p_value, 4),
                    }
                )

            # Save detailed results
            rows = []
            for s_name in perm_results:
                for i, roi in enumerate(perm_results[s_name]):
                    rows.append({"strategy": s_name, "permutation": i, "roi": round(roi, 2)})
                rows.append(
                    {"strategy": s_name, "permutation": -1, "roi": round(real_results[s_name], 2)}
                )
            res_df = pd.DataFrame(rows)
            res_path = str(SESSION_DIR / "experiments" / "permutation_results.csv")
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
            logger.exception("Step 4.13 failed")
            raise


if __name__ == "__main__":
    main()
