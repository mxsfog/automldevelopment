"""Step 4.15 -- Capped-odds permutation test.

Step 4.13 дал невалидный результат из-за extreme odds (perm mean = 1136%).
Здесь: тест с odds <= 10 для корректной оценки значимости.
Также: AUC permutation test (метрика не зависит от odds scale).
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

N_PERMUTATIONS = 50


def run_walkforward_capped(
    df: pd.DataFrame,
    retrain_schedule: list,
    shuffle_test: bool = False,
    perm_seed: int = 0,
    max_odds: float = 10.0,
) -> dict[str, float]:
    """Walk-forward with odds cap, return ROI and AUC."""
    cum_n = 0
    cum_profit = 0.0
    all_y_true: list[float] = []
    all_p_mean: list[float] = []

    for test_week in retrain_schedule:
        train_mask = df["year_week"] < test_week
        test_mask = df["year_week"] == test_week

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        if len(test_df) == 0:
            continue

        train_enc, _ = add_sport_market_features(train_df, train_df)
        test_enc, _ = add_sport_market_features(test_df, train_enc)

        # Cap odds for evaluation
        odds_mask = test_enc["Odds"] <= max_odds
        test_capped = test_enc[odds_mask].copy()
        if len(test_capped) == 0:
            continue

        if shuffle_test:
            rng = np.random.RandomState(perm_seed + hash(test_week) % 10000)
            test_capped["target"] = rng.permutation(test_capped["target"].values)

        x_train = train_enc[FEATURES_ENC].fillna(0)
        y_train = train_enc["target"]
        x_test = test_capped[FEATURES_ENC].fillna(0)

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

        odds = test_capped["Odds"].values
        ev = p_mean * odds - 1
        mask = (ev >= 0.05) & (p_std <= 0.02)

        n = mask.sum()
        if n > 0:
            sel = test_capped[mask]
            payout = (sel["target"] * sel["Odds"]).sum()
            roi = (payout - n) / n * 100
            cum_n += n
            cum_profit += n * roi / 100

        all_y_true.extend(test_capped["target"].values.tolist())
        all_p_mean.extend(p_mean.tolist())

    overall_roi = (cum_profit / cum_n * 100) if cum_n > 0 else 0.0
    auc = roc_auc_score(all_y_true, all_p_mean) if len(set(all_y_true)) > 1 else 0.5

    return {"roi_capped": overall_roi, "auc": auc, "n": cum_n}


def main() -> None:
    """Capped permutation test."""
    with mlflow.start_run(run_name="phase4/perm_capped") as run:
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
                    "method": "capped_permutation_test",
                    "n_permutations": N_PERMUTATIONS,
                    "max_odds": 10,
                }
            )

            # Real results
            logger.info("Computing real walk-forward ROI (odds<=10)...")
            real = run_walkforward_capped(df, retrain_schedule)
            logger.info(
                "Real: ROI=%.2f%% AUC=%.4f n=%d", real["roi_capped"], real["auc"], real["n"]
            )

            # Permutation results
            perm_rois: list[float] = []
            perm_aucs: list[float] = []

            for perm_i in range(N_PERMUTATIONS):
                if perm_i % 10 == 0:
                    logger.info("Permutation %d/%d...", perm_i + 1, N_PERMUTATIONS)
                res = run_walkforward_capped(
                    df, retrain_schedule, shuffle_test=True, perm_seed=perm_i * 1000
                )
                perm_rois.append(res["roi_capped"])
                perm_aucs.append(res["auc"])

            perm_rois_arr = np.array(perm_rois)
            perm_aucs_arr = np.array(perm_aucs)

            # P-values
            p_roi = (np.sum(perm_rois_arr >= real["roi_capped"]) + 1) / (N_PERMUTATIONS + 1)
            p_auc = (np.sum(perm_aucs_arr >= real["auc"]) + 1) / (N_PERMUTATIONS + 1)

            logger.info("\nCapped permutation test results (odds<=10):")
            logger.info(
                "  ROI: real=%.2f%% perm_mean=%.2f%% perm_std=%.2f%% p=%.3f %s",
                real["roi_capped"],
                perm_rois_arr.mean(),
                perm_rois_arr.std(),
                p_roi,
                "SIGNIFICANT" if p_roi < 0.05 else "NOT significant",
            )
            logger.info(
                "  AUC: real=%.4f perm_mean=%.4f perm_std=%.4f p=%.3f %s",
                real["auc"],
                perm_aucs_arr.mean(),
                perm_aucs_arr.std(),
                p_auc,
                "SIGNIFICANT" if p_auc < 0.05 else "NOT significant",
            )

            mlflow.log_metrics(
                {
                    "real_roi_capped": round(real["roi_capped"], 2),
                    "real_auc": round(real["auc"], 4),
                    "perm_roi_mean": round(float(perm_rois_arr.mean()), 2),
                    "perm_roi_std": round(float(perm_rois_arr.std()), 2),
                    "perm_auc_mean": round(float(perm_aucs_arr.mean()), 4),
                    "perm_auc_std": round(float(perm_aucs_arr.std()), 4),
                    "p_value_roi": round(float(p_roi), 4),
                    "p_value_auc": round(float(p_auc), 4),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.15 failed")
            raise


if __name__ == "__main__":
    main()
