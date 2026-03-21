"""Step 4.21 -- Adversarial validation + distribution shift analysis.

Проверка: отличается ли test set от train set по распределению?
Если adversarial classifier может отличить train от test с AUC > 0.7,
это объясняет падение ROI в walk-forward vs fixed split.

Также: анализ drift по ключевым фичам между блоками.
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
from sklearn.metrics import roc_auc_score

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


def main() -> None:
    """Adversarial validation + distribution shift."""
    with mlflow.start_run(run_name="phase4/adversarial_validation") as run:
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
                    "method": "adversarial_validation_drift",
                }
            )

            # Per-block adversarial validation
            logger.info("=== Adversarial Validation per WF block ===")
            block_results = []

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
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Adversarial: classify train(0) vs test(1)
                adv_x = pd.concat([x_train, x_test], ignore_index=True)
                adv_y = np.array([0] * len(x_train) + [1] * len(x_test))

                # Quick shuffle and split for adversarial AUC
                rng = np.random.RandomState(42)
                idx = rng.permutation(len(adv_x))
                split = int(0.7 * len(idx))
                tr_idx, val_idx = idx[:split], idx[split:]

                adv_model = CatBoostClassifier(
                    iterations=100, learning_rate=0.1, depth=4, random_seed=42, verbose=0
                )
                adv_model.fit(adv_x.iloc[tr_idx], adv_y[tr_idx])
                adv_pred = adv_model.predict_proba(adv_x.iloc[val_idx])[:, 1]
                adv_auc = roc_auc_score(adv_y[val_idx], adv_pred)

                # Feature drift: compare means
                drift_features = {}
                for feat in ["Odds", "ML_P_Model", "ML_P_Implied", "ML_Edge"]:
                    train_mean = x_train[feat].mean()
                    test_mean = x_test[feat].mean()
                    pct_change = (
                        (test_mean - train_mean) / abs(train_mean) * 100 if train_mean != 0 else 0
                    )
                    drift_features[feat] = round(pct_change, 2)

                # Target drift
                train_winrate = train_enc["target"].mean()
                test_winrate = test_enc["target"].mean()

                info = {
                    "block": i,
                    "week": test_week,
                    "n_train": len(x_train),
                    "n_test": len(x_test),
                    "adv_auc": round(adv_auc, 4),
                    "train_winrate": round(train_winrate, 4),
                    "test_winrate": round(test_winrate, 4),
                    "winrate_delta": round(test_winrate - train_winrate, 4),
                }
                info.update({f"drift_{k}": v for k, v in drift_features.items()})
                block_results.append(info)

                logger.info(
                    "  Block %d (week %d): adv_auc=%.3f train_wr=%.3f test_wr=%.3f "
                    "delta_wr=%.3f drift_odds=%.1f%% drift_edge=%.1f%%",
                    i,
                    test_week,
                    adv_auc,
                    train_winrate,
                    test_winrate,
                    test_winrate - train_winrate,
                    drift_features["Odds"],
                    drift_features["ML_Edge"],
                )

            results_df = pd.DataFrame(block_results)

            mean_adv_auc = results_df["adv_auc"].mean()
            mean_wr_delta = results_df["winrate_delta"].mean()
            logger.info(
                "\nSummary: mean_adv_auc=%.3f mean_wr_delta=%.4f",
                mean_adv_auc,
                mean_wr_delta,
            )

            if mean_adv_auc > 0.7:
                logger.info("SIGNIFICANT distribution shift detected!")
            else:
                logger.info("No significant distribution shift (adv AUC ~0.5)")

            mlflow.log_metrics(
                {
                    "mean_adv_auc": round(mean_adv_auc, 4),
                    "mean_winrate_delta": round(mean_wr_delta, 4),
                    "max_adv_auc": round(float(results_df["adv_auc"].max()), 4),
                    "min_adv_auc": round(float(results_df["adv_auc"].min()), 4),
                }
            )

            res_path = str(SESSION_DIR / "experiments" / "adversarial_results.csv")
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
            logger.exception("Step 4.21 failed")
            raise


if __name__ == "__main__":
    main()
