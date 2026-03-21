"""Step 4.9 -- Agreement filter with odds caps + final validation.

Проверка: ev_and_agree_p02 (лучшая стратегия) с odds caps.
Зависит ли 13.80% от extreme odds?

Также: сравнение нескольких вариантов p_std threshold.
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


def main() -> None:
    """Agreement filter with odds caps walk-forward."""
    with mlflow.start_run(run_name="phase4/agree_capped") as run:
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

            # Strategies: best agreement filter with various odds caps
            strategies = {
                "agree_p02_all": {"ev_thr": 0.05, "pstd": 0.02},
                "agree_p02_le50": {"ev_thr": 0.05, "pstd": 0.02, "max_odds": 50},
                "agree_p02_le10": {"ev_thr": 0.05, "pstd": 0.02, "max_odds": 10},
                "agree_p02_le5": {"ev_thr": 0.05, "pstd": 0.02, "max_odds": 5},
                "agree_p02_le3": {"ev_thr": 0.05, "pstd": 0.02, "max_odds": 3},
                "agree_p02_2_5": {"ev_thr": 0.05, "pstd": 0.02, "min_odds": 2, "max_odds": 5},
                "agree_p03_le10": {"ev_thr": 0.05, "pstd": 0.03, "max_odds": 10},
                "agree_p03_le5": {"ev_thr": 0.05, "pstd": 0.03, "max_odds": 5},
                "conf_ev015_le10": {"conf_ev_thr": 0.15, "max_odds": 10},
            }

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "agreement_capped_final",
                    "n_strategies": len(strategies),
                }
            )

            cumulative = {s: {"n": 0, "profit": 0.0} for s in strategies}
            all_blocks = []

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
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                cb = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0,
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

                block_info = {"block": i, "week": test_week, "n_test": len(test_df)}

                for s_name, cfg in strategies.items():
                    if "conf_ev_thr" in cfg:
                        mask = conf_ev >= cfg["conf_ev_thr"]
                    else:
                        mask = (ev >= cfg["ev_thr"]) & (p_std <= cfg["pstd"])

                    if "max_odds" in cfg:
                        mask &= odds <= cfg["max_odds"]
                    if "min_odds" in cfg:
                        mask &= odds >= cfg["min_odds"]

                    n = mask.sum()
                    if n > 0:
                        sel = test_enc[mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                    else:
                        roi = 0.0

                    block_info[f"{s_name}_roi"] = round(roi, 2)
                    block_info[f"{s_name}_n"] = int(n)
                    if n > 0:
                        cumulative[s_name]["n"] += n
                        cumulative[s_name]["profit"] += n * roi / 100

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nAgreement + caps results:")
            for s_name in strategies:
                total_n = cumulative[s_name]["n"]
                total_profit = cumulative[s_name]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"{s_name}_roi"
                if col in results_df.columns:
                    vals = results_df[col].values
                    n_pos = (vals > 0).sum()
                    mean_roi = vals.mean()
                    std_roi = vals.std()
                else:
                    n_pos = 0
                    mean_roi = std_roi = 0.0

                logger.info(
                    "  %s: overall=%.2f%% (n=%d) mean=%.2f%% std=%.2f%% pos=%d/%d",
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
                        f"wf_{s_name}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "agree_capped_results.csv")
            results_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.9 failed")
            raise


if __name__ == "__main__":
    main()
