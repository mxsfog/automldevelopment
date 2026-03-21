"""Step 4.6 -- Clean strategies without outlier dependence.

Предыдущий анализ показал: 70% P&L = 1 ставка с odds=490.9.
Нужно проверить, есть ли edge БЕЗ extreme odds.

Стратегии для walk-forward:
1. conf_ev_0.15 + odds <= 10 (убираем extreme parlays)
2. conf_ev_0.15 + odds <= 5 (только low odds)
3. conf_ev_0.15 + только parlays (где edge сильнее)
4. conf_ev_0.15 + только soccer (основной объём, 9% ROI)
5. conf_ev_0.20 + odds <= 10 (более строгий фильтр)
6. conf_ev_0.15 + odds 2-5 (оптимальный range по предыдущему анализу)
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
    """Walk-forward с cleaned strategies."""
    with mlflow.start_run(run_name="phase4/clean_strategies") as run:
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

            strategy_configs = {
                "baseline_cev015": {"cev_thr": 0.15},
                "cev015_odds_le10": {"cev_thr": 0.15, "max_odds": 10},
                "cev015_odds_le5": {"cev_thr": 0.15, "max_odds": 5},
                "cev015_parlays": {"cev_thr": 0.15, "parlay_only": True},
                "cev015_soccer": {"cev_thr": 0.15, "sport": "Soccer"},
                "cev020_odds_le10": {"cev_thr": 0.20, "max_odds": 10},
                "cev015_odds_2_5": {"cev_thr": 0.15, "min_odds": 2, "max_odds": 5},
                "cev015_odds_2_10": {"cev_thr": 0.15, "min_odds": 2, "max_odds": 10},
            }

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "clean_strategies",
                    "n_strategies": len(strategy_configs),
                    "n_test_blocks": len(retrain_schedule),
                }
            )

            cumulative = {s: {"n": 0, "profit": 0.0} for s in strategy_configs}
            all_blocks = []

            for i, test_week in enumerate(retrain_schedule):
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                # Feature engineering
                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train = train_enc[FEATURES_ENC].fillna(0)
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Train ensemble
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

                for s_name, cfg in strategy_configs.items():
                    # Base conf_ev filter
                    mask = conf_ev >= cfg["cev_thr"]

                    # Additional filters
                    if "max_odds" in cfg:
                        mask &= odds <= cfg["max_odds"]
                    if "min_odds" in cfg:
                        mask &= odds >= cfg["min_odds"]
                    if cfg.get("parlay_only"):
                        mask &= test_enc["Is_Parlay"].values == 1
                    if "sport" in cfg:
                        mask &= test_enc["Sport"].values == cfg["sport"]

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

            # Summary
            results_df = pd.DataFrame(all_blocks)
            logger.info("\nResults summary:")
            for s_name in strategy_configs:
                total_n = cumulative[s_name]["n"]
                total_profit = cumulative[s_name]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col_roi = f"{s_name}_roi"
                if col_roi in results_df.columns:
                    vals = results_df[col_roi].values
                    mean_roi = vals.mean()
                    std_roi = vals.std()
                    n_pos = (vals > 0).sum()
                    min_roi = vals.min()
                    max_roi = vals.max()
                else:
                    mean_roi = std_roi = min_roi = max_roi = 0.0
                    n_pos = 0

                logger.info(
                    "  %s: overall=%.2f%% (n=%d) mean=%.2f%% std=%.2f%% "
                    "pos=%d/%d min=%.1f%% max=%.1f%%",
                    s_name,
                    overall_roi,
                    total_n,
                    mean_roi,
                    std_roi,
                    n_pos,
                    len(all_blocks),
                    min_roi,
                    max_roi,
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

            res_path = str(SESSION_DIR / "experiments" / "clean_strategies_results.csv")
            results_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.6 failed")
            raise


if __name__ == "__main__":
    main()
