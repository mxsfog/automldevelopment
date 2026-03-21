"""Step 4.11 -- Rolling window walk-forward (fixed size vs expanding).

Гипотеза: фиксированное окно обучения (last N weeks) лучше адаптируется
к concept drift чем expanding window (all history).

Сравнение: expanding vs rolling 4/6/8 weeks.
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


def train_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 3-model ensemble, return (p_mean, p_std)."""
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
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
    return p_mean, p_std


def main() -> None:
    """Rolling vs expanding window walk-forward."""
    with mlflow.start_run(run_name="phase4/rolling_window") as run:
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

            window_sizes = {
                "expanding": None,
                "rolling_4w": 4,
                "rolling_6w": 6,
                "rolling_8w": 8,
            }

            # Strategies to evaluate per window
            selection_strategies = ["conf_ev_015", "ev005_agree_p02", "pmean055"]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "rolling_vs_expanding_window",
                    "n_window_types": len(window_sizes),
                    "n_strategies": len(selection_strategies),
                }
            )

            # cumulative[window_name][strategy] = {n, profit}
            cumulative: dict[str, dict[str, dict]] = {}
            for w_name in window_sizes:
                cumulative[w_name] = {s: {"n": 0, "profit": 0.0} for s in selection_strategies}

            all_blocks = []

            for i, test_week in enumerate(retrain_schedule):
                test_mask = df["year_week"] == test_week
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                block_info = {"block": i, "week": test_week, "n_test": len(test_df)}

                for w_name, w_size in window_sizes.items():
                    if w_size is None:
                        # Expanding: all data before test_week
                        train_mask = df["year_week"] < test_week
                    else:
                        # Rolling: only last w_size weeks
                        week_idx = unique_weeks.index(test_week)
                        start_idx = max(0, week_idx - w_size)
                        train_weeks = unique_weeks[start_idx:week_idx]
                        train_mask = df["year_week"].isin(train_weeks)

                    train_df = df[train_mask].copy()
                    if len(train_df) < 500:
                        for s_name in selection_strategies:
                            block_info[f"{w_name}_{s_name}_roi"] = 0.0
                            block_info[f"{w_name}_{s_name}_n"] = 0
                        continue

                    train_enc, _ = add_sport_market_features(train_df, train_df)
                    test_enc, _ = add_sport_market_features(test_df, train_enc)

                    x_train = train_enc[FEATURES_ENC].fillna(0)
                    y_train = train_enc["target"]
                    x_test = test_enc[FEATURES_ENC].fillna(0)

                    block_info[f"{w_name}_n_train"] = len(x_train)

                    p_mean, p_std = train_predict(x_train, y_train, x_test)

                    odds = test_enc["Odds"].values
                    ev = p_mean * odds - 1
                    conf = 1.0 / (1.0 + p_std * 10)
                    conf_ev = ev * conf

                    masks = {
                        "conf_ev_015": conf_ev >= 0.15,
                        "ev005_agree_p02": (ev >= 0.05) & (p_std <= 0.02),
                        "pmean055": p_mean >= 0.55,
                    }

                    for s_name in selection_strategies:
                        mask = masks[s_name]
                        n = mask.sum()
                        if n > 0:
                            sel = test_enc[mask]
                            payout = (sel["target"] * sel["Odds"]).sum()
                            roi = (payout - n) / n * 100
                        else:
                            roi = 0.0

                        key = f"{w_name}_{s_name}"
                        block_info[f"{key}_roi"] = round(roi, 2)
                        block_info[f"{key}_n"] = int(n)
                        if n > 0:
                            cumulative[w_name][s_name]["n"] += n
                            cumulative[w_name][s_name]["profit"] += n * roi / 100

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nRolling vs expanding window results:")

            for w_name in window_sizes:
                logger.info("  Window: %s", w_name)
                for s_name in selection_strategies:
                    total_n = cumulative[w_name][s_name]["n"]
                    total_profit = cumulative[w_name][s_name]["profit"]
                    overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                    col = f"{w_name}_{s_name}_roi"
                    if col in results_df.columns:
                        vals = results_df[col].values
                        n_pos = (vals > 0).sum()
                    else:
                        n_pos = 0

                    logger.info(
                        "    %s: overall=%.2f%% (n=%d) pos=%d/%d",
                        s_name,
                        overall_roi,
                        total_n,
                        n_pos,
                        len(all_blocks),
                    )
                    mlflow.log_metrics(
                        {
                            f"wf_{w_name}_{s_name}_roi": round(overall_roi, 2),
                            f"wf_{w_name}_{s_name}_n": total_n,
                            f"wf_{w_name}_{s_name}_pos": n_pos,
                        }
                    )

            res_path = str(SESSION_DIR / "experiments" / "rolling_window_results.csv")
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
            logger.exception("Step 4.11 failed")
            raise


if __name__ == "__main__":
    main()
