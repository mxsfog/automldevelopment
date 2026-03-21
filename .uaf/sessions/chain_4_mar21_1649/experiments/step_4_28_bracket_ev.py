"""Step 4.28 -- Bracket-specific EV thresholds + confidence requirements.

Step 4.17 showed calibration varies by odds bracket:
- [1-2]: gap +0.007, ROI +3.7%
- [2-3]: gap +0.025, ROI -0.4%
- [3-5]: gap -0.002, ROI -6.5%
- [5-10]: gap +0.003, ROI -2.9%
- [50-1000]: gap +0.008, ROI +132.8%

Hypothesis: bracket-specific thresholds can account for differential overconfidence.
E.g., require higher EV for brackets where model is more overconfident.
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
    "Odds", "USD", "Is_Parlay", "Outcomes_Count",
    "ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV",
    "ML_Winrate_Diff", "ML_Rating_Diff",
    "Outcome_Odds", "n_outcomes", "mean_outcome_odds",
    "max_outcome_odds", "min_outcome_odds",
    "Sport_target_enc", "Sport_count_enc",
    "Market_target_enc", "Market_count_enc",
]

# Bracket-specific strategies: (min_odds, max_odds, min_ev, max_p_std)
BRACKET_STRATEGIES: dict[str, list[tuple[float, float, float, float]]] = {
    "baseline": [
        (1.0, 1000.0, 0.05, 0.02),
    ],
    "bracket_v1": [
        (1.0, 2.0, 0.02, 0.015),
        (2.0, 5.0, 0.10, 0.015),
        (5.0, 10.0, 0.15, 0.02),
        (10.0, 1000.0, 0.05, 0.02),
    ],
    "bracket_v2": [
        (1.0, 2.0, 0.01, 0.01),
        (2.0, 5.0, 0.08, 0.01),
        (5.0, 50.0, 0.10, 0.02),
        (50.0, 1000.0, 0.05, 0.02),
    ],
    "high_ev_only": [
        (1.0, 1000.0, 0.15, 0.02),
    ],
    "low_odds_strict": [
        (1.0, 3.0, 0.03, 0.01),
        (3.0, 1000.0, 0.05, 0.02),
    ],
    "moderate_focus": [
        (1.0, 2.0, 0.02, 0.01),
        (2.0, 10.0, 0.12, 0.015),
    ],
}


def main() -> None:
    """Bracket-specific EV thresholds walk-forward."""
    with mlflow.start_run(run_name="phase4/bracket_ev") as run:
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
                    "method": "bracket_ev_thresholds",
                    "n_strategies": len(BRACKET_STRATEGIES),
                }
            )

            cumulative: dict[str, dict] = {
                k: {"n": 0, "profit": 0.0} for k in BRACKET_STRATEGIES
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
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                cb = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6,
                    random_seed=42, verbose=0,
                )
                cb.fit(x_train, y_train)

                lgbm = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=42, verbose=-1, min_child_samples=50,
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

                block_info: dict = {"block": i, "week": test_week, "n_test": len(test_df)}

                for s_name, brackets in BRACKET_STRATEGIES.items():
                    combined_mask = np.zeros(len(test_enc), dtype=bool)
                    for min_o, max_o, min_ev, max_pstd in brackets:
                        bracket_mask = (
                            (odds >= min_o)
                            & (odds < max_o)
                            & (ev >= min_ev)
                            & (p_std <= max_pstd)
                        )
                        combined_mask |= bracket_mask

                    n = combined_mask.sum()
                    roi = 0.0
                    if n > 0:
                        sel = test_enc[combined_mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                        cumulative[s_name]["n"] += n
                        cumulative[s_name]["profit"] += n * roi / 100

                    block_info[f"{s_name}_roi"] = round(roi, 2)
                    block_info[f"{s_name}_n"] = int(n)

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nBracket-specific EV results:")
            for s_name in BRACKET_STRATEGIES:
                total_n = cumulative[s_name]["n"]
                total_profit = cumulative[s_name]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"{s_name}_roi"
                n_pos = (results_df[col].values > 0).sum() if col in results_df.columns else 0

                logger.info(
                    "  %s: overall=%.2f%% (n=%d) pos=%d/%d",
                    s_name, overall_roi, total_n, n_pos, len(all_blocks),
                )
                mlflow.log_metrics(
                    {
                        f"wf_{s_name}_roi": round(overall_roi, 2),
                        f"wf_{s_name}_n": total_n,
                        f"wf_{s_name}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "bracket_ev_results.csv")
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
            logger.exception("Step 4.28 failed")
            raise


if __name__ == "__main__":
    main()
