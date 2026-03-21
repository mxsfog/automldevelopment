"""Step 4.7 -- Low-odds focused model + daily walk-forward.

Гипотеза: модель обученная только на odds 1-10 (без extreme parlays)
лучше калибрована для low-odds бетов. Предыдущие сессии (step 4.7)
пробовали segment models на fixed split, но не walk-forward.

Дополнительно: daily walk-forward для более гранулярной оценки.
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


def train_and_predict(
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


def compute_conf_ev_roi(
    df: pd.DataFrame,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    cev_thr: float = 0.15,
    min_odds: float = 0.0,
    max_odds: float = 1e6,
) -> dict:
    """ROI с conf_ev фильтром + odds range."""
    odds = df["Odds"].values
    ev = p_mean * odds - 1
    conf = 1.0 / (1.0 + p_std * 10)
    conf_ev = ev * conf

    mask = (conf_ev >= cev_thr) & (odds >= min_odds) & (odds <= max_odds)
    n = mask.sum()
    if n == 0:
        return {"roi": 0.0, "n": 0}
    sel = df[mask]
    payout = (sel["target"] * sel["Odds"]).sum()
    roi = (payout - n) / n * 100
    return {"roi": round(roi, 2), "n": int(n)}


def main() -> None:
    """Low-odds model + daily walk-forward."""
    with mlflow.start_run(run_name="phase4/lowodds_daily_wf") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            # Daily blocks
            df["date"] = df["Created_At"].dt.date
            unique_dates = sorted(df["date"].unique())
            n_days = len(unique_dates)
            min_train_days = int(n_days * 0.6)
            retrain_schedule = unique_dates[min_train_days:]

            # Retrain every 3 days (daily is too expensive)
            retrain_every = 3
            retrain_indices = list(range(0, len(retrain_schedule), retrain_every))

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward_daily",
                    "seed": 42,
                    "method": "lowodds_focused_model_daily",
                    "n_total_days": n_days,
                    "min_train_days": min_train_days,
                    "retrain_frequency": f"every_{retrain_every}_days",
                    "n_retrain_blocks": len(retrain_indices),
                }
            )

            strategies = [
                "full_model_cev015_odds2_5",
                "lowodds_model_cev015_odds2_5",
                "full_model_cev015_all",
                "lowodds_model_cev015_all",
            ]
            cumulative = {s: {"n": 0, "profit": 0.0} for s in strategies}
            block_results = []

            for block_idx in retrain_indices:
                test_start = retrain_schedule[block_idx]
                test_end_idx = min(block_idx + retrain_every, len(retrain_schedule))
                test_dates = retrain_schedule[block_idx:test_end_idx]

                train_mask = df["date"] < test_start
                test_mask = df["date"].isin(test_dates)

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0 or len(train_df) < 1000:
                    continue

                # Feature engineering
                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train_full = train_enc[FEATURES_ENC].fillna(0)
                y_train_full = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Low-odds model: train only on odds 1-10
                low_odds_mask = train_enc["Odds"] <= 10
                x_train_low = x_train_full[low_odds_mask]
                y_train_low = y_train_full[low_odds_mask]

                # Full model
                p_full_mean, p_full_std = train_and_predict(x_train_full, y_train_full, x_test)

                # Low-odds model
                if len(x_train_low) >= 500:
                    p_low_mean, p_low_std = train_and_predict(x_train_low, y_train_low, x_test)
                else:
                    p_low_mean, p_low_std = p_full_mean, p_full_std

                block_info = {
                    "block": block_idx,
                    "test_start": str(test_start),
                    "n_test": len(test_df),
                    "n_train_full": len(x_train_full),
                    "n_train_low": int(low_odds_mask.sum()),
                }

                # Evaluate strategies
                for s_name, p_m, p_s in [
                    ("full_model", p_full_mean, p_full_std),
                    ("lowodds_model", p_low_mean, p_low_std),
                ]:
                    # All odds
                    res_all = compute_conf_ev_roi(test_enc, p_m, p_s, 0.15)
                    block_info[f"{s_name}_cev015_all_roi"] = res_all["roi"]
                    block_info[f"{s_name}_cev015_all_n"] = res_all["n"]
                    cum_key = f"{s_name}_cev015_all"
                    if res_all["n"] > 0:
                        cumulative[cum_key]["n"] += res_all["n"]
                        cumulative[cum_key]["profit"] += res_all["n"] * res_all["roi"] / 100

                    # Odds 2-5
                    res_25 = compute_conf_ev_roi(test_enc, p_m, p_s, 0.15, 2, 5)
                    block_info[f"{s_name}_cev015_odds2_5_roi"] = res_25["roi"]
                    block_info[f"{s_name}_cev015_odds2_5_n"] = res_25["n"]
                    cum_key = f"{s_name}_cev015_odds2_5"
                    if res_25["n"] > 0:
                        cumulative[cum_key]["n"] += res_25["n"]
                        cumulative[cum_key]["profit"] += res_25["n"] * res_25["roi"] / 100

                block_results.append(block_info)

            # Summary
            results_df = pd.DataFrame(block_results)
            logger.info("Daily walk-forward results (%d blocks):", len(block_results))

            for s_name in strategies:
                total_n = cumulative[s_name]["n"]
                total_profit = cumulative[s_name]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"{s_name}_roi"
                if col in results_df.columns:
                    vals = results_df[col].values
                    n_pos = (vals > 0).sum()
                    n_blocks = len(vals)
                else:
                    n_pos = n_blocks = 0

                logger.info(
                    "  %s: overall=%.2f%% (n=%d), pos=%d/%d",
                    s_name,
                    overall_roi,
                    total_n,
                    n_pos,
                    n_blocks,
                )
                mlflow.log_metrics(
                    {
                        f"wf_{s_name}_roi": round(overall_roi, 2),
                        f"wf_{s_name}_n": total_n,
                        f"wf_{s_name}_pos": n_pos,
                        f"wf_{s_name}_blocks": n_blocks,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "lowodds_daily_results.csv")
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
            logger.exception("Step 4.7 failed")
            raise


if __name__ == "__main__":
    main()
