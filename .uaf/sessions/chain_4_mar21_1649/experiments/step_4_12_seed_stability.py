"""Step 4.12 -- Seed stability of agreement filter in walk-forward.

Проверка: стабилен ли результат ev005_agree_p02 (13.80%) по seeds?
chain_3 показала seed std=0.44% для edge strategy (хорошо).
Тест: 5 seeds × walk-forward для лучших стратегий.
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

SEEDS = [42, 123, 777, 2024, 31337]
STRATEGIES = ["ev005_agree_p02", "conf_ev_015", "pmean055"]


def main() -> None:
    """Seed stability walk-forward."""
    with mlflow.start_run(run_name="phase4/seed_stability") as run:
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
                    "method": "seed_stability",
                    "seeds": str(SEEDS),
                    "n_seeds": len(SEEDS),
                    "n_strategies": len(STRATEGIES),
                }
            )

            # seed_results[seed][strategy] = {n, profit}
            seed_results: dict[int, dict[str, dict]] = {}

            for seed in SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                seed_results[seed] = {s: {"n": 0, "profit": 0.0} for s in STRATEGIES}

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
                        iterations=200,
                        learning_rate=0.05,
                        depth=6,
                        random_seed=seed,
                        verbose=0,
                    )
                    cb.fit(x_train, y_train)

                    lgbm = LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=seed,
                        verbose=-1,
                        min_child_samples=50,
                    )
                    lgbm.fit(x_train, y_train)

                    scaler = StandardScaler()
                    x_train_s = scaler.fit_transform(x_train)
                    x_test_s = scaler.transform(x_test)
                    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
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
                        "pmean055": p_mean >= 0.55,
                    }

                    for s_name in STRATEGIES:
                        mask = masks[s_name]
                        n = mask.sum()
                        if n > 0:
                            sel = test_enc[mask]
                            payout = (sel["target"] * sel["Odds"]).sum()
                            roi_val = (payout - n) / n * 100
                            seed_results[seed][s_name]["n"] += n
                            seed_results[seed][s_name]["profit"] += n * roi_val / 100

                logger.info("Seed %d done", seed)

            # Summary
            logger.info("\nSeed stability results:")
            summary_rows = []
            for s_name in STRATEGIES:
                rois = []
                ns = []
                for seed in SEEDS:
                    sr = seed_results[seed][s_name]
                    total_n = sr["n"]
                    total_profit = sr["profit"]
                    roi = (total_profit / total_n * 100) if total_n > 0 else 0.0
                    rois.append(roi)
                    ns.append(total_n)
                    summary_rows.append(
                        {"strategy": s_name, "seed": seed, "roi": round(roi, 2), "n": total_n}
                    )

                mean_roi = np.mean(rois)
                std_roi = np.std(rois)
                logger.info(
                    "  %s: mean=%.2f%% std=%.2f%% [%s]",
                    s_name,
                    mean_roi,
                    std_roi,
                    ", ".join(f"{r:.1f}" for r in rois),
                )
                mlflow.log_metrics(
                    {
                        f"seed_{s_name}_mean_roi": round(mean_roi, 2),
                        f"seed_{s_name}_std_roi": round(std_roi, 2),
                        f"seed_{s_name}_min_roi": round(min(rois), 2),
                        f"seed_{s_name}_max_roi": round(max(rois), 2),
                    }
                )

            res_df = pd.DataFrame(summary_rows)
            res_path = str(SESSION_DIR / "experiments" / "seed_stability_results.csv")
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
            logger.exception("Step 4.12 failed")
            raise


if __name__ == "__main__":
    main()
