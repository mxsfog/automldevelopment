"""Step 4.14 -- Sport-specific models walk-forward.

Гипотеза: раздельные модели по топ-3 спортам (soccer, tennis, basketball)
могут дать лучшую калибровку для конкретного спорта.
chain_3 (step 4.4) делала сегментацию на fixed split, но не walk-forward.
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
    """Sport-specific models walk-forward."""
    with mlflow.start_run(run_name="phase4/sport_specific_wf") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            # Find top sports by volume
            sport_counts = df["Sport"].value_counts()
            top_sports = sport_counts.head(5).index.tolist()
            logger.info("Top sports: %s", dict(sport_counts.head(5)))

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
                    "method": "sport_specific_models",
                    "top_sports": str(top_sports),
                }
            )

            # Models to compare: "all" (baseline), plus each sport-specific
            model_types = ["all_data"] + [f"sport_{s}" for s in top_sports]
            strategies = ["ev005_agree_p02", "conf_ev_015"]

            cumulative: dict[str, dict[str, dict]] = {}
            for mt in model_types:
                cumulative[mt] = {s: {"n": 0, "profit": 0.0} for s in strategies}

            all_blocks = []

            for i, test_week in enumerate(retrain_schedule):
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()
                if len(test_df) == 0:
                    continue

                block_info = {"block": i, "week": test_week, "n_test": len(test_df)}

                # Baseline: all data model
                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train = train_enc[FEATURES_ENC].fillna(0)
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                p_mean_all, p_std_all = train_predict(x_train, y_train, x_test)

                odds = test_enc["Odds"].values
                ev_all = p_mean_all * odds - 1
                conf_all = 1.0 / (1.0 + p_std_all * 10)
                conf_ev_all = ev_all * conf_all

                for s_name, mask in [
                    ("ev005_agree_p02", (ev_all >= 0.05) & (p_std_all <= 0.02)),
                    ("conf_ev_015", conf_ev_all >= 0.15),
                ]:
                    n = mask.sum()
                    roi = 0.0
                    if n > 0:
                        sel = test_enc[mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                        cumulative["all_data"][s_name]["n"] += n
                        cumulative["all_data"][s_name]["profit"] += n * roi / 100
                    block_info[f"all_data_{s_name}_roi"] = round(roi, 2)
                    block_info[f"all_data_{s_name}_n"] = int(n)

                # Sport-specific models
                for sport in top_sports:
                    sport_train = train_df[train_df["Sport"] == sport].copy()
                    sport_test_mask = test_enc["Sport"] == sport
                    sport_test = test_enc[sport_test_mask].copy()

                    mt = f"sport_{sport}"

                    if len(sport_train) < 200 or len(sport_test) == 0:
                        for s_name in strategies:
                            block_info[f"{mt}_{s_name}_roi"] = 0.0
                            block_info[f"{mt}_{s_name}_n"] = 0
                        continue

                    sport_train_enc, _ = add_sport_market_features(sport_train, sport_train)
                    sport_test_enc, _ = add_sport_market_features(sport_test, sport_train_enc)

                    x_tr = sport_train_enc[FEATURES_ENC].fillna(0)
                    y_tr = sport_train_enc["target"]
                    x_te = sport_test_enc[FEATURES_ENC].fillna(0)

                    p_mean_s, p_std_s = train_predict(x_tr, y_tr, x_te)

                    odds_s = sport_test_enc["Odds"].values
                    ev_s = p_mean_s * odds_s - 1
                    conf_s = 1.0 / (1.0 + p_std_s * 10)
                    conf_ev_s = ev_s * conf_s

                    for s_name, mask in [
                        ("ev005_agree_p02", (ev_s >= 0.05) & (p_std_s <= 0.02)),
                        ("conf_ev_015", conf_ev_s >= 0.15),
                    ]:
                        n = mask.sum()
                        roi = 0.0
                        if n > 0:
                            sel = sport_test_enc[mask]
                            payout = (sel["target"] * sel["Odds"]).sum()
                            roi = (payout - n) / n * 100
                            cumulative[mt][s_name]["n"] += n
                            cumulative[mt][s_name]["profit"] += n * roi / 100
                        block_info[f"{mt}_{s_name}_roi"] = round(roi, 2)
                        block_info[f"{mt}_{s_name}_n"] = int(n)

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nSport-specific model results:")

            for mt in model_types:
                logger.info("  Model: %s", mt)
                for s_name in strategies:
                    total_n = cumulative[mt][s_name]["n"]
                    total_profit = cumulative[mt][s_name]["profit"]
                    overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                    col = f"{mt}_{s_name}_roi"
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
                    # Sanitize metric name
                    safe_mt = mt.replace(" ", "_").replace("-", "_")
                    mlflow.log_metrics(
                        {
                            f"wf_{safe_mt}_{s_name}_roi": round(overall_roi, 2),
                            f"wf_{safe_mt}_{s_name}_n": total_n,
                        }
                    )

            res_path = str(SESSION_DIR / "experiments" / "sport_specific_results.csv")
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
            logger.exception("Step 4.14 failed")
            raise


if __name__ == "__main__":
    main()
