"""Step 4.23 -- Drop high-drift features to reduce covariate shift.

Step 4.21: adv AUC=0.878 (shift). Step 4.22: time-weighting не помогает.
Гипотеза: если убрать фичи, которые adversarial classifier использует
для отличения train от test, модель станет более устойчивой к shift.

Подход: train adversarial model, rank features by importance, drop top-K drifters.
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


def get_adversarial_importance(
    x_train: pd.DataFrame, x_test: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """Train adversarial classifier and return feature importances."""
    adv_x = pd.concat([x_train[features], x_test[features]], ignore_index=True)
    adv_y = np.array([0] * len(x_train) + [1] * len(x_test))

    adv_model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=4, random_seed=42, verbose=0
    )
    adv_model.fit(adv_x, adv_y)

    importances = adv_model.get_feature_importance()
    return dict(zip(features, importances, strict=True))


def main() -> None:
    """Drop drift features walk-forward."""
    with mlflow.start_run(run_name="phase4/drop_drift_features") as run:
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

            # Strategies: drop top-K adversarial features
            drop_counts = [0, 3, 5, 7]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "method": "drop_drift_features",
                    "drop_counts": str(drop_counts),
                }
            )

            cumulative: dict[int, dict] = {
                k: {"n": 0, "profit": 0.0} for k in drop_counts
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

                # Get adversarial importance for this block
                adv_imp = get_adversarial_importance(
                    train_enc, test_enc, FEATURES_ENC
                )
                sorted_feats = sorted(adv_imp.items(), key=lambda x: -x[1])

                if i == 0:
                    logger.info("Adversarial feature importance (block 0):")
                    for feat, imp in sorted_feats:
                        logger.info("  %s: %.2f", feat, imp)

                block_info: dict = {"block": i, "week": test_week, "n_test": len(test_df)}

                for n_drop in drop_counts:
                    if n_drop == 0:
                        feats = FEATURES_ENC
                    else:
                        drop_set = {f for f, _ in sorted_feats[:n_drop]}
                        feats = [f for f in FEATURES_ENC if f not in drop_set]

                    x_train = train_enc[feats].fillna(0)
                    y_train = train_enc["target"]
                    x_test = test_enc[feats].fillna(0)

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
                    mask = (ev >= 0.05) & (p_std <= 0.02)

                    n = mask.sum()
                    roi = 0.0
                    if n > 0:
                        sel = test_enc[mask]
                        payout = (sel["target"] * sel["Odds"]).sum()
                        roi = (payout - n) / n * 100
                        cumulative[n_drop]["n"] += n
                        cumulative[n_drop]["profit"] += n * roi / 100

                    # Adversarial AUC for this block with reduced features
                    if n_drop > 0:
                        adv_x = pd.concat(
                            [train_enc[feats].fillna(0), test_enc[feats].fillna(0)],
                            ignore_index=True,
                        )
                        adv_y_arr = np.array([0] * len(train_enc) + [1] * len(test_enc))
                        rng = np.random.RandomState(42)
                        idx = rng.permutation(len(adv_x))
                        split = int(0.7 * len(idx))
                        adv_m = CatBoostClassifier(
                            iterations=100, learning_rate=0.1, depth=4,
                            random_seed=42, verbose=0,
                        )
                        adv_m.fit(adv_x.iloc[idx[:split]], adv_y_arr[idx[:split]])
                        adv_p = adv_m.predict_proba(adv_x.iloc[idx[split:]])[:, 1]
                        adv_auc = roc_auc_score(adv_y_arr[idx[split:]], adv_p)
                        block_info[f"drop{n_drop}_adv_auc"] = round(adv_auc, 4)

                    block_info[f"drop{n_drop}_roi"] = round(roi, 2)
                    block_info[f"drop{n_drop}_n"] = int(n)

                all_blocks.append(block_info)

            results_df = pd.DataFrame(all_blocks)
            logger.info("\nDrop drift features results (ev005_agree_p02):")
            for n_drop in drop_counts:
                total_n = cumulative[n_drop]["n"]
                total_profit = cumulative[n_drop]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col = f"drop{n_drop}_roi"
                n_pos = 0
                if col in results_df.columns:
                    n_pos = (results_df[col].values > 0).sum()

                adv_col = f"drop{n_drop}_adv_auc"
                mean_adv = ""
                if adv_col in results_df.columns:
                    mean_adv = f" adv_auc={results_df[adv_col].mean():.3f}"

                logger.info(
                    "  drop_%d: overall=%.2f%% (n=%d) pos=%d/%d%s",
                    n_drop,
                    overall_roi,
                    total_n,
                    n_pos,
                    len(all_blocks),
                    mean_adv,
                )
                mlflow.log_metrics(
                    {
                        f"wf_drop{n_drop}_roi": round(overall_roi, 2),
                        f"wf_drop{n_drop}_n": total_n,
                        f"wf_drop{n_drop}_pos": n_pos,
                    }
                )

            res_path = str(SESSION_DIR / "experiments" / "drop_drift_results.csv")
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
            logger.exception("Step 4.23 failed")
            raise


if __name__ == "__main__":
    main()
