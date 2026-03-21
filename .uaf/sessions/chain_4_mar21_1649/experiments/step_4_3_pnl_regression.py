"""Step 4.3 -- P&L Regression + Walk-forward.

Вместо классификации (won/lost) предсказываем P&L на ставку:
  target_pnl = odds - 1  (если won)
  target_pnl = -1        (если lost)

Гипотеза: регрессионная модель может напрямую оценить expected value,
минуя проблему несоответствия между accuracy и ROI.

Дополнительно: dual-model (classification ensemble + regression) --
фильтруем conf_ev > 0, ранжируем по predicted_pnl.
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
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
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
    """P&L regression + classification ensemble walk-forward."""
    with mlflow.start_run(run_name="phase4/pnl_regression") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            # P&L target: (odds - 1) if won, -1 if lost
            df["pnl"] = np.where(df["target"] == 1, df["Odds"] - 1, -1.0)

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
                    "method": "pnl_regression_plus_classification",
                    "n_test_blocks": len(retrain_schedule),
                    "regression_target": "pnl = (odds-1)*won - 1*lost",
                }
            )

            # Strategies to compare:
            # 1. Classification ensemble conf_ev_0.15 (baseline)
            # 2. Regression: bet where predicted_pnl > threshold
            # 3. Dual: classification filter + regression ranking
            strategy_names = [
                "clf_conf_ev_0.15",
                "reg_pnl_0.05",
                "reg_pnl_0.10",
                "reg_pnl_0.20",
                "dual_cev_reg_top50",
                "dual_cev_reg_top75",
            ]

            cumulative = {s: {"n": 0, "profit": 0.0} for s in strategy_names}
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
                y_train_cls = train_enc["target"]
                y_train_reg = train_enc["pnl"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Classification ensemble
                cb_cls = CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
                )
                cb_cls.fit(x_train, y_train_cls)

                lgbm_cls = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm_cls.fit(x_train, y_train_cls)

                scaler = StandardScaler()
                x_train_s = scaler.fit_transform(x_train)
                x_test_s = scaler.transform(x_test)
                lr_cls = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr_cls.fit(x_train_s, y_train_cls)

                p_cb = cb_cls.predict_proba(x_test)[:, 1]
                p_lgbm = lgbm_cls.predict_proba(x_test)[:, 1]
                p_lr = lr_cls.predict_proba(x_test_s)[:, 1]
                p_mean = (p_cb + p_lgbm + p_lr) / 3
                p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)

                # Regression ensemble (P&L prediction)
                # Cap extreme pnl values to reduce noise
                y_reg_capped = y_train_reg.clip(-1, 20)

                cb_reg = CatBoostRegressor(
                    iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
                )
                cb_reg.fit(x_train, y_reg_capped)

                lgbm_reg = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                    min_child_samples=50,
                )
                lgbm_reg.fit(x_train, y_reg_capped)

                lr_reg = LinearRegression()
                lr_reg.fit(x_train_s, y_reg_capped)

                pnl_cb = cb_reg.predict(x_test)
                pnl_lgbm = lgbm_reg.predict(x_test)
                pnl_lr = lr_reg.predict(x_test_s)
                pnl_mean = (pnl_cb + pnl_lgbm + pnl_lr) / 3

                # Metrics
                odds_test = test_enc["Odds"].values
                target_test = test_enc["target"].values
                actual_pnl = test_enc["pnl"].values

                try:
                    auc = roc_auc_score(target_test, p_mean)
                except ValueError:
                    auc = 0.5
                mae = mean_absolute_error(actual_pnl, pnl_mean)

                # Strategy 1: clf conf_ev_0.15
                ev = p_mean * odds_test - 1
                conf = 1.0 / (1.0 + p_std * 10)
                conf_ev = ev * conf
                mask_cev = conf_ev >= 0.15

                # Strategy 2-4: regression threshold
                mask_reg_005 = pnl_mean >= 0.05
                mask_reg_010 = pnl_mean >= 0.10
                mask_reg_020 = pnl_mean >= 0.20

                # Strategy 5-6: dual (conf_ev filter + regression ranking)
                mask_cev_any = conf_ev >= 0.05  # looser filter
                dual_indices = np.where(mask_cev_any)[0]
                if len(dual_indices) > 0:
                    pnl_of_selected = pnl_mean[dual_indices]
                    # Top 50%
                    cutoff_50 = np.percentile(pnl_of_selected, 50)
                    cutoff_75 = np.percentile(pnl_of_selected, 25)  # top 75%
                    mask_dual_50 = np.zeros(len(test_enc), dtype=bool)
                    mask_dual_75 = np.zeros(len(test_enc), dtype=bool)
                    mask_dual_50[dual_indices[pnl_of_selected >= cutoff_50]] = True
                    mask_dual_75[dual_indices[pnl_of_selected >= cutoff_75]] = True
                else:
                    mask_dual_50 = np.zeros(len(test_enc), dtype=bool)
                    mask_dual_75 = np.zeros(len(test_enc), dtype=bool)

                masks = {
                    "clf_conf_ev_0.15": mask_cev,
                    "reg_pnl_0.05": mask_reg_005,
                    "reg_pnl_0.10": mask_reg_010,
                    "reg_pnl_0.20": mask_reg_020,
                    "dual_cev_reg_top50": mask_dual_50,
                    "dual_cev_reg_top75": mask_dual_75,
                }

                block_info = {
                    "block": i,
                    "week": test_week,
                    "n_test": len(test_df),
                    "auc": round(auc, 4),
                    "mae_pnl": round(mae, 4),
                }

                for s_name, mask in masks.items():
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
                logger.info(
                    "Block %d (wk %d): AUC=%.3f MAE=%.2f | "
                    "clf_cev=%.1f%%(n=%d) reg_010=%.1f%%(n=%d) dual50=%.1f%%(n=%d)",
                    i,
                    test_week,
                    auc,
                    mae,
                    block_info["clf_conf_ev_0.15_roi"],
                    block_info["clf_conf_ev_0.15_n"],
                    block_info["reg_pnl_0.10_roi"],
                    block_info["reg_pnl_0.10_n"],
                    block_info["dual_cev_reg_top50_roi"],
                    block_info["dual_cev_reg_top50_n"],
                )

            # Summary
            results_df = pd.DataFrame(all_blocks)
            for s_name in strategy_names:
                total_n = cumulative[s_name]["n"]
                total_profit = cumulative[s_name]["profit"]
                overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                col_roi = f"{s_name}_roi"
                if col_roi in results_df.columns:
                    block_rois = results_df[col_roi].values
                    mean_roi = block_rois.mean()
                    std_roi = block_rois.std()
                    n_pos = (block_rois > 0).sum()
                else:
                    mean_roi = std_roi = 0.0
                    n_pos = 0

                logger.info(
                    "%s: overall=%.2f%% (n=%d), mean=%.2f%% std=%.2f%%, pos=%d/%d",
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
                        f"wf_{s_name}_std": round(std_roi, 2),
                        f"wf_{s_name}_pos": n_pos,
                    }
                )

            # Save
            res_path = str(SESSION_DIR / "experiments" / "pnl_regression_results.csv")
            results_df.to_csv(res_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.3 failed")
            raise


if __name__ == "__main__":
    main()
