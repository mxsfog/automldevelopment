"""Step 4.33 -- Risk analysis: drawdown, concentration, Sharpe.

For deployment decisions: how risky is the 4m_blend50 strategy?
- Max drawdown per block
- Profit concentration (top-N bets as % of total PnL)
- Block-by-block Sharpe ratio
- What happens if biggest winner didn't happen?
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
from xgboost import XGBClassifier

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


def main() -> None:
    """Risk analysis for 4m_blend50."""
    with mlflow.start_run(run_name="phase4/risk_analysis") as run:
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
                    "method": "risk_analysis",
                    "strategy": "4m_blend50",
                }
            )

            all_selected: list[dict] = []
            block_results: list[dict] = []

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

                xgb = XGBClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=42, verbosity=0, min_child_weight=50,
                    use_label_encoder=False, eval_metric="logloss",
                )
                xgb.fit(x_train, y_train)

                preds = [
                    cb.predict_proba(x_test)[:, 1],
                    lgbm.predict_proba(x_test)[:, 1],
                    lr.predict_proba(x_test_s)[:, 1],
                    xgb.predict_proba(x_test)[:, 1],
                ]
                p_model = np.mean(preds, axis=0)
                p_std = np.std(preds, axis=0)

                odds = test_enc["Odds"].values
                p_implied = 1.0 / odds
                p_final = 0.5 * p_model + 0.5 * p_implied

                ev = p_final * odds - 1
                mask = (ev >= 0.05) & (p_std <= 0.02)

                n = mask.sum()
                if n > 0:
                    sel = test_enc[mask].copy()
                    sel_pnl = sel["target"] * sel["Odds"] - 1
                    total_profit = sel_pnl.sum()
                    roi = total_profit / n * 100

                    # Per-bet PnL
                    for _, row in sel.iterrows():
                        pnl = row["target"] * row["Odds"] - 1
                        all_selected.append({
                            "block": i, "week": test_week,
                            "odds": row["Odds"], "target": row["target"],
                            "pnl": pnl, "sport": row.get("Sport", ""),
                        })

                    # Cumulative within block
                    cumsum = sel_pnl.values.cumsum()
                    max_dd = 0.0
                    peak = 0.0
                    for val in cumsum:
                        if val > peak:
                            peak = val
                        dd = peak - val
                        if dd > max_dd:
                            max_dd = dd

                    block_results.append({
                        "block": i, "week": test_week, "n": n,
                        "profit": round(total_profit, 2),
                        "roi": round(roi, 2),
                        "max_dd": round(max_dd, 2),
                        "max_odds": round(sel["Odds"].max(), 2),
                        "mean_odds": round(sel["Odds"].mean(), 2),
                        "win_rate": round(sel["target"].mean(), 4),
                    })

            # Aggregate analysis
            all_df = pd.DataFrame(all_selected)
            block_df = pd.DataFrame(block_results)

            total_profit = all_df["pnl"].sum()
            total_n = len(all_df)
            overall_roi = total_profit / total_n * 100

            logger.info("=== Risk Analysis: 4m_blend50 ===")
            logger.info("Total bets: %d, Total profit: %.2f, ROI: %.2f%%",
                        total_n, total_profit, overall_roi)

            # 1. Profit concentration
            all_df_sorted = all_df.sort_values("pnl", ascending=False)
            top1_pnl = all_df_sorted.iloc[0]["pnl"]
            top5_pnl = all_df_sorted.head(5)["pnl"].sum()
            top10_pnl = all_df_sorted.head(10)["pnl"].sum()
            top1_pct = top1_pnl / total_profit * 100 if total_profit > 0 else 0
            top5_pct = top5_pnl / total_profit * 100 if total_profit > 0 else 0
            top10_pct = top10_pnl / total_profit * 100 if total_profit > 0 else 0

            logger.info("\n--- Profit Concentration ---")
            logger.info(
                "Top 1 bet: PnL=%.2f (%.1f%% of total), odds=%.1f",
                top1_pnl, top1_pct, all_df_sorted.iloc[0]["odds"],
            )
            logger.info("Top 5 bets: PnL=%.2f (%.1f%% of total)", top5_pnl, top5_pct)
            logger.info("Top 10 bets: PnL=%.2f (%.1f%% of total)", top10_pnl, top10_pct)

            # 2. Without top-1 bet
            without_top1 = all_df_sorted.iloc[1:]
            roi_no_top1 = without_top1["pnl"].sum() / len(without_top1) * 100
            logger.info(
                "\nWithout top-1 bet: ROI=%.2f%% (n=%d)", roi_no_top1, len(without_top1)
            )

            # 3. Block-by-block
            logger.info("\n--- Block Results ---")
            for _, row in block_df.iterrows():
                logger.info(
                    "  Block %d: ROI=%.2f%% n=%d max_dd=%.2f "
                    "max_odds=%.1f win_rate=%.3f",
                    row["block"], row["roi"], row["n"],
                    row["max_dd"], row["max_odds"], row["win_rate"],
                )

            # 4. Sharpe-like ratio (mean/std of block ROIs)
            block_rois = block_df["roi"].values
            sharpe = np.mean(block_rois) / np.std(block_rois) if np.std(block_rois) > 0 else 0
            logger.info(
                "\nBlock Sharpe: %.2f (mean=%.2f, std=%.2f)",
                sharpe, np.mean(block_rois), np.std(block_rois),
            )

            # 5. Odds distribution of selected bets
            logger.info("\n--- Odds Distribution (selected bets) ---")
            brackets = [(1, 2), (2, 5), (5, 10), (10, 50), (50, 1000)]
            for lo, hi in brackets:
                bracket = all_df[(all_df["odds"] >= lo) & (all_df["odds"] < hi)]
                if len(bracket) > 0:
                    b_pnl = bracket["pnl"].sum()
                    b_pct = b_pnl / total_profit * 100 if total_profit > 0 else 0
                    logger.info(
                        "  [%d-%d]: n=%d, PnL=%.2f (%.1f%%), wr=%.3f",
                        lo, hi, len(bracket), b_pnl, b_pct,
                        bracket["target"].mean(),
                    )

            mlflow.log_metrics({
                "risk_total_n": total_n,
                "risk_overall_roi": round(overall_roi, 2),
                "risk_top1_pct": round(top1_pct, 2),
                "risk_top5_pct": round(top5_pct, 2),
                "risk_top10_pct": round(top10_pct, 2),
                "risk_roi_no_top1": round(roi_no_top1, 2),
                "risk_sharpe": round(sharpe, 2),
            })

            res_path = str(SESSION_DIR / "experiments" / "risk_analysis.csv")
            all_df.to_csv(res_path, index=False)
            block_path = str(SESSION_DIR / "experiments" / "risk_blocks.csv")
            block_df.to_csv(block_path, index=False)
            mlflow.log_artifact(res_path)
            mlflow.log_artifact(block_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.33 failed")
            raise


if __name__ == "__main__":
    main()
