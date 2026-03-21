"""Step 4.5 -- Walk-forward segment analysis.

Анализ: где именно conf_ev_0.15 генерирует edge в walk-forward?
Разбивка по: sport, odds range, is_parlay, conf_ev magnitude.

Цель: найти стабильные profitable сегменты для более узкого фильтра.
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
    """Walk-forward с сегментным анализом conf_ev отобранных ставок."""
    with mlflow.start_run(run_name="phase4/wf_segment_analysis") as run:
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
                    "method": "segment_analysis_walk_forward",
                    "n_test_blocks": len(retrain_schedule),
                }
            )

            # Collect all conf_ev selected bets across blocks
            all_selected = []

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

                # Collect all test bets with predictions
                test_enc = test_enc.copy()
                test_enc["p_mean"] = p_mean
                test_enc["p_std"] = p_std
                test_enc["ev"] = ev
                test_enc["conf"] = conf
                test_enc["conf_ev"] = conf_ev
                test_enc["block"] = i
                test_enc["week"] = test_week
                test_enc["selected"] = conf_ev >= 0.15

                all_selected.append(test_enc)

            all_df = pd.concat(all_selected, ignore_index=True)
            selected_df = all_df[all_df["selected"]].copy()
            logger.info("Total bets across blocks: %d", len(all_df))
            logger.info("Selected by conf_ev>=0.15: %d", len(selected_df))

            # Overall ROI
            n_total = len(selected_df)
            payout_total = (selected_df["target"] * selected_df["Odds"]).sum()
            roi_total = (payout_total - n_total) / n_total * 100
            logger.info("Overall walk-forward ROI: %.2f%% (n=%d)", roi_total, n_total)

            # Segment analyses
            segments = {}

            # 1. By Sport
            logger.info("\n--- BY SPORT ---")
            sport_stats = []
            for sport, grp in selected_df.groupby("Sport"):
                n = len(grp)
                payout = (grp["target"] * grp["Odds"]).sum()
                roi = (payout - n) / n * 100
                wr = grp["target"].mean()
                avg_odds = grp["Odds"].mean()
                sport_stats.append(
                    {
                        "sport": sport,
                        "n": n,
                        "roi": round(roi, 2),
                        "win_rate": round(wr, 4),
                        "avg_odds": round(avg_odds, 2),
                    }
                )
                logger.info(
                    "  %s: n=%d, ROI=%.2f%%, WR=%.2f, odds=%.2f", sport, n, roi, wr, avg_odds
                )
            segments["sport"] = sport_stats

            # 2. By Odds range
            logger.info("\n--- BY ODDS RANGE ---")
            odds_bins = [1, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 10000.0]
            odds_labels = ["1-1.5", "1.5-2", "2-3", "3-5", "5-10", "10-50", "50+"]
            selected_df["odds_bin"] = pd.cut(
                selected_df["Odds"], bins=odds_bins, labels=odds_labels
            )
            odds_stats = []
            for obin, grp in selected_df.groupby("odds_bin", observed=True):
                n = len(grp)
                if n == 0:
                    continue
                payout = (grp["target"] * grp["Odds"]).sum()
                roi = (payout - n) / n * 100
                wr = grp["target"].mean()
                odds_stats.append(
                    {
                        "odds_range": str(obin),
                        "n": n,
                        "roi": round(roi, 2),
                        "win_rate": round(wr, 4),
                    }
                )
                logger.info("  %s: n=%d, ROI=%.2f%%, WR=%.2f", obin, n, roi, wr)
            segments["odds_range"] = odds_stats

            # 3. By Is_Parlay
            logger.info("\n--- BY PARLAY ---")
            parlay_stats = []
            for is_p, grp in selected_df.groupby("Is_Parlay"):
                n = len(grp)
                payout = (grp["target"] * grp["Odds"]).sum()
                roi = (payout - n) / n * 100
                label = "parlay" if is_p else "single"
                parlay_stats.append({"type": label, "n": n, "roi": round(roi, 2)})
                logger.info("  %s: n=%d, ROI=%.2f%%", label, n, roi)
            segments["parlay"] = parlay_stats

            # 4. By conf_ev magnitude
            logger.info("\n--- BY CONF_EV MAGNITUDE ---")
            cev_bins = [0.15, 0.20, 0.30, 0.50, 1.0, 100.0]
            cev_labels = ["0.15-0.20", "0.20-0.30", "0.30-0.50", "0.50-1.0", "1.0+"]
            selected_df["cev_bin"] = pd.cut(
                selected_df["conf_ev"], bins=cev_bins, labels=cev_labels
            )
            cev_stats = []
            for cbin, grp in selected_df.groupby("cev_bin", observed=True):
                n = len(grp)
                if n == 0:
                    continue
                payout = (grp["target"] * grp["Odds"]).sum()
                roi = (payout - n) / n * 100
                cev_stats.append({"cev_range": str(cbin), "n": n, "roi": round(roi, 2)})
                logger.info("  %s: n=%d, ROI=%.2f%%", cbin, n, roi)
            segments["conf_ev_magnitude"] = cev_stats

            # 5. Profit concentration (Gini-like)
            logger.info("\n--- PROFIT CONCENTRATION ---")
            selected_df["pnl"] = np.where(selected_df["target"] == 1, selected_df["Odds"] - 1, -1)
            sorted_pnl = selected_df["pnl"].sort_values()
            cumsum = sorted_pnl.cumsum()
            total_pnl = cumsum.iloc[-1]
            # Top-1, top-5, top-10 contribution
            top_pnl = selected_df.nlargest(10, "pnl")
            top1_pct = top_pnl.iloc[0]["pnl"] / total_pnl * 100 if total_pnl > 0 else 0
            top5_pct = top_pnl.head(5)["pnl"].sum() / total_pnl * 100 if total_pnl > 0 else 0
            top10_pct = top_pnl["pnl"].sum() / total_pnl * 100 if total_pnl > 0 else 0
            logger.info(
                "Top-1: %.1f%%, Top-5: %.1f%%, Top-10: %.1f%% of total P&L",
                top1_pct,
                top5_pct,
                top10_pct,
            )
            logger.info(
                "Top-10 bets:\n%s",
                top_pnl[["Odds", "target", "pnl", "Sport", "conf_ev"]].to_string(),
            )

            # N bets with negative P&L
            n_negative = (selected_df["pnl"] < 0).sum()
            n_positive = (selected_df["pnl"] > 0).sum()
            logger.info("Positive P&L bets: %d, Negative: %d", n_positive, n_negative)

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "wf_overall_roi": round(roi_total, 2),
                    "wf_n_selected": n_total,
                    "wf_top1_pct": round(top1_pct, 1),
                    "wf_top5_pct": round(top5_pct, 1),
                    "wf_top10_pct": round(top10_pct, 1),
                    "wf_n_positive_bets": n_positive,
                    "wf_n_negative_bets": n_negative,
                }
            )

            for stat_list in [sport_stats, odds_stats, parlay_stats, cev_stats]:
                for item in stat_list:
                    key = next(iter(item.values()))
                    safe_key = str(key).replace("+", "plus").replace(" ", "_")
                    mlflow.log_metric(f"seg_{safe_key}_roi", item["roi"])
                    mlflow.log_metric(f"seg_{safe_key}_n", item["n"])

            # Save full data
            seg_path = str(SESSION_DIR / "experiments" / "wf_segments.json")
            with open(seg_path, "w") as f:
                json.dump(segments, f, indent=2)
            mlflow.log_artifact(seg_path)

            selected_path = str(SESSION_DIR / "experiments" / "wf_selected_bets.csv")
            selected_df.to_csv(selected_path, index=False)
            mlflow.log_artifact(selected_path)
            mlflow.log_artifact(__file__)

            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
