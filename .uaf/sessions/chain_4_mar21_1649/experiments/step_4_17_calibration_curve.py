"""Step 4.17 -- Calibration curve analysis + break-even analysis.

Step 4.15 показал: AUC=0.73 (значимый) но ROI@odds<=10 = -0.14% (хуже random).
Почему? Возможно, модель overconfident — завышает P(win), создавая ложный EV.

Анализ:
1. Calibration curve: predicted P(win) vs actual win rate по децилям
2. Break-even: для каждого odds bracket, нужный win rate vs фактический
3. EV decomposition: где EV > 0 реален, а где артефакт miscalibration
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
    """Calibration and break-even analysis walk-forward."""
    with mlflow.start_run(run_name="phase4/calibration_curve") as run:
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
                    "method": "calibration_breakeven_analysis",
                }
            )

            # Collect all test predictions
            all_preds: list[dict] = []

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
                    iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
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

                for j in range(len(test_enc)):
                    all_preds.append(
                        {
                            "p_mean": p_mean[j],
                            "p_std": p_std[j],
                            "target": test_enc.iloc[j]["target"],
                            "odds": test_enc.iloc[j]["Odds"],
                            "p_implied": 1.0 / test_enc.iloc[j]["Odds"],
                            "week": test_week,
                        }
                    )

            pred_df = pd.DataFrame(all_preds)
            pred_df["ev"] = pred_df["p_mean"] * pred_df["odds"] - 1

            # 1. Calibration curve by probability deciles
            logger.info("\n=== Calibration Curve (probability deciles) ===")
            pred_df["p_decile"] = pd.qcut(pred_df["p_mean"], 10, labels=False, duplicates="drop")
            cal_rows = []
            for dec in sorted(pred_df["p_decile"].unique()):
                sub = pred_df[pred_df["p_decile"] == dec]
                predicted = sub["p_mean"].mean()
                actual = sub["target"].mean()
                n = len(sub)
                gap = predicted - actual
                cal_rows.append(
                    {
                        "decile": int(dec),
                        "predicted": round(predicted, 4),
                        "actual": round(actual, 4),
                        "gap": round(gap, 4),
                        "n": n,
                    }
                )
                logger.info(
                    "  D%d: predicted=%.3f actual=%.3f gap=%.3f n=%d",
                    dec,
                    predicted,
                    actual,
                    gap,
                    n,
                )

            # 2. Calibration for EV-selected bets only
            logger.info("\n=== Calibration for EV-selected bets (ev>=0.05, p_std<=0.02) ===")
            ev_sel = pred_df[(pred_df["ev"] >= 0.05) & (pred_df["p_std"] <= 0.02)]
            if len(ev_sel) > 0:
                ev_pred = ev_sel["p_mean"].mean()
                ev_actual = ev_sel["target"].mean()
                ev_gap = ev_pred - ev_actual
                breakeven = 1.0 / ev_sel["odds"].mean()
                logger.info(
                    "  Selected: predicted=%.4f actual=%.4f gap=%.4f breakeven=%.4f n=%d",
                    ev_pred,
                    ev_actual,
                    ev_gap,
                    breakeven,
                    len(ev_sel),
                )
                mlflow.log_metrics(
                    {
                        "evsel_predicted_winrate": round(ev_pred, 4),
                        "evsel_actual_winrate": round(ev_actual, 4),
                        "evsel_calibration_gap": round(ev_gap, 4),
                        "evsel_breakeven_winrate": round(breakeven, 4),
                    }
                )

            # 3. Calibration for EV-selected bets at low odds
            logger.info("\n=== Calibration for EV-selected at odds<=10 ===")
            ev_sel_low = ev_sel[ev_sel["odds"] <= 10]
            if len(ev_sel_low) > 0:
                ev_pred_l = ev_sel_low["p_mean"].mean()
                ev_actual_l = ev_sel_low["target"].mean()
                ev_gap_l = ev_pred_l - ev_actual_l
                breakeven_l = 1.0 / ev_sel_low["odds"].mean()
                logger.info(
                    "  Selected (odds<=10): predicted=%.4f actual=%.4f gap=%.4f "
                    "breakeven=%.4f n=%d",
                    ev_pred_l,
                    ev_actual_l,
                    ev_gap_l,
                    breakeven_l,
                    len(ev_sel_low),
                )
                mlflow.log_metrics(
                    {
                        "evsel_low_predicted": round(ev_pred_l, 4),
                        "evsel_low_actual": round(ev_actual_l, 4),
                        "evsel_low_gap": round(ev_gap_l, 4),
                        "evsel_low_breakeven": round(breakeven_l, 4),
                    }
                )

            # 4. Break-even analysis by odds bracket
            logger.info("\n=== Break-even Analysis ===")
            odds_brackets = [(1, 2), (2, 3), (3, 5), (5, 10), (10, 50), (50, 1000)]
            be_rows = []
            for lo, hi in odds_brackets:
                sub = pred_df[(pred_df["odds"] >= lo) & (pred_df["odds"] < hi)]
                if len(sub) < 10:
                    continue
                avg_odds = sub["odds"].mean()
                breakeven_wr = 1.0 / avg_odds
                actual_wr = sub["target"].mean()
                predicted_wr = sub["p_mean"].mean()
                roi = (actual_wr * avg_odds - 1) * 100
                ev_mean = sub["ev"].mean()
                be_rows.append(
                    {
                        "bracket": f"{lo}-{hi}",
                        "n": len(sub),
                        "avg_odds": round(avg_odds, 2),
                        "breakeven_wr": round(breakeven_wr, 4),
                        "predicted_wr": round(predicted_wr, 4),
                        "actual_wr": round(actual_wr, 4),
                        "pred_gap": round(predicted_wr - actual_wr, 4),
                        "roi": round(roi, 2),
                        "mean_ev": round(ev_mean, 4),
                    }
                )
                logger.info(
                    "  [%d-%d]: n=%d avg_odds=%.2f breakeven=%.3f "
                    "predicted=%.3f actual=%.3f gap=%.3f ROI=%.1f%%",
                    lo,
                    hi,
                    len(sub),
                    avg_odds,
                    breakeven_wr,
                    predicted_wr,
                    actual_wr,
                    predicted_wr - actual_wr,
                    roi,
                )

            # Save results
            cal_df = pd.DataFrame(cal_rows)
            cal_path = str(SESSION_DIR / "experiments" / "calibration_curve.csv")
            cal_df.to_csv(cal_path, index=False)

            be_df = pd.DataFrame(be_rows)
            be_path = str(SESSION_DIR / "experiments" / "breakeven_analysis.csv")
            be_df.to_csv(be_path, index=False)

            mlflow.log_artifact(cal_path)
            mlflow.log_artifact(be_path)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.17 failed")
            raise


if __name__ == "__main__":
    main()
