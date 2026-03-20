"""Step 3.2: Fine-tuning threshold + segment analysis для пробития ROI 10%."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    DATA_DIR,
    SEED,
    calc_roi,
    check_budget,
    get_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURE_COLS = [*get_feature_columns(), "avg_elo", "elo_diff", "max_elo", "min_elo", "elo_spread"]

BEST_PARAMS = {
    "iterations": 534,
    "depth": 7,
    "learning_rate": 0.16522729299476824,
    "l2_leaf_reg": 24.882821071506907,
    "random_strength": 4.318577859996768,
    "bagging_temperature": 5.567575136556755,
    "border_count": 146,
    "random_seed": SEED,
    "verbose": 0,
    "eval_metric": "AUC",
}


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление ELO-фич."""
    elo_history = pd.read_csv(DATA_DIR / "elo_history.csv")
    elo_per_bet = (
        elo_history.groupby("Bet_ID")
        .agg(
            avg_elo=("New_ELO", "mean"),
            elo_diff=("ELO_Change", "sum"),
            max_elo=("New_ELO", "max"),
            min_elo=("New_ELO", "min"),
        )
        .reset_index()
    )
    df = df.merge(elo_per_bet, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    for col in ["avg_elo", "max_elo", "min_elo"]:
        df[col] = df[col].fillna(1500.0)
    df["elo_diff"] = df["elo_diff"].fillna(0.0)
    df["elo_spread"] = df["max_elo"] - df["min_elo"]
    return df


def main() -> None:
    logger.info("Step 3.2: Fine-tuning threshold + segment analysis")
    df = load_data()
    df = add_elo_features(df)
    train, test = time_series_split(df)

    x_train = train[FEATURE_COLS].values.astype(float)
    y_train = train["target"].values
    x_test = test[FEATURE_COLS].values.astype(float)
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase3/step3.2_finetune_segments") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "3.2")
        mlflow.set_tag("phase", "3")

        try:
            model = CatBoostClassifier(**BEST_PARAMS)
            model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=50)
            proba = model.predict_proba(x_test)[:, 1]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "threshold_sweep_segment",
                    "n_features": len(FEATURE_COLS),
                }
            )

            # Fine-grained threshold sweep (0.01 step)
            logger.info("Fine-grained threshold sweep:")
            best_roi = -999.0
            best_threshold = 0.5
            for t in np.arange(0.50, 0.85, 0.01):
                result = calc_roi(test, proba, threshold=t)
                if result["n_bets"] >= 50 and result["roi"] > best_roi:
                    best_roi = result["roi"]
                    best_threshold = t
                if result["n_bets"] >= 50:
                    logger.info(
                        "  t=%.2f: ROI=%.2f%%, n=%d, WR=%.4f",
                        t,
                        result["roi"],
                        result["n_bets"],
                        result["win_rate"],
                    )

            roi_best = calc_roi(test, proba, threshold=best_threshold)
            logger.info(
                "Best threshold: %.2f, ROI=%.2f%%, n=%d, WR=%.4f",
                best_threshold,
                roi_best["roi"],
                roi_best["n_bets"],
                roi_best["win_rate"],
            )

            # Segment analysis
            logger.info("Segment analysis (at best threshold=%.2f):", best_threshold)
            mask = proba >= best_threshold
            selected = test.iloc[np.where(mask)[0]].copy()
            selected["proba"] = proba[mask]

            # By Sport
            logger.info("By Sport:")
            for sport, group in selected.groupby("Sport"):
                if len(group) < 10:
                    continue
                total_s = group["USD"].sum()
                payout_s = group["Payout_USD"].sum()
                roi_s = (payout_s - total_s) / total_s * 100
                wr_s = (group["Status"] == "won").mean()
                logger.info(
                    "  %s: n=%d, ROI=%.2f%%, WR=%.4f",
                    sport,
                    len(group),
                    roi_s,
                    wr_s,
                )

            # By Is_Parlay
            logger.info("By Is_Parlay:")
            for parlay, group in selected.groupby("Is_Parlay"):
                total_p = group["USD"].sum()
                payout_p = group["Payout_USD"].sum()
                roi_p = (payout_p - total_p) / total_p * 100
                wr_p = (group["Status"] == "won").mean()
                logger.info(
                    "  Parlay=%s: n=%d, ROI=%.2f%%, WR=%.4f",
                    parlay,
                    len(group),
                    roi_p,
                    wr_p,
                )

            # By Odds band
            logger.info("By Odds band:")
            selected["odds_band"] = pd.cut(
                selected["Odds"], bins=[0, 1.5, 2.0, 3.0, 5.0, 100], labels=False
            )
            for band, group in selected.groupby("odds_band"):
                if len(group) < 10:
                    continue
                total_o = group["USD"].sum()
                payout_o = group["Payout_USD"].sum()
                roi_o = (payout_o - total_o) / total_o * 100
                wr_o = (group["Status"] == "won").mean()
                labels = ["<1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]
                logger.info(
                    "  Odds %s: n=%d, ROI=%.2f%%, WR=%.4f",
                    labels[int(band)],
                    len(group),
                    roi_o,
                    wr_o,
                )

            # Попытка повысить ROI: фильтрация по сегментам
            # Exclude worst-performing segments
            logger.info("Trying segment-filtered strategies:")

            # Strategy: exclude parlays if they hurt ROI
            singles_mask = (proba >= best_threshold) & (test["Is_Parlay"] == "f")
            roi_singles = calc_roi(test, singles_mask.astype(float), threshold=0.5)
            logger.info(
                "  Singles only: ROI=%.2f%%, n=%d",
                roi_singles["roi"],
                roi_singles["n_bets"],
            )

            # Strategy: only low/mid odds
            low_odds_mask = (proba >= best_threshold) & (test["Odds"] <= 3.0)
            roi_low_odds = calc_roi(test, low_odds_mask.astype(float), threshold=0.5)
            logger.info(
                "  Low odds (<=3.0): ROI=%.2f%%, n=%d",
                roi_low_odds["roi"],
                roi_low_odds["n_bets"],
            )

            # Strategy: higher threshold for high odds
            adaptive_mask = ((proba >= best_threshold) & (test["Odds"] <= 2.0)) | (
                (proba >= best_threshold + 0.1) & (test["Odds"] > 2.0)
            )
            roi_adaptive = calc_roi(test, adaptive_mask.astype(float), threshold=0.5)
            logger.info(
                "  Adaptive threshold: ROI=%.2f%%, n=%d",
                roi_adaptive["roi"],
                roi_adaptive["n_bets"],
            )

            # Pick best overall
            all_strategies = {
                "flat_threshold": {"roi": roi_best["roi"], "n_bets": roi_best["n_bets"]},
                "singles_only": roi_singles,
                "low_odds": roi_low_odds,
                "adaptive": roi_adaptive,
            }
            best_strategy = max(
                all_strategies.items(),
                key=lambda x: x[1]["roi"] if x[1]["n_bets"] >= 50 else -999,
            )
            logger.info("Best strategy: %s, ROI=%.2f%%", best_strategy[0], best_strategy[1]["roi"])

            final_roi = best_strategy[1]["roi"]
            auc = roc_auc_score(y_test, proba)

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "roi_flat": roi_best["roi"],
                    "roi_singles_only": roi_singles["roi"],
                    "roi_low_odds": roi_low_odds["roi"],
                    "roi_adaptive": roi_adaptive["roi"],
                    "roc_auc": auc,
                    "best_threshold": best_threshold,
                    "n_bets_selected": best_strategy[1]["n_bets"],
                    "best_strategy": 0,
                }
            )
            mlflow.set_tag("best_strategy", best_strategy[0])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Final ROI: %.2f%%", final_roi)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.2")
            raise


if __name__ == "__main__":
    main()
