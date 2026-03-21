"""Step 1.3 — Linear baseline (LogisticRegression).

Логистическая регрессия на базовых числовых фичах + порог по вероятности.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

FEATURES = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Is_Parlay_num",
    "Outcomes_Count",
]


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(Sport=("Sport", "first"), Market=("Market", "first"))
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()

    df["Is_Parlay_num"] = (df["Is_Parlay"] == "t").astype(int)
    df["ML_Team_Stats_Found_num"] = (df["ML_Team_Stats_Found"] == "t").astype(int)

    logger.info("После фильтрации: %d строк", len(df))
    return df


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split по индексу."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """ROI на выбранных ставках."""
    sel = df[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def find_best_threshold(proba: np.ndarray, df: pd.DataFrame) -> float:
    """Подбор порога вероятности на val для максимизации ROI."""
    best_roi = -999.0
    best_thr = 0.5
    for thr in np.arange(0.35, 0.85, 0.02):
        mask = proba >= thr
        result = calc_roi(df, mask)
        if result["n_bets"] >= 20 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr


def main():
    df = load_data()
    train_full, test = time_series_split(df)

    # Val = последние 20% train
    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    x_train = train[FEATURES].fillna(0).values
    x_val = val[FEATURES].fillna(0).values
    x_test = test[FEATURES].fillna(0).values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    with mlflow.start_run(run_name="phase1/step_1_3_logreg") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.3")
            mlflow.set_tag("phase", "1")

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "LogisticRegression",
                    "features": str(FEATURES),
                    "n_features": len(FEATURES),
                    "C": 1.0,
                }
            )

            model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
            model.fit(x_train, y_train)

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            auc_val = roc_auc_score(y_val, proba_val)
            auc_test = roc_auc_score(y_test, proba_test)
            logger.info("AUC: val=%.4f, test=%.4f", auc_val, auc_test)

            # Порог подбираем на val
            threshold = find_best_threshold(proba_val, val)
            logger.info("Best threshold from val: %.2f", threshold)

            # ROI на val с лучшим порогом
            val_mask = proba_val >= threshold
            roi_val = calc_roi(val, val_mask)
            logger.info("Val ROI: %.2f%% (%d bets)", roi_val["roi"], roi_val["n_bets"])

            # ROI на test — применяем порог один раз
            test_mask = proba_test >= threshold
            roi_test = calc_roi(test, test_mask)
            logger.info("Test ROI: %.2f%% (%d bets)", roi_test["roi"], roi_test["n_bets"])

            # ROI с дефолтным порогом 0.5
            default_mask = proba_test >= 0.5
            roi_default = calc_roi(test, default_mask)
            logger.info(
                "Test ROI (threshold=0.5): %.2f%% (%d bets)",
                roi_default["roi"],
                roi_default["n_bets"],
            )

            # Feature importance
            for fname, coef in zip(FEATURES, model.coef_[0], strict=True):
                logger.info("  %s: %.4f", fname, coef)

            mlflow.log_metrics(
                {
                    "roi": roi_test["roi"],
                    "n_bets": roi_test["n_bets"],
                    "roi_val": roi_val["roi"],
                    "roi_default_threshold": roi_default["roi"],
                    "auc_val": auc_val,
                    "auc_test": auc_test,
                    "threshold": threshold,
                }
            )
            mlflow.log_params({"threshold": threshold})

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={roi_test['roi']}")
            print(f"RESULT:auc={auc_test}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.3")
            logger.exception("Step 1.3 failed")
            raise


if __name__ == "__main__":
    main()
