"""Step 1.3 — LogisticRegression + Kelly: linear baseline."""

import json
import logging
import os
import random
import sys
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
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass


def load_data() -> pd.DataFrame:
    """Загрузка данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)
    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Базовые числовые фичи."""
    return np.column_stack(
        [
            df["Odds"].fillna(2.0).values,
            np.log(df["Odds"].fillna(2.0).clip(1.001).values),
            1.0 / df["Odds"].fillna(2.0).clip(1.001).values,
            df["USD"].fillna(0).values,
            np.log1p(df["USD"].fillna(0).clip(0).values),
            (df["Is_Parlay"] == "t").astype(int).values,
            df["Outcomes_Count"].fillna(1).values,
            df["ML_P_Model"].fillna(-1).values,
            df["ML_P_Implied"].fillna(-1).values,
            df["ML_Edge"].fillna(0.0).values,
            df["ML_EV"].clip(-100, 1000).fillna(0.0).values,
            df["ML_Edge"].fillna(0.0).clip(0).values,
            df["ML_EV"].clip(-100, 1000).fillna(0.0).clip(0).values,
        ]
    )


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """ROI на выбранных ставках."""
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def find_threshold(
    df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 200
) -> tuple[float, float]:
    """Поиск Kelly-порога на val."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.60, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def main() -> None:
    """LogisticRegression + Kelly."""
    with mlflow.start_run(run_name="phase1/step1.3_logreg") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

            x_tr = build_features(train_df)
            x_vl = build_features(val_df)
            x_te = build_features(test_df)
            y_tr = (train_df["Status"] == "won").astype(int).values
            y_vl = (val_df["Status"] == "won").astype(int).values
            y_te = (test_df["Status"] == "won").astype(int).values

            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_vl = scaler.transform(x_vl)
            x_te = scaler.transform(x_te)

            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            clf.fit(x_tr, y_tr)

            pv = clf.predict_proba(x_vl)[:, 1]
            pt = clf.predict_proba(x_te)[:, 1]
            auc_val = roc_auc_score(y_vl, pv)
            auc_test = roc_auc_score(y_te, pt)

            pm_val = (val_df["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values
            k_v = compute_kelly(pv, val_df["Odds"].values)
            k_t = compute_kelly(pt, test_df["Odds"].values)
            k_v[~pm_val] = -999
            k_t[~pm_test] = -999

            t_best, roi_val = find_threshold(val_df, k_v)
            roi_test, n_bets = calc_roi(test_df, k_t >= t_best)

            logger.info(
                "LogReg Kelly: val=%.2f%%, test=%.2f%% (%d bets), t=%.3f, AUC=%.4f",
                roi_val,
                roi_test,
                n_bets,
                t_best,
                auc_test,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "model": "logistic_regression",
                    "C": 0.1,
                    "threshold": t_best,
                }
            )
            mlflow.log_metrics(
                {
                    "auc_val": float(auc_val),
                    "auc_test": float(auc_test),
                    "roi_val": float(roi_val),
                    "roi_test": float(roi_test),
                    "n_bets": n_bets,
                    "kelly_threshold": float(t_best),
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT roi_val={roi_val:.2f}% roi_test={roi_test:.2f}% "
                f"n_bets={n_bets} auc={auc_test:.4f} run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
