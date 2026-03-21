"""Step 1.3 — LogisticRegression linear baseline."""

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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}


def load_data() -> pd.DataFrame:
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Market"]
    outcomes_first = outcomes_first[safe_cols]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df = df.sort_values("Created_At").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["odds"] = df["Odds"].fillna(2.0)
    feats["log_odds"] = np.log(df["Odds"].clip(1.001)).fillna(0)
    feats["implied_prob"] = (1.0 / df["Odds"].clip(1.001)).fillna(0.5)
    feats["usd"] = df["USD"].fillna(0)
    feats["log_usd"] = np.log1p(df["USD"].clip(0)).fillna(0)
    feats["is_parlay"] = (df["Is_Parlay"] == "t").astype(int)
    feats["outcomes_count"] = df["Outcomes_Count"].fillna(1)
    feats["ml_p_model"] = df["ML_P_Model"].fillna(-1)
    feats["ml_p_implied"] = df["ML_P_Implied"].fillna(-1)
    feats["ml_edge"] = df["ML_Edge"].fillna(0.0)
    feats["ml_ev"] = df["ML_EV"].clip(-100, 1000).fillna(0.0)
    feats["ml_team_stats_found"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
    feats["ml_winrate_diff"] = df["ML_Winrate_Diff"].fillna(0.0)
    feats["ml_rating_diff"] = df["ML_Rating_Diff"].fillna(0.0)
    feats["hour"] = df["Created_At"].dt.hour
    feats["day_of_week"] = df["Created_At"].dt.dayofweek
    return feats


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 100) -> float:
    best_roi, best_t = -999.0, 0.5
    for t in np.arange(0.40, 0.93, 0.01):
        mask = proba >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best threshold=%.2f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


with mlflow.start_run(run_name="phase1/step1.3_logreg") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")

    try:
        budget_file_path = os.environ.get("UAF_BUDGET_STATUS_FILE", "")
        if budget_file_path:
            try:
                budget_status = json.loads(Path(budget_file_path).read_text())
                if budget_status.get("hard_stop"):
                    mlflow.set_tag("status", "budget_stopped")
                    sys.exit(0)
            except FileNotFoundError:
                pass

        df = load_data()
        n = len(df)
        train_end = int(n * 0.8)
        val_start = int(n * 0.64)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:train_end]
        test_df = df.iloc[train_end:]

        X_train = build_features(train_df)
        X_val = build_features(val_df)
        X_test = build_features(test_df)

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        model.fit(X_train_s, y_train)

        pv = model.predict_proba(X_val_s)[:, 1]
        pt = model.predict_proba(X_test_s)[:, 1]

        auc_v = roc_auc_score(y_val, pv)
        auc_t = roc_auc_score(y_test, pt)

        threshold = find_best_threshold(val_df, pv)
        roi_val, n_val = calc_roi(val_df, pv >= threshold)
        roi_test, n_test = calc_roi(test_df, pt >= threshold)

        # CV
        cv_rois = []
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n
            fold_train = df.iloc[:fold_start]
            fold_val_cv = df.iloc[fold_start:fold_end]
            xft = build_features(fold_train)
            xfv = build_features(fold_val_cv)
            yft = (fold_train["Status"] == "won").astype(int)
            sc = StandardScaler()
            xft_s = sc.fit_transform(xft)
            xfv_s = sc.transform(xfv)
            m = LogisticRegression(max_iter=300, random_state=42, C=1.0)
            m.fit(xft_s, yft)
            pf = m.predict_proba(xfv_s)[:, 1]
            mask_f = pf >= threshold
            roi_f, _ = calc_roi(fold_val_cv, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_f)
            logger.info("CV Fold %d: ROI=%.2f%%", fold_idx, roi_f)

        roi_mean = float(np.mean(cv_rois))
        roi_std = float(np.std(cv_rois))

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "LogisticRegression",
                "C": 1.0,
                "threshold": threshold,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "roi_test": roi_test,
                "roi_val": roi_val,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "auc_val": auc_v,
                "auc_test": auc_t,
                "n_bets_test": n_test,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.1")

        print("\n=== Step 1.3 LogisticRegression ===")
        print(f"AUC val/test: {auc_v:.4f} / {auc_t:.4f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"ROI test: {roi_test:.2f}% ({n_test} ставок)")
        print(f"ROI val:  {roi_val:.2f}% ({n_val} ставок)")
        print(f"CV ROI:   {roi_mean:.2f}% +/- {roi_std:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
