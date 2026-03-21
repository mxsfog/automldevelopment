"""Step 1.3 - Linear baseline (LogisticRegression).

Hypothesis: LogisticRegression с базовыми фичами — linear baseline.
"""

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
SESSION_DIR = os.environ["UAF_SESSION_DIR"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Selection", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df = df.sort_values("Created_At").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Базовый feature engineering."""
    feats = pd.DataFrame(index=df.index)

    # Числовые фичи из bets.csv
    feats["Odds"] = df["Odds"]
    feats["USD"] = df["USD"]
    feats["log_odds"] = np.log(df["Odds"].clip(1.001))
    feats["log_usd"] = np.log1p(df["USD"].clip(0))
    feats["implied_prob"] = 1.0 / df["Odds"].clip(1.001)
    feats["is_parlay"] = (df["Is_Parlay"] == "t").astype(int)
    feats["outcomes_count"] = df["Outcomes_Count"].fillna(1)

    # ML фичи платформы (доступны до ставки)
    feats["ml_p_model"] = df["ML_P_Model"].fillna(df["ML_P_Model"].median())
    feats["ml_p_implied"] = df["ML_P_Implied"].fillna(df["ML_P_Implied"].median())
    feats["ml_edge"] = df["ML_Edge"].fillna(0.0)
    feats["ml_ev"] = df["ML_EV"].clip(-100, 1000).fillna(0.0)
    feats["ml_team_stats_found"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
    feats["ml_winrate_diff"] = df["ML_Winrate_Diff"].fillna(0.0)
    feats["ml_rating_diff"] = df["ML_Rating_Diff"].fillna(0.0)

    # Временные фичи
    feats["hour"] = df["Created_At"].dt.hour
    feats["day_of_week"] = df["Created_At"].dt.dayofweek

    # Категориальные (label encoding)
    feats["sport_enc"] = pd.Categorical(df["Sport"].fillna("unknown")).codes
    feats["market_enc"] = pd.Categorical(df["Market"].fillna("unknown")).codes

    return feats


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """Вычислить ROI на выбранных ставках."""
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray) -> float:
    """Найти лучший порог вероятности на val-сете."""
    best_roi = -999.0
    best_t = 0.5
    for t in np.arange(0.4, 0.85, 0.02):
        mask = proba >= t
        if mask.sum() < 200:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t


with mlflow.start_run(run_name="phase1/step1.3_logreg") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")

    try:
        budget_file_path = os.environ.get("UAF_BUDGET_STATUS_FILE", "")
        if budget_file_path:
            try:
                status = json.loads(Path(budget_file_path).read_text())
                if status.get("hard_stop"):
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

        logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

        X_train = build_features(train_df)
        X_val = build_features(val_df)
        X_test = build_features(test_df)

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        feature_names = X_train.columns.tolist()
        logger.info("Фичи: %d", len(feature_names))

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        model = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train_sc, y_train)

        proba_val = model.predict_proba(X_val_sc)[:, 1]
        proba_test = model.predict_proba(X_test_sc)[:, 1]

        auc_val = roc_auc_score(y_val, proba_val)
        auc_test = roc_auc_score(y_test, proba_test)
        logger.info("AUC val=%.4f, test=%.4f", auc_val, auc_test)

        # Порог выбираем на val
        best_threshold = find_best_threshold(val_df, proba_val)
        logger.info("Лучший порог по val: %.2f", best_threshold)

        mask_val = proba_val >= best_threshold
        mask_test = proba_test >= best_threshold

        roi_val, n_val = calc_roi(val_df, mask_val)
        roi_test, n_test = calc_roi(test_df, mask_test)

        logger.info(
            "ROI val=%.2f%% (%d ставок), test=%.2f%% (%d ставок)",
            roi_val,
            n_val,
            roi_test,
            n_test,
        )

        # CV по времени (5 фолдов expanding window)
        cv_rois = []
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n

            fold_train = df.iloc[:fold_start]
            fold_val_cv = df.iloc[fold_start:fold_end]

            X_ft = build_features(fold_train)
            X_fv = build_features(fold_val_cv)
            y_ft = (fold_train["Status"] == "won").astype(int)

            sc = StandardScaler()
            X_ft_sc = sc.fit_transform(X_ft)
            X_fv_sc = sc.transform(X_fv)

            m = LogisticRegression(
                C=1.0, max_iter=500, random_state=42, class_weight="balanced", n_jobs=-1
            )
            m.fit(X_ft_sc, y_ft)

            proba_fv = m.predict_proba(X_fv_sc)[:, 1]
            # Используем тот же порог, что нашли на val
            mask_fv = proba_fv >= best_threshold
            if mask_fv.sum() < 50:
                continue
            roi_fold, _ = calc_roi(fold_val_cv, mask_fv)
            cv_rois.append(roi_fold)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_fold)
            logger.info("CV Fold %d: ROI=%.2f%%", fold_idx, roi_fold)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else roi_test
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "LogisticRegression",
                "C": 1.0,
                "n_features": len(feature_names),
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "best_threshold": best_threshold,
            }
        )
        mlflow.log_metrics(
            {
                "auc_val": auc_val,
                "auc_test": auc_test,
                "roi_val": roi_val,
                "roi_test": roi_test,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_val": n_val,
                "n_bets_test": n_test,
            }
        )

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (roi_test - (-3.07)) / 20.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        print("\n=== Step 1.3 Results ===")
        print(f"AUC val={auc_val:.4f}, test={auc_test:.4f}")
        print(f"Best threshold (by val): {best_threshold:.2f}")
        print(f"ROI val:  {roi_val:.2f}% ({n_val} ставок)")
        print(f"ROI test: {roi_test:.2f}% ({n_test} ставок)")
        print(f"CV ROI:   {roi_mean:.2f}% ± {roi_std:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
