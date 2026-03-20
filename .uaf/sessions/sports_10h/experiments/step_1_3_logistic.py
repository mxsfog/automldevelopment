"""Step 1.3: Linear baseline (LogisticRegression).

Гипотеза: LogisticRegression с базовыми числовыми фичами даёт
линейный baseline для ROI-метрики.
Фичи: Odds, ML_P_Model, ML_P_Implied, ML_Edge, ML_EV, Is_Parlay, Outcomes_Count.
ROI считается через отбор ставок с predicted_proba > threshold (оптимизируется).
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.warning("Budget hard stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

FEATURE_COLS = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "is_parlay_num",
    "Outcomes_Count",
]


def compute_roi_at_threshold(
    df_val: pd.DataFrame, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """ROI на ставках где P(won) > threshold."""
    selected = proba >= threshold
    if selected.sum() == 0:
        return {"roi": 0.0, "n_selected": 0, "coverage": 0.0}
    sel = df_val[selected]
    total_staked = sel["USD"].sum()
    total_returned = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_returned - total_staked) / total_staked * 100
    return {
        "roi": roi,
        "n_selected": int(selected.sum()),
        "coverage": selected.mean(),
    }


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка фичей."""
    df = df.copy()
    df["is_parlay_num"] = (df["Is_Parlay"] == "t").astype(int)
    for col in ["ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV"]:
        df[col] = df[col].fillna(0.0)
    return df


def main() -> None:
    logger.info("Загрузка данных")
    bets = pd.read_csv(DATA_DIR / "bets.csv", low_memory=False)

    exclude = ["pending", "cancelled", "error", "cashout"]
    df = bets[~bets["Status"].isin(exclude)].copy()
    logger.info("После фильтрации: %d строк", len(df))

    df["Created_At"] = pd.to_datetime(df["Created_At"])
    df = df.sort_values("Created_At").reset_index(drop=True)
    df = prepare_data(df)

    y_binary = (df["Status"] == "won").astype(int)

    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step1.3_logistic") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("phase", "1")

        try:
            fold_aucs = []
            thresholds_to_try = [0.50, 0.55, 0.60, 0.65, 0.70]
            threshold_roi_sums: dict[float, list[float]] = {t: [] for t in thresholds_to_try}

            for fold_idx in range(n_splits):
                train_end = fold_size * (fold_idx + 1)
                val_start = train_end
                val_end = train_end + fold_size
                if fold_idx == n_splits - 1:
                    val_end = n

                train = df.iloc[:train_end]
                val = df.iloc[val_start:val_end]

                feat_train = train[FEATURE_COLS].values
                y_train = y_binary.iloc[:train_end].values
                feat_val = val[FEATURE_COLS].values
                y_val = y_binary.iloc[val_start:val_end].values

                scaler = StandardScaler()
                feat_train_s = scaler.fit_transform(feat_train)
                feat_val_s = scaler.transform(feat_val)

                model = LogisticRegression(random_state=SEED, max_iter=1000, C=1.0, solver="lbfgs")
                model.fit(feat_train_s, y_train)
                proba = model.predict_proba(feat_val_s)[:, 1]

                auc = roc_auc_score(y_val, proba)
                fold_aucs.append(auc)

                for threshold in thresholds_to_try:
                    stats = compute_roi_at_threshold(val, proba, threshold)
                    threshold_roi_sums[threshold].append(stats["roi"])

                mlflow.log_metric(f"auc_fold_{fold_idx}", round(auc, 4))
                mlflow.set_tag("fold_idx", str(fold_idx))

                logger.info(
                    "Fold %d: train=%d, val=%d, auc=%.4f",
                    fold_idx,
                    len(train),
                    len(val),
                    auc,
                )

            # Найдём лучший порог по среднему ROI
            best_threshold = 0.5
            best_roi_mean = -999.0
            for threshold in thresholds_to_try:
                roi_mean = np.mean(threshold_roi_sums[threshold])
                roi_std = np.std(threshold_roi_sums[threshold])
                mlflow.log_metrics(
                    {
                        f"roi_mean_t{int(threshold * 100)}": round(roi_mean, 4),
                        f"roi_std_t{int(threshold * 100)}": round(roi_std, 4),
                    }
                )
                logger.info(
                    "Threshold=%.2f: roi=%.4f +/- %.4f",
                    threshold,
                    roi_mean,
                    roi_std,
                )
                if roi_mean > best_roi_mean:
                    best_roi_mean = roi_mean
                    best_threshold = threshold

            auc_mean = np.mean(fold_aucs)
            auc_std = np.std(fold_aucs)

            mlflow.log_params(
                {
                    "model": "LogisticRegression",
                    "features": ",".join(FEATURE_COLS),
                    "C": 1.0,
                    "best_threshold": best_threshold,
                    "seed": SEED,
                    "validation_scheme": "time_series",
                    "n_splits": n_splits,
                    "n_samples_total": n,
                }
            )

            mlflow.log_metrics(
                {
                    "roi_mean": round(best_roi_mean, 4),
                    "auc_mean": round(auc_mean, 4),
                    "auc_std": round(auc_std, 4),
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.2")

            logger.info(
                "Best: threshold=%.2f, roi=%.4f, auc=%.4f +/- %.4f",
                best_threshold,
                best_roi_mean,
                auc_mean,
                auc_std,
            )
            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT: threshold={best_threshold}, roi_mean={best_roi_mean:.4f}, "
                f"auc_mean={auc_mean:.4f}"
            )
            print(f"RUN_ID: {run.info.run_id}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception during logistic regression")
            logger.exception("Step 1.3 failed")
            raise


if __name__ == "__main__":
    main()
