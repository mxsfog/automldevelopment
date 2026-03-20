"""Step 1.4: Non-linear baseline (CatBoost с дефолтами).

Гипотеза: CatBoost с дефолтными параметрами даёт strong non-linear baseline.
Фичи: числовые из bets + категориальные Sport, Market из outcomes.
ROI через отбор ставок по predicted probability > threshold.
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
from sklearn.metrics import roc_auc_score

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

NUM_FEATURES = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "Outcomes_Count",
    "USD",
]
CAT_FEATURES = ["Sport", "Market", "is_parlay_str"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def compute_roi_at_threshold(
    df_val: pd.DataFrame, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """ROI на ставках где P(won) > threshold."""
    selected = proba >= threshold
    if selected.sum() == 0:
        return {"roi": 0.0, "n_selected": 0, "coverage": 0.0}
    sel = df_val.iloc[selected] if isinstance(selected, np.ndarray) else df_val[selected]
    total_staked = sel["USD"].sum()
    total_returned = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_returned - total_staked) / total_staked * 100
    return {
        "roi": roi,
        "n_selected": int(selected.sum()),
        "coverage": float(selected.mean()),
    }


def prepare_data(bets: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Подготовка данных: join outcomes, создание фичей."""
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = bets[~bets["Status"].isin(exclude)].copy()

    df["Created_At"] = pd.to_datetime(df["Created_At"])
    df = df.sort_values("Created_At").reset_index(drop=True)

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(Sport=("Sport", "first"), Market=("Market", "first"))
        .reset_index()
    )
    df = df.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")

    df["is_parlay_str"] = df["Is_Parlay"].astype(str)

    for col in ["ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV"]:
        df[col] = df[col].fillna(0.0)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)

    return df


def main() -> None:
    logger.info("Загрузка данных")
    bets = pd.read_csv(DATA_DIR / "bets.csv", low_memory=False)
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv", low_memory=False)
    df = prepare_data(bets, outcomes)
    logger.info("Подготовленный датасет: %d строк", len(df))

    y_binary = (df["Status"] == "won").astype(int)
    cat_indices = [ALL_FEATURES.index(c) for c in CAT_FEATURES]

    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase1/step1.4_catboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("phase", "1")

        try:
            fold_aucs = []
            thresholds_to_try = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
            threshold_roi_sums: dict[float, list[float]] = {t: [] for t in thresholds_to_try}

            for fold_idx in range(n_splits):
                train_end = fold_size * (fold_idx + 1)
                val_start = train_end
                val_end = train_end + fold_size
                if fold_idx == n_splits - 1:
                    val_end = n

                train_df = df.iloc[:train_end]
                val_df = df.iloc[val_start:val_end]

                feat_train = train_df[ALL_FEATURES]
                y_train = y_binary.iloc[:train_end].values
                feat_val = val_df[ALL_FEATURES]
                y_val = y_binary.iloc[val_start:val_end].values

                model = CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.1,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=SEED,
                    cat_features=cat_indices,
                    verbose=0,
                    auto_class_weights="Balanced",
                )
                model.fit(feat_train, y_train, eval_set=(feat_val, y_val), verbose=0)
                proba = model.predict_proba(feat_val)[:, 1]

                auc = roc_auc_score(y_val, proba)
                fold_aucs.append(auc)

                for threshold in thresholds_to_try:
                    stats = compute_roi_at_threshold(val_df, proba, threshold)
                    threshold_roi_sums[threshold].append(stats["roi"])

                mlflow.log_metric(f"auc_fold_{fold_idx}", round(auc, 4))
                mlflow.set_tag("fold_idx", str(fold_idx))

                logger.info(
                    "Fold %d: train=%d, val=%d, auc=%.4f",
                    fold_idx,
                    len(train_df),
                    len(val_df),
                    auc,
                )

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

            # Feature importance
            fi = model.get_feature_importance()
            fi_dict = dict(zip(ALL_FEATURES, fi, strict=True))
            fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
            logger.info("Feature importance: %s", fi_sorted)
            for fname, fval in fi_sorted:
                mlflow.log_metric(f"fi_{fname}", round(fval, 4))

            mlflow.log_params(
                {
                    "model": "CatBoost",
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.1,
                    "auto_class_weights": "Balanced",
                    "features": ",".join(ALL_FEATURES),
                    "cat_features": ",".join(CAT_FEATURES),
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
            mlflow.set_tag("convergence_signal", "0.3")

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
            mlflow.set_tag("failure_reason", "exception during catboost training")
            logger.exception("Step 1.4 failed")
            raise


if __name__ == "__main__":
    main()
