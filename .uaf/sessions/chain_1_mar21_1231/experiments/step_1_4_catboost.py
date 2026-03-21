"""Step 1.4 — Non-linear baseline (CatBoost default).

CatBoost с дефолтными параметрами на базовых числовых и категориальных фичах.
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

NUM_FEATURES = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcomes_Count",
    "USD",
]

CAT_FEATURES = [
    "Sport",
    "Market",
    "Is_Parlay",
    "ML_Team_Stats_Found",
]

FEATURES = NUM_FEATURES + CAT_FEATURES


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

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
    for thr in np.arange(0.35, 0.90, 0.01):
        mask = proba >= thr
        result = calc_roi(df, mask)
        if result["n_bets"] >= 20 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr


def main():
    df = load_data()
    train_full, test = time_series_split(df)

    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    x_train = train[FEATURES].copy()
    x_val = val[FEATURES].copy()
    x_test = test[FEATURES].copy()

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    with mlflow.start_run(run_name="phase1/step_1_4_catboost") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "1.4")
            mlflow.set_tag("phase", "1")

            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=100,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )

            model.fit(
                x_train,
                y_train,
                eval_set=(x_val, y_val),
                early_stopping_rounds=100,
            )

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            auc_val = roc_auc_score(y_val, proba_val)
            auc_test = roc_auc_score(y_test, proba_test)
            logger.info("AUC: val=%.4f, test=%.4f", auc_val, auc_test)

            threshold = find_best_threshold(proba_val, val)
            logger.info("Best threshold from val: %.2f", threshold)

            val_mask = proba_val >= threshold
            roi_val = calc_roi(val, val_mask)
            logger.info("Val ROI: %.2f%% (%d bets)", roi_val["roi"], roi_val["n_bets"])

            test_mask = proba_test >= threshold
            roi_test = calc_roi(test, test_mask)
            logger.info("Test ROI: %.2f%% (%d bets)", roi_test["roi"], roi_test["n_bets"])

            default_mask = proba_test >= 0.5
            roi_default = calc_roi(test, default_mask)
            logger.info(
                "Test ROI (threshold=0.5): %.2f%% (%d bets)",
                roi_default["roi"],
                roi_default["n_bets"],
            )

            # Feature importance
            importances = model.get_feature_importance()
            for fname, imp in sorted(zip(FEATURES, importances, strict=True), key=lambda x: -x[1]):
                logger.info("  %s: %.2f", fname, imp)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost",
                    "iterations": model.tree_count_,
                    "n_features": len(FEATURES),
                    "features": str(FEATURES),
                    "threshold": threshold,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "auto_class_weights": "Balanced",
                }
            )

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

            # Сохранение модели если ROI > 0
            if roi_test["roi"] > 0:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": roi_test["roi"],
                    "auc": auc_test,
                    "threshold": threshold,
                    "n_bets": roi_test["n_bets"],
                    "feature_names": FEATURES,
                    "params": {
                        "iterations": model.tree_count_,
                        "depth": 6,
                        "learning_rate": 0.05,
                    },
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved to %s", models_dir)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={roi_test['roi']}")
            print(f"RESULT:auc={auc_test}")
            print(f"RESULT:threshold={threshold}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 1.4")
            logger.exception("Step 1.4 failed")
            raise


if __name__ == "__main__":
    main()
