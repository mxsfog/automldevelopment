"""Step 1.1 - Constant baseline (DummyClassifier).

Hypothesis: DummyClassifier (most_frequent) задаёт lower bound по roi.
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
from sklearn.dummy import DummyClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# UAF-SECTION: MLFLOW-INIT
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

    # Фильтруем статусы
    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    logger.info("После фильтрации статусов: %d строк", len(bets))

    # Join с outcomes (берём первый outcome для каждого бета — sport, market, etc.)
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    # ВАЖНО: из outcomes берём только признаки, известные до исхода ставки
    # Payout_USD, final_score, result_at — запрещены
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Selection", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df = df.sort_values("Created_At").reset_index(drop=True)

    logger.info("Итоговый датасет: %d строк, %d колонок", len(df), df.shape[1])
    return df


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


def time_series_split(df: pd.DataFrame, test_size: float = 0.2):
    """Разбивка по времени: train/val/test."""
    n = len(df)
    train_end = int(n * 0.8)
    # val = последние 20% train (для выбора порогов)
    val_start = int(n * 0.64)  # последние 20% от 80%

    train_df = df.iloc[:train_end]
    val_df = df.iloc[val_start:train_end]
    test_df = df.iloc[train_end:]

    logger.info(
        "Split: train=%d, val=%d (subset of train), test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


with mlflow.start_run(run_name="phase1/step1.1_dummy") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")

    try:
        # Проверка бюджета
        budget_file_path = os.environ.get("UAF_BUDGET_STATUS_FILE", "")
        if budget_file_path:
            try:
                status = json.loads(Path(budget_file_path).read_text())
                if status.get("hard_stop"):
                    mlflow.set_tag("status", "budget_stopped")
                    logger.info("hard_stop=True, выход")
                    sys.exit(0)
            except FileNotFoundError:
                pass

        df = load_data()
        train_df, val_df, test_df = time_series_split(df)

        # Бинарная цель
        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "DummyClassifier(most_frequent)",
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )

        # DummyClassifier — most_frequent предсказывает всегда "won" (класс 1)
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(np.zeros((len(y_train), 1)), y_train)

        # Предсказание на train/val/test
        pred_train = dummy.predict(np.zeros((len(y_train), 1)))
        pred_val = dummy.predict(np.zeros((len(y_val), 1)))
        pred_test = dummy.predict(np.zeros((len(y_test), 1)))

        # Winrate (доля won в данных)
        winrate_train = y_train.mean()
        winrate_val = y_val.mean()
        winrate_test = y_test.mean()

        logger.info(
            "Winrate train=%.4f, val=%.4f, test=%.4f", winrate_train, winrate_val, winrate_test
        )

        # ROI если ставим на все (most_frequent = всегда ставим)
        mask_all_train = np.ones(len(train_df), dtype=bool)
        mask_all_val = np.ones(len(val_df), dtype=bool)
        mask_all_test = np.ones(len(test_df), dtype=bool)

        roi_train, n_train = calc_roi(train_df, mask_all_train)
        roi_val, n_val = calc_roi(val_df, mask_all_val)
        roi_test, n_test = calc_roi(test_df, mask_all_test)

        logger.info(
            "ROI: train=%.2f%% (%d ставок), val=%.2f%% (%d), test=%.2f%% (%d)",
            roi_train,
            n_train,
            roi_val,
            n_val,
            roi_test,
            n_test,
        )

        # CV по времени: 5 фолдов
        n = len(df)
        fold_size = n // 5
        cv_rois = []
        for fold_idx in range(5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n
            # train = до fold_start (минимум 10% данных)
            if fold_start < n // 10:
                continue
            fold_val = df.iloc[fold_start:fold_end]
            mask_fold = np.ones(len(fold_val), dtype=bool)
            roi_fold, _ = calc_roi(fold_val, mask_fold)
            cv_rois.append(roi_fold)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_fold)
            logger.info("Fold %d roi=%.2f%%", fold_idx, roi_fold)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else roi_test
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        mlflow.log_metrics(
            {
                "roi_train": roi_train,
                "roi_val": roi_val,
                "roi_test": roi_test,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_test": n_test,
                "winrate_train": winrate_train,
                "winrate_val": winrate_val,
                "winrate_test": winrate_test,
            }
        )

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.0")

        print("\n=== Step 1.1 Results ===")
        print(f"ROI test (all bets):  {roi_test:.2f}% ({n_test} ставок)")
        print(f"ROI val (all bets):   {roi_val:.2f}% ({n_val} ставок)")
        print(f"CV ROI mean:          {roi_mean:.2f}% ± {roi_std:.2f}%")
        print(f"Winrate test:         {winrate_test:.4f}")
        print(f"MLflow run_id:        {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
