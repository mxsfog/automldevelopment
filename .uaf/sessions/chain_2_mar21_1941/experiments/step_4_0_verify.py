"""Step 4.0 - Chain Verification.

Загружаем pipeline.pkl из предыдущей сессии и воспроизводим точный ROI.
BestPipeline и build_features должны быть определены в __main__
для корректной десериализации joblib.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

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
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_1_mar21_1859/models/best")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Базовый feature engineering (step 1.4) — нужен для десериализации pipeline."""
    feats = pd.DataFrame(index=df.index)

    feats["Odds"] = df["Odds"]
    feats["USD"] = df["USD"]
    feats["log_odds"] = np.log(df["Odds"].clip(1.001))
    feats["log_usd"] = np.log1p(df["USD"].clip(0))
    feats["implied_prob"] = 1.0 / df["Odds"].clip(1.001)
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
    feats["month"] = df["Created_At"].dt.month

    feats["odds_times_stake"] = feats["Odds"] * feats["USD"]
    feats["ml_edge_pos"] = feats["ml_edge"].clip(0)
    feats["ml_ev_pos"] = feats["ml_ev"].clip(0)

    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")

    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


class BestPipeline:
    """Полный пайплайн: feature engineering + предсказание + оценка."""

    def __init__(self, model, feature_names, cat_features, threshold, framework="catboost"):
        self.model = model
        self.feature_names = feature_names
        self.cat_features = cat_features
        self.threshold = threshold
        self.sport_filter = []
        self.framework = framework

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, _ = build_features(df)
        return feats[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        features = self._build_features(df)
        return self.model.predict_proba(features)[:, 1]

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Вычислить ROI на RAW DataFrame."""
        proba = self.predict_proba(df)
        mask = proba >= self.threshold
        selected = df[mask].copy()
        if len(selected) == 0:
            return {"roi": -100.0, "n_selected": 0, "threshold": self.threshold}
        won_mask = selected["Status"] == "won"
        total_stake = selected["USD"].sum()
        total_payout = selected.loc[won_mask, "Payout_USD"].sum()
        roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
        return {"roi": roi, "n_selected": int(mask.sum()), "threshold": self.threshold}


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
    logger.info("Датасет: %d строк, %d колонок", len(df), df.shape[1])
    return df


with mlflow.start_run(run_name="chain/verify") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "chain_verify")
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

        meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        expected_roi = meta["roi"]
        logger.info("Expected roi from metadata: %.4f%%", expected_roi)

        mlflow.log_params(
            {
                "prev_session": meta.get("session_id", "unknown"),
                "prev_step": meta.get("step", "unknown"),
                "expected_roi": expected_roi,
                "threshold": meta["threshold"],
                "framework": meta["framework"],
                "n_features": len(meta["feature_names"]),
            }
        )

        df = load_data()
        n = len(df)
        train_end = int(n * 0.8)
        test_df = df.iloc[train_end:]
        logger.info("Test set: %d строк", len(test_df))

        pipeline = joblib.load(PREV_BEST_DIR / "pipeline.pkl")
        logger.info("Pipeline загружен из %s", PREV_BEST_DIR)

        result = pipeline.evaluate(test_df)
        reproduced_roi = result["roi"]
        n_selected = result["n_selected"]

        logger.info(
            "Reproduced ROI=%.4f%%, n_selected=%d (expected=%.4f%%)",
            reproduced_roi,
            n_selected,
            expected_roi,
        )

        diff = abs(reproduced_roi - expected_roi)
        is_ok = diff < 1.0

        mlflow.log_metrics(
            {
                "reproduced_roi": reproduced_roi,
                "expected_roi": expected_roi,
                "roi_diff": diff,
                "n_selected": n_selected,
            }
        )
        mlflow.set_tag("reproduced_roi", str(round(reproduced_roi, 4)))
        mlflow.set_tag("verification_ok", str(is_ok))

        assert is_ok, (
            f"ROI mismatch: got {reproduced_roi:.2f}, expected {expected_roi:.2f}. "
            "Pipeline не воспроизводит предыдущий результат!"
        )

        mlflow.set_tag("status", "success")
        logger.info("Chain verification PASSED. ROI diff=%.4f%%", diff)

        print("\n=== Step 4.0 Chain Verification ===")
        print(f"Expected ROI:   {expected_roi:.4f}%")
        print(f"Reproduced ROI: {reproduced_roi:.4f}%")
        print(f"Diff:           {diff:.4f}%")
        print(f"N selected:     {n_selected}")
        print(f"Status:         {'PASS' if is_ok else 'FAIL'}")
        print(f"MLflow run_id:  {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
