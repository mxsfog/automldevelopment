"""Step 4.0 — Chain Verification: воспроизвести ROI из chain_6_mar21_2236.

BestPipeline и все вспомогательные функции скопированы из step_1_4_catboost.py
для корректной десериализации pickle.
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_6_mar21_2236/models/best")

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Стандартный feature set (chain_1 compatible)."""
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
    feats["elo_max"] = df["elo_max"].fillna(-1)
    feats["elo_min"] = df["elo_min"].fillna(-1)
    feats["elo_diff"] = df["elo_diff"].fillna(0.0)
    feats["elo_ratio"] = df["elo_ratio"].fillna(1.0)
    feats["elo_mean"] = df["elo_mean"].fillna(-1)
    feats["elo_std"] = df["elo_std"].fillna(0.0)
    feats["k_factor_mean"] = df["k_factor_mean"].fillna(-1)
    feats["has_elo"] = df["elo_count"].notna().astype(int)
    feats["elo_count"] = df["elo_count"].fillna(0)
    feats["ml_edge_x_elo_diff"] = feats["ml_edge"] * feats["elo_diff"].clip(0, 500) / 500
    feats["elo_implied_agree"] = (
        feats["implied_prob"] - 1.0 / feats["elo_ratio"].clip(0.5, 2.0)
    ).abs()
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


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


class BestPipeline:
    """Полный пайплайн: feature engineering + предсказание + оценка метрики."""

    def __init__(
        self,
        model: CatBoostClassifier,
        feature_names: list[str],
        cat_features: list[str],
        threshold: float,
        sport_filter: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.cat_features = cat_features
        self.threshold = threshold
        self.sport_filter = sport_filter

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, _ = build_features(df)
        return feats[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Вероятности для RAW DataFrame."""
        x = self._build_features(df)
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, df: pd.DataFrame) -> dict:
        """ROI и метрики на RAW DataFrame."""
        if self.sport_filter:
            df = df[~df["Sport"].isin(self.sport_filter)].copy()
        proba = self.predict_proba(df)
        odds = df["Odds"].values
        kelly = compute_kelly(proba, odds)
        lead_hours = (
            pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
            - pd.to_datetime(df["Created_At"], utc=True)
        ).dt.total_seconds() / 3600.0
        kelly[lead_hours <= 0] = -999
        mask = kelly >= self.threshold
        roi, n_selected = calc_roi(df, mask)
        return {"roi": roi, "n_selected": n_selected, "threshold": self.threshold}


def load_raw_data() -> pd.DataFrame:
    """Загрузка и объединение данных (RAW — до feature engineering)."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            elo_max=("Old_ELO", "max"),
            elo_min=("Old_ELO", "min"),
            elo_mean=("Old_ELO", "mean"),
            elo_std=("Old_ELO", "std"),
            elo_count=("Old_ELO", "count"),
            k_factor_mean=("K_Factor", "mean"),
        )
        .reset_index()
    )
    elo_agg["elo_diff"] = elo_agg["elo_max"] - elo_agg["elo_min"]
    elo_agg["elo_ratio"] = elo_agg["elo_max"] / elo_agg["elo_min"].clip(1.0)
    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    return df


with mlflow.start_run(run_name="chain/verify") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.0")

    try:
        meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        expected_roi = meta["roi"]
        logger.info("Ожидаемый ROI из metadata: %.4f%%", expected_roi)

        mlflow.log_params(
            {
                "prev_session": "chain_6_mar21_2236",
                "expected_roi": expected_roi,
                "threshold": meta["threshold"],
                "n_bets_expected": meta["n_bets"],
                "validation_scheme": "time_series",
                "seed": 42,
            }
        )

        pipeline_path = PREV_BEST_DIR / "pipeline.pkl"
        logger.info("Загрузка pipeline: %s", pipeline_path)
        pipeline = joblib.load(pipeline_path)

        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        logger.info("Всего строк: %d", len(df_raw))

        n = len(df_raw)
        test_start = int(n * 0.80)
        test_df = df_raw.iloc[test_start:].copy()
        logger.info("Test set: %d строк", len(test_df))

        mlflow.log_params(
            {
                "n_samples_train": test_start,
                "n_samples_val": len(test_df),
            }
        )

        result = pipeline.evaluate(test_df)
        reproduced_roi = result["roi"]
        n_selected = result["n_selected"]

        logger.info("Воспроизведённый ROI: %.4f%%", reproduced_roi)
        logger.info("N ставок: %d (ожидалось %d)", n_selected, meta["n_bets"])

        mlflow.log_metrics(
            {
                "reproduced_roi": reproduced_roi,
                "n_selected": n_selected,
                "roi_delta": reproduced_roi - expected_roi,
            }
        )

        delta = abs(reproduced_roi - expected_roi)
        if delta > 1.0:
            logger.error(
                "ROI mismatch: got %.4f, expected %.4f, delta=%.4f",
                reproduced_roi,
                expected_roi,
                delta,
            )
            mlflow.set_tag("status", "mismatch")
            mlflow.set_tag("mismatch_delta", f"{delta:.4f}")
            sys.exit(1)

        logger.info("Верификация успешна. Delta=%.4f%%", delta)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("reproduced_roi", f"{reproduced_roi:.4f}")
        mlflow.set_tag("convergence_signal", "0.0")
        mlflow.log_artifact(__file__)

        print(
            f"VERIFY OK: reproduced_roi={reproduced_roi:.4f}%, "
            f"expected={expected_roi:.4f}%, delta={delta:.4f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception during verification")
        raise
