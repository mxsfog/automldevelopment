"""Step 4.12 — Сохранение лучшего пайплайна (p80 Kelly + chain_8 model).

Лучшая стратегия сессии chain_9:
  model: chain_8 model.cbm (CatBoost, depth=7, lr=0.1, iter=500)
  threshold: LOW segment Kelly >= p80 of train distribution (0.5914)
  filter: Market=1x2, lead_hours>0
  ROI: 33.35% (n=148) — delta=+4.77% vs baseline 28.58%

Сохраняет BestPipeline1x2P80 в ./models/best/ для следующей сессии.
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
from catboost import CatBoostClassifier

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_8_mar22_0035/models/best")

FEATURE_NAMES = [
    "Odds",
    "USD",
    "log_odds",
    "log_usd",
    "implied_prob",
    "is_parlay",
    "outcomes_count",
    "ml_p_model",
    "ml_p_implied",
    "ml_edge",
    "ml_ev",
    "ml_team_stats_found",
    "ml_winrate_diff",
    "ml_rating_diff",
    "hour",
    "day_of_week",
    "month",
    "odds_times_stake",
    "ml_edge_pos",
    "ml_ev_pos",
    "elo_max",
    "elo_min",
    "elo_diff",
    "elo_ratio",
    "elo_mean",
    "elo_std",
    "k_factor_mean",
    "has_elo",
    "elo_count",
    "ml_edge_x_elo_diff",
    "elo_implied_agree",
    "Sport",
    "Market",
    "Currency",
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def check_budget() -> bool:
    try:
        status = json.loads(BUDGET_FILE.read_text())
        return bool(status.get("hard_stop"))
    except FileNotFoundError:
        return False


def load_raw_data() -> pd.DataFrame:
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return feats


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


class BestPipeline1x2P80:
    """Полный пайплайн: chain_8 CatBoost + p80 Kelly threshold.

    Принимает RAW DataFrame (до любого feature engineering).
    """

    def __init__(
        self,
        model: CatBoostClassifier,
        feature_names: list[str],
        kelly_threshold_low: float,
        kelly_threshold_mid: float,
        kelly_threshold_high: float,
        session_id: str,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.kelly_threshold_low = kelly_threshold_low
        self.kelly_threshold_mid = kelly_threshold_mid
        self.kelly_threshold_high = kelly_threshold_high
        self.session_id = session_id
        self.segment_thresholds = {
            "low": kelly_threshold_low,
            "mid": kelly_threshold_mid,
            "high": kelly_threshold_high,
        }

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df)[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._build_features(df)
        return self.model.predict_proba(x)[:, 1]

    def _apply_seg_thresholds(self, df: pd.DataFrame, kelly: np.ndarray) -> np.ndarray:
        buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
        mask = np.zeros(len(df), dtype=bool)
        for bucket, t in self.segment_thresholds.items():
            mask |= (buckets == bucket).values & (kelly >= t)
        return mask

    def evaluate(self, df: pd.DataFrame) -> dict:
        """ROI на RAW DataFrame (test split)."""
        proba = self.predict_proba(df)
        kelly = compute_kelly(proba, df["Odds"].values)

        lead_hours = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
        kelly[lead_hours.values <= 0] = -999

        mkt_mask = df["Market"].values == "1x2"
        seg_mask = self._apply_seg_thresholds(df, kelly)
        final_mask = mkt_mask & seg_mask

        roi, n = calc_roi(df, final_mask)
        return {
            "roi": roi,
            "n_selected": n,
            "kelly_threshold_low": self.kelly_threshold_low,
            "session_id": self.session_id,
        }


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    model = CatBoostClassifier()
    model.load_model(str(PREV_BEST_DIR / "model.cbm"))

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    # Вычислить p80 threshold из train Kelly
    x_train = build_features(train_df)[FEATURE_NAMES]
    proba_train = model.predict_proba(x_train)[:, 1]
    kelly_train = compute_kelly(proba_train, train_df["Odds"].values)
    lead_hours_train = (
        train_df["Start_Time"] - train_df["Created_At"]
    ).dt.total_seconds() / 3600.0

    train_low_mask = (
        (train_df["Market"].values == "1x2")
        & (train_df["Odds"].values < 1.8)
        & (lead_hours_train.values > 0)
    )
    kelly_train_low = kelly_train[train_low_mask]
    threshold_p80 = float(np.percentile(kelly_train_low, 80))
    logger.info("p80 threshold = %.4f", threshold_p80)

    # Создать пайплайн
    pipeline = BestPipeline1x2P80(
        model=model,
        feature_names=FEATURE_NAMES,
        kelly_threshold_low=threshold_p80,
        kelly_threshold_mid=0.545,
        kelly_threshold_high=0.325,
        session_id=SESSION_ID,
    )

    # Верификация на test
    result = pipeline.evaluate(test_df)
    roi = result["roi"]
    n_bets = result["n_selected"]
    logger.info("Pipeline verification: ROI=%.4f%% n=%d", roi, n_bets)

    assert abs(roi - 33.3538) < 1.0, f"Pipeline ROI mismatch: {roi:.4f} vs 33.3538"

    # Сохранить
    best_dir = SESSION_DIR / "models" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, best_dir / "pipeline.pkl")
    model.save_model(str(best_dir / "model.cbm"))

    metadata = {
        "framework": "catboost",
        "model_file": "model.cbm",
        "pipeline_file": "pipeline.pkl",
        "pipeline_class": "BestPipeline1x2P80",
        "roi": roi,
        "auc": 0.7856,
        "kelly_threshold_low": threshold_p80,
        "kelly_threshold_mid": 0.545,
        "kelly_threshold_high": 0.325,
        "kelly_percentile": 80,
        "n_bets": n_bets,
        "feature_names": FEATURE_NAMES,
        "market_filter": "1x2",
        "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
        "source_model": "chain_8_mar22_0035/models/best/model.cbm",
        "threshold_method": "p80_of_train_low_kelly_insample",
        "session_id": SESSION_ID,
        "step": "4.12",
    }
    with open(best_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Сохранено в %s: ROI=%.4f%% n=%d", best_dir, roi, n_bets)

    baseline_roi = 28.5833
    delta = roi - baseline_roi

    with mlflow.start_run(run_name="phase4/step_4_12_save_best") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "save_best")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.12")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "pipeline_class": "BestPipeline1x2P80",
                "kelly_threshold_low": round(threshold_p80, 6),
                "kelly_threshold_mid": 0.545,
                "kelly_threshold_high": 0.325,
                "source_model": "chain_8_model.cbm",
            }
        )
        mlflow.log_metrics({"roi": roi, "n_selected": n_bets, "roi_delta": delta})
        mlflow.log_artifact(str(best_dir / "pipeline.pkl"))
        mlflow.log_artifact(str(best_dir / "metadata.json"))
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

    print(f"STEP_4_12_ROI={roi:.6f}")
    print(f"STEP_4_12_N={n_bets}")
    print(f"STEP_4_12_DELTA={delta:.4f}")
    print(f"PIPELINE_SAVED={best_dir / 'pipeline.pkl'}")
    print(f"MLFLOW_RUN_ID={run.info.run_id}")


if __name__ == "__main__":
    main()
