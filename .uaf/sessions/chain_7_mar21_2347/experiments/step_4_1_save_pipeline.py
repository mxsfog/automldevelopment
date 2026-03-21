"""Step 4.1 — Save pipeline with segment thresholds."""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_6_mar21_2236/models/best")


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


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


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


class BestPipelineSegmented:
    """Пайплайн с раздельными Kelly-порогами по odds-bucket."""

    def __init__(
        self,
        model: CatBoostClassifier,
        feature_names: list[str],
        cat_features: list[str],
        segment_thresholds: dict[str, float],
        sport_filter: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.cat_features = cat_features
        self.segment_thresholds = segment_thresholds
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
        kelly[lead_hours.values <= 0] = -999

        # Odds buckets
        buckets = pd.cut(
            df["Odds"],
            bins=[0, 1.8, 3.0, np.inf],
            labels=["low", "mid", "high"],
        )

        mask = np.zeros(len(df), dtype=bool)
        for bucket, threshold in self.segment_thresholds.items():
            seg_mask = (buckets == bucket).values & (kelly >= threshold)
            mask |= seg_mask

        roi, n_selected = calc_roi(df, mask)
        return {
            "roi": roi,
            "n_selected": n_selected,
            "segment_thresholds": self.segment_thresholds,
        }


# Сохраняем pipeline
meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
model = CatBoostClassifier()
model.load_model(str(PREV_BEST_DIR / "model.cbm"))

# Thresholds из step_4_1 (optimized on val)
segment_thresholds = {"low": 0.495, "mid": 0.635, "high": 0.195}

pipeline = BestPipelineSegmented(
    model=model,
    feature_names=meta["feature_names"],
    cat_features=["Sport", "Market", "Currency"],
    segment_thresholds=segment_thresholds,
    sport_filter=[],
)

best_dir = SESSION_DIR / "models" / "best"
best_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, best_dir / "pipeline.pkl")
model.save_model(str(best_dir / "model.cbm"))

new_meta = {
    "framework": "catboost",
    "model_file": "model.cbm",
    "pipeline_file": "pipeline.pkl",
    "pipeline_class": "BestPipelineSegmented",
    "roi": 25.8347,
    "auc": meta["auc"],
    "threshold": None,
    "segment_thresholds": segment_thresholds,
    "n_bets": 362,
    "feature_names": meta["feature_names"],
    "params": meta["params"],
    "sport_filter": [],
    "session_id": os.environ["UAF_SESSION_ID"],
    "step": "4.1",
}
with open(best_dir / "metadata.json", "w") as f:
    json.dump(new_meta, f, indent=2)

logger.info("Saved pipeline.pkl + metadata.json. roi=%.4f", new_meta["roi"])
print(f"Saved to {best_dir}")
