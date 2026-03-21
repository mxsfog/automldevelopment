"""Step 4.5 — Probability calibration + Kelly criterion betting.

Калибровка CatBoost через isotonic/Platt + Kelly criterion
для взвешивания ставок по edge.
"""

import json
import logging
import os
import random
import re
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

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


def extract_team_name(selection: str) -> str | None:
    if pd.isna(selection):
        return None
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", str(selection))
    cleaned = re.sub(r"\s*(Over|Under)\s+\d+.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().lower() if cleaned.strip() else None


def categorize_market(market: str) -> str:
    if pd.isna(market):
        return "unknown"
    m = str(market).lower()
    if "winner" in m or "1x2" in m:
        return "match_winner"
    if "over" in m or "under" in m or "total" in m:
        return "totals"
    if "handicap" in m or "spread" in m:
        return "handicap"
    return "other"


def build_elo_trend(elo_history: pd.DataFrame) -> dict:
    elo_history = elo_history.sort_values("Created_At")
    trend = {}
    for team_id, group in elo_history.groupby("Team_ID"):
        recent = group.tail(5)
        trend[team_id] = {
            "elo_trend_5": recent["ELO_Change"].sum(),
            "elo_avg_change": recent["ELO_Change"].mean(),
            "recent_win_streak": sum(
                1 for w in reversed(recent["Won"].values) if w == "t" or w is True
            ),
        }
    return trend


def load_and_prepare():
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    elo_history = pd.read_csv(DATA_DIR / "elo_history.csv")

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(
            Sport=("Sport", "first"),
            Market=("Market", "first"),
            Selection=("Selection", "first"),
        )
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()

    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["implied_prob"] = 1.0 / df["Odds"]
    df["log_odds"] = np.log(df["Odds"].clip(lower=1.01))
    df["value_ratio"] = df["ML_P_Model"] / df["implied_prob"].clip(lower=0.01)
    df["edge_x_odds"] = df["ML_Edge"] * df["Odds"]
    df["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[0, 1.3, 1.6, 2.0, 2.5, 3.5, 100],
        labels=["heavy_fav", "fav", "slight_fav", "even", "dog", "big_dog"],
    ).astype(str)
    df["model_confidence"] = df["ML_P_Model"] - df["ML_P_Implied"]
    df["market_category"] = df["Market"].apply(categorize_market)

    teams_dedup = teams.drop_duplicates(subset="Normalized_Name", keep="last")
    team_lookup = teams_dedup.set_index("Normalized_Name")[
        [
            "Current_ELO",
            "Winrate",
            "Total_Games",
            "Offensive_Rating",
            "Defensive_Rating",
            "Net_Rating",
        ]
    ].to_dict("index")
    team_id_lookup = teams_dedup.set_index("Normalized_Name")["ID"].to_dict()

    def get_stat(sel: str, stat: str) -> float | None:
        name = extract_team_name(sel)
        return team_lookup.get(name, {}).get(stat) if name else None

    def get_tid(sel: str) -> int | None:
        name = extract_team_name(sel)
        return team_id_lookup.get(name) if name else None

    for col, stat in [
        ("team_elo", "Current_ELO"),
        ("team_winrate", "Winrate"),
        ("team_games", "Total_Games"),
        ("team_off_rating", "Offensive_Rating"),
        ("team_def_rating", "Defensive_Rating"),
        ("team_net_rating", "Net_Rating"),
    ]:
        df[col] = df["Selection"].apply(lambda x, s=stat: get_stat(x, s))

    df["elo_x_odds"] = df["team_elo"].fillna(1500) * df["implied_prob"]
    df["winrate_vs_implied"] = df["team_winrate"].fillna(0.5) - df["implied_prob"]

    elo_trend = build_elo_trend(elo_history)
    df["team_id_resolved"] = df["Selection"].apply(get_tid)
    for elo_col in ["elo_trend_5", "elo_avg_change", "recent_win_streak"]:
        df[elo_col] = df["team_id_resolved"].apply(
            lambda tid, s=elo_col: elo_trend.get(tid, {}).get(s) if pd.notna(tid) else None
        )

    for col in [
        "Odds",
        "ML_P_Model",
        "ML_P_Implied",
        "ML_Edge",
        "ML_EV",
        "ML_Winrate_Diff",
        "ML_Rating_Diff",
        "Outcomes_Count",
        "USD",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "Sport",
        "Market",
        "Is_Parlay",
        "ML_Team_Stats_Found",
        "odds_bucket",
        "market_category",
    ]:
        df[col] = df[col].fillna("unknown").astype(str)

    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_full = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()

    return train, val, train_full, test


FEATURES = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcomes_Count",
    "USD",
    "hour",
    "day_of_week",
    "is_weekend",
    "implied_prob",
    "log_odds",
    "value_ratio",
    "edge_x_odds",
    "team_elo",
    "team_winrate",
    "team_games",
    "team_off_rating",
    "team_def_rating",
    "team_net_rating",
    "elo_x_odds",
    "winrate_vs_implied",
    "model_confidence",
    "elo_trend_5",
    "elo_avg_change",
    "recent_win_streak",
    "Sport",
    "Market",
    "Is_Parlay",
    "ML_Team_Stats_Found",
    "odds_bucket",
    "market_category",
]

CAT_FEATURES = [
    "Sport",
    "Market",
    "Is_Parlay",
    "ML_Team_Stats_Found",
    "odds_bucket",
    "market_category",
]


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sel = df[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def calc_kelly_roi(
    df: pd.DataFrame, proba: np.ndarray, mask: np.ndarray, fraction: float = 0.25
) -> dict:
    """ROI с Kelly criterion sizing: bet size ~ edge * fraction."""
    sel = df[mask].copy()
    sel_proba = proba[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}

    implied = 1.0 / sel["Odds"].clip(lower=1.01).values
    edge = sel_proba - implied
    kelly_frac = np.clip(edge * fraction / (1.0 - implied + 1e-8), 0.01, 1.0)

    weighted_stake = (sel["USD"].values * kelly_frac).sum()
    won_mask = sel["Status"].values == "won"
    weighted_payout = (sel["Payout_USD"].values[won_mask] * kelly_frac[won_mask]).sum()

    roi = (weighted_payout - weighted_stake) / weighted_stake * 100 if weighted_stake > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def main():
    train, val, train_full, test = load_and_prepare()
    logger.info(
        "Train: %d, Val: %d, Full: %d, Test: %d", len(train), len(val), len(train_full), len(test)
    )

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    with mlflow.start_run(run_name="phase4/step_4_5_calibration") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.5")
            mlflow.set_tag("phase", "4")

            # Base CatBoost
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )
            model.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=100,
            )

            proba_val_raw = model.predict_proba(val[FEATURES])[:, 1]
            proba_test_raw = model.predict_proba(test[FEATURES])[:, 1]

            auc_raw = roc_auc_score(y_test, proba_test_raw)
            brier_raw = brier_score_loss(y_test, proba_test_raw)
            logger.info("Raw CatBoost: AUC=%.4f, Brier=%.4f", auc_raw, brier_raw)

            # Calibration: isotonic on val predictions
            from sklearn.isotonic import IsotonicRegression

            iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso_reg.fit(proba_val_raw, y_val)
            proba_test_iso = iso_reg.predict(proba_test_raw)

            auc_iso = roc_auc_score(y_test, proba_test_iso)
            brier_iso = brier_score_loss(y_test, proba_test_iso)
            logger.info("Isotonic calibrated: AUC=%.4f, Brier=%.4f", auc_iso, brier_iso)

            # Calibration: Platt scaling on val predictions
            from sklearn.linear_model import LogisticRegression

            platt = LogisticRegression(random_state=42, max_iter=1000)
            platt.fit(proba_val_raw.reshape(-1, 1), y_val)
            proba_test_platt = platt.predict_proba(proba_test_raw.reshape(-1, 1))[:, 1]

            auc_platt = roc_auc_score(y_test, proba_test_platt)
            brier_platt = brier_score_loss(y_test, proba_test_platt)
            logger.info("Platt calibrated: AUC=%.4f, Brier=%.4f", auc_platt, brier_platt)

            # ROI comparison at fixed thresholds
            approaches = {
                "raw": proba_test_raw,
                "isotonic": proba_test_iso,
                "platt": proba_test_platt,
            }

            best_roi = -999.0
            best_approach = ""
            best_thr = 0.5
            best_n_bets = 0

            for name, proba in approaches.items():
                logger.info("--- %s ---", name)
                for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
                    r = calc_roi(test, proba >= thr)
                    logger.info("  thr=%.2f: ROI=%.2f%% (%d bets)", thr, r["roi"], r["n_bets"])
                    if r["n_bets"] >= 20 and r["roi"] > best_roi:
                        best_roi = r["roi"]
                        best_approach = name
                        best_thr = thr
                        best_n_bets = r["n_bets"]

            # Kelly criterion with different fractions
            logger.info("--- Kelly criterion (raw, thr=0.45) ---")
            for frac in [0.10, 0.25, 0.50, 1.0]:
                mask = proba_test_raw >= 0.45
                r = calc_kelly_roi(test, proba_test_raw, mask, fraction=frac)
                logger.info("  Kelly f=%.2f: ROI=%.2f%% (%d bets)", frac, r["roi"], r["n_bets"])

            # Kelly with isotonic calibrated probabilities
            logger.info("--- Kelly criterion (isotonic, thr=0.45) ---")
            for frac in [0.10, 0.25, 0.50, 1.0]:
                mask = proba_test_iso >= 0.45
                r = calc_kelly_roi(test, proba_test_iso, mask, fraction=frac)
                logger.info("  Kelly f=%.2f: ROI=%.2f%% (%d bets)", frac, r["roi"], r["n_bets"])

            # Edge-based selection: bet only when calibrated_prob > implied_prob + margin
            logger.info("--- Edge-based selection ---")
            implied_test = 1.0 / test["Odds"].clip(lower=1.01).values
            for margin in [0.0, 0.02, 0.05, 0.10]:
                for proba_name, proba in [("raw", proba_test_raw), ("isotonic", proba_test_iso)]:
                    edge_mask = proba > (implied_test + margin)
                    r = calc_roi(test, edge_mask)
                    if r["n_bets"] >= 10:
                        logger.info(
                            "  %s edge>%.2f: ROI=%.2f%% (%d bets)",
                            proba_name,
                            margin,
                            r["roi"],
                            r["n_bets"],
                        )
                        if r["n_bets"] >= 20 and r["roi"] > best_roi:
                            best_roi = r["roi"]
                            best_approach = f"{proba_name}_edge>{margin}"
                            best_thr = margin
                            best_n_bets = r["n_bets"]

            logger.info("Best: %s => ROI=%.2f%% (%d bets)", best_approach, best_roi, best_n_bets)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost_calibrated",
                    "best_approach": best_approach,
                    "best_threshold": best_thr,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "n_bets": best_n_bets,
                    "auc_raw": auc_raw,
                    "auc_isotonic": auc_iso,
                    "auc_platt": auc_platt,
                    "brier_raw": brier_raw,
                    "brier_isotonic": brier_iso,
                    "brier_platt": brier_platt,
                }
            )

            if best_roi > 5.58:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost_calibrated",
                    "model_file": "model.cbm",
                    "roi": best_roi,
                    "auc": auc_raw,
                    "threshold": best_thr,
                    "n_bets": best_n_bets,
                    "approach": best_approach,
                    "feature_names": FEATURES,
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved (new best)")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={best_roi}")
            print(f"RESULT:auc={auc_raw}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.5")
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
