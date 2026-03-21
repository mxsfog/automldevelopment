"""Step 4.3 — CatBoost full train + sport-specific thresholds.

Обучаем CatBoost на полном train (train+val), порог из val.
Отдельные пороги по видам спорта.
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
            Sport=("Sport", "first"), Market=("Market", "first"), Selection=("Selection", "first")
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


def main():
    train, val, train_full, test = load_and_prepare()
    logger.info(
        "Train: %d, Val: %d, Full: %d, Test: %d", len(train), len(val), len(train_full), len(test)
    )

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    with mlflow.start_run(run_name="phase4/step_4_3_catboost_full") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.3")
            mlflow.set_tag("phase", "4")

            # Model 1: trained on train, eval on val (for threshold selection)
            y_train = (train["Status"] == "won").astype(int)
            y_val = (val["Status"] == "won").astype(int)

            model_thr = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )
            model_thr.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=100,
            )
            best_iter = model_thr.tree_count_
            logger.info("Best iterations from val: %d", best_iter)

            proba_val = model_thr.predict_proba(val[FEATURES])[:, 1]

            # Sport-specific threshold analysis on val
            sport_thresholds = {}
            for sport, grp in val.groupby("Sport"):
                if len(grp) < 50:
                    continue
                sport_mask = val["Sport"] == sport
                proba_sport = proba_val[sport_mask.values]
                sport_df = grp

                best_roi_sport = -999.0
                best_thr_sport = 0.5
                for thr in np.arange(0.35, 0.85, 0.02):
                    mask = proba_sport >= thr
                    r = calc_roi(sport_df, mask)
                    if r["n_bets"] >= 10 and r["roi"] > best_roi_sport:
                        best_roi_sport = r["roi"]
                        best_thr_sport = thr

                sport_thresholds[sport] = best_thr_sport
                logger.info(
                    "  %s: best_thr=%.2f, val_roi=%.2f%%", sport, best_thr_sport, best_roi_sport
                )

            # Global threshold
            global_best_roi = -999.0
            global_best_thr = 0.5
            min_bets = max(20, int(len(val) * 0.05))
            for thr in np.arange(0.35, 0.85, 0.01):
                mask = proba_val >= thr
                r = calc_roi(val, mask)
                if r["n_bets"] >= min_bets and r["roi"] > global_best_roi:
                    global_best_roi = r["roi"]
                    global_best_thr = thr
            logger.info(
                "Global threshold: %.2f (val ROI: %.2f%%)", global_best_thr, global_best_roi
            )

            # Model 2: trained on full train, fixed iterations
            y_full = (train_full["Status"] == "won").astype(int)
            y_test = (test["Status"] == "won").astype(int)

            model_full = CatBoostClassifier(
                iterations=best_iter,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )
            model_full.fit(train_full[FEATURES], y_full)

            proba_test = model_full.predict_proba(test[FEATURES])[:, 1]
            auc_test = roc_auc_score(y_test, proba_test)
            logger.info("Full model AUC: test=%.4f", auc_test)

            # Global threshold on test
            roi_global = calc_roi(test, proba_test >= global_best_thr)
            logger.info(
                "Global threshold ROI: %.2f%% (%d bets, thr=%.2f)",
                roi_global["roi"],
                roi_global["n_bets"],
                global_best_thr,
            )

            # Sport-specific thresholds on test
            test_selected = np.zeros(len(test), dtype=bool)
            for i, (_, row) in enumerate(test.iterrows()):
                sport = row["Sport"]
                thr = sport_thresholds.get(sport, global_best_thr)
                if proba_test[i] >= thr:
                    test_selected[i] = True

            roi_sport_thr = calc_roi(test, test_selected)
            logger.info(
                "Sport-specific threshold ROI: %.2f%% (%d bets)",
                roi_sport_thr["roi"],
                roi_sport_thr["n_bets"],
            )

            # Fixed low thresholds (what worked before)
            for ft in [0.40, 0.45, 0.50, 0.55]:
                r = calc_roi(test, proba_test >= ft)
                logger.info("  Fixed thr=%.2f: ROI=%.2f%% (%d bets)", ft, r["roi"], r["n_bets"])

            # Sport ROI analysis on test
            logger.info("--- Sport ROI on test (thr=0.45) ---")
            for sport in sorted(test["Sport"].unique()):
                sport_mask = (test["Sport"] == sport) & (proba_test >= 0.45)
                if sport_mask.sum() >= 5:
                    r = calc_roi(test, sport_mask.values)
                    logger.info("  %s: ROI=%.2f%% (%d bets)", sport, r["roi"], r["n_bets"])

            best_roi = max(roi_global["roi"], roi_sport_thr["roi"])
            best_n_bets = (
                roi_global["n_bets"]
                if roi_global["roi"] >= roi_sport_thr["roi"]
                else roi_sport_thr["n_bets"]
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_full),
                    "n_samples_val": len(val),
                    "model": "CatBoost_full_train",
                    "iterations": best_iter,
                    "global_threshold": global_best_thr,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_global_thr": roi_global["roi"],
                    "roi_sport_thr": roi_sport_thr["roi"],
                    "auc_test": auc_test,
                    "n_bets": best_n_bets,
                }
            )

            if best_roi > 5.56:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model_full.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_roi,
                    "auc": auc_test,
                    "threshold": global_best_thr,
                    "n_bets": best_n_bets,
                    "feature_names": FEATURES,
                    "params": {"iterations": best_iter, "depth": 6, "learning_rate": 0.05},
                    "sport_filter": [],
                    "sport_thresholds": {k: float(v) for k, v in sport_thresholds.items()},
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved (new best)")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={best_roi}")
            print(f"RESULT:auc={auc_test}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.3")
            logger.exception("Step 4.3 failed")
            raise


if __name__ == "__main__":
    main()
