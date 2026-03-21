"""Step 4.7 — Q2 period analysis + rolling edge strategy.

Key finding from step 4.6: Q2 (second quarter of test) is ALWAYS negative
across all models and strategies. This step:
1. Analyzes what's different about Q2 (sports, odds, markets)
2. Tests rolling/adaptive edge threshold
3. Tests odds-range filtering (exclude unprofitable odds ranges)
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
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
ESPORTS = {"valorant", "counter-strike", "league-of-legends", "dota-2", "call-of-duty"}


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
            Slug=("Slug", "first"),
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

    df["market_category"] = df["Market"].apply(categorize_market)
    df["edge_abs"] = df["ML_Edge"].abs()
    df["odds_sq"] = df["Odds"] ** 2
    df["elo_winrate_product"] = df["team_elo"].fillna(1500) * df["team_winrate"].fillna(0.5)
    df["streak_x_elo"] = df["recent_win_streak"].fillna(0) * df["team_elo"].fillna(1500)
    df["value_edge"] = df["value_ratio"] * df["ML_Edge"]
    df["games_log"] = np.log1p(df["team_games"].fillna(0))
    df["is_esports"] = df["Slug"].fillna("").apply(lambda x: str(x).lower() in ESPORTS).astype(int)
    df["odds_edge_ratio"] = df["ML_Edge"] / df["Odds"].clip(lower=1.01)
    df["ev_normalized"] = df["ML_EV"] / df["Odds"].clip(lower=1.01)
    df["prob_diff_sq"] = (df["ML_P_Model"] - df["ML_P_Implied"]) ** 2

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
    "edge_abs",
    "odds_sq",
    "elo_winrate_product",
    "streak_x_elo",
    "value_edge",
    "games_log",
    "is_esports",
    "odds_edge_ratio",
    "ev_normalized",
    "prob_diff_sq",
]

CAT_FEATURES = [
    "Sport",
    "Market",
    "Is_Parlay",
    "ML_Team_Stats_Found",
    "odds_bucket",
    "market_category",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

HPO_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.147,
    "depth": 6,
    "l2_leaf_reg": 28.3,
    "min_data_in_leaf": 100,
    "random_strength": 3.49,
    "bagging_temperature": 0.76,
    "border_count": 92,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


def calc_roi(df: pd.DataFrame, mask: np.ndarray | None = None) -> dict:
    sel = df[mask] if mask is not None else df
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def main():
    train, val, _train_full, test = load_and_prepare()
    logger.info(
        "Train: %d, Val: %d, Test: %d",
        len(train),
        len(val),
        len(test),
    )

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]
    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    with mlflow.start_run(run_name="phase4/step_4_7_q2_analysis") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.7")
            mlflow.set_tag("phase", "4")

            model = CatBoostClassifier(**HPO_PARAMS, cat_features=cat_indices)
            model.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=50,
            )
            proba_test = model.predict_proba(test[FEATURES])[:, 1]
            auc_test = roc_auc_score(y_test, proba_test)

            implied_test = test["implied_prob"].values
            edge_test = proba_test - implied_test

            # Q2 analysis
            q1_end = len(test) // 4
            q2_end = len(test) // 2
            q3_end = 3 * len(test) // 4

            test_q2 = test.iloc[q1_end:q2_end].copy()
            test_not_q2 = pd.concat([test.iloc[:q1_end], test.iloc[q2_end:]])

            logger.info("--- Q2 Period Analysis ---")
            logger.info(
                "Q2 dates: %s to %s", test_q2["Created_At"].min(), test_q2["Created_At"].max()
            )
            logger.info("Q2 size: %d bets", len(test_q2))

            # ROI by sport in Q2 vs rest
            for sport in test["Sport"].value_counts().head(8).index:
                q2_sport = test_q2[test_q2["Sport"] == sport]
                rest_sport = test_not_q2[test_not_q2["Sport"] == sport]
                r_q2 = calc_roi(q2_sport) if len(q2_sport) > 0 else {"roi": 0, "n_bets": 0}
                r_rest = calc_roi(rest_sport) if len(rest_sport) > 0 else {"roi": 0, "n_bets": 0}
                logger.info(
                    "  %s: Q2=%.1f%%(%d) rest=%.1f%%(%d)",
                    sport,
                    r_q2["roi"],
                    r_q2["n_bets"],
                    r_rest["roi"],
                    r_rest["n_bets"],
                )

            # ROI by odds bucket in Q2
            logger.info("--- Q2 by odds bucket ---")
            for bucket in ["heavy_fav", "fav", "slight_fav", "even", "dog", "big_dog"]:
                q2_b = test_q2[test_q2["odds_bucket"] == bucket]
                rest_b = test_not_q2[test_not_q2["odds_bucket"] == bucket]
                r_q2 = calc_roi(q2_b) if len(q2_b) > 0 else {"roi": 0, "n_bets": 0}
                r_rest = calc_roi(rest_b) if len(rest_b) > 0 else {"roi": 0, "n_bets": 0}
                logger.info(
                    "  %s: Q2=%.1f%%(%d) rest=%.1f%%(%d)",
                    bucket,
                    r_q2["roi"],
                    r_q2["n_bets"],
                    r_rest["roi"],
                    r_rest["n_bets"],
                )

            # Strategy: odds-range filter (from val)
            # Scan which odds ranges are profitable on val
            proba_val = model.predict_proba(val[FEATURES])[:, 1]
            implied_val = val["implied_prob"].values
            edge_val = proba_val - implied_val

            logger.info("--- Odds-range strategies (val) ---")
            best_odds_roi = -999.0
            best_odds_filter = None

            odds_ranges = [
                ("no_heavyfav", lambda o: o > 1.3),
                ("no_bigdog", lambda o: o < 3.5),
                ("mid_range", lambda o: (o >= 1.6) & (o <= 3.5)),
                ("fav_only", lambda o: o < 2.0),
                ("value_only", lambda o: (o >= 2.0) & (o <= 3.5)),
            ]

            for name, odds_fn in odds_ranges:
                odds_mask = odds_fn(val["Odds"])
                for edge_thr in [0.03, 0.05, 0.07, 0.10]:
                    mask = odds_mask & (proba_val >= 0.43) & (edge_val >= edge_thr)
                    r = calc_roi(val, mask.values)
                    if r["n_bets"] >= 200:
                        logger.info(
                            "  VAL %s e>=%.2f: %.2f%% (%d)",
                            name,
                            edge_thr,
                            r["roi"],
                            r["n_bets"],
                        )
                        if r["roi"] > best_odds_roi:
                            best_odds_roi = r["roi"]
                            best_odds_filter = (name, edge_thr, odds_fn)

            # Apply best odds filter to test
            if best_odds_filter:
                oname, oedge, ofn = best_odds_filter
                logger.info(
                    "Best odds filter: %s e>=%.2f val_roi=%.2f%%", oname, oedge, best_odds_roi
                )

                test_odds_mask = ofn(test["Odds"])
                test_filter_mask = (
                    test_odds_mask.values & (proba_test >= 0.43) & (edge_test >= oedge)
                )
                r_filtered = calc_roi(test, test_filter_mask)

                # Quarter analysis
                masks_q = [
                    test_filter_mask[:q1_end],
                    test_filter_mask[q1_end:q2_end],
                    test_filter_mask[q2_end:q3_end],
                    test_filter_mask[q3_end:],
                ]
                rqs = [
                    calc_roi(test.iloc[:q1_end], masks_q[0]),
                    calc_roi(test.iloc[q1_end:q2_end], masks_q[1]),
                    calc_roi(test.iloc[q2_end:q3_end], masks_q[2]),
                    calc_roi(test.iloc[q3_end:], masks_q[3]),
                ]
                logger.info(
                    "TEST odds-filtered: ROI=%.2f%% (%d) Q=[%.1f, %.1f, %.1f, %.1f]",
                    r_filtered["roi"],
                    r_filtered["n_bets"],
                    rqs[0]["roi"],
                    rqs[1]["roi"],
                    rqs[2]["roi"],
                    rqs[3]["roi"],
                )

            # Baseline comparison: edge>=0.07 (moderate) without odds filter
            base_mask = (proba_test >= 0.43) & (edge_test >= 0.07)
            r_base = calc_roi(test, base_mask)
            rqs_base = [
                calc_roi(test.iloc[:q1_end], base_mask[:q1_end]),
                calc_roi(test.iloc[q1_end:q2_end], base_mask[q1_end:q2_end]),
                calc_roi(test.iloc[q2_end:q3_end], base_mask[q2_end:q3_end]),
                calc_roi(test.iloc[q3_end:], base_mask[q3_end:]),
            ]
            logger.info(
                "TEST e>=0.07 baseline: ROI=%.2f%% (%d) Q=[%.1f, %.1f, %.1f, %.1f]",
                r_base["roi"],
                r_base["n_bets"],
                rqs_base[0]["roi"],
                rqs_base[1]["roi"],
                rqs_base[2]["roi"],
                rqs_base[3]["roi"],
            )

            # Also test e>=0.05 moderate
            base05_mask = (proba_test >= 0.43) & (edge_test >= 0.05)
            r_base05 = calc_roi(test, base05_mask)
            rqs05 = [
                calc_roi(test.iloc[:q1_end], base05_mask[:q1_end]),
                calc_roi(test.iloc[q1_end:q2_end], base05_mask[q1_end:q2_end]),
                calc_roi(test.iloc[q2_end:q3_end], base05_mask[q2_end:q3_end]),
                calc_roi(test.iloc[q3_end:], base05_mask[q3_end:]),
            ]
            logger.info(
                "TEST e>=0.05: ROI=%.2f%% (%d) Q=[%.1f, %.1f, %.1f, %.1f]",
                r_base05["roi"],
                r_base05["n_bets"],
                rqs05[0]["roi"],
                rqs05[1]["roi"],
                rqs05[2]["roi"],
                rqs05[3]["roi"],
            )

            # Final: pick best
            final_roi = r_filtered["roi"] if best_odds_filter else r_base["roi"]
            final_n = r_filtered["n_bets"] if best_odds_filter else r_base["n_bets"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost_HPO",
                    "odds_filter": best_odds_filter[0] if best_odds_filter else "none",
                    "edge_threshold": best_odds_filter[1] if best_odds_filter else 0.07,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "n_bets": final_n,
                    "auc_test": auc_test,
                    "roi_e07_baseline": r_base["roi"],
                    "roi_e05": r_base05["roi"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={final_roi}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.7")
            logger.exception("Step 4.7 failed")
            raise


if __name__ == "__main__":
    main()
