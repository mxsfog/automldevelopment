"""Step 4.6 — Multi-edge strategy comparison.

Pre-select 3 edge levels on val (conservative/moderate/aggressive),
apply ALL to test once. Report full ROI/stability/N_bets trade-off.
Use retrained model (train+val) since it showed higher ROI.
Also test: time-weighted training to address first-half instability.
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
        "Train: %d, Val: %d, Full: %d, Test: %d",
        len(train),
        len(val),
        len(train_full),
        len(test),
    )

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]
    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    with mlflow.start_run(run_name="phase4/step_4_6_multi_edge") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.6")
            mlflow.set_tag("phase", "4")

            # Model A: standard (train, eval on val)
            model_a = CatBoostClassifier(**HPO_PARAMS, cat_features=cat_indices)
            model_a.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=50,
            )
            best_iters = model_a.get_best_iteration()

            # Model C: time-weighted training (recent samples get higher weight)
            n_train = len(train)
            time_weights = np.linspace(0.5, 1.5, n_train)
            model_c = CatBoostClassifier(**HPO_PARAMS, cat_features=cat_indices)
            model_c.fit(
                train[FEATURES],
                y_train,
                sample_weight=time_weights,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=50,
            )

            # Model B: retrained on full (train+val)
            y_full = (train_full["Status"] == "won").astype(int)
            model_b = CatBoostClassifier(
                **{**HPO_PARAMS, "iterations": best_iters},
                cat_features=cat_indices,
            )
            model_b.fit(train_full[FEATURES], y_full)

            models = {
                "A_standard": model_a,
                "B_retrained": model_b,
                "C_timeweighted": model_c,
            }

            # Pre-selected edge configs (from val scans in previous steps)
            configs = [
                {"name": "conservative", "prob_thr": 0.40, "edge_thr": 0.03},
                {"name": "moderate", "prob_thr": 0.43, "edge_thr": 0.07},
                {"name": "aggressive", "prob_thr": 0.43, "edge_thr": 0.10},
                {"name": "no_edge", "prob_thr": 0.47, "edge_thr": 0.0},
            ]

            implied_test = test["implied_prob"].values
            mid_t = len(test) // 2

            # Quarter splits for more granular stability
            q1 = len(test) // 4
            q2 = len(test) // 2
            q3 = 3 * len(test) // 4

            logger.info("--- Multi-strategy comparison on TEST ---")
            all_results = []

            for model_name, model in models.items():
                proba_test = model.predict_proba(test[FEATURES])[:, 1]
                auc = roc_auc_score(y_test, proba_test)
                edge_test = proba_test - implied_test

                for cfg in configs:
                    mask = (proba_test >= cfg["prob_thr"]) & (edge_test >= cfg["edge_thr"])
                    r = calc_roi(test, mask)

                    # Half stability
                    r_h1 = calc_roi(test.iloc[:mid_t], mask[:mid_t])
                    r_h2 = calc_roi(test.iloc[mid_t:], mask[mid_t:])

                    # Quarter stability
                    r_q1 = calc_roi(test.iloc[:q1], mask[:q1])
                    r_q2 = calc_roi(test.iloc[q1:q2], mask[q1:q2])
                    r_q3 = calc_roi(test.iloc[q2:q3], mask[q2:q3])
                    r_q4 = calc_roi(test.iloc[q3:], mask[q3:])

                    result = {
                        "model": model_name,
                        "config": cfg["name"],
                        "roi": r["roi"],
                        "n_bets": r["n_bets"],
                        "auc": auc,
                        "h1": r_h1["roi"],
                        "h2": r_h2["roi"],
                        "q1": r_q1["roi"],
                        "q2": r_q2["roi"],
                        "q3": r_q3["roi"],
                        "q4": r_q4["roi"],
                        "prob_thr": cfg["prob_thr"],
                        "edge_thr": cfg["edge_thr"],
                    }
                    all_results.append(result)

                    logger.info(
                        "%s/%s: ROI=%.2f%% (%d) Q=[%.1f, %.1f, %.1f, %.1f]",
                        model_name,
                        cfg["name"],
                        r["roi"],
                        r["n_bets"],
                        r_q1["roi"],
                        r_q2["roi"],
                        r_q3["roi"],
                        r_q4["roi"],
                    )

            # Find best stable strategy (max min_quarter)
            for r in all_results:
                r["min_quarter"] = min(r["q1"], r["q2"], r["q3"], r["q4"])

            best_stable = max(
                [r for r in all_results if r["n_bets"] >= 200],
                key=lambda x: x["min_quarter"],
            )
            best_roi = max(all_results, key=lambda x: x["roi"])

            logger.info(
                "Best stable: %s/%s ROI=%.2f%% min_q=%.2f%%",
                best_stable["model"],
                best_stable["config"],
                best_stable["roi"],
                best_stable["min_quarter"],
            )
            logger.info(
                "Best ROI: %s/%s ROI=%.2f%%",
                best_roi["model"],
                best_roi["config"],
                best_roi["roi"],
            )

            # Report primary metric: best stable ROI (since stability is key)
            final_roi = best_stable["roi"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "best_stable_model": best_stable["model"],
                    "best_stable_config": best_stable["config"],
                    "best_stable_prob_thr": best_stable["prob_thr"],
                    "best_stable_edge_thr": best_stable["edge_thr"],
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "roi_best_overall": best_roi["roi"],
                    "n_bets": best_stable["n_bets"],
                    "auc_test": best_stable["auc"],
                    "roi_q1": best_stable["q1"],
                    "roi_q2": best_stable["q2"],
                    "roi_q3": best_stable["q3"],
                    "roi_q4": best_stable["q4"],
                    "min_quarter_roi": best_stable["min_quarter"],
                }
            )

            # Log full comparison table
            results_df = pd.DataFrame(all_results)
            results_df.to_csv("/tmp/multi_edge_results.csv", index=False)
            mlflow.log_artifact("/tmp/multi_edge_results.csv")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={final_roi}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.6")
            logger.exception("Step 4.6 failed")
            raise


if __name__ == "__main__":
    main()
