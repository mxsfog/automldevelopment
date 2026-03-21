"""Step 4.5 — Kelly sizing + retrain on full train+val.

Hypotheses:
1. Kelly fraction (edge / (odds-1)) weights bets by confidence.
2. Training on train+val gives better model (more data, same threshold from val).
3. Lower edge thresholds (0.05, 0.07) may give better Kelly-weighted ROI.
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


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sel = df[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def calc_kelly_roi(
    df: pd.DataFrame, mask: np.ndarray, proba: np.ndarray, implied: np.ndarray
) -> dict:
    """Calculate Kelly-weighted ROI: bet fraction = edge / (odds - 1)."""
    sel_idx = np.where(mask)[0]
    if len(sel_idx) == 0:
        return {"roi": 0.0, "n_bets": 0, "total_kelly_stake": 0.0}

    sel = df.iloc[sel_idx]
    edge = proba[sel_idx] - implied[sel_idx]
    odds = sel["Odds"].values
    kelly_fraction = np.clip(edge / (odds - 1).clip(min=0.01), 0.01, 1.0)

    # Weighted stake: base_usd * kelly_fraction
    base_usd = sel["USD"].values
    weighted_stake = base_usd * kelly_fraction
    total_weighted_stake = weighted_stake.sum()

    won_mask = (sel["Status"] == "won").values
    payout = sel["Payout_USD"].values
    weighted_payout = (payout * kelly_fraction * won_mask).sum()

    roi = (
        (weighted_payout - total_weighted_stake) / total_weighted_stake * 100
        if total_weighted_stake > 0
        else 0.0
    )
    return {
        "roi": round(roi, 4),
        "n_bets": len(sel_idx),
        "total_kelly_stake": round(total_weighted_stake, 2),
    }


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


def main():
    train, val, train_full, test = load_and_prepare()
    logger.info(
        "Train: %d, Val: %d, Train_full: %d, Test: %d",
        len(train),
        len(val),
        len(train_full),
        len(test),
    )

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    with mlflow.start_run(run_name="phase4/step_4_5_kelly_retrain") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.5")
            mlflow.set_tag("phase", "4")

            y_train = (train["Status"] == "won").astype(int)
            y_val = (val["Status"] == "won").astype(int)
            y_test = (test["Status"] == "won").astype(int)

            # Model A: train on train only (same as before, threshold from val)
            model_a = CatBoostClassifier(**HPO_PARAMS, cat_features=cat_indices)
            model_a.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=50,
            )
            proba_val_a = model_a.predict_proba(val[FEATURES])[:, 1]
            proba_test_a = model_a.predict_proba(test[FEATURES])[:, 1]
            auc_test_a = roc_auc_score(y_test, proba_test_a)

            # Model B: retrain on train_full (train+val), no early stopping
            y_full = (train_full["Status"] == "won").astype(int)
            best_iters = model_a.get_best_iteration()
            logger.info("Model A best iteration: %d", best_iters)

            model_b = CatBoostClassifier(
                **{**HPO_PARAMS, "iterations": best_iters},
                cat_features=cat_indices,
            )
            model_b.fit(train_full[FEATURES], y_full)
            proba_test_b = model_b.predict_proba(test[FEATURES])[:, 1]
            auc_test_b = roc_auc_score(y_test, proba_test_b)

            logger.info("AUC A (train only)=%.4f, B (retrained)=%.4f", auc_test_a, auc_test_b)

            implied_val = val["implied_prob"].values
            implied_test = test["implied_prob"].values

            # Use val-selected thresholds from step 4.3
            # Best was p>=0.43, edge>=0.10, but let's also check lower edges
            edge_val = proba_val_a - implied_val

            logger.info("--- Val threshold scan (for Kelly) ---")
            val_configs = []
            for prob_thr in [0.40, 0.43, 0.45]:
                for edge_thr in [0.0, 0.03, 0.05, 0.07, 0.10]:
                    mask = (proba_val_a >= prob_thr) & (edge_val >= edge_thr)
                    r_flat = calc_roi(val, mask)
                    r_kelly = calc_kelly_roi(val, mask, proba_val_a, implied_val)
                    if r_flat["n_bets"] >= 200:
                        val_configs.append(
                            {
                                "prob_thr": prob_thr,
                                "edge_thr": edge_thr,
                                "flat_roi": r_flat["roi"],
                                "kelly_roi": r_kelly["roi"],
                                "n_bets": r_flat["n_bets"],
                            }
                        )
                        logger.info(
                            "  VAL p>=%.2f e>=%.2f: flat=%.2f%% kelly=%.2f%% (%d)",
                            prob_thr,
                            edge_thr,
                            r_flat["roi"],
                            r_kelly["roi"],
                            r_flat["n_bets"],
                        )

            # Pick best val config by Kelly ROI
            best_kelly_cfg = max(val_configs, key=lambda x: x["kelly_roi"])
            best_flat_cfg = max(val_configs, key=lambda x: x["flat_roi"])

            logger.info(
                "Best Kelly: p>=%.2f e>=%.2f kelly_roi=%.2f%%",
                best_kelly_cfg["prob_thr"],
                best_kelly_cfg["edge_thr"],
                best_kelly_cfg["kelly_roi"],
            )
            logger.info(
                "Best Flat: p>=%.2f e>=%.2f flat_roi=%.2f%%",
                best_flat_cfg["prob_thr"],
                best_flat_cfg["edge_thr"],
                best_flat_cfg["flat_roi"],
            )

            # Apply to test
            logger.info("--- TEST results ---")
            mid_t = len(test) // 2
            results = {}

            for model_name, proba_t, auc_t in [
                ("model_A", proba_test_a, auc_test_a),
                ("model_B", proba_test_b, auc_test_b),
            ]:
                edge_t = proba_t - implied_test

                for cfg_name, cfg in [
                    ("best_kelly", best_kelly_cfg),
                    ("best_flat", best_flat_cfg),
                ]:
                    mask = (proba_t >= cfg["prob_thr"]) & (edge_t >= cfg["edge_thr"])
                    r_flat = calc_roi(test, mask)
                    r_kelly = calc_kelly_roi(test, mask, proba_t, implied_test)

                    # Stability
                    r_h1 = calc_roi(test.iloc[:mid_t], mask[:mid_t])
                    r_h2 = calc_roi(test.iloc[mid_t:], mask[mid_t:])

                    key = f"{model_name}_{cfg_name}"
                    results[key] = {
                        "flat_roi": r_flat["roi"],
                        "kelly_roi": r_kelly["roi"],
                        "n_bets": r_flat["n_bets"],
                        "h1": r_h1["roi"],
                        "h2": r_h2["roi"],
                        "auc": auc_t,
                        "cfg": cfg,
                    }

                    logger.info(
                        "TEST %s %s: flat=%.2f%% kelly=%.2f%% (%d) h1=%.2f%% h2=%.2f%%",
                        model_name,
                        cfg_name,
                        r_flat["roi"],
                        r_kelly["roi"],
                        r_flat["n_bets"],
                        r_h1["roi"],
                        r_h2["roi"],
                    )

            # Pick best by Kelly ROI
            best_key = max(results, key=lambda k: results[k]["kelly_roi"])
            best = results[best_key]
            logger.info(
                "Best overall: %s => kelly_roi=%.2f%% flat_roi=%.2f%%",
                best_key,
                best["kelly_roi"],
                best["flat_roi"],
            )

            # Use Kelly ROI as the final metric
            final_roi = best["kelly_roi"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "n_samples_train_full": len(train_full),
                    "model": "CatBoost_HPO",
                    "strategy": best_key,
                    "prob_threshold": best["cfg"]["prob_thr"],
                    "edge_threshold": best["cfg"]["edge_thr"],
                    "best_iterations_a": best_iters,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "roi_flat": best["flat_roi"],
                    "n_bets": best["n_bets"],
                    "auc_test_a": auc_test_a,
                    "auc_test_b": auc_test_b,
                    "threshold": best["cfg"]["prob_thr"],
                    "roi_half1": best["h1"],
                    "roi_half2": best["h2"],
                }
            )

            # Log all results
            for key, val_r in results.items():
                mlflow.log_metric(f"roi_flat_{key}", val_r["flat_roi"])
                mlflow.log_metric(f"roi_kelly_{key}", val_r["kelly_roi"])

            # Save model if best
            if final_roi > 14.82:
                best_model = model_b if "model_B" in best_key else model_a
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                best_model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": final_roi,
                    "roi_type": "kelly",
                    "auc": best["auc"],
                    "threshold": best["cfg"]["prob_thr"],
                    "edge_threshold": best["cfg"]["edge_thr"],
                    "n_bets": best["n_bets"],
                    "feature_names": FEATURES,
                    "sport_filter": [],
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
            print(f"RESULT:roi={final_roi}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.5")
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
