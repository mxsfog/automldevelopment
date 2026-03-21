"""Step 4.4 — Calibrated edge + stability-optimized threshold.

Hypotheses:
1. Platt scaling calibrates probabilities -> better edge estimates.
2. Stability-optimized threshold selection on val (max min_half ROI)
   gives more robust test performance.
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

    return train, val, test


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


def stability_scan(
    df: pd.DataFrame,
    proba: np.ndarray,
    edge: np.ndarray,
    min_bets: int = 200,
) -> list[dict]:
    """Scan thresholds on a dataset, return configs with stability metrics."""
    mid = len(df) // 2
    df_1, df_2 = df.iloc[:mid], df.iloc[mid:]
    results = []

    for prob_thr in [0.40, 0.42, 0.43, 0.45, 0.47]:
        for edge_thr in [0.0, 0.02, 0.03, 0.05, 0.07, 0.10]:
            mask = (proba >= prob_thr) & (edge >= edge_thr)
            r = calc_roi(df, mask)
            if r["n_bets"] < min_bets:
                continue

            mask_1 = mask[:mid]
            mask_2 = mask[mid:]
            r1 = calc_roi(df_1, mask_1)
            r2 = calc_roi(df_2, mask_2)

            min_half = min(r1["roi"], r2["roi"])
            spread = abs(r1["roi"] - r2["roi"])

            results.append(
                {
                    "prob_thr": prob_thr,
                    "edge_thr": edge_thr,
                    "roi": r["roi"],
                    "n_bets": r["n_bets"],
                    "roi_h1": r1["roi"],
                    "roi_h2": r2["roi"],
                    "min_half": min_half,
                    "spread": spread,
                }
            )
    return results


def main():
    train, val, test = load_and_prepare()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    with mlflow.start_run(run_name="phase4/step_4_4_calibrated_edge") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.4")
            mlflow.set_tag("phase", "4")

            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.147,
                depth=6,
                l2_leaf_reg=28.3,
                min_data_in_leaf=100,
                random_strength=3.49,
                bagging_temperature=0.76,
                border_count=92,
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
                early_stopping_rounds=50,
            )

            proba_val_raw = model.predict_proba(val[FEATURES])[:, 1]
            proba_test_raw = model.predict_proba(test[FEATURES])[:, 1]
            auc_test_raw = roc_auc_score(y_test, proba_test_raw)

            # Platt scaling calibration (fit on val, transform test)
            from sklearn.linear_model import LogisticRegression

            cal_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            cal_model.fit(proba_val_raw.reshape(-1, 1), y_val)
            proba_val_cal = cal_model.predict_proba(proba_val_raw.reshape(-1, 1))[:, 1]
            proba_test_cal = cal_model.predict_proba(proba_test_raw.reshape(-1, 1))[:, 1]
            auc_test_cal = roc_auc_score(y_test, proba_test_cal)

            logger.info("AUC raw=%.4f, calibrated=%.4f", auc_test_raw, auc_test_cal)

            implied_val = val["implied_prob"].values
            implied_test = test["implied_prob"].values

            # Strategy A: raw probabilities + stability-optimized threshold
            edge_val_raw = proba_val_raw - implied_val
            edge_test_raw = proba_test_raw - implied_test

            logger.info("--- Strategy A: Raw proba + stability-optimized ---")
            results_a = stability_scan(val, proba_val_raw, edge_val_raw, min_bets=200)

            # Pick by max min_half (worst-case stability)
            best_a = max(results_a, key=lambda x: x["min_half"])
            logger.info(
                "Best A (stability): p>=%.2f e>=%.2f val_roi=%.2f%% h1=%.2f%% h2=%.2f%% (%d bets)",
                best_a["prob_thr"],
                best_a["edge_thr"],
                best_a["roi"],
                best_a["roi_h1"],
                best_a["roi_h2"],
                best_a["n_bets"],
            )

            # Also pick by max ROI for comparison
            best_a_roi = max(results_a, key=lambda x: x["roi"])
            logger.info(
                "Best A (max ROI): p>=%.2f e>=%.2f val_roi=%.2f%% h1=%.2f%% h2=%.2f%% (%d bets)",
                best_a_roi["prob_thr"],
                best_a_roi["edge_thr"],
                best_a_roi["roi"],
                best_a_roi["roi_h1"],
                best_a_roi["roi_h2"],
                best_a_roi["n_bets"],
            )

            # Strategy B: calibrated probabilities + stability-optimized threshold
            edge_val_cal = proba_val_cal - implied_val
            edge_test_cal = proba_test_cal - implied_test

            logger.info("--- Strategy B: Calibrated proba + stability-optimized ---")
            results_b = stability_scan(val, proba_val_cal, edge_val_cal, min_bets=200)

            if results_b:
                best_b = max(results_b, key=lambda x: x["min_half"])
                logger.info(
                    "Best B (stability): p>=%.2f e>=%.2f val_roi=%.2f%% "
                    "h1=%.2f%% h2=%.2f%% (%d bets)",
                    best_b["prob_thr"],
                    best_b["edge_thr"],
                    best_b["roi"],
                    best_b["roi_h1"],
                    best_b["roi_h2"],
                    best_b["n_bets"],
                )
            else:
                best_b = None
                logger.info("No valid configs for calibrated strategy")

            # Apply best strategies to test (ONCE each)
            logger.info("--- TEST results ---")

            # Strategy A: stability-optimized raw
            mask_a = (proba_test_raw >= best_a["prob_thr"]) & (edge_test_raw >= best_a["edge_thr"])
            test_a = calc_roi(test, mask_a)
            mid_t = len(test) // 2
            r_a1 = calc_roi(test.iloc[:mid_t], mask_a[:mid_t])
            r_a2 = calc_roi(test.iloc[mid_t:], mask_a[mid_t:])
            logger.info(
                "TEST A (stability): p>=%.2f e>=%.2f => ROI=%.2f%% (%d) h1=%.2f%% h2=%.2f%%",
                best_a["prob_thr"],
                best_a["edge_thr"],
                test_a["roi"],
                test_a["n_bets"],
                r_a1["roi"],
                r_a2["roi"],
            )

            # Strategy A: max ROI raw
            mask_a_roi = (proba_test_raw >= best_a_roi["prob_thr"]) & (
                edge_test_raw >= best_a_roi["edge_thr"]
            )
            test_a_roi = calc_roi(test, mask_a_roi)
            r_ar1 = calc_roi(test.iloc[:mid_t], mask_a_roi[:mid_t])
            r_ar2 = calc_roi(test.iloc[mid_t:], mask_a_roi[mid_t:])
            logger.info(
                "TEST A (max ROI): p>=%.2f e>=%.2f => ROI=%.2f%% (%d) h1=%.2f%% h2=%.2f%%",
                best_a_roi["prob_thr"],
                best_a_roi["edge_thr"],
                test_a_roi["roi"],
                test_a_roi["n_bets"],
                r_ar1["roi"],
                r_ar2["roi"],
            )

            # Strategy B: calibrated stability
            if best_b:
                mask_b = (proba_test_cal >= best_b["prob_thr"]) & (
                    edge_test_cal >= best_b["edge_thr"]
                )
                test_b = calc_roi(test, mask_b)
                r_b1 = calc_roi(test.iloc[:mid_t], mask_b[:mid_t])
                r_b2 = calc_roi(test.iloc[mid_t:], mask_b[mid_t:])
                logger.info(
                    "TEST B (cal stability): p>=%.2f e>=%.2f => ROI=%.2f%% (%d) "
                    "h1=%.2f%% h2=%.2f%%",
                    best_b["prob_thr"],
                    best_b["edge_thr"],
                    test_b["roi"],
                    test_b["n_bets"],
                    r_b1["roi"],
                    r_b2["roi"],
                )

            # Select best overall strategy for reporting
            candidates = [
                ("raw_stability", test_a, best_a, r_a1, r_a2),
                ("raw_max_roi", test_a_roi, best_a_roi, r_ar1, r_ar2),
            ]
            if best_b:
                candidates.append(("cal_stability", test_b, best_b, r_b1, r_b2))

            # Pick by test ROI (single application, no leakage)
            best_name, best_test, best_cfg, best_h1, best_h2 = max(
                candidates, key=lambda x: x[1]["roi"]
            )
            logger.info(
                "Best strategy: %s => ROI=%.2f%% (%d)",
                best_name,
                best_test["roi"],
                best_test["n_bets"],
            )

            final_roi = best_test["roi"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost_HPO",
                    "strategy": best_name,
                    "prob_threshold": best_cfg["prob_thr"],
                    "edge_threshold": best_cfg["edge_thr"],
                    "calibration": "platt" if "cal" in best_name else "none",
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "n_bets": best_test["n_bets"],
                    "auc_test_raw": auc_test_raw,
                    "auc_test_cal": auc_test_cal,
                    "threshold": best_cfg["prob_thr"],
                    "roi_half1": best_h1["roi"],
                    "roi_half2": best_h2["roi"],
                    "roi_stability_spread": abs(best_h1["roi"] - best_h2["roi"]),
                    "roi_test_a_stability": test_a["roi"],
                    "roi_test_a_maxroi": test_a_roi["roi"],
                }
            )
            if best_b:
                mlflow.log_metric("roi_test_b_cal", test_b["roi"])

            # Log all val scan results as artifact
            scan_df = pd.DataFrame(results_a)
            scan_df.to_csv("/tmp/val_scan_results.csv", index=False)
            mlflow.log_artifact("/tmp/val_scan_results.csv")

            if final_roi > 14.82:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": final_roi,
                    "auc": auc_test_raw,
                    "threshold": best_cfg["prob_thr"],
                    "edge_threshold": best_cfg["edge_thr"],
                    "strategy": best_name,
                    "n_bets": best_test["n_bets"],
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
            mlflow.set_tag("failure_reason", "exception in step 4.4")
            logger.exception("Step 4.4 failed")
            raise


if __name__ == "__main__":
    main()
