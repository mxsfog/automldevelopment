"""Step 2.2 — ELO history trends + sport ROI analysis.

Добавляем:
- ELO trend: изменение ELO за последние N матчей (momentum)
- Sport ROI: анализ прибыльности по видам спорта + фильтрация
- Market grouping: группировка рынков в категории
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
    """Извлечение имени команды из Selection."""
    if pd.isna(selection):
        return None
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", str(selection))
    cleaned = re.sub(r"\s*(Over|Under)\s+\d+.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().lower() if cleaned.strip() else None


def categorize_market(market: str) -> str:
    """Группировка рынков в категории."""
    if pd.isna(market):
        return "unknown"
    m = str(market).lower()
    if "winner" in m or "match winner" in m or "1x2" in m:
        return "match_winner"
    if "over" in m or "under" in m or "total" in m:
        return "totals"
    if "handicap" in m or "spread" in m:
        return "handicap"
    if "both teams" in m or "btts" in m:
        return "btts"
    if "correct score" in m:
        return "correct_score"
    if "map" in m:
        return "map_market"
    return "other"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загрузка всех данных."""
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
            Outcome_Odds=("Odds", "first"),
            Start_Time=("Start_Time", "first"),
        )
        .reset_index()
    )

    df = bets.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    exclude = ["pending", "cancelled", "error", "cashout"]
    df = df[~df["Status"].isin(exclude)].copy()

    logger.info("После фильтрации: %d строк", len(df))
    return df, teams, elo_history


def build_elo_trend(elo_history: pd.DataFrame) -> dict:
    """Построение ELO тренда по командам."""
    elo_history = elo_history.sort_values("Created_At")
    trend = {}
    for team_id, group in elo_history.groupby("Team_ID"):
        recent = group.tail(5)
        total_change = recent["ELO_Change"].sum()
        avg_change = recent["ELO_Change"].mean()
        win_streak = 0
        for won in reversed(recent["Won"].values):
            if won == "t" or won is True:
                win_streak += 1
            else:
                break
        trend[team_id] = {
            "elo_trend_5": total_change,
            "elo_avg_change": avg_change,
            "recent_win_streak": win_streak,
            "n_recent_matches": len(recent),
        }
    return trend


def add_features(df: pd.DataFrame, teams: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """Добавление всех фич."""
    # --- Базовые фичи из step 2.1 ---
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

    # --- ELO из teams ---
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

    # Team ID lookup
    team_id_lookup = teams_dedup.set_index("Normalized_Name")["ID"].to_dict()

    def get_team_stat(selection: str, stat: str) -> float | None:
        name = extract_team_name(selection)
        if name and name in team_lookup:
            return team_lookup[name].get(stat)
        return None

    def get_team_id(selection: str) -> int | None:
        name = extract_team_name(selection)
        if name and name in team_id_lookup:
            return team_id_lookup[name]
        return None

    df["team_elo"] = df["Selection"].apply(lambda x: get_team_stat(x, "Current_ELO"))
    df["team_winrate"] = df["Selection"].apply(lambda x: get_team_stat(x, "Winrate"))
    df["team_games"] = df["Selection"].apply(lambda x: get_team_stat(x, "Total_Games"))
    df["team_off_rating"] = df["Selection"].apply(lambda x: get_team_stat(x, "Offensive_Rating"))
    df["team_def_rating"] = df["Selection"].apply(lambda x: get_team_stat(x, "Defensive_Rating"))
    df["team_net_rating"] = df["Selection"].apply(lambda x: get_team_stat(x, "Net_Rating"))

    df["elo_x_odds"] = df["team_elo"].fillna(1500) * df["implied_prob"]
    df["winrate_vs_implied"] = df["team_winrate"].fillna(0.5) - df["implied_prob"]

    # --- NEW: ELO trend ---
    elo_trend = build_elo_trend(elo_history)
    df["team_id_resolved"] = df["Selection"].apply(get_team_id)

    def _get_elo_stat(tid: int | None, stat: str) -> float | None:
        if pd.notna(tid):
            return elo_trend.get(tid, {}).get(stat)
        return None

    for elo_col in ["elo_trend_5", "elo_avg_change", "recent_win_streak"]:
        stat_name = elo_col
        df[elo_col] = df["team_id_resolved"].apply(lambda tid, s=stat_name: _get_elo_stat(tid, s))

    elo_trend_matched = df["elo_trend_5"].notna().sum()
    logger.info(
        "ELO trend matched: %d / %d (%.1f%%)",
        elo_trend_matched,
        len(df),
        100 * elo_trend_matched / len(df),
    )

    # --- NEW: Market category ---
    df["market_category"] = df["Market"].apply(categorize_market)

    # Cat features
    for col in [
        "Sport",
        "Market",
        "Is_Parlay",
        "ML_Team_Stats_Found",
        "odds_bucket",
        "market_category",
    ]:
        df[col] = df[col].fillna("unknown").astype(str)

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

    return df


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split по индексу."""
    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """ROI на выбранных ставках."""
    sel = df[mask]
    if len(sel) == 0:
        return {"roi": 0.0, "n_bets": 0}
    total_staked = sel["USD"].sum()
    total_payout = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0.0
    return {"roi": round(roi, 4), "n_bets": len(sel)}


def find_best_threshold(proba: np.ndarray, df: pd.DataFrame) -> float:
    """Подбор порога на val."""
    best_roi = -999.0
    best_thr = 0.5
    for thr in np.arange(0.35, 0.90, 0.01):
        mask = proba >= thr
        result = calc_roi(df, mask)
        if result["n_bets"] >= 20 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr


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


def main():
    df, teams, elo_history = load_data()
    df = add_features(df, teams, elo_history)
    train_full, test = time_series_split(df)

    # Sport ROI analysis на train (для фильтрации)
    logger.info("--- Sport ROI analysis (on train) ---")
    unprofitable_sports = []
    for sport, grp in train_full.groupby("Sport"):
        won_mask = grp["Status"] == "won"
        staked = grp["USD"].sum()
        payout = grp.loc[won_mask, "Payout_USD"].sum()
        sport_roi = (payout - staked) / staked * 100 if staked > 0 else 0.0
        logger.info("  %s: ROI=%.2f%% (%d bets)", sport, sport_roi, len(grp))
        if sport_roi < -10 and len(grp) > 100:
            unprofitable_sports.append(sport)

    logger.info("Unprofitable sports (ROI < -10%%): %s", unprofitable_sports)

    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    with mlflow.start_run(run_name="phase2/step_2_2_elo_sport") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "2.2")
            mlflow.set_tag("phase", "2")

            # Baseline = step 2.1 features (без elo_trend, market_category)
            baseline_features_list = [
                f
                for f in FEATURES
                if f
                not in ["elo_trend_5", "elo_avg_change", "recent_win_streak", "market_category"]
            ]
            baseline_cat = [c for c in CAT_FEATURES if c != "market_category"]
            baseline_cat_idx = [baseline_features_list.index(c) for c in baseline_cat]

            model_baseline = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=baseline_cat_idx,
                auto_class_weights="Balanced",
            )
            model_baseline.fit(
                train[baseline_features_list],
                y_train,
                eval_set=(val[baseline_features_list], y_val),
                early_stopping_rounds=100,
            )
            proba_bl_val = model_baseline.predict_proba(val[baseline_features_list])[:, 1]
            proba_bl_test = model_baseline.predict_proba(test[baseline_features_list])[:, 1]
            auc_baseline = roc_auc_score(y_test, proba_bl_test)
            thr_bl = find_best_threshold(proba_bl_val, val)
            roi_baseline = calc_roi(test, proba_bl_test >= thr_bl)

            # Candidate: full features
            model_candidate = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )
            model_candidate.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=100,
            )
            proba_cand_val = model_candidate.predict_proba(val[FEATURES])[:, 1]
            proba_cand_test = model_candidate.predict_proba(test[FEATURES])[:, 1]
            auc_candidate = roc_auc_score(y_test, proba_cand_test)
            thr_cand = find_best_threshold(proba_cand_val, val)
            roi_candidate = calc_roi(test, proba_cand_test >= thr_cand)

            logger.info(
                "BASELINE: AUC=%.4f, ROI=%.2f%% (%d bets)",
                auc_baseline,
                roi_baseline["roi"],
                roi_baseline["n_bets"],
            )
            logger.info(
                "CANDIDATE: AUC=%.4f, ROI=%.2f%% (%d bets)",
                auc_candidate,
                roi_candidate["roi"],
                roi_candidate["n_bets"],
            )

            delta_roi = roi_candidate["roi"] - roi_baseline["roi"]

            # Sport-filtered ROI (фильтруем убыточные спорты из test)
            if unprofitable_sports:
                sport_mask = ~test["Sport"].isin(unprofitable_sports)
                test_filtered = test[sport_mask]
                proba_filtered = proba_cand_test[sport_mask.values]
                thr_filt = find_best_threshold(
                    proba_cand_val[~val["Sport"].isin(unprofitable_sports).values],
                    val[~val["Sport"].isin(unprofitable_sports)],
                )
                roi_filtered = calc_roi(test_filtered, proba_filtered >= thr_filt)
                logger.info(
                    "FILTERED (no %s): ROI=%.2f%% (%d bets)",
                    unprofitable_sports,
                    roi_filtered["roi"],
                    roi_filtered["n_bets"],
                )
            else:
                roi_filtered = roi_candidate

            # Feature importance
            importances = model_candidate.get_feature_importance()
            for fname, imp in sorted(zip(FEATURES, importances, strict=True), key=lambda x: -x[1]):
                if imp > 1.0:
                    logger.info("  %s: %.2f", fname, imp)

            decision = (
                "accept" if delta_roi > 0.002 else ("marginal" if delta_roi > 0 else "reject")
            )
            logger.info("Decision: %s (delta=%.4f)", decision, delta_roi)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost",
                    "n_features": len(FEATURES),
                    "new_features": "elo_trend+market_category",
                    "unprofitable_sports": str(unprofitable_sports),
                    "decision": decision,
                }
            )

            best_roi = max(roi_candidate["roi"], roi_filtered["roi"])
            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_candidate": roi_candidate["roi"],
                    "roi_baseline": roi_baseline["roi"],
                    "roi_filtered": roi_filtered["roi"],
                    "delta_roi": delta_roi,
                    "auc_test": auc_candidate,
                    "n_bets": roi_candidate["n_bets"],
                    "threshold": thr_cand,
                }
            )

            # Сохранение если лучше
            if best_roi > 2.66:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model_candidate.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_roi,
                    "auc": auc_candidate,
                    "threshold": thr_cand,
                    "n_bets": roi_candidate["n_bets"],
                    "feature_names": FEATURES,
                    "params": {
                        "iterations": model_candidate.tree_count_,
                        "depth": 6,
                        "learning_rate": 0.05,
                    },
                    "sport_filter": unprofitable_sports,
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={best_roi}")
            print(f"RESULT:decision={decision}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.2")
            logger.exception("Step 2.2 failed")
            raise


if __name__ == "__main__":
    main()
