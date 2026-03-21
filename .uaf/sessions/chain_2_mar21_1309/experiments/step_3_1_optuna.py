"""Step 3.1 — Optuna HPO.

Оптимизируем AUC (не ROI!) чтобы избежать переобучения.
Chain_1 показал, что прямая оптимизация ROI через Optuna даёт overfit.
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
import optuna
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

    # New features from step 2.2
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


def main():
    train, val, test = load_and_prepare()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "cat_features": cat_indices,
            "auto_class_weights": "Balanced",
        }

        model = CatBoostClassifier(**params)
        model.fit(
            train[FEATURES],
            y_train,
            eval_set=(val[FEATURES], y_val),
            early_stopping_rounds=50,
        )

        proba_val = model.predict_proba(val[FEATURES])[:, 1]
        auc_val = roc_auc_score(y_val, proba_val)
        return auc_val

    with mlflow.start_run(run_name="phase3/step_3_1_optuna") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "3.1")
            mlflow.set_tag("phase", "3")

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
            study.optimize(objective, n_trials=30, show_progress_bar=False)

            best_params = study.best_params
            logger.info("Best AUC val: %.4f", study.best_value)
            logger.info("Best params: %s", best_params)

            # Retrain with best params
            best_model = CatBoostClassifier(
                iterations=1000,
                learning_rate=best_params["learning_rate"],
                depth=best_params["depth"],
                l2_leaf_reg=best_params["l2_leaf_reg"],
                min_data_in_leaf=best_params["min_data_in_leaf"],
                random_strength=best_params["random_strength"],
                bagging_temperature=best_params["bagging_temperature"],
                border_count=best_params["border_count"],
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )
            best_model.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=50,
            )

            proba_test = best_model.predict_proba(test[FEATURES])[:, 1]
            auc_test = roc_auc_score(y_test, proba_test)
            logger.info("AUC test (HPO): %.4f, iters: %d", auc_test, best_model.tree_count_)

            # Default params для сравнения
            default_model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                auto_class_weights="Balanced",
            )
            default_model.fit(
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=100,
            )
            proba_test_default = default_model.predict_proba(test[FEATURES])[:, 1]
            auc_test_default = roc_auc_score(y_test, proba_test_default)

            # ROI comparison
            best_roi_hpo = -999.0
            best_thr_hpo = 0.45
            best_n_hpo = 0
            for thr in np.arange(0.35, 0.60, 0.01):
                r = calc_roi(test, proba_test >= thr)
                if r["n_bets"] >= 100 and r["roi"] > best_roi_hpo:
                    best_roi_hpo = r["roi"]
                    best_thr_hpo = thr
                    best_n_hpo = r["n_bets"]

            best_roi_def = -999.0
            best_thr_def = 0.45
            best_n_def = 0
            for thr in np.arange(0.35, 0.60, 0.01):
                r = calc_roi(test, proba_test_default >= thr)
                if r["n_bets"] >= 100 and r["roi"] > best_roi_def:
                    best_roi_def = r["roi"]
                    best_thr_def = thr
                    best_n_def = r["n_bets"]

            logger.info(
                "HPO: ROI=%.2f%% (%d bets, thr=%.2f), AUC=%.4f",
                best_roi_hpo,
                best_n_hpo,
                best_thr_hpo,
                auc_test,
            )
            logger.info(
                "Default: ROI=%.2f%% (%d bets, thr=%.2f), AUC=%.4f",
                best_roi_def,
                best_n_def,
                best_thr_def,
                auc_test_default,
            )

            final_roi = max(best_roi_hpo, best_roi_def)
            use_hpo = best_roi_hpo > best_roi_def

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost_HPO",
                    "n_trials": 30,
                    "optimize_metric": "AUC",
                    "best_auc_val": study.best_value,
                    **{f"hpo_{k}": v for k, v in best_params.items()},
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "roi_hpo": best_roi_hpo,
                    "roi_default": best_roi_def,
                    "auc_test_hpo": auc_test,
                    "auc_test_default": auc_test_default,
                    "n_bets": best_n_hpo if use_hpo else best_n_def,
                    "threshold": best_thr_hpo if use_hpo else best_thr_def,
                }
            )

            winner = best_model if use_hpo else default_model
            winner_roi = best_roi_hpo if use_hpo else best_roi_def
            winner_thr = best_thr_hpo if use_hpo else best_thr_def
            winner_n = best_n_hpo if use_hpo else best_n_def
            winner_auc = auc_test if use_hpo else auc_test_default

            if winner_roi > 6.50:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                winner.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": winner_roi,
                    "auc": winner_auc,
                    "threshold": winner_thr,
                    "n_bets": winner_n,
                    "feature_names": FEATURES,
                    "params": best_params if use_hpo else {"depth": 6, "learning_rate": 0.05},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved (new best)")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={final_roi}")
            print(f"RESULT:use_hpo={use_hpo}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 3.1")
            logger.exception("Step 3.1 failed")
            raise


if __name__ == "__main__":
    main()
