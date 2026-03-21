"""Step 4.1 — LightGBM + robust threshold optimization.

LightGBM как альтернатива CatBoost + более робастный подбор порога:
- Порог по ROI на val, но с ограничением n_bets >= 5% от val
- Также пробуем фиксированные пороги для стабильности
"""

import json
import logging
import os
import random
import re
import sys
import traceback
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

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


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полная загрузка и подготовка данных."""
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

    # Features
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

    # ELO
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

    # Split
    df = df.sort_values("Created_At").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_full = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()

    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))
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


def find_best_threshold_robust(
    proba: np.ndarray, df: pd.DataFrame, min_bets_frac: float = 0.05
) -> tuple[float, float]:
    """Робастный подбор порога: минимум 5% ставок от выборки."""
    min_bets = max(20, int(len(df) * min_bets_frac))
    best_roi = -999.0
    best_thr = 0.5
    for thr in np.arange(0.35, 0.85, 0.01):
        mask = proba >= thr
        result = calc_roi(df, mask)
        if result["n_bets"] >= min_bets and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr, best_roi


def main():
    train, val, test = load_and_prepare()

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    # Label encode categorical features for LightGBM
    label_encoders: dict[str, LabelEncoder] = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], val[col], test[col]]).fillna("unknown").astype(str)
        le.fit(all_vals)
        train[col] = le.transform(train[col].fillna("unknown").astype(str))
        val[col] = le.transform(val[col].fillna("unknown").astype(str))
        test[col] = le.transform(test[col].fillna("unknown").astype(str))
        label_encoders[col] = le

    x_train = train[FEATURES].values
    x_val = val[FEATURES].values
    x_test = test[FEATURES].values

    with mlflow.start_run(run_name="phase4/step_4_1_lgbm") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.1")
            mlflow.set_tag("phase", "4")

            cat_feature_indices = [FEATURES.index(c) for c in CAT_FEATURES]

            train_data = lgb.Dataset(
                x_train,
                label=y_train,
                feature_name=FEATURES,
                categorical_feature=[FEATURES[i] for i in cat_feature_indices],
            )
            val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": 8,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 1.0,
                "lambda_l2": 10.0,
                "is_unbalance": True,
                "seed": 42,
                "verbose": -1,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
            )

            proba_val = model.predict(x_val)
            proba_test = model.predict(x_test)

            auc_val = roc_auc_score(y_val, proba_val)
            auc_test = roc_auc_score(y_test, proba_test)
            logger.info("AUC: val=%.4f, test=%.4f", auc_val, auc_test)

            # Robust threshold
            threshold, val_roi = find_best_threshold_robust(proba_val, val)
            logger.info("Best threshold (robust): %.2f, val ROI: %.2f%%", threshold, val_roi)

            roi_test = calc_roi(test, proba_test >= threshold)
            logger.info(
                "Test ROI: %.2f%% (%d bets), threshold=%.2f",
                roi_test["roi"],
                roi_test["n_bets"],
                threshold,
            )

            # Фиксированные пороги
            for fixed_thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                r = calc_roi(test, proba_test >= fixed_thr)
                logger.info(
                    "  threshold=%.2f: ROI=%.2f%% (%d bets)", fixed_thr, r["roi"], r["n_bets"]
                )

            # Feature importance
            importance = model.feature_importance(importance_type="gain")
            for fname, imp in sorted(zip(FEATURES, importance, strict=True), key=lambda x: -x[1]):
                if imp > 100:
                    logger.info("  %s: %.0f", fname, imp)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "LightGBM",
                    "n_features": len(FEATURES),
                    "threshold": threshold,
                    "num_boost_round": model.best_iteration,
                    **{k: v for k, v in params.items() if k not in ["verbose"]},
                }
            )

            mlflow.log_metrics(
                {
                    "roi": roi_test["roi"],
                    "n_bets": roi_test["n_bets"],
                    "roi_val": val_roi,
                    "auc_val": auc_val,
                    "auc_test": auc_test,
                    "threshold": threshold,
                }
            )

            if roi_test["roi"] > 5.32:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(models_dir / "model.lgb"))
                metadata = {
                    "framework": "lgbm",
                    "model_file": "model.lgb",
                    "roi": roi_test["roi"],
                    "auc": auc_test,
                    "threshold": threshold,
                    "n_bets": roi_test["n_bets"],
                    "feature_names": FEATURES,
                    "params": params,
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
            print(f"RESULT:roi={roi_test['roi']}")
            print(f"RESULT:auc={auc_test}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.1")
            logger.exception("Step 4.1 failed")
            raise


if __name__ == "__main__":
    main()
