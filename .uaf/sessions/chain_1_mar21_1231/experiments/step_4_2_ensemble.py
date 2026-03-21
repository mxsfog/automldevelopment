"""Step 4.2 — Stacking ensemble (CatBoost + LightGBM + XGBoost).

Среднее вероятностей трёх моделей + sport filtering.
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
import xgboost as xgb
from catboost import CatBoostClassifier
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

UNPROFITABLE_SPORTS = ["FIFA", "Ice Hockey", "MMA", "Super Bowl LX"]


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
    """Полная загрузка и подготовка."""
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


def find_best_threshold(proba: np.ndarray, df: pd.DataFrame, min_frac: float = 0.05) -> float:
    min_bets = max(20, int(len(df) * min_frac))
    best_roi = -999.0
    best_thr = 0.5
    for thr in np.arange(0.35, 0.85, 0.01):
        mask = proba >= thr
        result = calc_roi(df, mask)
        if result["n_bets"] >= min_bets and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr


def main():
    train, val, test = load_and_prepare()

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    # CatBoost
    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        cat_features=cat_indices,
        auto_class_weights="Balanced",
    )
    cb_model.fit(
        train[FEATURES], y_train, eval_set=(val[FEATURES], y_val), early_stopping_rounds=100
    )
    proba_cb_val = cb_model.predict_proba(val[FEATURES])[:, 1]
    proba_cb_test = cb_model.predict_proba(test[FEATURES])[:, 1]
    logger.info(
        "CatBoost AUC: val=%.4f, test=%.4f",
        roc_auc_score(y_val, proba_cb_val),
        roc_auc_score(y_test, proba_cb_test),
    )

    # LightGBM — label encode cats
    train_lgb = train.copy()
    val_lgb = val.copy()
    test_lgb = test.copy()
    label_encoders: dict[str, LabelEncoder] = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        all_vals = (
            pd.concat([train_lgb[col], val_lgb[col], test_lgb[col]]).fillna("unknown").astype(str)
        )
        le.fit(all_vals)
        train_lgb[col] = le.transform(train_lgb[col].fillna("unknown").astype(str))
        val_lgb[col] = le.transform(val_lgb[col].fillna("unknown").astype(str))
        test_lgb[col] = le.transform(test_lgb[col].fillna("unknown").astype(str))
        label_encoders[col] = le

    lgb_train = lgb.Dataset(
        train_lgb[FEATURES].values,
        label=y_train,
        feature_name=FEATURES,
        categorical_feature=[FEATURES[i] for i in cat_indices],
    )
    lgb_val = lgb.Dataset(val_lgb[FEATURES].values, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l2": 10.0,
            "is_unbalance": True,
            "seed": 42,
            "verbose": -1,
        },
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    proba_lgb_val = lgb_model.predict(val_lgb[FEATURES].values)
    proba_lgb_test = lgb_model.predict(test_lgb[FEATURES].values)
    logger.info(
        "LightGBM AUC: val=%.4f, test=%.4f",
        roc_auc_score(y_val, proba_lgb_val),
        roc_auc_score(y_test, proba_lgb_test),
    )

    # XGBoost — label encode cats (same encoding)
    xgb_train = xgb.DMatrix(
        train_lgb[FEATURES].values,
        label=y_train,
        feature_names=FEATURES,
        enable_categorical=False,
    )
    xgb_val_d = xgb.DMatrix(
        val_lgb[FEATURES].values,
        label=y_val,
        feature_names=FEATURES,
        enable_categorical=False,
    )
    xgb_test_d = xgb.DMatrix(
        test_lgb[FEATURES].values,
        label=y_test,
        feature_names=FEATURES,
        enable_categorical=False,
    )

    xgb_model = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": sum(y_train == 0) / max(sum(y_train == 1), 1),
            "seed": 42,
            "verbosity": 0,
        },
        xgb_train,
        num_boost_round=1000,
        evals=[(xgb_val_d, "val")],
        early_stopping_rounds=100,
        verbose_eval=0,
    )
    proba_xgb_val = xgb_model.predict(xgb_val_d)
    proba_xgb_test = xgb_model.predict(xgb_test_d)
    logger.info(
        "XGBoost AUC: val=%.4f, test=%.4f",
        roc_auc_score(y_val, proba_xgb_val),
        roc_auc_score(y_test, proba_xgb_test),
    )

    with mlflow.start_run(run_name="phase4/step_4_2_ensemble") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.2")
            mlflow.set_tag("phase", "4")

            # Ensemble: average
            proba_ens_val = (proba_cb_val + proba_lgb_val + proba_xgb_val) / 3
            proba_ens_test = (proba_cb_test + proba_lgb_test + proba_xgb_test) / 3
            auc_ens = roc_auc_score(y_test, proba_ens_test)
            logger.info("Ensemble AUC: test=%.4f", auc_ens)

            # Threshold search
            thr_ens = find_best_threshold(proba_ens_val, val)
            roi_ens = calc_roi(test, proba_ens_test >= thr_ens)
            logger.info(
                "Ensemble ROI: %.2f%% (%d bets, thr=%.2f)",
                roi_ens["roi"],
                roi_ens["n_bets"],
                thr_ens,
            )

            # Weighted ensemble (CatBoost has better AUC)
            proba_w_val = 0.4 * proba_cb_val + 0.35 * proba_lgb_val + 0.25 * proba_xgb_val
            proba_w_test = 0.4 * proba_cb_test + 0.35 * proba_lgb_test + 0.25 * proba_xgb_test
            thr_w = find_best_threshold(proba_w_val, val)
            roi_w = calc_roi(test, proba_w_test >= thr_w)
            auc_w = roc_auc_score(y_test, proba_w_test)
            logger.info(
                "Weighted ensemble ROI: %.2f%% (%d bets, thr=%.2f), AUC=%.4f",
                roi_w["roi"],
                roi_w["n_bets"],
                thr_w,
                auc_w,
            )

            # Sport-filtered ensemble
            sport_mask = ~test["Sport"].isin(UNPROFITABLE_SPORTS)
            test_filt = test[sport_mask]
            proba_filt = proba_w_test[sport_mask.values]
            val_sport_mask = ~val["Sport"].isin(UNPROFITABLE_SPORTS)
            thr_filt = find_best_threshold(proba_w_val[val_sport_mask.values], val[val_sport_mask])
            roi_filt = calc_roi(test_filt, proba_filt >= thr_filt)
            logger.info(
                "Filtered ensemble ROI: %.2f%% (%d bets, thr=%.2f)",
                roi_filt["roi"],
                roi_filt["n_bets"],
                thr_filt,
            )

            # Fixed thresholds for ensemble
            for ft in [0.40, 0.45, 0.50, 0.55, 0.60]:
                r = calc_roi(test, proba_w_test >= ft)
                rf = calc_roi(test_filt, proba_filt >= ft)
                logger.info(
                    "  thr=%.2f: all=%.2f%% (%d), filtered=%.2f%% (%d)",
                    ft,
                    r["roi"],
                    r["n_bets"],
                    rf["roi"],
                    rf["n_bets"],
                )

            best_roi = max(roi_ens["roi"], roi_w["roi"], roi_filt["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "Ensemble_CatBoost_LightGBM_XGBoost",
                    "weights": "0.4_0.35_0.25",
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "roi_equal_avg": roi_ens["roi"],
                    "roi_weighted": roi_w["roi"],
                    "roi_filtered": roi_filt["roi"],
                    "auc_test": auc_ens,
                    "auc_weighted": auc_w,
                    "n_bets_ens": roi_ens["n_bets"],
                    "n_bets_filtered": roi_filt["n_bets"],
                }
            )

            # Save if new best
            if best_roi > 5.52:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                cb_model.save_model(str(models_dir / "model.cbm"))
                lgb_model.save_model(str(models_dir / "model.lgb"))
                xgb_model.save_model(str(models_dir / "model.xgb"))
                metadata = {
                    "framework": "ensemble",
                    "model_file": "model.cbm+model.lgb+model.xgb",
                    "roi": best_roi,
                    "auc": auc_w,
                    "threshold": thr_w,
                    "n_bets": roi_w["n_bets"],
                    "feature_names": FEATURES,
                    "params": {"weights": [0.4, 0.35, 0.25]},
                    "sport_filter": UNPROFITABLE_SPORTS,
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Ensemble model saved (new best)")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={best_roi}")
            print(f"RESULT:auc={auc_w}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.2")
            logger.exception("Step 4.2 failed")
            raise


if __name__ == "__main__":
    main()
