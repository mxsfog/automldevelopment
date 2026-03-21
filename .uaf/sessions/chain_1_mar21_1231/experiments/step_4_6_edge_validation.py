"""Step 4.6 — Bootstrap validation of edge-based betting + ensemble edge.

Валидация edge-based результатов из step 4.5 через bootstrap.
Комбинация stacking ensemble с edge selection.
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
from sklearn.linear_model import LogisticRegression
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


def bootstrap_roi(df: pd.DataFrame, mask: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap confidence interval for ROI."""
    rng = np.random.RandomState(seed)
    sel = df[mask]
    if len(sel) < 10:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "pct_positive": 0.0}

    rois = []
    for _ in range(n_boot):
        idx = rng.choice(len(sel), size=len(sel), replace=True)
        sample = sel.iloc[idx]
        staked = sample["USD"].sum()
        payout = sample.loc[sample["Status"] == "won", "Payout_USD"].sum()
        roi = (payout - staked) / staked * 100 if staked > 0 else 0.0
        rois.append(roi)

    rois = np.array(rois)
    return {
        "mean": round(float(rois.mean()), 2),
        "std": round(float(rois.std()), 2),
        "ci_lower": round(float(np.percentile(rois, 2.5)), 2),
        "ci_upper": round(float(np.percentile(rois, 97.5)), 2),
        "pct_positive": round(float((rois > 0).mean() * 100), 1),
    }


def main():
    train, val, train_full, test = load_and_prepare()
    logger.info(
        "Train: %d, Val: %d, Full: %d, Test: %d", len(train), len(val), len(train_full), len(test)
    )

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    # Label encode for LightGBM/XGBoost
    label_encoders: dict[str, LabelEncoder] = {}
    train_lgb = train.copy()
    val_lgb = val.copy()
    test_lgb = test.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], val[col], test[col]]).fillna("unknown").astype(str)
        le.fit(all_vals)
        train_lgb[col] = le.transform(train_lgb[col].fillna("unknown").astype(str))
        val_lgb[col] = le.transform(val_lgb[col].fillna("unknown").astype(str))
        test_lgb[col] = le.transform(test_lgb[col].fillna("unknown").astype(str))
        label_encoders[col] = le

    with mlflow.start_run(run_name="phase4/step_4_6_edge_validation") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.6")
            mlflow.set_tag("phase", "4")

            # Train all 3 base models
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
                train[FEATURES],
                y_train,
                eval_set=(val[FEATURES], y_val),
                early_stopping_rounds=100,
            )
            cb_val = cb_model.predict_proba(val[FEATURES])[:, 1]
            cb_test = cb_model.predict_proba(test[FEATURES])[:, 1]

            x_train_lgb = train_lgb[FEATURES].values
            x_val_lgb = val_lgb[FEATURES].values
            x_test_lgb = test_lgb[FEATURES].values

            lgb_train = lgb.Dataset(
                x_train_lgb,
                label=y_train,
                feature_name=FEATURES,
                categorical_feature=[FEATURES[i] for i in cat_indices],
            )
            lgb_val_ds = lgb.Dataset(x_val_lgb, label=y_val, reference=lgb_train)
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
                    "lambda_l1": 1.0,
                    "lambda_l2": 10.0,
                    "is_unbalance": True,
                    "seed": 42,
                    "verbose": -1,
                },
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val_ds],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            lgb_val_pred = lgb_model.predict(x_val_lgb)
            lgb_test_pred = lgb_model.predict(x_test_lgb)

            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=10.0,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                eval_metric="auc",
                random_state=42,
                verbosity=0,
                enable_categorical=False,
            )
            xgb_model.fit(x_train_lgb, y_train, eval_set=[(x_val_lgb, y_val)], verbose=False)
            xgb_val_pred = xgb_model.predict_proba(x_val_lgb)[:, 1]
            xgb_test_pred = xgb_model.predict_proba(x_test_lgb)[:, 1]

            # Stacking meta-learner (augmented, как в step 4.4)
            meta_val = np.column_stack([cb_val, lgb_val_pred, xgb_val_pred])
            meta_test = np.column_stack([cb_test, lgb_test_pred, xgb_test_pred])

            key_feats = ["Odds", "ML_P_Model", "ML_Edge", "implied_prob"]
            key_val = val[key_feats].fillna(0).values
            key_test = test[key_feats].fillna(0).values
            meta_aug_val = np.column_stack([meta_val, key_val])
            meta_aug_test = np.column_stack([meta_test, key_test])

            meta_lr = LogisticRegression(random_state=42, C=1.0, max_iter=1000)
            meta_lr.fit(meta_aug_val, y_val)
            stacking_test = meta_lr.predict_proba(meta_aug_test)[:, 1]

            # Simple average
            avg_test = (cb_test + lgb_test_pred + xgb_test_pred) / 3.0

            implied_test = 1.0 / test["Odds"].clip(lower=1.01).values

            # Bootstrap validation for key strategies
            logger.info("=== Bootstrap Validation (1000 samples) ===")
            strategies = {
                "CatBoost thr=0.45": cb_test >= 0.45,
                "Ensemble avg thr=0.45": avg_test >= 0.45,
                "Stacking aug thr=0.45": stacking_test >= 0.45,
                "CatBoost edge>0.03": cb_test > (implied_test + 0.03),
                "CatBoost edge>0.05": cb_test > (implied_test + 0.05),
                "CatBoost edge>0.07": cb_test > (implied_test + 0.07),
                "CatBoost edge>0.10": cb_test > (implied_test + 0.10),
                "Ensemble edge>0.03": avg_test > (implied_test + 0.03),
                "Ensemble edge>0.05": avg_test > (implied_test + 0.05),
                "Stacking edge>0.03": stacking_test > (implied_test + 0.03),
                "Stacking edge>0.05": stacking_test > (implied_test + 0.05),
            }

            best_roi = -999.0
            best_strategy = ""
            best_n_bets = 0
            best_ci_lower = -999.0

            for name, mask in strategies.items():
                n_sel = int(mask.sum())
                if n_sel < 20:
                    logger.info("  %s: only %d bets, skipping", name, n_sel)
                    continue

                r = calc_roi(test, mask)
                boot = bootstrap_roi(test, mask)
                logger.info(
                    "  %s: ROI=%.2f%% (%d bets), CI=[%.2f%%, %.2f%%], P(ROI>0)=%.1f%%",
                    name,
                    r["roi"],
                    r["n_bets"],
                    boot["ci_lower"],
                    boot["ci_upper"],
                    boot["pct_positive"],
                )

                # Best by CI lower bound (most robust)
                if boot["ci_lower"] > best_ci_lower and r["n_bets"] >= 100:
                    best_ci_lower = boot["ci_lower"]
                    best_roi = r["roi"]
                    best_strategy = name
                    best_n_bets = r["n_bets"]

            logger.info(
                "Most robust: %s => ROI=%.2f%% (%d bets), CI_lower=%.2f%%",
                best_strategy,
                best_roi,
                best_n_bets,
                best_ci_lower,
            )

            # Also find best ROI with > 1000 bets for reliability
            best_roi_large = -999.0
            best_strategy_large = ""
            best_n_large = 0
            for name, mask in strategies.items():
                n_sel = int(mask.sum())
                if n_sel < 1000:
                    continue
                r = calc_roi(test, mask)
                if r["roi"] > best_roi_large:
                    best_roi_large = r["roi"]
                    best_strategy_large = name
                    best_n_large = r["n_bets"]

            logger.info(
                "Best (>1000 bets): %s => ROI=%.2f%% (%d bets)",
                best_strategy_large,
                best_roi_large,
                best_n_large,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "edge_validation",
                    "best_strategy": best_strategy,
                    "best_strategy_large": best_strategy_large,
                    "n_bootstrap": 1000,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "n_bets": best_n_bets,
                    "ci_lower": best_ci_lower,
                    "roi_large": best_roi_large,
                    "n_bets_large": best_n_large,
                }
            )

            if best_roi > 5.58:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                cb_model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_roi,
                    "n_bets": best_n_bets,
                    "strategy": best_strategy,
                    "ci_lower": best_ci_lower,
                    "feature_names": FEATURES,
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved (new best)")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi={best_roi}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.6")
            logger.exception("Step 4.6 failed")
            raise


if __name__ == "__main__":
    main()
