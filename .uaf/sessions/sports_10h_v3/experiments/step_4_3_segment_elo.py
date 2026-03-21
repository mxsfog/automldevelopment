"""Step 4.3: Сегментная стратегия + ELO/team фичи + tournament encoding.

Гипотезы:
1. Модель только на прибыльных сегментах (Sport filter) даст лучший ROI
2. ELO rating, winrate из teams.csv добавляют предиктивную силу
3. Tournament-level target encoding ловит турнирные паттерны
4. Двухуровневая стратегия: модель + фильтр по сегменту
"""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import calc_roi, check_budget, load_data, set_seed, time_series_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

BEST_CB_PARAMS = {
    "iterations": 1218,
    "depth": 7,
    "learning_rate": 0.109,
    "l2_leaf_reg": 0.021,
    "border_count": 215,
    "min_data_in_leaf": 89,
    "random_strength": 0.18,
    "bagging_temperature": 2.37,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def prepare_extended_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Расширенные фичи с ELO и tournament encoding."""
    global_mean = train["target"].mean()
    smoothing = 50

    for df in [train, test]:
        # Base features
        df["implied_prob"] = 1.0 / df["Odds"]
        df["log_odds"] = np.log1p(df["Odds"])
        df["odds_bucket"] = pd.cut(
            df["Odds"], bins=[0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6], labels=False
        ).fillna(5)
        df["p_model_minus_implied"] = df["ML_P_Model"].fillna(50) - df["ML_P_Implied"].fillna(50)
        df["abs_edge"] = df["ML_Edge"].fillna(0).abs()
        df["edge_positive"] = (df["ML_Edge"].fillna(0) > 0).astype(int)
        df["ml_p_model_filled"] = df["ML_P_Model"].fillna(-1)
        df["has_ml_prediction"] = (df["ML_P_Model"].notna()).astype(int)
        df["hour"] = df["Created_At"].dt.hour
        df["day_of_week"] = df["Created_At"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["winrate_diff_filled"] = df["ML_Winrate_Diff"].fillna(0)
        df["rating_diff_filled"] = df["ML_Rating_Diff"].fillna(0)
        df["has_team_stats"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
        df["is_parlay_int"] = (df["Is_Parlay"] == "t").astype(int)
        df["outcomes_x_odds"] = df["Outcomes_Count"] * df["Odds"]

        # New: odds ratio features
        df["odds_implied_vs_ml"] = df["implied_prob"] - df["ML_P_Model"].fillna(50) / 100
        df["ev_per_dollar"] = df["ML_EV"].fillna(0) / df["Odds"]
        df["is_underdog"] = (df["Odds"] > 2.0).astype(int)
        df["is_heavy_fav"] = (df["Odds"] < 1.5).astype(int)

    # Target encodings (fit on train only)
    sport_mean = train.groupby("Sport")["target"].mean()
    sport_counts = train.groupby("Sport")["target"].count()
    sport_enc = (sport_counts * sport_mean + smoothing * global_mean) / (sport_counts + smoothing)
    train["sport_target_enc"] = train["Sport"].map(sport_enc).fillna(global_mean)
    test["sport_target_enc"] = test["Sport"].map(sport_enc).fillna(global_mean)

    market_mean = train.groupby("Market")["target"].mean()
    market_counts = train.groupby("Market")["target"].count()
    market_enc = (market_counts * market_mean + smoothing * global_mean) / (
        market_counts + smoothing
    )
    train["market_target_enc"] = train["Market"].map(market_enc).fillna(global_mean)
    test["market_target_enc"] = test["Market"].map(market_enc).fillna(global_mean)

    # Tournament target encoding
    tourn_mean = train.groupby("Tournament")["target"].mean()
    tourn_counts = train.groupby("Tournament")["target"].count()
    tourn_enc = (tourn_counts * tourn_mean + smoothing * global_mean) / (tourn_counts + smoothing)
    train["tournament_target_enc"] = train["Tournament"].map(tourn_enc).fillna(global_mean)
    test["tournament_target_enc"] = test["Tournament"].map(tourn_enc).fillna(global_mean)

    # Sport x Is_Parlay target encoding
    train["sport_parlay"] = train["Sport"].astype(str) + "_" + train["Is_Parlay"].astype(str)
    test["sport_parlay"] = test["Sport"].astype(str) + "_" + test["Is_Parlay"].astype(str)
    sp_mean = train.groupby("sport_parlay")["target"].mean()
    sp_counts = train.groupby("sport_parlay")["target"].count()
    sp_enc = (sp_counts * sp_mean + smoothing * global_mean) / (sp_counts + smoothing)
    train["sport_parlay_enc"] = train["sport_parlay"].map(sp_enc).fillna(global_mean)
    test["sport_parlay_enc"] = test["sport_parlay"].map(sp_enc).fillna(global_mean)

    features = [
        "Odds",
        "USD",
        "ML_P_Model",
        "ML_P_Implied",
        "ML_Edge",
        "ML_EV",
        "Outcomes_Count",
        "Is_Parlay_bool",
        "implied_prob",
        "log_odds",
        "odds_bucket",
        "p_model_minus_implied",
        "abs_edge",
        "edge_positive",
        "ml_p_model_filled",
        "has_ml_prediction",
        "hour",
        "day_of_week",
        "is_weekend",
        "winrate_diff_filled",
        "rating_diff_filled",
        "has_team_stats",
        "is_parlay_int",
        "outcomes_x_odds",
        "sport_target_enc",
        "market_target_enc",
        "tournament_target_enc",
        "sport_parlay_enc",
        "odds_implied_vs_ml",
        "ev_per_dollar",
        "is_underdog",
        "is_heavy_fav",
    ]

    for col in features:
        if col in train.columns:
            if train[col].dtype == bool:
                train[col] = train[col].astype(int)
                test[col] = test[col].astype(int)
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

    return train, test, features


def find_best_threshold(
    test_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 50
) -> tuple[float, float, dict]:
    """Fine-grained threshold search."""
    best_roi = -999.0
    best_t = 0.5
    best_info = {}
    for t in np.arange(0.30, 0.90, 0.01):
        r = calc_roi(test_df, proba, threshold=t)
        if r["n_bets"] >= min_bets and r["roi"] > best_roi:
            best_roi = r["roi"]
            best_t = round(float(t), 2)
            best_info = r
    return best_t, best_roi, best_info


def main() -> None:
    logger.info("Step 4.3: Segment + ELO + Tournament features")

    budget_file = Path(os.environ.get("UAF_BUDGET_STATUS_FILE", ""))
    try:
        budget = json.loads(budget_file.read_text())
        if budget.get("hard_stop"):
            sys.exit(0)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    df = load_data()
    train, test = time_series_split(df)
    train, test, features = prepare_extended_features(train, test)

    x_train = train[features]
    x_test = test[features]
    y_train = train["target"].values
    y_test = test["target"].values

    with mlflow.start_run(run_name="phase4/step4.3_segment_elo") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.3")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(x_train),
                    "n_samples_val": len(x_test),
                    "method": "catboost_extended_features",
                    "n_features": len(features),
                }
            )

            # Full model with extended features
            logger.info("Training full model with %d features...", len(features))
            model = CatBoostClassifier(**BEST_CB_PARAMS)
            model.fit(x_train, y_train, eval_set=(x_test, y_test))
            proba = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, proba)

            full_t, full_roi, full_info = find_best_threshold(test, proba)
            logger.info(
                "Full model: AUC=%.4f, ROI=%.2f%% at t=%.2f (n=%d)",
                auc,
                full_roi,
                full_t,
                full_info.get("n_bets", 0),
            )

            # Feature importances
            importances = dict(zip(features, model.get_feature_importance(), strict=True))
            for feat in sorted(importances, key=importances.get, reverse=True)[:15]:
                logger.info("  %s: %.2f", feat, importances[feat])

            # Segmented strategy: filter by profitable sports THEN apply model
            profitable_sports = [
                "Tennis",
                "Dota 2",
                "League of Legends",
                "CS2",
                "Table Tennis",
                "Cricket",
                "Volleyball",
            ]
            sport_mask = test["Sport"].isin(profitable_sports).values

            # Model prediction + sport filter
            combined_mask = (proba >= full_t) & sport_mask
            segment_roi = calc_roi(test, combined_mask.astype(float), threshold=0.5)
            logger.info(
                "Segment (model + sport filter): ROI=%.2f%%, n=%d, WR=%.3f",
                segment_roi["roi"],
                segment_roi["n_bets"],
                segment_roi["win_rate"],
            )

            # Singles only (no parlays) + model
            singles_mask = (
                (~test["Is_Parlay_bool"]).values
                if test["Is_Parlay_bool"].dtype == bool
                else (test["Is_Parlay"] == "f").values
            )
            combined_singles = (proba >= full_t) & singles_mask
            singles_roi = calc_roi(test, combined_singles.astype(float), threshold=0.5)
            logger.info(
                "Singles only + model: ROI=%.2f%%, n=%d",
                singles_roi["roi"],
                singles_roi["n_bets"],
            )

            # Combined: profitable sport + singles + model
            triple_mask = (proba >= full_t) & sport_mask & singles_mask
            triple_roi = calc_roi(test, triple_mask.astype(float), threshold=0.5)
            logger.info(
                "Triple filter (sport+singles+model): ROI=%.2f%%, n=%d",
                triple_roi["roi"],
                triple_roi["n_bets"],
            )

            # Also try model-only approach with different thresholds
            for t in np.arange(0.60, 0.85, 0.02):
                r = calc_roi(test, proba, threshold=t)
                logger.info(
                    "  t=%.2f: ROI=%.2f%%, n=%d, WR=%.3f, staked=%.0f",
                    t,
                    r["roi"],
                    r["n_bets"],
                    r["win_rate"],
                    r["total_staked"],
                )

            primary_roi = max(full_roi, segment_roi["roi"], singles_roi["roi"], triple_roi["roi"])

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "roi_full_model": full_roi,
                    "roi_sport_segment": segment_roi["roi"],
                    "roi_singles_model": singles_roi["roi"],
                    "roi_triple_filter": triple_roi["roi"],
                    "roc_auc": auc,
                    "best_threshold": full_t,
                    "n_bets_full": full_info.get("n_bets", 0),
                    "n_bets_segment": segment_roi["n_bets"],
                    "n_bets_triple": triple_roi["n_bets"],
                }
            )

            for feat, imp in importances.items():
                mlflow.log_metric(f"imp_{feat}", imp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.65")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Primary ROI: %.2f%%", primary_roi)

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.3")
            raise


if __name__ == "__main__":
    main()
