"""Step 2.1 — Feature Engineering: time + odds-derived + ELO.

Добавляем:
- Временные: hour, day_of_week, is_weekend
- Odds-derived: implied_prob, value_ratio (ML_P / implied), log_odds
- ELO: team ELO, opponent ELO, ELO diff, winrate из teams.csv
- Interaction: edge_x_odds, value_x_sport
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


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных с ELO."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", parse_dates=["Created_At"])
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")

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
    return df, teams


def extract_team_name(selection: str) -> str | None:
    """Извлечение имени команды из Selection."""
    if pd.isna(selection):
        return None
    # Убираем handicap/over/under суффиксы
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", str(selection))
    cleaned = re.sub(r"\s*(Over|Under)\s+\d+.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().lower() if cleaned.strip() else None


def add_features(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Добавление всех инженерных фич."""
    # --- Временные фичи ---
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Odds-derived фичи ---
    df["implied_prob"] = 1.0 / df["Odds"]
    df["log_odds"] = np.log(df["Odds"].clip(lower=1.01))
    df["value_ratio"] = df["ML_P_Model"] / df["implied_prob"].clip(lower=0.01)
    df["edge_x_odds"] = df["ML_Edge"] * df["Odds"]
    df["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[0, 1.3, 1.6, 2.0, 2.5, 3.5, 100],
        labels=["heavy_fav", "fav", "slight_fav", "even", "dog", "big_dog"],
    ).astype(str)

    # --- ELO фичи из teams ---
    teams_dedup = teams.drop_duplicates(subset="Normalized_Name", keep="last")
    team_lookup = teams_dedup.set_index("Normalized_Name")[
        [
            "Current_ELO",
            "Winrate",
            "Total_Games",
            "Recent_Form",
            "Offensive_Rating",
            "Defensive_Rating",
            "Net_Rating",
        ]
    ].to_dict("index")

    def get_team_stat(selection: str, stat: str) -> float | None:
        name = extract_team_name(selection)
        if name and name in team_lookup:
            return team_lookup[name].get(stat)
        return None

    df["team_elo"] = df["Selection"].apply(lambda x: get_team_stat(x, "Current_ELO"))
    df["team_winrate"] = df["Selection"].apply(lambda x: get_team_stat(x, "Winrate"))
    df["team_games"] = df["Selection"].apply(lambda x: get_team_stat(x, "Total_Games"))
    df["team_off_rating"] = df["Selection"].apply(lambda x: get_team_stat(x, "Offensive_Rating"))
    df["team_def_rating"] = df["Selection"].apply(lambda x: get_team_stat(x, "Defensive_Rating"))
    df["team_net_rating"] = df["Selection"].apply(lambda x: get_team_stat(x, "Net_Rating"))

    # ELO match count
    elo_matched = df["team_elo"].notna().sum()
    logger.info("ELO matched: %d / %d (%.1f%%)", elo_matched, len(df), 100 * elo_matched / len(df))

    # --- Derived interaction features ---
    df["elo_x_odds"] = df["team_elo"].fillna(1500) * df["implied_prob"]
    df["winrate_vs_implied"] = df["team_winrate"].fillna(0.5) - df["implied_prob"]
    df["model_confidence"] = df["ML_P_Model"] - df["ML_P_Implied"]

    # Cat features
    for col in ["Sport", "Market", "Is_Parlay", "ML_Team_Stats_Found", "odds_bucket"]:
        df[col] = df[col].fillna("unknown").astype(str)

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
    """Подбор порога вероятности на val для максимизации ROI."""
    best_roi = -999.0
    best_thr = 0.5
    for thr in np.arange(0.35, 0.90, 0.01):
        mask = proba >= thr
        result = calc_roi(df, mask)
        if result["n_bets"] >= 20 and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_thr = thr
    return best_thr


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
]

CAT_FEATURES = ["Sport", "Market", "Is_Parlay", "ML_Team_Stats_Found", "odds_bucket"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def main():
    df, teams = load_data()
    df = add_features(df, teams)
    train_full, test = time_series_split(df)

    val_split = int(len(train_full) * 0.8)
    train = train_full.iloc[:val_split].copy()
    val = train_full.iloc[val_split:].copy()
    logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

    y_train = (train["Status"] == "won").astype(int)
    y_val = (val["Status"] == "won").astype(int)
    y_test = (test["Status"] == "won").astype(int)

    x_train = train[FEATURES].copy()
    x_val = val[FEATURES].copy()
    x_test = test[FEATURES].copy()

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

    with mlflow.start_run(run_name="phase2/step_2_1_features") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "2.1")
            mlflow.set_tag("phase", "2")

            # Baseline: те же параметры что в 1.4
            baseline_features = [
                "Odds",
                "ML_P_Model",
                "ML_P_Implied",
                "ML_Edge",
                "ML_EV",
                "ML_Winrate_Diff",
                "ML_Rating_Diff",
                "Outcomes_Count",
                "USD",
                "Sport",
                "Market",
                "Is_Parlay",
                "ML_Team_Stats_Found",
            ]
            baseline_cat = ["Sport", "Market", "Is_Parlay", "ML_Team_Stats_Found"]
            baseline_cat_idx = [baseline_features.index(c) for c in baseline_cat]

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
                train[baseline_features],
                y_train,
                eval_set=(val[baseline_features], y_val),
                early_stopping_rounds=100,
            )
            proba_baseline_val = model_baseline.predict_proba(val[baseline_features])[:, 1]
            proba_baseline_test = model_baseline.predict_proba(test[baseline_features])[:, 1]
            auc_baseline = roc_auc_score(y_test, proba_baseline_test)
            thr_baseline = find_best_threshold(proba_baseline_val, val)
            roi_baseline = calc_roi(test, proba_baseline_test >= thr_baseline)
            logger.info(
                "BASELINE: AUC=%.4f, ROI=%.2f%% (%d bets, thr=%.2f)",
                auc_baseline,
                roi_baseline["roi"],
                roi_baseline["n_bets"],
                thr_baseline,
            )

            # Candidate: с новыми фичами
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
                x_train,
                y_train,
                eval_set=(x_val, y_val),
                early_stopping_rounds=100,
            )
            proba_candidate_val = model_candidate.predict_proba(x_val)[:, 1]
            proba_candidate_test = model_candidate.predict_proba(x_test)[:, 1]
            auc_candidate = roc_auc_score(y_test, proba_candidate_test)
            thr_candidate = find_best_threshold(proba_candidate_val, val)
            roi_candidate = calc_roi(test, proba_candidate_test >= thr_candidate)
            logger.info(
                "CANDIDATE: AUC=%.4f, ROI=%.2f%% (%d bets, thr=%.2f)",
                auc_candidate,
                roi_candidate["roi"],
                roi_candidate["n_bets"],
                thr_candidate,
            )

            delta_roi = roi_candidate["roi"] - roi_baseline["roi"]
            delta_auc = auc_candidate - auc_baseline
            logger.info("Delta ROI: %.4f, Delta AUC: %.4f", delta_roi, delta_auc)

            # Feature importance (candidate)
            importances = model_candidate.get_feature_importance()
            for fname, imp in sorted(zip(FEATURES, importances, strict=True), key=lambda x: -x[1]):
                if imp > 0.5:
                    logger.info("  %s: %.2f", fname, imp)

            decision = (
                "accept" if delta_roi > 0.002 else ("marginal" if delta_roi > 0 else "reject")
            )
            logger.info("Decision: %s (delta_roi=%.4f)", decision, delta_roi)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(val),
                    "model": "CatBoost",
                    "n_features_baseline": len(baseline_features),
                    "n_features_candidate": len(FEATURES),
                    "new_features": "time+odds_derived+elo+interactions",
                    "decision": decision,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": roi_candidate["roi"],
                    "roi_baseline": roi_baseline["roi"],
                    "delta_roi": delta_roi,
                    "auc_test": auc_candidate,
                    "auc_baseline": auc_baseline,
                    "delta_auc": delta_auc,
                    "n_bets": roi_candidate["n_bets"],
                    "threshold": thr_candidate,
                }
            )

            # Сохранение если лучше
            if roi_candidate["roi"] > roi_baseline["roi"]:
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                model_candidate.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": roi_candidate["roi"],
                    "auc": auc_candidate,
                    "threshold": thr_candidate,
                    "n_bets": roi_candidate["n_bets"],
                    "feature_names": FEATURES,
                    "params": {
                        "iterations": model_candidate.tree_count_,
                        "depth": 6,
                        "learning_rate": 0.05,
                    },
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(models_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Model saved to %s", models_dir)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")

            logger.info("Run ID: %s", run.info.run_id)
            print(f"RESULT:run_id={run.info.run_id}")
            print(f"RESULT:roi_baseline={roi_baseline['roi']}")
            print(f"RESULT:roi_candidate={roi_candidate['roi']}")
            print(f"RESULT:delta={delta_roi}")
            print(f"RESULT:decision={decision}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.1")
            logger.exception("Step 2.1 failed")
            raise


if __name__ == "__main__":
    main()
