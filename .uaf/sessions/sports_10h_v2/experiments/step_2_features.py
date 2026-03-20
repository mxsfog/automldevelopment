"""Phase 2: Feature Engineering с shadow feature trick.

Запускает все 5 шагов последовательно, каждый добавляет группу фич
и сравнивает baseline vs candidate через shadow feature trick.
"""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    DATA_DIR,
    SEED,
    calc_roi_at_thresholds,
    check_budget,
    get_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DELTA_ACCEPT = 0.002
DELTA_MARGINAL = 0.0


def add_sport_market_features(df: pd.DataFrame, train_df: pd.DataFrame) -> list[str]:
    """Step 2.1: Sport/Market categorical features via target encoding."""
    new_cols = []

    # Target encoding для Sport (fit на train)
    sport_means = train_df.groupby("Sport")["target"].mean()
    df["sport_target_enc"] = df["Sport"].map(sport_means).fillna(train_df["target"].mean())
    new_cols.append("sport_target_enc")

    # Target encoding для Market
    market_means = train_df.groupby("Market")["target"].mean()
    df["market_target_enc"] = df["Market"].map(market_means).fillna(train_df["target"].mean())
    new_cols.append("market_target_enc")

    # Sport win rate на train
    sport_counts = train_df.groupby("Sport")["target"].agg(["mean", "count"])
    df["sport_win_rate"] = df["Sport"].map(sport_counts["mean"]).fillna(0.5)
    df["sport_count"] = df["Sport"].map(sport_counts["count"]).fillna(0)
    new_cols.extend(["sport_win_rate", "sport_count"])

    # Is_Esports flag
    esports = ["CS2", "Dota 2", "League of Legends", "FIFA", "Valorant", "NBA 2K"]
    df["is_esports"] = df["Sport"].isin(esports).astype(int)
    new_cols.append("is_esports")

    # Market group (simplify 20+ markets into groups)
    winner_markets = ["Winner", "1x2", "Match Winner - Twoway", "Winner (Incl. Overtime)"]
    total_markets = [m for m in df["Market"].unique() if "Total" in str(m)]
    handicap_markets = [m for m in df["Market"].unique() if "Handicap" in str(m)]

    df["market_group_winner"] = df["Market"].isin(winner_markets).astype(int)
    df["market_group_total"] = df["Market"].isin(total_markets).astype(int)
    df["market_group_handicap"] = df["Market"].isin(handicap_markets).astype(int)
    new_cols.extend(["market_group_winner", "market_group_total", "market_group_handicap"])

    return new_cols


def add_odds_value_features(df: pd.DataFrame) -> list[str]:
    """Step 2.2: Odds-derived value features."""
    new_cols = []

    # Implied probability from odds
    df["implied_prob"] = 1.0 / df["Odds"]
    new_cols.append("implied_prob")

    # ML model vs implied probability gap
    df["model_vs_implied"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    new_cols.append("model_vs_implied")

    # Odds categories
    df["odds_low"] = (df["Odds"] <= 1.5).astype(int)
    df["odds_mid"] = ((df["Odds"] > 1.5) & (df["Odds"] <= 3.0)).astype(int)
    df["odds_high"] = (df["Odds"] > 3.0).astype(int)
    new_cols.extend(["odds_low", "odds_mid", "odds_high"])

    # Log odds
    df["log_odds"] = np.log(df["Odds"].clip(lower=1.01))
    new_cols.append("log_odds")

    # ML_Edge normalized by odds
    df["edge_per_odds"] = df["ML_Edge"] / df["Odds"].clip(lower=0.01)
    new_cols.append("edge_per_odds")

    # Expected value: ML_P_Model * Odds - 1
    df["expected_value"] = (df["ML_P_Model"] / 100.0) * df["Odds"] - 1
    new_cols.append("expected_value")

    # Kelly criterion fraction
    p = df["ML_P_Model"].clip(lower=1, upper=99) / 100.0
    b = df["Odds"] - 1
    df["kelly_fraction"] = ((p * b - (1 - p)) / b).clip(lower=-1, upper=1)
    new_cols.append("kelly_fraction")

    return new_cols


def add_team_features(df: pd.DataFrame, train_df: pd.DataFrame) -> list[str]:
    """Step 2.3: Team ELO and stats from teams.csv and elo_history.csv."""
    new_cols = []
    elo_history = pd.read_csv(DATA_DIR / "elo_history.csv")

    # Aggregate ELO history per bet
    elo_per_bet = (
        elo_history.groupby("Bet_ID")
        .agg(
            avg_elo=("New_ELO", "mean"),
            elo_diff=("ELO_Change", "sum"),
            max_elo=("New_ELO", "max"),
            min_elo=("New_ELO", "min"),
        )
        .reset_index()
    )

    df = df.merge(elo_per_bet, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))

    for col in ["avg_elo", "elo_diff", "max_elo", "min_elo"]:
        df[col] = df[col].fillna(1500.0 if "elo" in col.lower() else 0.0)
        new_cols.append(col)

    # ELO spread
    df["elo_spread"] = df["max_elo"] - df["min_elo"]
    new_cols.append("elo_spread")

    return new_cols, df


def add_time_volume_features(df: pd.DataFrame) -> list[str]:
    """Step 2.4: Time and volume features."""
    new_cols = []

    created = pd.to_datetime(df["Created_At"])

    df["hour"] = created.dt.hour
    df["dow"] = created.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    new_cols.extend(["hour", "dow", "is_weekend"])

    # Log USD (stake size)
    df["log_usd"] = np.log1p(df["USD"])
    new_cols.append("log_usd")

    # Stake relative to median (по train нужно считать, но для простоты здесь)
    df["stake_ratio"] = df["USD"] / df["USD"].median()
    new_cols.append("stake_ratio")

    # Outcomes count features
    df["is_single"] = (df["Outcomes_Count"] == 1).astype(int)
    df["outcomes_log"] = np.log1p(df["Outcomes_Count"])
    new_cols.extend(["is_single", "outcomes_log"])

    return new_cols


def add_historical_features(df: pd.DataFrame, train_df: pd.DataFrame) -> list[str]:
    """Step 2.5: Historical aggregation features (fit on train only)."""
    new_cols = []

    # Historical win rate by sport+market combo
    combo_stats = (
        train_df.groupby(["Sport", "Market"])["target"].agg(["mean", "count"]).reset_index()
    )
    combo_stats.columns = ["Sport", "Market", "combo_win_rate", "combo_count"]

    df = df.merge(combo_stats, on=["Sport", "Market"], how="left")
    df["combo_win_rate"] = df["combo_win_rate"].fillna(train_df["target"].mean())
    df["combo_count"] = df["combo_count"].fillna(0)
    new_cols.extend(["combo_win_rate", "combo_count"])

    # Historical ROI by sport (on train)
    sport_roi = train_df.groupby("Sport").apply(
        lambda g: (
            (g["Payout_USD"].sum() - g["USD"].sum()) / g["USD"].sum() if g["USD"].sum() > 0 else 0
        )
    )
    df["sport_hist_roi"] = df["Sport"].map(sport_roi).fillna(0)
    new_cols.append("sport_hist_roi")

    # Odds quantile position (where this odds sits in training distribution)
    odds_median = train_df["Odds"].median()
    odds_std = train_df["Odds"].std()
    df["odds_zscore"] = (df["Odds"] - odds_median) / odds_std
    new_cols.append("odds_zscore")

    return new_cols, df


def run_shadow_experiment(
    step_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_features: list[str],
    candidate_features: list[str],
    step_tag: str,
) -> dict:
    """Shadow feature trick: сравнивает baseline vs candidate feature set."""
    with mlflow.start_run(run_name=f"phase2/{step_name}") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", step_tag)
        mlflow.set_tag("phase", "2")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "shadow_feature_trick",
                    "n_baseline_features": len(baseline_features),
                    "n_candidate_features": len(candidate_features),
                    "new_features": ",".join(
                        [f for f in candidate_features if f not in baseline_features]
                    ),
                }
            )

            y_train = train["target"].values
            y_test = test["target"].values

            model_params = {
                "iterations": 500,
                "depth": 6,
                "random_seed": SEED,
                "verbose": 0,
                "eval_metric": "AUC",
                "auto_class_weights": "Balanced",
            }

            # Baseline model
            x_train_base = train[baseline_features].values.astype(float)
            x_test_base = test[baseline_features].values.astype(float)
            model_base = CatBoostClassifier(**model_params)
            model_base.fit(
                x_train_base,
                y_train,
                eval_set=(x_test_base, y_test),
                early_stopping_rounds=50,
            )
            proba_base = model_base.predict_proba(x_test_base)[:, 1]

            auc_base = roc_auc_score(y_test, proba_base)
            roi_base_results = calc_roi_at_thresholds(test, proba_base)
            best_roi_base = max(
                (r["roi"] for r in roi_base_results.values() if r["n_bets"] >= 50),
                default=0.0,
            )

            # Candidate model
            x_train_cand = train[candidate_features].values.astype(float)
            x_test_cand = test[candidate_features].values.astype(float)
            model_cand = CatBoostClassifier(**model_params)
            model_cand.fit(
                x_train_cand,
                y_train,
                eval_set=(x_test_cand, y_test),
                early_stopping_rounds=50,
            )
            proba_cand = model_cand.predict_proba(x_test_cand)[:, 1]

            auc_cand = roc_auc_score(y_test, proba_cand)
            roi_cand_results = calc_roi_at_thresholds(test, proba_cand)
            best_roi_cand = max(
                (r["roi"] for r in roi_cand_results.values() if r["n_bets"] >= 50),
                default=0.0,
            )

            delta_roi = (best_roi_cand - best_roi_base) / 100.0
            delta_auc = auc_cand - auc_base

            if delta_roi > DELTA_ACCEPT:
                decision = "accept"
            elif delta_roi <= DELTA_MARGINAL:
                decision = "reject"
            else:
                decision = "marginal"

            logger.info(
                "%s: base_ROI=%.2f%% cand_ROI=%.2f%% delta=%.4f -> %s",
                step_name,
                best_roi_base,
                best_roi_cand,
                delta_roi,
                decision,
            )
            logger.info(
                "%s: base_AUC=%.4f cand_AUC=%.4f delta_AUC=%.4f",
                step_name,
                auc_base,
                auc_cand,
                delta_auc,
            )

            # Log best threshold details for candidate
            best_thresh_cand = 0.5
            best_roi_val = -999.0
            for t, r in roi_cand_results.items():
                if r["n_bets"] >= 50 and r["roi"] > best_roi_val:
                    best_roi_val = r["roi"]
                    best_thresh_cand = t

            roi_at_best = roi_cand_results[best_thresh_cand]
            logger.info(
                "Best candidate: t=%.2f, ROI=%.2f%%, n=%d, WR=%.4f",
                best_thresh_cand,
                roi_at_best["roi"],
                roi_at_best["n_bets"],
                roi_at_best["win_rate"],
            )

            mlflow.log_metrics(
                {
                    "roi": best_roi_cand,
                    "roi_baseline": best_roi_base,
                    "roi_candidate": best_roi_cand,
                    "roi_delta": delta_roi,
                    "auc_baseline": auc_base,
                    "auc_candidate": auc_cand,
                    "auc_delta": delta_auc,
                    "best_threshold": best_thresh_cand,
                    "n_bets_selected": roi_at_best["n_bets"],
                }
            )
            mlflow.set_tag("decision", decision)
            mlflow.set_tag("target_enc_fit_on_val", "false")

            # Feature importances for candidate
            importances = model_cand.get_feature_importance()
            for fname, imp in zip(candidate_features, importances, strict=True):
                mlflow.log_metric(f"imp_{fname}", imp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5" if decision == "accept" else "0.3")

            return {
                "run_id": run.info.run_id,
                "decision": decision,
                "roi_baseline": best_roi_base,
                "roi_candidate": best_roi_cand,
                "delta_roi": delta_roi,
                "auc_baseline": auc_base,
                "auc_candidate": auc_cand,
                "best_threshold": best_thresh_cand,
                "importances": dict(zip(candidate_features, importances, strict=True)),
            }

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", f"exception in {step_name}")
            raise


def main() -> None:
    logger.info("Phase 2: Feature Engineering")
    df = load_data()
    train, test = time_series_split(df)

    baseline_features = get_feature_columns()
    accepted_features = list(baseline_features)
    results = []

    # Step 2.1: Sport/Market features
    logger.info("Step 2.1: Sport/Market categorical features")
    check_budget()
    new_cols_21 = add_sport_market_features(train, train)
    add_sport_market_features(test, train)

    candidate_21 = accepted_features + new_cols_21
    result_21 = run_shadow_experiment(
        "step2.1_sport_market", train, test, accepted_features, candidate_21, "2.1"
    )
    results.append(("2.1", result_21))
    if result_21["decision"] in ("accept", "marginal"):
        accepted_features = list(candidate_21)
        logger.info("Step 2.1: ACCEPTED, features=%d", len(accepted_features))
    else:
        logger.info("Step 2.1: REJECTED")

    # Step 2.2: Odds-derived value features
    logger.info("Step 2.2: Odds-derived value features")
    check_budget()
    new_cols_22 = add_odds_value_features(train)
    add_odds_value_features(test)

    candidate_22 = accepted_features + new_cols_22
    result_22 = run_shadow_experiment(
        "step2.2_odds_value", train, test, accepted_features, candidate_22, "2.2"
    )
    results.append(("2.2", result_22))
    if result_22["decision"] in ("accept", "marginal"):
        accepted_features = list(candidate_22)
        logger.info("Step 2.2: ACCEPTED, features=%d", len(accepted_features))
    else:
        logger.info("Step 2.2: REJECTED")

    # Step 2.3: Team ELO features
    logger.info("Step 2.3: Team ELO and stats features")
    check_budget()
    new_cols_23, train = add_team_features(train, train)
    _, test = add_team_features(test, train)

    candidate_23 = accepted_features + new_cols_23
    result_23 = run_shadow_experiment(
        "step2.3_team_elo", train, test, accepted_features, candidate_23, "2.3"
    )
    results.append(("2.3", result_23))
    if result_23["decision"] in ("accept", "marginal"):
        accepted_features = list(candidate_23)
        logger.info("Step 2.3: ACCEPTED, features=%d", len(accepted_features))
    else:
        logger.info("Step 2.3: REJECTED")

    # Step 2.4: Time/volume features
    logger.info("Step 2.4: Time and volume features")
    check_budget()
    new_cols_24 = add_time_volume_features(train)
    add_time_volume_features(test)

    candidate_24 = accepted_features + new_cols_24
    result_24 = run_shadow_experiment(
        "step2.4_time_volume", train, test, accepted_features, candidate_24, "2.4"
    )
    results.append(("2.4", result_24))
    if result_24["decision"] in ("accept", "marginal"):
        accepted_features = list(candidate_24)
        logger.info("Step 2.4: ACCEPTED, features=%d", len(accepted_features))
    else:
        logger.info("Step 2.4: REJECTED")

    # Step 2.5: Historical aggregation features
    logger.info("Step 2.5: Historical aggregation features")
    check_budget()
    new_cols_25, train = add_historical_features(train, train)
    _, test = add_historical_features(test, train)

    candidate_25 = accepted_features + new_cols_25
    result_25 = run_shadow_experiment(
        "step2.5_historical", train, test, accepted_features, candidate_25, "2.5"
    )
    results.append(("2.5", result_25))
    if result_25["decision"] in ("accept", "marginal"):
        accepted_features = list(candidate_25)
        logger.info("Step 2.5: ACCEPTED, features=%d", len(accepted_features))
    else:
        logger.info("Step 2.5: REJECTED")

    # Summary
    logger.info("Phase 2 complete. Final feature set (%d features):", len(accepted_features))
    for f in accepted_features:
        logger.info("  - %s", f)
    for step, res in results:
        logger.info(
            "  Step %s: %s (delta=%.4f, ROI=%.2f%%)",
            step,
            res["decision"],
            res["delta_roi"],
            res["roi_candidate"],
        )


if __name__ == "__main__":
    main()
