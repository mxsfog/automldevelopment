"""Phase 2: Feature Engineering -- ELO + chain_1 features via Shadow Feature Trick."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    load_elo_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
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


def build_elo_features(bets: pd.DataFrame) -> pd.DataFrame:
    """Построение ELO-фич для каждой ставки."""
    elo, teams = load_elo_data()
    logger.info("ELO history: %d rows, Teams: %d rows", len(elo), len(teams))

    # Агрегация ELO по ставкам: для каждого Bet_ID берем ELO обоих участников
    # Группируем по Bet_ID, берем team и opponent
    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            team_elo_mean=("Old_ELO", "mean"),
            team_elo_max=("Old_ELO", "max"),
            team_elo_min=("Old_ELO", "min"),
            elo_change_mean=("ELO_Change", "mean"),
            elo_change_abs_mean=("ELO_Change", lambda x: x.abs().mean()),
            k_factor_mean=("K_Factor", "mean"),
            n_elo_records=("ID", "count"),
        )
        .reset_index()
    )

    # Для ставок с 2 участниками (team vs opponent), берем разницу ELO
    elo_pairs = elo[elo.duplicated("Bet_ID", keep=False)].copy()
    if len(elo_pairs) > 0:
        # Берем первую запись (team) и вторую (opponent) для каждого Bet_ID
        elo_first = elo_pairs.groupby("Bet_ID").first().reset_index()
        elo_last = elo_pairs.groupby("Bet_ID").last().reset_index()

        elo_diff = pd.DataFrame(
            {
                "Bet_ID": elo_first["Bet_ID"],
                "elo_diff": elo_first["Old_ELO"].values - elo_last["Old_ELO"].values,
                "elo_diff_abs": (elo_first["Old_ELO"].values - elo_last["Old_ELO"].values).abs()
                if hasattr(elo_first["Old_ELO"].values - elo_last["Old_ELO"].values, "abs")
                else np.abs(elo_first["Old_ELO"].values - elo_last["Old_ELO"].values),
            }
        )
        elo_agg = elo_agg.merge(elo_diff, on="Bet_ID", how="left")
    else:
        elo_agg["elo_diff"] = np.nan
        elo_agg["elo_diff_abs"] = np.nan

    # Join team stats
    # Для каждого Bet_ID берем статистику Team из teams.csv
    elo_with_team = elo.merge(
        teams, left_on="Team_ID", right_on="ID", how="left", suffixes=("", "_team")
    )

    team_stats = (
        elo_with_team.groupby("Bet_ID")
        .agg(
            team_winrate_mean=("Winrate", "mean"),
            team_winrate_max=("Winrate", "max"),
            team_total_games_mean=("Total_Games", "mean"),
            team_form_diversity=("Recent_Form", "nunique"),
            team_offensive_mean=("Offensive_Rating", "mean"),
            team_defensive_mean=("Defensive_Rating", "mean"),
            team_net_rating_mean=("Net_Rating", "mean"),
            team_goals_diff_mean=("Goal_Differential", "mean"),
        )
        .reset_index()
    )

    elo_agg = elo_agg.merge(team_stats, on="Bet_ID", how="left")

    # Merge to bets
    bets_out = bets.merge(
        elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo")
    )
    # has_elo flag
    bets_out["has_elo"] = bets_out["team_elo_mean"].notna().astype(float)

    # Fill NaN for bets without ELO data
    elo_feature_cols = get_elo_features()
    for col in elo_feature_cols:
        if col in bets_out.columns:
            bets_out[col] = bets_out[col].fillna(0.0)

    logger.info(
        "ELO features joined: %d/%d bets have ELO data (%.1f%%)",
        int(bets_out["has_elo"].sum()),
        len(bets_out),
        bets_out["has_elo"].mean() * 100,
    )
    return bets_out


def get_elo_features() -> list[str]:
    """ELO-фичи."""
    return [
        "team_elo_mean",
        "team_elo_max",
        "team_elo_min",
        "elo_change_mean",
        "elo_change_abs_mean",
        "k_factor_mean",
        "n_elo_records",
        "elo_diff",
        "elo_diff_abs",
        "has_elo",
        "team_winrate_mean",
        "team_winrate_max",
        "team_total_games_mean",
        "team_offensive_mean",
        "team_defensive_mean",
        "team_net_rating_mean",
        "team_goals_diff_mean",
    ]


def run_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    run_name: str,
    step: str,
    description: str,
) -> dict:
    """Запуск одного эксперимента CatBoost с заданными фичами."""
    check_budget()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", step)
        mlflow.set_tag("phase", "2")

        try:
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val_df = train.iloc[val_split:]

            imp = SimpleImputer(strategy="median")
            x_fit = imp.fit_transform(train_fit[feature_cols])
            x_val = imp.transform(val_df[feature_cols])
            x_test = imp.transform(test[feature_cols])

            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(test["target"], proba_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "catboost_shadow",
                    "description": description,
                    "n_features": len(feature_cols),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "best_iteration": model.best_iteration_,
                }
            )

            roi_thresholds = calc_roi_at_thresholds(test, proba_test)
            for t, r in roi_thresholds.items():
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            # Feature importance
            fi = dict(zip(feature_cols, model.feature_importances_, strict=False))
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted[:15]:
                logger.info("  FI: %s = %.3f", fname, fval)
                mlflow.log_metric(f"fi_{fname}", fval)

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                    "best_threshold": best_t,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            logger.info(
                "%s: ROI=%.2f%% AUC=%.4f t=%.2f n=%d iter=%d",
                run_name,
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                model.best_iteration_,
            )
            return {
                "run_id": run.info.run_id,
                "roi": roi_result["roi"],
                "auc": auc,
                "threshold": best_t,
                "n_bets": roi_result["n_bets"],
                "best_iteration": model.best_iteration_,
                "roi_at_thresholds": {t: r["roi"] for t, r in roi_thresholds.items()},
            }

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", f"exception in {step}")
            raise


def main() -> None:
    """Shadow Feature Trick: baseline vs candidate с ELO-фичами."""
    logger.info("Phase 2: Feature Engineering with ELO data")

    # Load and enrich data
    df = load_data()
    df = add_engineered_features(df)
    df = build_elo_features(df)

    train, test = time_series_split(df)

    # Step 2.1: Baseline -- base + engineered features (chain_1 proven set)
    base_feats = get_base_features()
    eng_feats = get_engineered_features()
    baseline_feats = base_feats + eng_feats

    logger.info("Step 2.1: Baseline features (%d)", len(baseline_feats))
    res_baseline = run_experiment(
        train,
        test,
        baseline_feats,
        run_name="phase2/step2.1_baseline",
        step="2.1",
        description="base + chain_1 engineered features",
    )

    # Step 2.2: Candidate -- baseline + ELO features
    elo_feats = get_elo_features()
    candidate_feats = baseline_feats + elo_feats

    logger.info("Step 2.2: Candidate features (%d)", len(candidate_feats))
    res_candidate = run_experiment(
        train,
        test,
        candidate_feats,
        run_name="phase2/step2.2_elo_candidate",
        step="2.2",
        description="base + engineered + ELO features",
    )

    # Shadow Feature Trick evaluation
    delta_roi = res_candidate["roi"] - res_baseline["roi"]
    delta_auc = res_candidate["auc"] - res_baseline["auc"]
    logger.info(
        "Shadow Feature Trick: delta_roi=%.2f%%, delta_auc=%.4f",
        delta_roi,
        delta_auc,
    )

    if delta_roi > 0.2:
        logger.info("ELO features ACCEPTED (delta > 0.2%%)")
    elif delta_roi > 0:
        logger.info("ELO features MARGINAL (0 < delta <= 0.2%%)")
    else:
        logger.info("ELO features REJECTED (delta <= 0)")

    # Step 2.3: ELO-only subset -- only bets with ELO data
    logger.info("Step 2.3: ELO-only subset (has_elo=1)")
    train_elo = train[train["has_elo"] == 1.0].copy()
    test_elo = test[test["has_elo"] == 1.0].copy()
    logger.info("ELO subset: train=%d, test=%d", len(train_elo), len(test_elo))

    if len(test_elo) > 100:
        res_elo_only = run_experiment(
            train_elo,
            test_elo,
            candidate_feats,
            run_name="phase2/step2.3_elo_only",
            step="2.3",
            description="ELO-only subset with all features",
        )
    else:
        logger.info("Too few ELO-only test samples, skipping step 2.3")
        res_elo_only = None

    # Step 2.4: Interaction features (ELO x odds-based)
    logger.info("Step 2.4: Interaction features")
    df["elo_x_implied"] = df["team_elo_mean"] * df["implied_prob"]
    df["elo_x_edge"] = df["team_elo_mean"] * df["ML_Edge"]
    df["elo_diff_x_odds"] = df["elo_diff"] * df["log_odds"]
    df["winrate_x_value"] = df["team_winrate_mean"] * df["value_ratio"]
    df["net_rating_x_edge"] = df["team_net_rating_mean"] * df["edge_abs"]

    interaction_feats = [
        "elo_x_implied",
        "elo_x_edge",
        "elo_diff_x_odds",
        "winrate_x_value",
        "net_rating_x_edge",
    ]
    for col in interaction_feats:
        df[col] = df[col].fillna(0.0)

    train2, test2 = time_series_split(df)
    interaction_candidate = candidate_feats + interaction_feats

    res_interaction = run_experiment(
        train2,
        test2,
        interaction_candidate,
        run_name="phase2/step2.4_interactions",
        step="2.4",
        description="all features + ELO interactions",
    )

    delta_int = res_interaction["roi"] - res_candidate["roi"]
    logger.info("Interaction delta: %.2f%%", delta_int)

    # Summary
    logger.info("Phase 2 Summary:")
    logger.info("  Baseline (chain_1 feats): ROI=%.2f%%", res_baseline["roi"])
    logger.info("  + ELO features: ROI=%.2f%% (delta=%.2f%%)", res_candidate["roi"], delta_roi)
    if res_elo_only:
        logger.info("  ELO-only subset: ROI=%.2f%%", res_elo_only["roi"])
    logger.info("  + Interactions: ROI=%.2f%% (delta=%.2f%%)", res_interaction["roi"], delta_int)

    best_roi = max(res_baseline["roi"], res_candidate["roi"], res_interaction["roi"])
    logger.info("  Best Phase 2 ROI: %.2f%%", best_roi)


if __name__ == "__main__":
    main()
