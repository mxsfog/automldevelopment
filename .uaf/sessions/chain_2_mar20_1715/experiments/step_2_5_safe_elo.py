"""Step 2.5: Safe ELO features (no ELO_Change leakage)."""

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


def build_safe_elo_features(bets: pd.DataFrame) -> pd.DataFrame:
    """ELO-фичи без leakage (только Old_ELO, K_Factor, team stats)."""
    elo, teams = load_elo_data()
    logger.info("ELO history: %d rows, Teams: %d rows", len(elo), len(teams))

    # SAFE aggregation: only Old_ELO and K_Factor (pre-match known)
    # REMOVED: ELO_Change, New_ELO (these depend on match result = leakage)
    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            team_elo_mean=("Old_ELO", "mean"),
            team_elo_max=("Old_ELO", "max"),
            team_elo_min=("Old_ELO", "min"),
            k_factor_mean=("K_Factor", "mean"),
            n_elo_records=("ID", "count"),
        )
        .reset_index()
    )

    # ELO difference between teams (using Old_ELO only)
    elo_pairs = elo[elo.duplicated("Bet_ID", keep=False)].copy()
    if len(elo_pairs) > 0:
        elo_first = elo_pairs.groupby("Bet_ID").first().reset_index()
        elo_last = elo_pairs.groupby("Bet_ID").last().reset_index()
        elo_diff = pd.DataFrame(
            {
                "Bet_ID": elo_first["Bet_ID"],
                "elo_diff": elo_first["Old_ELO"].values - elo_last["Old_ELO"].values,
                "elo_diff_abs": np.abs(elo_first["Old_ELO"].values - elo_last["Old_ELO"].values),
            }
        )
        elo_agg = elo_agg.merge(elo_diff, on="Bet_ID", how="left")
    else:
        elo_agg["elo_diff"] = np.nan
        elo_agg["elo_diff_abs"] = np.nan

    # Team stats from teams.csv (static reference)
    elo_with_team = elo.merge(
        teams, left_on="Team_ID", right_on="ID", how="left", suffixes=("", "_team")
    )
    team_stats = (
        elo_with_team.groupby("Bet_ID")
        .agg(
            team_winrate_mean=("Winrate", "mean"),
            team_winrate_max=("Winrate", "max"),
            team_winrate_diff=("Winrate", lambda x: x.max() - x.min()),
            team_total_games_mean=("Total_Games", "mean"),
            team_current_elo_mean=("Current_ELO", "mean"),
            team_elo_vs_current=("Old_ELO", lambda x: x.mean()),
        )
        .reset_index()
    )

    elo_agg = elo_agg.merge(team_stats, on="Bet_ID", how="left")

    # Merge
    bets_out = bets.merge(
        elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo")
    )
    bets_out["has_elo"] = bets_out["team_elo_mean"].notna().astype(float)

    # Derived safe features
    bets_out["elo_spread"] = bets_out["team_elo_max"] - bets_out["team_elo_min"]
    bets_out["elo_mean_vs_1500"] = bets_out["team_elo_mean"] - 1500.0

    safe_cols = get_safe_elo_features()
    for col in safe_cols:
        if col in bets_out.columns:
            bets_out[col] = bets_out[col].fillna(0.0)

    n_with = int(bets_out["has_elo"].sum())
    logger.info(
        "Safe ELO features: %d/%d bets have ELO (%.1f%%)",
        n_with,
        len(bets_out),
        n_with / len(bets_out) * 100,
    )
    return bets_out


def get_safe_elo_features() -> list[str]:
    """Безопасные ELO-фичи (без leakage)."""
    return [
        "team_elo_mean",
        "team_elo_max",
        "team_elo_min",
        "k_factor_mean",
        "n_elo_records",
        "elo_diff",
        "elo_diff_abs",
        "has_elo",
        "team_winrate_mean",
        "team_winrate_max",
        "team_winrate_diff",
        "team_total_games_mean",
        "team_current_elo_mean",
        "elo_spread",
        "elo_mean_vs_1500",
    ]


def main() -> None:
    """Safe ELO features without leakage."""
    logger.info("Step 2.5: Safe ELO features (no ELO_Change)")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)

    train, test = time_series_split(df)

    base_feats = get_base_features()
    eng_feats = get_engineered_features()
    safe_elo = get_safe_elo_features()

    # Baseline: base + engineered (no ELO)
    baseline_feats = base_feats + eng_feats
    # Candidate: base + engineered + safe ELO
    candidate_feats = baseline_feats + safe_elo

    for feat_set, feat_list, name, step_id in [
        ("baseline", baseline_feats, "phase2/step2.5a_baseline", "2.5a"),
        ("candidate", candidate_feats, "phase2/step2.5b_safe_elo", "2.5b"),
    ]:
        check_budget()
        logger.info("Running %s (%d features)", feat_set, len(feat_list))

        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", step_id)
            mlflow.set_tag("phase", "2")

            try:
                val_split = int(len(train) * 0.8)
                train_fit = train.iloc[:val_split]
                val_df = train.iloc[val_split:]

                imp = SimpleImputer(strategy="median")
                x_fit = imp.fit_transform(train_fit[feat_list])
                x_val = imp.transform(val_df[feat_list])
                x_test = imp.transform(test[feat_list])

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
                        "method": f"catboost_{feat_set}",
                        "n_features": len(feat_list),
                        "n_samples_train": len(train_fit),
                        "n_samples_val": len(val_df),
                        "n_samples_test": len(test),
                        "best_iteration": model.best_iteration_,
                        "leakage_free": "true",
                    }
                )

                roi_thresholds = calc_roi_at_thresholds(test, proba_test)
                for t, r in roi_thresholds.items():
                    mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

                fi = dict(zip(feat_list, model.feature_importances_, strict=False))
                fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                for fname, fval in fi_sorted[:15]:
                    logger.info("  FI: %s = %.3f", fname, fval)

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
                    "%s: ROI=%.2f%% AUC=%.4f t=%.2f n=%d iter=%d run=%s",
                    feat_set,
                    roi_result["roi"],
                    auc,
                    best_t,
                    roi_result["n_bets"],
                    model.best_iteration_,
                    run.info.run_id,
                )

            except Exception:
                mlflow.set_tag("status", "failed")
                mlflow.log_text(traceback.format_exc(), "traceback.txt")
                raise

    # Step 2.5c: ELO-only subset with safe features
    logger.info("Step 2.5c: ELO-only subset, safe features")
    train_elo = train[train["has_elo"] == 1.0].copy()
    test_elo = test[test["has_elo"] == 1.0].copy()
    logger.info("ELO subset: train=%d, test=%d", len(train_elo), len(test_elo))

    if len(test_elo) > 100:
        check_budget()
        with mlflow.start_run(run_name="phase2/step2.5c_elo_only_safe") as run:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "2.5c")
            mlflow.set_tag("phase", "2")

            try:
                val_split = int(len(train_elo) * 0.8)
                train_fit = train_elo.iloc[:val_split]
                val_df = train_elo.iloc[val_split:]

                imp = SimpleImputer(strategy="median")
                x_fit = imp.fit_transform(train_fit[candidate_feats])
                x_val = imp.transform(val_df[candidate_feats])
                x_test = imp.transform(test_elo[candidate_feats])

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
                roi_result = calc_roi(test_elo, proba_test, threshold=best_t)
                auc = roc_auc_score(test_elo["target"], proba_test)

                mlflow.log_params(
                    {
                        "validation_scheme": "time_series",
                        "seed": 42,
                        "method": "catboost_elo_only_safe",
                        "n_features": len(candidate_feats),
                        "n_samples_train": len(train_fit),
                        "n_samples_val": len(val_df),
                        "n_samples_test": len(test_elo),
                        "best_iteration": model.best_iteration_,
                        "leakage_free": "true",
                    }
                )

                roi_thresholds = calc_roi_at_thresholds(test_elo, proba_test)
                for t, r in roi_thresholds.items():
                    logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                    mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

                fi = dict(zip(candidate_feats, model.feature_importances_, strict=False))
                fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                for fname, fval in fi_sorted[:15]:
                    logger.info("  FI: %s = %.3f", fname, fval)

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
                    "ELO-only safe: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                    roi_result["roi"],
                    auc,
                    best_t,
                    roi_result["n_bets"],
                    run.info.run_id,
                )

            except Exception:
                mlflow.set_tag("status", "failed")
                mlflow.log_text(traceback.format_exc(), "traceback.txt")
                raise


if __name__ == "__main__":
    main()
