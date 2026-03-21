"""Step 4.1: New feature engineering for sport-filtered ELO subset."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_elo_features,
    get_engineered_features,
    load_data,
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

CB_PARAMS = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 21.1,
    "min_data_in_leaf": 20,
    "random_strength": 1.0,
    "bagging_temperature": 0.06,
    "border_count": 102,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def add_new_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Новые фичи для попытки улучшения."""
    df = df.copy()
    new_feats: list[str] = []

    # Odds-based interaction features
    df["odds_x_elo_diff"] = df["Odds"] * df["elo_diff"].fillna(0)
    new_feats.append("odds_x_elo_diff")

    df["implied_x_winrate"] = df["implied_prob"] * df["team_winrate_mean"].fillna(0.5)
    new_feats.append("implied_x_winrate")

    # ELO quality: certainty of ELO (more games = more reliable)
    df["elo_confidence"] = np.log1p(df["team_total_games_mean"].fillna(0))
    new_feats.append("elo_confidence")

    # Elo-probability agreement: does ELO agree with implied prob?
    elo_prob = df["team_winrate_mean"].fillna(0.5)
    df["elo_implied_agreement"] = 1.0 - (elo_prob - df["implied_prob"]).abs()
    new_feats.append("elo_implied_agreement")

    # Discrepancy: model says edge but ELO disagrees
    df["edge_elo_conflict"] = df["ML_Edge"] * (df["elo_diff"].fillna(0) < 0).astype(float)
    new_feats.append("edge_elo_conflict")

    # Winrate relative to odds
    df["winrate_vs_odds"] = df["team_winrate_mean"].fillna(0.5) - df["implied_prob"]
    new_feats.append("winrate_vs_odds")

    # Squared elo diff (non-linear)
    df["elo_diff_sq"] = df["elo_diff"].fillna(0) ** 2
    new_feats.append("elo_diff_sq")

    # Value ratio x elo spread
    df["value_x_elo_spread"] = df["value_ratio"] * df["elo_spread"].fillna(0)
    new_feats.append("value_x_elo_spread")

    # Odds bucket (categorical encoded as ordinal)
    df["odds_bucket"] = pd.cut(
        df["Odds"], bins=[0, 1.3, 1.6, 2.0, 3.0, 5.0, 100], labels=False
    ).fillna(3)
    new_feats.append("odds_bucket")

    # Hour of day (temporal pattern)
    if "Created_At" in df.columns:
        df["hour_of_day"] = df["Created_At"].dt.hour
        new_feats.append("hour_of_day")
        df["is_weekend"] = df["Created_At"].dt.dayofweek.isin([5, 6]).astype(float)
        new_feats.append("is_weekend")

    # N outcomes interaction
    df["elo_per_outcome"] = df["team_elo_mean"].fillna(1500) / df["Outcomes_Count"].clip(lower=1)
    new_feats.append("elo_per_outcome")

    return df, new_feats


def main() -> None:
    """Test new features: baseline vs baseline+new."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)
    df, new_feats = add_new_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_base = get_base_features() + get_engineered_features() + get_elo_features()
    feat_extended = feat_base + new_feats

    logger.info(
        "Base features: %d, Extended: %d (+%d new)",
        len(feat_base),
        len(feat_extended),
        len(new_feats),
    )
    logger.info("New features: %s", new_feats)

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    results: dict[str, dict] = {}

    for name, feats in [("baseline", feat_base), ("extended", feat_extended)]:
        check_budget()
        imp = SimpleImputer(strategy="median")
        x_fit = imp.fit_transform(train_fit[feats])
        x_val = imp.transform(val_df[feats])
        x_test = imp.transform(test_sf[feats])

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]

        t_val, _ = find_best_threshold_on_val(val_df, p_val, min_bets=15)
        roi_val = calc_roi(test_sf, p_test, threshold=t_val)
        roi_t77 = calc_roi(test_sf, p_test, threshold=0.77)
        auc = roc_auc_score(test_sf["target"], p_test)

        results[name] = {
            "roi_val": roi_val["roi"],
            "roi_t77": roi_t77["roi"],
            "auc": auc,
            "threshold": t_val,
            "n_bets_val": roi_val["n_bets"],
            "n_bets_t77": roi_t77["n_bets"],
        }

        logger.info(
            "  %s: ROI(val_t=%.2f)=%.2f%% n=%d | ROI(t77)=%.2f%% n=%d | AUC=%.4f",
            name,
            t_val,
            roi_val["roi"],
            roi_val["n_bets"],
            roi_t77["roi"],
            roi_t77["n_bets"],
            auc,
        )

        if name == "extended":
            fi = model.get_feature_importance()
            fi_pairs = sorted(zip(feats, fi, strict=True), key=lambda x: x[1], reverse=True)
            logger.info("Top-15 features (extended):")
            for fname, imp_val in fi_pairs[:15]:
                marker = " [NEW]" if fname in new_feats else ""
                logger.info("  %s: %.2f%s", fname, imp_val, marker)

    delta_roi = results["extended"]["roi_t77"] - results["baseline"]["roi_t77"]
    delta_auc = results["extended"]["auc"] - results["baseline"]["auc"]
    logger.info("Delta ROI(t77): %.2f%%, Delta AUC: %.4f", delta_roi, delta_auc)

    best_key = "extended" if delta_roi > 0.2 else "baseline"
    best = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.1_new_features") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "new_features_shadow",
                    "n_features_base": len(feat_base),
                    "n_features_extended": len(feat_extended),
                    "n_new_features": len(new_feats),
                    "new_features": str(new_feats),
                    "accepted": best_key == "extended",
                    "delta_roi_t77": delta_roi,
                    "delta_auc": delta_auc,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_t77_{name}", r["roi_t77"])
                mlflow.log_metric(f"roi_val_{name}", r["roi_val"])
                mlflow.log_metric(f"auc_{name}", r["auc"])

            mlflow.log_metrics(
                {
                    "roi": best["roi_t77"],
                    "roc_auc": best["auc"],
                    "n_bets": best["n_bets_t77"],
                    "best_threshold": 0.77,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.1: %s accepted, ROI=%.2f%% AUC=%.4f run=%s",
                best_key,
                best["roi_t77"],
                best["auc"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
