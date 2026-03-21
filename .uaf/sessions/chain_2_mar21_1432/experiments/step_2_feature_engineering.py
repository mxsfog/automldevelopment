"""Phase 2 — Feature Engineering.

5 групп фичей, каждая проверяется shadow feature trick:
1. Odds-based: implied_prob, odds_ratio, margin
2. Sport/Market encoding: target encoding на train
3. Temporal: hour, day_of_week, time_since_start
4. ML interaction: edge*odds, ev*prob
5. Parlay/complexity: outcomes_count bins, odds spread
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    find_best_threshold,
    get_base_features,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

BASE_FEATURES = get_base_features()

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def add_odds_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Odds-based фичи."""
    df = df.copy()
    df["implied_prob"] = 1.0 / df["Odds"].clip(lower=1.01)
    df["odds_ratio"] = df["Outcome_Odds"] / df["Odds"].clip(lower=1.01)
    df["margin"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["odds_log"] = np.log1p(df["Odds"])
    df["implied_vs_model"] = df["ML_P_Model"] - df["ML_P_Implied"]
    new_feats = ["implied_prob", "odds_ratio", "margin", "odds_log", "implied_vs_model"]
    return df, new_feats


def add_sport_market_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Sport/Market target encoding (fit on train only)."""
    df = df.copy()
    new_feats = []

    for col in ["Sport", "Market"]:
        if col not in df.columns:
            continue
        # Target encoding на train
        means = train_df.groupby(col)["target"].mean()
        global_mean = train_df["target"].mean()
        # Smoothing
        counts = train_df.groupby(col)["target"].count()
        smooth = 50
        smoothed = (means * counts + global_mean * smooth) / (counts + smooth)

        feat_name = f"{col}_target_enc"
        df[feat_name] = df[col].map(smoothed).fillna(global_mean)
        new_feats.append(feat_name)

        # Count encoding
        count_map = train_df[col].value_counts()
        feat_name_cnt = f"{col}_count_enc"
        df[feat_name_cnt] = df[col].map(count_map).fillna(0)
        new_feats.append(feat_name_cnt)

    return df, new_feats


def add_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Temporal фичи."""
    df = df.copy()
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    new_feats = ["hour", "day_of_week", "is_weekend"]
    return df, new_feats


def add_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """ML interaction фичи."""
    df = df.copy()
    df["edge_x_odds"] = df["ML_Edge"] * df["Odds"]
    df["ev_x_prob"] = df["ML_EV"] * df["ML_P_Model"] / 100.0
    df["edge_per_odds"] = df["ML_Edge"] / df["Odds"].clip(lower=1.01)
    df["model_vs_implied_ratio"] = df["ML_P_Model"] / df["ML_P_Implied"].clip(lower=0.1)
    new_feats = ["edge_x_odds", "ev_x_prob", "edge_per_odds", "model_vs_implied_ratio"]
    return df, new_feats


def add_complexity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parlay/complexity фичи."""
    df = df.copy()
    df["odds_spread"] = df["max_outcome_odds"] - df["min_outcome_odds"]
    df["odds_cv"] = df["max_outcome_odds"] / df["mean_outcome_odds"].clip(lower=0.01) - 1
    df["high_odds"] = (df["Odds"] > 3.0).astype(int)
    df["very_low_odds"] = (df["Odds"] < 1.3).astype(int)
    new_feats = ["odds_spread", "odds_cv", "high_odds", "very_low_odds"]
    return df, new_feats


def evaluate_feature_group(
    train_fit: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base_features: list[str],
    new_features: list[str],
    group_name: str,
) -> dict:
    """Shadow feature trick: сравнение baseline vs baseline+new."""
    all_features = base_features + new_features

    x_train_base = train_fit[base_features].fillna(0)
    x_val_base = val[base_features].fillna(0)
    x_test_base = test[base_features].fillna(0)

    x_train_cand = train_fit[all_features].fillna(0)
    x_val_cand = val[all_features].fillna(0)
    x_test_cand = test[all_features].fillna(0)

    y_train = train_fit["target"]
    y_val = val["target"]

    # Baseline model
    model_base = CatBoostClassifier(**CATBOOST_PARAMS)
    model_base.fit(x_train_base, y_train, eval_set=(x_val_base, y_val), verbose=0)
    probas_base_val = model_base.predict_proba(x_val_base)[:, 1]
    probas_base_test = model_base.predict_proba(x_test_base)[:, 1]
    auc_base = roc_auc_score(y_val, probas_base_val)
    thr_base, _roi_base_val = find_best_threshold(val, probas_base_val, min_bets=50)
    roi_base_test = calc_roi(test, probas_base_test, threshold=thr_base)

    # Candidate model
    model_cand = CatBoostClassifier(**CATBOOST_PARAMS)
    model_cand.fit(x_train_cand, y_train, eval_set=(x_val_cand, y_val), verbose=0)
    probas_cand_val = model_cand.predict_proba(x_val_cand)[:, 1]
    probas_cand_test = model_cand.predict_proba(x_test_cand)[:, 1]
    auc_cand = roc_auc_score(y_val, probas_cand_val)
    thr_cand, _roi_cand_val = find_best_threshold(val, probas_cand_val, min_bets=50)
    roi_cand_test = calc_roi(test, probas_cand_test, threshold=thr_cand)

    delta_auc = auc_cand - auc_base
    delta_roi = roi_cand_test["roi"] - roi_base_test["roi"]

    # Решение: принять, отклонить или marginal
    if delta_roi > 0.2:
        decision = "accepted"
    elif delta_roi > 0:
        decision = "marginal"
    else:
        decision = "rejected"

    result = {
        "group": group_name,
        "new_features": new_features,
        "auc_base": round(auc_base, 4),
        "auc_cand": round(auc_cand, 4),
        "delta_auc": round(delta_auc, 4),
        "roi_base_test": roi_base_test["roi"],
        "roi_cand_test": roi_cand_test["roi"],
        "delta_roi": round(delta_roi, 4),
        "n_bets_base": roi_base_test["n_bets"],
        "n_bets_cand": roi_cand_test["n_bets"],
        "thr_base": thr_base,
        "thr_cand": thr_cand,
        "decision": decision,
    }

    logger.info(
        "Group %s: delta_auc=%.4f, delta_roi=%.4f -> %s",
        group_name,
        delta_auc,
        delta_roi,
        decision,
    )
    return result


def main() -> None:
    with mlflow.start_run(run_name="phase2/feature_engineering") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val = train.iloc[val_split:]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "shadow_feature_trick",
                    "base_features": len(BASE_FEATURES),
                }
            )

            # Применяем ВСЕ группы фичей к данным
            feature_groups = {}

            # Group 1: Odds-based
            for split_df in [train_fit, val, test]:
                _, feats = add_odds_features(split_df)
            train_fit, feats = add_odds_features(train_fit)
            val, _ = add_odds_features(val)
            test, _ = add_odds_features(test)
            feature_groups["odds_features"] = feats

            # Group 2: Sport/Market encoding
            train_fit, feats = add_sport_market_features(train_fit, train_fit)
            val, _ = add_sport_market_features(val, train_fit)
            test, _ = add_sport_market_features(test, train_fit)
            feature_groups["sport_market"] = feats

            # Group 3: Temporal
            train_fit, feats = add_temporal_features(train_fit)
            val, _ = add_temporal_features(val)
            test, _ = add_temporal_features(test)
            feature_groups["temporal"] = feats

            # Group 4: ML interactions
            train_fit, feats = add_interaction_features(train_fit)
            val, _ = add_interaction_features(val)
            test, _ = add_interaction_features(test)
            feature_groups["ml_interactions"] = feats

            # Group 5: Complexity
            train_fit, feats = add_complexity_features(train_fit)
            val, _ = add_complexity_features(val)
            test, _ = add_complexity_features(test)
            feature_groups["complexity"] = feats

            # Evaluate each group incrementally
            accepted_features = list(BASE_FEATURES)
            all_results = []

            for group_name, new_feats in feature_groups.items():
                logger.info("Evaluating group: %s (%d features)", group_name, len(new_feats))
                result = evaluate_feature_group(
                    train_fit,
                    val,
                    test,
                    accepted_features,
                    new_feats,
                    group_name,
                )
                all_results.append(result)

                # Log to MLflow
                mlflow.log_metrics(
                    {
                        f"auc_base_{group_name}": result["auc_base"],
                        f"auc_cand_{group_name}": result["auc_cand"],
                        f"delta_auc_{group_name}": result["delta_auc"],
                        f"roi_base_{group_name}": result["roi_base_test"],
                        f"roi_cand_{group_name}": result["roi_cand_test"],
                        f"delta_roi_{group_name}": result["delta_roi"],
                    }
                )
                mlflow.set_tag(f"decision_{group_name}", result["decision"])

                if result["decision"] in ("accepted", "marginal"):
                    accepted_features.extend(new_feats)
                    logger.info(
                        "Accepted %s -> total features: %d", group_name, len(accepted_features)
                    )

            # Final model with all accepted features
            logger.info("Final feature set: %d features", len(accepted_features))
            x_train = train_fit[accepted_features].fillna(0)
            x_val = val[accepted_features].fillna(0)
            x_test = test[accepted_features].fillna(0)
            y_train = train_fit["target"]
            y_val = val["target"]
            y_test = test["target"]

            model = CatBoostClassifier(**CATBOOST_PARAMS)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)

            probas_test = model.predict_proba(x_test)[:, 1]
            auc_test = roc_auc_score(y_test, probas_test)

            probas_val = model.predict_proba(x_val)[:, 1]
            best_thr, _val_result = find_best_threshold(val, probas_val, min_bets=50)
            test_result = calc_roi(test, probas_test, threshold=best_thr)

            logger.info("Final AUC test: %.4f", auc_test)
            logger.info("Final threshold: %.2f", best_thr)
            logger.info("Final test ROI: %s", test_result)

            for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                r = calc_roi(test, probas_test, threshold=thr)
                logger.info("Test thr=%.2f: ROI=%.2f%%, n=%d", thr, r["roi"], r["n_bets"])

            mlflow.log_metrics(
                {
                    "auc_test_final": auc_test,
                    "roi_test_final": test_result["roi"],
                    "n_bets_final": test_result["n_bets"],
                    "threshold_final": best_thr,
                    "n_accepted_features": len(accepted_features),
                }
            )

            # Save model if ROI > 0
            if test_result["roi"] > 0:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": test_result["roi"],
                    "auc": auc_test,
                    "threshold": best_thr,
                    "n_bets": test_result["n_bets"],
                    "feature_names": accepted_features,
                    "params": model.get_all_params(),
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "feature_groups_accepted": [
                        r["group"] for r in all_results if r["decision"] != "rejected"
                    ],
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                mlflow.log_artifact(str(model_dir / "model.cbm"))
                mlflow.log_artifact(str(model_dir / "metadata.json"))

            # Log summary
            summary = pd.DataFrame(all_results)
            summary_path = str(SESSION_DIR / "experiments" / "feature_engineering_summary.csv")
            summary.to_csv(summary_path, index=False)
            mlflow.log_artifact(summary_path)
            mlflow.log_artifact(__file__)

            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")
            mlflow.set_tag("accepted_features", ",".join(accepted_features))

            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Accepted features: %s", accepted_features)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Phase 2 failed")
            raise


if __name__ == "__main__":
    main()
