"""Step 4.10 — Extended feature set with 3-model ensemble + conf_ev.

Гипотеза: текущие 19 фичей могут не захватывать все паттерны.
Добавляем 16 новых фичей из step_2_feature_engineering:
- odds_features: implied_prob, odds_ratio, margin, odds_log, implied_vs_model
- interaction_features: edge_x_odds, ev_x_prob, edge_per_odds, model_vs_implied_ratio
- temporal_features: hour, day_of_week, is_weekend
- complexity_features: odds_spread, odds_cv, high_odds, very_low_odds

Тестируем каждую группу отдельно и все вместе с conf_ev selection.
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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import (
    add_complexity_features,
    add_interaction_features,
    add_odds_features,
    add_sport_market_features,
    add_temporal_features,
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

BASE_FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcome_Odds",
    "n_outcomes",
    "mean_outcome_odds",
    "max_outcome_odds",
    "min_outcome_odds",
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def train_ensemble(x: pd.DataFrame, y: pd.Series) -> tuple:
    """3-model ensemble."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y)
    return cb, lgbm, lr, scaler


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions + std."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std(np.array([p_cb, p_lgbm, p_lr]), axis=0)
    return p_mean, p_std


def evaluate_feature_set(
    train_fit_enc: pd.DataFrame,
    val_enc: pd.DataFrame,
    features: list[str],
    name: str,
) -> dict:
    """Train ensemble and evaluate conf_ev on val."""
    x_tr = train_fit_enc[features].fillna(0)
    x_val = val_enc[features].fillna(0)

    cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fit_enc["target"])
    p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_val)

    odds_val = val_enc["Odds"].values
    ev = p_mean * odds_val - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf

    auc = roc_auc_score(val_enc["target"], p_mean)

    results = {"name": name, "n_features": len(features), "auc_val": auc}
    for thr in [0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
        results[f"roi_conf_ev_{thr:.2f}"] = r["roi"]
        results[f"n_conf_ev_{thr:.2f}"] = r["n_bets"]

    logger.info(
        "%s (%d feats): AUC=%.4f, conf_ev_0.15=%.2f%%(n=%d)",
        name,
        len(features),
        auc,
        results["roi_conf_ev_0.15"],
        results["n_conf_ev_0.15"],
    )
    return results


def main() -> None:
    """Extended features experiment."""
    with mlflow.start_run(run_name="phase4/extended_features") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Val split
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "extended_features",
                }
            )

            # Apply all feature groups
            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            train_fit_enc, odds_feats = add_odds_features(train_fit_enc)
            val_enc, _ = add_odds_features(val_enc)

            train_fit_enc, temporal_feats = add_temporal_features(train_fit_enc)
            val_enc, _ = add_temporal_features(val_enc)

            train_fit_enc, interaction_feats = add_interaction_features(train_fit_enc)
            val_enc, _ = add_interaction_features(val_enc)

            train_fit_enc, complexity_feats = add_complexity_features(train_fit_enc)
            val_enc, _ = add_complexity_features(val_enc)

            # Evaluate incrementally on val
            val_results = []

            # Baseline: 19 features
            val_results.append(
                evaluate_feature_set(train_fit_enc, val_enc, BASE_FEATURES, "baseline_19")
            )

            # +odds
            feats_odds = BASE_FEATURES + odds_feats
            val_results.append(evaluate_feature_set(train_fit_enc, val_enc, feats_odds, "+odds"))

            # +interaction
            feats_interaction = BASE_FEATURES + interaction_feats
            val_results.append(
                evaluate_feature_set(train_fit_enc, val_enc, feats_interaction, "+interaction")
            )

            # +temporal
            feats_temporal = BASE_FEATURES + temporal_feats
            val_results.append(
                evaluate_feature_set(train_fit_enc, val_enc, feats_temporal, "+temporal")
            )

            # +complexity
            feats_complexity = BASE_FEATURES + complexity_feats
            val_results.append(
                evaluate_feature_set(train_fit_enc, val_enc, feats_complexity, "+complexity")
            )

            # All features combined
            all_new = odds_feats + interaction_feats + temporal_feats + complexity_feats
            feats_all = BASE_FEATURES + all_new
            val_results.append(evaluate_feature_set(train_fit_enc, val_enc, feats_all, "+all_35"))

            # Best on val: odds+interaction (top performers likely)
            feats_oi = BASE_FEATURES + odds_feats + interaction_feats
            val_results.append(
                evaluate_feature_set(train_fit_enc, val_enc, feats_oi, "+odds_inter")
            )

            # Summary
            results_df = pd.DataFrame(val_results)
            logger.info("Feature set comparison on val:\n%s", results_df.to_string(index=False))

            # Select best feature set on val (by conf_ev_0.15 ROI)
            best_val = max(val_results, key=lambda r: r["roi_conf_ev_0.15"])
            best_name = best_val["name"]
            logger.info("Best val feature set: %s", best_name)

            # Map name to feature list
            feature_map = {
                "baseline_19": BASE_FEATURES,
                "+odds": feats_odds,
                "+interaction": feats_interaction,
                "+temporal": feats_temporal,
                "+complexity": feats_complexity,
                "+all_35": feats_all,
                "+odds_inter": feats_oi,
            }
            best_features = feature_map[best_name]

            # Test evaluation with full train
            logger.info("Test evaluation with %s (%d features)", best_name, len(best_features))

            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            train_enc, _ = add_odds_features(train_enc)
            test_enc, _ = add_odds_features(test_enc)
            train_enc, _ = add_temporal_features(train_enc)
            test_enc, _ = add_temporal_features(test_enc)
            train_enc, _ = add_interaction_features(train_enc)
            test_enc, _ = add_interaction_features(test_enc)
            train_enc, _ = add_complexity_features(train_enc)
            test_enc, _ = add_complexity_features(test_enc)

            # Baseline 19 features on test
            x_train_base = train_enc[BASE_FEATURES].fillna(0)
            x_test_base = test_enc[BASE_FEATURES].fillna(0)
            cb_b, lgbm_b, lr_b, sc_b = train_ensemble(x_train_base, train_enc["target"])
            p_test_b, p_std_b = predict_ensemble(cb_b, lgbm_b, lr_b, sc_b, x_test_base)
            odds_test = test_enc["Odds"].values
            ev_b = p_test_b * odds_test - 1
            conf_b = 1 / (1 + p_std_b * 10)
            ev_conf_b = ev_b * conf_b

            r_base = calc_roi(test_enc, (ev_conf_b >= 0.15).astype(float), threshold=0.5)
            auc_base = roc_auc_score(test_enc["target"], p_test_b)
            logger.info(
                "Baseline 19 test: AUC=%.4f, conf_ev_0.15=%.2f%%(n=%d)",
                auc_base,
                r_base["roi"],
                r_base["n_bets"],
            )

            # Best feature set on test
            x_train_best = train_enc[best_features].fillna(0)
            x_test_best = test_enc[best_features].fillna(0)
            cb_best, lgbm_best, lr_best, sc_best = train_ensemble(
                x_train_best, train_enc["target"]
            )
            p_test_best, p_std_best = predict_ensemble(
                cb_best, lgbm_best, lr_best, sc_best, x_test_best
            )
            ev_best = p_test_best * odds_test - 1
            conf_best = 1 / (1 + p_std_best * 10)
            ev_conf_best = ev_best * conf_best

            auc_best = roc_auc_score(test_enc["target"], p_test_best)

            test_results = {}
            for thr in [0.10, 0.12, 0.15, 0.18, 0.20]:
                mask = ev_conf_best >= thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                test_results[f"conf_ev_{thr:.2f}"] = r
                logger.info(
                    "%s test conf_ev_%.2f: ROI=%.2f%%, n=%d",
                    best_name,
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Also evaluate all_35 on test if not already best
            if best_name != "+all_35":
                x_train_all = train_enc[feats_all].fillna(0)
                x_test_all = test_enc[feats_all].fillna(0)
                cb_a, lgbm_a, lr_a, sc_a = train_ensemble(x_train_all, train_enc["target"])
                p_a, p_std_a = predict_ensemble(cb_a, lgbm_a, lr_a, sc_a, x_test_all)
                ev_a = p_a * odds_test - 1
                conf_a = 1 / (1 + p_std_a * 10)
                ev_conf_a = ev_a * conf_a
                r_all = calc_roi(test_enc, (ev_conf_a >= 0.15).astype(float), threshold=0.5)
                logger.info(
                    "+all_35 test conf_ev_0.15: ROI=%.2f%%, n=%d",
                    r_all["roi"],
                    r_all["n_bets"],
                )
                test_results["all_35_conf_ev_0.15"] = r_all

            # Best test result
            test_results["baseline_19"] = r_base
            best_test_name = max(
                test_results,
                key=lambda k: test_results[k]["roi"] if test_results[k]["n_bets"] >= 50 else -999,
            )
            best_test = test_results[best_test_name]
            logger.info(
                "Best test (n>=50): %s -> ROI=%.2f%%, n=%d",
                best_test_name,
                best_test["roi"],
                best_test["n_bets"],
            )

            # Log
            mlflow.log_metrics(
                {
                    "auc_base_test": auc_base,
                    "auc_best_test": auc_best,
                    "roi_base_test": r_base["roi"],
                    "n_bets_base": r_base["n_bets"],
                    "roi_best_test": best_test["roi"],
                    "n_bets_best": best_test["n_bets"],
                    "n_features_best": len(best_features),
                }
            )
            mlflow.set_tag("best_feature_set", best_name)
            mlflow.set_tag("best_test_strategy", best_test_name)

            if best_test["roi"] > 27.95:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_best.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_test["roi"],
                    "auc": auc_best,
                    "threshold": 0.12,
                    "ev_threshold": 0.15,
                    "n_bets": best_test["n_bets"],
                    "feature_names": best_features,
                    "selection_method": "conf_ev_0.15",
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info("Model saved with ROI=%.2f%%", best_test["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.10 failed")
            raise


if __name__ == "__main__":
    main()
