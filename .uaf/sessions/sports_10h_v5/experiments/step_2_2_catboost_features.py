"""Step 2.2: CatBoost с расширенными фичами + target encoding для Sport/Market."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
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


def add_target_encoding(
    train_fit: pd.DataFrame, train_val: pd.DataFrame, test: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Target encoding fit на train_fit, transform на val/test (anti-leakage)."""
    enc_map = train_fit.groupby(col)["target"].mean()
    global_mean = train_fit["target"].mean()

    feat_name = f"te_{col.lower()}"
    train_fit = train_fit.copy()
    train_val = train_val.copy()
    test = test.copy()

    train_fit[feat_name] = train_fit[col].map(enc_map).fillna(global_mean)
    train_val[feat_name] = train_val[col].map(enc_map).fillna(global_mean)
    test[feat_name] = test[col].map(enc_map).fillna(global_mean)

    return train_fit, train_val, test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создание новых фичей (без target encoding)."""
    df = df.copy()

    df["implied_prob"] = 1.0 / df["Odds"]
    df["log_odds"] = np.log1p(df["Odds"])
    df["has_ml_prediction"] = df["ML_P_Model"].notna().astype(int)
    df["ml_confidence"] = df["ML_P_Model"].fillna(0) / 100.0
    df["ml_prob_vs_implied"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["is_weekend"] = pd.to_datetime(df["Created_At"]).dt.dayofweek.ge(5).astype(int)
    df["hour"] = pd.to_datetime(df["Created_At"]).dt.hour
    df["log_usd"] = np.log1p(df["USD"])
    df["parlay_leg_avg_odds"] = np.where(
        df["Outcomes_Count"] > 1,
        df["Odds"] ** (1.0 / df["Outcomes_Count"]),
        df["Odds"],
    )

    return df


def get_numeric_features() -> list[str]:
    """Числовые фичи для CatBoost."""
    return [
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
        "has_ml_prediction",
        "ml_confidence",
        "ml_prob_vs_implied",
        "is_weekend",
        "hour",
        "log_usd",
        "parlay_leg_avg_odds",
    ]


def main() -> None:
    logger.info("Step 2.2: CatBoost with engineered features + target encoding")
    df = load_data()
    df = engineer_features(df)
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    # Target encoding для Sport и Market (fit только на train_fit)
    train_fit, train_val, test = add_target_encoding(train_fit, train_val, test, "Sport")
    train_fit, train_val, test = add_target_encoding(train_fit, train_val, test, "Market")

    feature_cols = [*get_numeric_features(), "te_sport", "te_market"]

    with mlflow.start_run(run_name="phase2/step2.2_catboost_features") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "2.2")
        mlflow.set_tag("phase", "2")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(train_val),
                    "n_samples_test": len(test),
                    "method": "catboost_with_features",
                    "features": ",".join(feature_cols),
                    "n_features": len(feature_cols),
                    "iterations": 1000,
                    "depth": 6,
                    "target_encoding": "Sport,Market",
                }
            )

            x_fit = train_fit[feature_cols].values.astype(float)
            y_fit = train_fit["target"].values
            x_val = train_val[feature_cols].values.astype(float)
            y_val = train_val["target"].values
            x_test = test[feature_cols].values.astype(float)
            y_test = test["target"].values

            model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                random_seed=42,
                verbose=100,
                eval_metric="AUC",
                auto_class_weights="Balanced",
                l2_leaf_reg=5,
            )

            model.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=100)

            # Threshold on val
            proba_val = model.predict_proba(x_val)[:, 1]
            best_t, val_roi = find_best_threshold_on_val(train_val, proba_val)
            logger.info("Best threshold from val: %.2f, val ROI=%.2f%%", best_t, val_roi)

            proba_test = model.predict_proba(x_test)[:, 1]
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(y_test, proba_test)

            # Также попробуем пороги вручную
            for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
                r = calc_roi(test, proba_test, threshold=t)
                logger.info(
                    "  Threshold %.2f: ROI=%.2f%% n=%d WR=%.4f",
                    t,
                    r["roi"],
                    r["n_bets"],
                    r["win_rate"],
                )
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "best_threshold": best_t,
                    "val_roi": val_roi,
                    "n_bets_selected": roi_result["n_bets"],
                    "pct_selected": roi_result["pct_selected"],
                    "win_rate_selected": roi_result["win_rate"],
                    "best_iteration": model.get_best_iteration(),
                }
            )

            importances = model.get_feature_importance()
            logger.info("Feature importances:")
            ranked = sorted(zip(feature_cols, importances, strict=True), key=lambda x: -x[1])
            for fname, imp in ranked:
                mlflow.log_metric(f"importance_{fname}", imp)
                logger.info("  %s: %.2f", fname, imp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")

            logger.info("Run ID: %s", run.info.run_id)
            logger.info(
                "ROI: %.2f%% at threshold=%.2f (n=%d, WR=%.4f)",
                roi_result["roi"],
                best_t,
                roi_result["n_bets"],
                roi_result["win_rate"],
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 2.2")
            raise


if __name__ == "__main__":
    main()
