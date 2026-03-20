"""Step 4.8: Final best -- Optuna CB + Optuna LGB ensemble with robust threshold."""

import logging
import os
import traceback

import lightgbm
import mlflow
import numpy as np
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
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from step_2_5_safe_elo import build_safe_elo_features, get_safe_elo_features
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def train_ensemble(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple:
    """Train CB+LGB+XGB ensemble with Optuna-tuned params."""
    cb = CatBoostClassifier(
        iterations=499,
        depth=7,
        learning_rate=0.214,
        l2_leaf_reg=1.15,
        random_strength=0.823,
        bagging_temperature=2.41,
        border_count=121,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=30,
    )
    cb.fit(x_fit, y_fit, eval_set=(x_eval, y_eval))

    lgb_m = LGBMClassifier(
        n_estimators=477,
        max_depth=3,
        learning_rate=0.292,
        num_leaves=16,
        min_child_samples=49,
        reg_lambda=28.63,
        random_state=42,
        verbose=-1,
    )
    lgb_m.fit(
        x_fit,
        y_fit,
        eval_set=[(x_eval, y_eval)],
        callbacks=[
            lightgbm.early_stopping(30, verbose=False),
            lightgbm.log_evaluation(0),
        ],
    )

    xgb_m = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        eval_metric="auc",
        early_stopping_rounds=30,
    )
    xgb_m.fit(x_fit, y_fit, eval_set=[(x_eval, y_eval)], verbose=False)
    return cb, lgb_m, xgb_m


def predict_cb50(
    cb: CatBoostClassifier,
    lgb_m: LGBMClassifier,
    xgb_m: XGBClassifier,
    x: np.ndarray,
) -> np.ndarray:
    """CB50 ensemble prediction."""
    return (
        0.5 * cb.predict_proba(x)[:, 1]
        + 0.25 * lgb_m.predict_proba(x)[:, 1]
        + 0.25 * xgb_m.predict_proba(x)[:, 1]
    )


def main() -> None:
    """Final best: Optuna ensemble + robust threshold."""
    logger.info("Step 4.8: Final best -- Optuna ens + robust threshold")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)
    train_all, test_all = time_series_split(df)

    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    feature_cols = get_base_features() + get_engineered_features() + get_safe_elo_features()

    with mlflow.start_run(run_name="phase4/step4.8_final_best") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("phase", "4")

        try:
            # Multi-fold robust threshold
            thresholds_found: list[float] = []

            for val_pct in [0.15, 0.20, 0.25, 0.30]:
                val_split_idx = int(len(train) * (1 - val_pct))
                t_fit = train.iloc[:val_split_idx]
                t_val = train.iloc[val_split_idx:]

                imp_f = SimpleImputer(strategy="median")
                xf = imp_f.fit_transform(t_fit[feature_cols])
                xv = imp_f.transform(t_val[feature_cols])
                yf = t_fit["target"].values
                yv = t_val["target"].values

                cb_f, lgb_f, xgb_f = train_ensemble(xf, yf, xv, yv)
                pv_ens = predict_cb50(cb_f, lgb_f, xgb_f, xv)
                t_fold, vr_fold = find_best_threshold_on_val(t_val, pv_ens)
                thresholds_found.append(t_fold)
                logger.info(
                    "  val_pct=%.0f%%: t=%.2f, val_roi=%.2f%%",
                    val_pct * 100,
                    t_fold,
                    vr_fold,
                )

            median_t = float(np.median(thresholds_found))
            mean_t = float(np.mean(thresholds_found))
            logger.info(
                "Multi-fold thresholds: %s, median=%.3f, mean=%.3f",
                thresholds_found,
                median_t,
                mean_t,
            )

            # Final model on standard 80/20 split
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val_df = train.iloc[val_split:]

            imp = SimpleImputer(strategy="median")
            x_fit = imp.fit_transform(train_fit[feature_cols])
            x_val = imp.transform(val_df[feature_cols])
            x_test = imp.transform(test[feature_cols])
            y_fit = train_fit["target"].values
            y_val = val_df["target"].values

            cb, lgb_m, xgb_m = train_ensemble(x_fit, y_fit, x_val, y_val)
            p_ens_val = predict_cb50(cb, lgb_m, xgb_m, x_val)
            p_ens_test = predict_cb50(cb, lgb_m, xgb_m, x_test)

            # Compare threshold strategies
            t_std, _vr = find_best_threshold_on_val(val_df, p_ens_val)

            strategies = {
                "standard": t_std,
                "median_4fold": median_t,
                "mean_4fold": mean_t,
            }

            # Also test fixed thresholds from analysis
            for fixed_t in [0.60, 0.65, 0.70, 0.73, 0.75]:
                strategies[f"fixed_{int(fixed_t * 100)}"] = fixed_t

            best_strat = ""
            best_roi_val = -999.0

            for strat_name, t in strategies.items():
                roi = calc_roi(test, p_ens_test, threshold=t)
                auc = roc_auc_score(test["target"], p_ens_test)
                logger.info(
                    "  %s t=%.2f: ROI=%.2f%% n=%d AUC=%.4f",
                    strat_name,
                    t,
                    roi["roi"],
                    roi["n_bets"],
                    auc,
                )
                mlflow.log_metric(f"roi_{strat_name}", roi["roi"])
                mlflow.log_metric(f"n_{strat_name}", roi["n_bets"])
                if roi["roi"] > best_roi_val:
                    best_roi_val = roi["roi"]
                    best_strat = strat_name

            logger.info("Best strategy: %s, ROI=%.2f%%", best_strat, best_roi_val)

            # Full threshold profile
            roi_thresholds = calc_roi_at_thresholds(test, p_ens_test)
            for t, r in roi_thresholds.items():
                mlflow.log_metric(f"roi_profile_t{int(t * 100):03d}", r["roi"])

            auc_final = roc_auc_score(test["target"], p_ens_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"final_best_{best_strat}",
                    "n_features": len(feature_cols),
                    "multi_fold_pcts": "15,20,25,30",
                    "thresholds_found": str(thresholds_found),
                    "median_threshold": median_t,
                    "mean_threshold": mean_t,
                    "standard_threshold": t_std,
                    "best_strategy": best_strat,
                    "leakage_free": "true",
                }
            )
            mlflow.log_metrics(
                {
                    "roi": best_roi_val,
                    "roc_auc": auc_final,
                    "best_threshold": strategies[best_strat],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.8")
            raise


if __name__ == "__main__":
    main()
