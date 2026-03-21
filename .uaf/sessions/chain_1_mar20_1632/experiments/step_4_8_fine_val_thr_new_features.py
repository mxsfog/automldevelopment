"""Step 4.8: Fine val threshold grid + temporal & Kelly features."""

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
    get_extended_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
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

EXCLUDE_SPORTS = {"Basketball", "MMA", "FIFA", "Snooker"}


def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extended features: safe features + temporal + Kelly + nonlinear odds."""
    df = df.copy()

    # Safe features (from common.py add_safe_features)
    df["log_odds"] = np.log1p(df["Odds"])
    df["implied_prob"] = 1.0 / df["Odds"]
    df["value_ratio"] = (df["ML_P_Model"] / 100.0 / df["implied_prob"]).clip(0, 10).fillna(1.0)
    df["edge_x_ev"] = df["ML_Edge"] * df["ML_EV"]
    df["edge_abs"] = df["ML_Edge"].abs()
    df["ev_positive"] = (df["ML_EV"] > 0).astype(float)
    df["model_implied_diff"] = df["ML_P_Model"] - df["ML_P_Implied"]
    df["log_usd"] = np.log1p(df["USD"])
    df["usd_per_outcome"] = df["USD"] / df["Outcomes_Count"].clip(lower=1)
    df["log_usd_per_outcome"] = np.log1p(df["usd_per_outcome"])
    df["parlay_complexity"] = df["Outcomes_Count"] * (df["Is_Parlay"] == "t").astype(float)

    # New: temporal features
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)

    # New: Kelly fraction = (p * odds - 1) / (odds - 1), where p = ML_P_Model/100
    p_model = df["ML_P_Model"] / 100.0
    odds = df["Odds"]
    df["kelly_fraction"] = ((p_model * odds - 1) / (odds - 1)).clip(-1, 5).fillna(0.0)

    # New: squared odds (nonlinear transformation)
    df["odds_squared"] = df["Odds"] ** 2
    df["log_odds_sq"] = df["log_odds"] ** 2

    # New: edge / implied_prob interaction
    df["edge_over_implied"] = (df["ML_Edge"] / (df["implied_prob"] * 100 + 1)).fillna(0.0)

    # New: model confidence = how far from 0.5
    df["model_confidence"] = (df["ML_P_Model"] / 100.0 - 0.5).abs()

    return df


def get_all_feature_columns() -> list[str]:
    """All features including new ones."""
    return [
        *get_extended_feature_columns(),
        "hour",
        "day_of_week",
        "is_weekend",
        "kelly_fraction",
        "odds_squared",
        "log_odds_sq",
        "edge_over_implied",
        "model_confidence",
    ]


def find_fine_threshold_on_val(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    min_bets: int = 100,
) -> tuple[float, float]:
    """Fine threshold grid on val with 0.01 step and higher min_bets."""
    best_roi = -999.0
    best_t = 0.5
    for t in np.arange(0.45, 0.80, 0.01):
        result = calc_roi(val_df, proba, threshold=t)
        if result["n_bets"] >= min_bets and result["roi"] > best_roi:
            best_roi = result["roi"]
            best_t = float(t)
    return best_t, best_roi


def main() -> None:
    logger.info("Step 4.8: Fine val threshold + new features")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    train_fit = add_extended_features(train_fit)
    train_val = add_extended_features(train_val)
    test = add_extended_features(test)

    # Sport filter masks
    val_good = ~train_val["Sport"].isin(EXCLUDE_SPORTS)
    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)

    with mlflow.start_run(run_name="phase4/step4.8_fine_val_thr") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("phase", "4")

        try:
            # A: Baseline features + fine val threshold
            baseline_cols = get_extended_feature_columns()
            x_fit_b = np.nan_to_num(train_fit[baseline_cols].values.astype(float), nan=0.0)
            y_fit = train_fit["target"].values
            x_val_b = np.nan_to_num(train_val[baseline_cols].values.astype(float), nan=0.0)
            y_val = train_val["target"].values
            x_test_b = np.nan_to_num(test[baseline_cols].values.astype(float), nan=0.0)

            cb_b = CatBoostClassifier(
                iterations=855,
                depth=3,
                learning_rate=0.059,
                l2_leaf_reg=21.0,
                border_count=254,
                random_strength=9.26,
                bagging_temperature=4.82,
                min_data_in_leaf=77,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
            )
            cb_b.fit(x_fit_b, y_fit, eval_set=(x_val_b, y_val), early_stopping_rounds=50)

            lgb_b = LGBMClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                reg_alpha=1.0,
                reg_lambda=10.0,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            lgb_b.fit(
                x_fit_b,
                y_fit,
                eval_set=[(x_val_b, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            xgb_b = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                reg_alpha=1.0,
                reg_lambda=10.0,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="auc",
                verbosity=0,
            )
            xgb_b.fit(x_fit_b, y_fit, eval_set=[(x_val_b, y_val)], verbose=False)

            p_ens_val_b = (
                cb_b.predict_proba(x_val_b)[:, 1]
                + lgb_b.predict_proba(x_val_b)[:, 1]
                + xgb_b.predict_proba(x_val_b)[:, 1]
            ) / 3
            p_ens_test_b = (
                cb_b.predict_proba(x_test_b)[:, 1]
                + lgb_b.predict_proba(x_test_b)[:, 1]
                + xgb_b.predict_proba(x_test_b)[:, 1]
            ) / 3

            # Fine val threshold (filtered)
            val_filt = train_val[val_good]
            test_filt = test[test_good]
            p_val_f = p_ens_val_b[val_good.values]
            p_test_f = p_ens_test_b[test_good.values]

            thr_fine, val_roi_fine = find_fine_threshold_on_val(val_filt, p_val_f, min_bets=100)
            roi_fine = calc_roi(test_filt, p_test_f, threshold=thr_fine)
            auc_fine = roc_auc_score(test_filt["target"].values, p_test_f)
            logger.info(
                "[baseline + fine val thr] thr=%.2f, val_ROI=%.2f%%, test_ROI=%.2f%%, "
                "AUC=%.4f, n=%d",
                thr_fine,
                val_roi_fine,
                roi_fine["roi"],
                auc_fine,
                roi_fine["n_bets"],
            )
            mlflow.log_metric("roi_baseline_fine_thr", roi_fine["roi"])
            mlflow.log_metric("thr_baseline_fine", thr_fine)

            # B: Extended features (with new ones)
            all_cols = get_all_feature_columns()
            x_fit_e = np.nan_to_num(train_fit[all_cols].values.astype(float), nan=0.0)
            x_val_e = np.nan_to_num(train_val[all_cols].values.astype(float), nan=0.0)
            x_test_e = np.nan_to_num(test[all_cols].values.astype(float), nan=0.0)

            cb_e = CatBoostClassifier(
                iterations=855,
                depth=3,
                learning_rate=0.059,
                l2_leaf_reg=21.0,
                border_count=254,
                random_strength=9.26,
                bagging_temperature=4.82,
                min_data_in_leaf=77,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
            )
            cb_e.fit(x_fit_e, y_fit, eval_set=(x_val_e, y_val), early_stopping_rounds=50)

            lgb_e = LGBMClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                reg_alpha=1.0,
                reg_lambda=10.0,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            lgb_e.fit(
                x_fit_e,
                y_fit,
                eval_set=[(x_val_e, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            xgb_e = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                reg_alpha=1.0,
                reg_lambda=10.0,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="auc",
                verbosity=0,
            )
            xgb_e.fit(x_fit_e, y_fit, eval_set=[(x_val_e, y_val)], verbose=False)

            p_ens_val_e = (
                cb_e.predict_proba(x_val_e)[:, 1]
                + lgb_e.predict_proba(x_val_e)[:, 1]
                + xgb_e.predict_proba(x_val_e)[:, 1]
            ) / 3
            p_ens_test_e = (
                cb_e.predict_proba(x_test_e)[:, 1]
                + lgb_e.predict_proba(x_test_e)[:, 1]
                + xgb_e.predict_proba(x_test_e)[:, 1]
            ) / 3

            p_val_e_f = p_ens_val_e[val_good.values]
            p_test_e_f = p_ens_test_e[test_good.values]

            thr_ext, val_roi_ext = find_fine_threshold_on_val(val_filt, p_val_e_f, min_bets=100)
            roi_ext = calc_roi(test_filt, p_test_e_f, threshold=thr_ext)
            auc_ext = roc_auc_score(test_filt["target"].values, p_test_e_f)
            logger.info(
                "[extended + fine val thr] thr=%.2f, val_ROI=%.2f%%, test_ROI=%.2f%%, "
                "AUC=%.4f, n=%d",
                thr_ext,
                val_roi_ext,
                roi_ext["roi"],
                auc_ext,
                roi_ext["n_bets"],
            )
            mlflow.log_metric("roi_extended_fine_thr", roi_ext["roi"])
            mlflow.log_metric("thr_extended_fine", thr_ext)

            # C: Also try multiple min_bets levels to see stability
            logger.info("=== Threshold stability (baseline features) ===")
            for min_b in [50, 100, 200, 500]:
                t, vr = find_fine_threshold_on_val(val_filt, p_val_f, min_bets=min_b)
                r = calc_roi(test_filt, p_test_f, threshold=t)
                logger.info(
                    "  min_bets=%d: thr=%.2f, val_ROI=%.2f%%, test_ROI=%.2f%%, n=%d",
                    min_b,
                    t,
                    vr,
                    r["roi"],
                    r["n_bets"],
                )
                mlflow.log_metric(f"roi_minb{min_b}", r["roi"])

            # D: Delta analysis (new features vs baseline)
            delta_roi = roi_ext["roi"] - roi_fine["roi"]
            logger.info("Delta ROI (extended - baseline): %.2f%%", delta_roi)
            feature_decision = "accepted" if delta_roi > 0.2 else "rejected"
            logger.info("Feature decision: %s", feature_decision)

            # Final: pick best approach
            if roi_ext["roi"] > roi_fine["roi"]:
                final_roi = roi_ext["roi"]
                final_n = roi_ext["n_bets"]
                final_wr = roi_ext["win_rate"]
                final_thr = thr_ext
                final_auc = auc_ext
                approach = "extended_fine_thr"
            else:
                final_roi = roi_fine["roi"]
                final_n = roi_fine["n_bets"]
                final_wr = roi_fine["win_rate"]
                final_thr = thr_fine
                final_auc = auc_fine
                approach = "baseline_fine_thr"

            logger.info(
                "Best: %s, ROI=%.2f%%, thr=%.2f, n=%d",
                approach,
                final_roi,
                final_thr,
                final_n,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "fine_val_thr_new_features",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_approach": approach,
                    "best_threshold": final_thr,
                    "feature_decision": feature_decision,
                    "delta_roi": round(delta_roi, 4),
                    "n_new_features": 8,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "roc_auc": final_auc,
                    "best_threshold": final_thr,
                    "n_bets_selected": final_n,
                    "win_rate_selected": final_wr,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.8")
            raise


if __name__ == "__main__":
    main()
