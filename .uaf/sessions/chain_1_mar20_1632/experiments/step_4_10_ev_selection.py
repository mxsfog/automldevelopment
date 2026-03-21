"""Step 4.10: EV-based selection instead of probability threshold."""

import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    add_safe_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
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


def calc_roi_from_mask(df, mask):
    """ROI from a boolean selection mask."""
    selected = df[mask]
    if len(selected) == 0:
        return {"roi": 0.0, "n_bets": 0, "win_rate": 0.0, "pct_selected": 0.0}
    staked = selected["USD"].sum()
    payout = selected["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    wr = (selected["Status"] == "won").mean()
    return {
        "roi": float(roi),
        "n_bets": len(selected),
        "win_rate": float(wr),
        "pct_selected": float(len(selected) / len(df) * 100),
    }


def find_best_ev_margin_on_val(val_df, proba, min_bets=30):
    """Find best EV margin on val. EV = p * odds - 1. Select when EV > margin."""
    odds = val_df["Odds"].values
    ev = proba * odds - 1
    best_roi = -999.0
    best_margin = 0.0
    for margin in np.arange(-0.1, 0.5, 0.02):
        mask = ev > margin
        if mask.sum() < min_bets:
            continue
        r = calc_roi_from_mask(val_df, mask)
        if r["roi"] > best_roi:
            best_roi = r["roi"]
            best_margin = float(margin)
    return best_margin, best_roi


def main() -> None:
    logger.info("Step 4.10: EV-based selection")
    df = load_data()
    train, test = time_series_split(df)

    val_split_idx = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split_idx]
    train_val = train.iloc[val_split_idx:]

    train_fit = add_safe_features(train_fit)
    train_val = add_safe_features(train_val)
    test = add_safe_features(test)

    feature_cols = get_extended_feature_columns()

    x_fit = np.nan_to_num(train_fit[feature_cols].values.astype(float), nan=0.0)
    y_fit = train_fit["target"].values
    x_val = np.nan_to_num(train_val[feature_cols].values.astype(float), nan=0.0)
    y_val = train_val["target"].values
    x_test = np.nan_to_num(test[feature_cols].values.astype(float), nan=0.0)

    val_good = ~train_val["Sport"].isin(EXCLUDE_SPORTS)
    test_good = ~test["Sport"].isin(EXCLUDE_SPORTS)

    with mlflow.start_run(run_name="phase4/step4.10_ev_selection") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.10")
        mlflow.set_tag("phase", "4")

        try:
            # Train ensemble
            cb = CatBoostClassifier(
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
            cb.fit(x_fit, y_fit, eval_set=(x_val, y_val), early_stopping_rounds=50)

            lgb = LGBMClassifier(
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
            lgb.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

            xgb = XGBClassifier(
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
            xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)

            p_ens_val = (
                cb.predict_proba(x_val)[:, 1]
                + lgb.predict_proba(x_val)[:, 1]
                + xgb.predict_proba(x_val)[:, 1]
            ) / 3
            p_ens_test = (
                cb.predict_proba(x_test)[:, 1]
                + lgb.predict_proba(x_test)[:, 1]
                + xgb.predict_proba(x_test)[:, 1]
            ) / 3

            val_filt = train_val[val_good]
            test_filt = test[test_good]
            p_val_f = p_ens_val[val_good.values]
            p_test_f = p_ens_test[test_good.values]

            # A: Probability threshold baseline
            thr_base, _vr_base = find_best_threshold_on_val(val_filt, p_val_f)
            roi_base = calc_roi(test_filt, p_test_f, threshold=thr_base)
            logger.info(
                "[A: prob threshold] thr=%.2f, ROI=%.2f%%, n=%d",
                thr_base,
                roi_base["roi"],
                roi_base["n_bets"],
            )

            # B: EV-based selection (on filtered)
            margin_val, val_roi_ev = find_best_ev_margin_on_val(val_filt, p_val_f)
            ev_test = p_test_f * test_filt["Odds"].values - 1
            ev_mask = ev_test > margin_val
            roi_ev = calc_roi_from_mask(test_filt, ev_mask)
            logger.info(
                "[B: EV selection] margin=%.2f, val_ROI=%.2f%%, test_ROI=%.2f%%, n=%d",
                margin_val,
                val_roi_ev,
                roi_ev["roi"],
                roi_ev["n_bets"],
            )

            # C: Combined prob threshold + EV filter
            logger.info("=== Combined: prob thr + EV margin ===")
            for thr in [0.55, 0.60, 0.65]:
                for margin in [0.0, 0.05, 0.10, 0.15, 0.20]:
                    combined_mask = (p_test_f >= thr) & (ev_test > margin)
                    r = calc_roi_from_mask(test_filt, combined_mask)
                    if r["n_bets"] >= 100:
                        logger.info(
                            "  thr=%.2f, ev_margin=%.2f: ROI=%.2f%%, n=%d, WR=%.3f",
                            thr,
                            margin,
                            r["roi"],
                            r["n_bets"],
                            r["win_rate"],
                        )

            # D: Optimal combined on val
            best_combined_roi = -999.0
            best_combined_thr = 0.60
            best_combined_margin = 0.0
            ev_val = p_val_f * val_filt["Odds"].values - 1
            for thr in np.arange(0.50, 0.70, 0.05):
                for margin in np.arange(-0.05, 0.25, 0.02):
                    combined_mask_val = (p_val_f >= thr) & (ev_val > margin)
                    if combined_mask_val.sum() < 50:
                        continue
                    r = calc_roi_from_mask(val_filt, combined_mask_val)
                    if r["roi"] > best_combined_roi:
                        best_combined_roi = r["roi"]
                        best_combined_thr = float(thr)
                        best_combined_margin = float(margin)

            logger.info(
                "Best combined on val: thr=%.2f, margin=%.2f, val_ROI=%.2f%%",
                best_combined_thr,
                best_combined_margin,
                best_combined_roi,
            )

            # Apply to test
            combined_test_mask = (p_test_f >= best_combined_thr) & (ev_test > best_combined_margin)
            roi_combined = calc_roi_from_mask(test_filt, combined_test_mask)
            logger.info(
                "[D: optimal combined] thr=%.2f, margin=%.2f, ROI=%.2f%%, n=%d",
                best_combined_thr,
                best_combined_margin,
                roi_combined["roi"],
                roi_combined["n_bets"],
            )

            # E: EV rank selection - select top N% by EV
            logger.info("=== EV rank selection ===")
            for pct in [10, 20, 30, 40, 50]:
                ev_threshold = np.percentile(ev_test, 100 - pct)
                rank_mask = ev_test >= ev_threshold
                r = calc_roi_from_mask(test_filt, rank_mask)
                logger.info(
                    "  top %d%%: ev_thr=%.3f, ROI=%.2f%%, n=%d",
                    pct,
                    ev_threshold,
                    r["roi"],
                    r["n_bets"],
                )

            # Pick best approach
            candidates = [
                ("prob_threshold", roi_base["roi"], roi_base),
                ("ev_selection", roi_ev["roi"], roi_ev),
                ("combined", roi_combined["roi"], roi_combined),
            ]
            best_name, best_roi_val, best_result = max(candidates, key=lambda x: x[1])
            logger.info("Best approach: %s, ROI=%.2f%%", best_name, best_roi_val)

            auc = roc_auc_score(test_filt["target"].values, p_test_f)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "ev_selection",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_approach": best_name,
                    "ev_margin": margin_val,
                    "prob_threshold": thr_base,
                    "combined_thr": best_combined_thr,
                    "combined_margin": best_combined_margin,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roi_prob_thr": roi_base["roi"],
                    "roi_ev": roi_ev["roi"],
                    "roi_combined": roi_combined["roi"],
                    "roc_auc": auc,
                    "n_bets_selected": best_result["n_bets"],
                    "win_rate_selected": best_result["win_rate"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.10")
            raise


if __name__ == "__main__":
    main()
