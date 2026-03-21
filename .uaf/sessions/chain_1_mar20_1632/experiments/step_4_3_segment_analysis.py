"""Step 4.3: Segment analysis -- ROI по Sport, Market, Is_Parlay, Odds ranges."""

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
    get_extended_feature_columns,
    load_data,
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
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


def main() -> None:
    logger.info("Step 4.3: Segment analysis")
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
    _y_test = test["target"].values

    with mlflow.start_run(run_name="phase4/step4.3_segment_analysis") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.3")
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

            p_test = (
                cb.predict_proba(x_test)[:, 1]
                + lgb.predict_proba(x_test)[:, 1]
                + xgb.predict_proba(x_test)[:, 1]
            ) / 3

            threshold = 0.60
            selected_mask = p_test >= threshold
            test_df = test.copy()
            test_df["proba"] = p_test
            test_df["selected"] = selected_mask

            overall = calc_roi(test_df, p_test, threshold=threshold)
            logger.info(
                "Overall: ROI=%.2f%%, n=%d, WR=%.4f",
                overall["roi"],
                overall["n_bets"],
                overall["win_rate"],
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "segment_analysis",
                    "threshold": threshold,
                    "n_samples_test": len(test),
                }
            )

            # Segment analysis by Sport
            logger.info("=== ROI by Sport ===")
            selected = test_df[test_df["selected"]]
            sport_groups = selected.groupby("Sport")
            sport_results = []
            for sport, group in sport_groups:
                if len(group) < 10:
                    continue
                staked = group["USD"].sum()
                payout = group["Payout_USD"].sum()
                roi = (payout - staked) / staked * 100
                wr = (group["Status"] == "won").mean()
                sport_results.append((sport, roi, len(group), wr, staked))
                logger.info(
                    "  %s: ROI=%.2f%%, n=%d, WR=%.3f, staked=%.0f",
                    sport,
                    roi,
                    len(group),
                    wr,
                    staked,
                )

            for sport, roi, n, _wr, _staked in sorted(sport_results, key=lambda x: -x[1]):
                mlflow.log_metric(f"seg_roi_sport_{sport}", roi)
                mlflow.log_metric(f"seg_n_sport_{sport}", n)

            # Segment by Is_Parlay
            logger.info("=== ROI by Is_Parlay ===")
            for is_p in ["f", "t"]:
                mask = selected["Is_Parlay"] == is_p
                if mask.sum() < 10:
                    continue
                g = selected[mask]
                staked = g["USD"].sum()
                payout = g["Payout_USD"].sum()
                roi = (payout - staked) / staked * 100
                logger.info("  Is_Parlay=%s: ROI=%.2f%%, n=%d", is_p, roi, len(g))
                mlflow.log_metric(f"seg_roi_parlay_{is_p}", roi)

            # Segment by Odds ranges
            logger.info("=== ROI by Odds range ===")
            odds_bins = [
                (1.0, 1.3),
                (1.3, 1.5),
                (1.5, 1.8),
                (1.8, 2.0),
                (2.0, 2.5),
                (2.5, 3.0),
                (3.0, 5.0),
                (5.0, 100.0),
            ]
            for lo, hi in odds_bins:
                mask = (selected["Odds"] >= lo) & (selected["Odds"] < hi)
                if mask.sum() < 10:
                    continue
                g = selected[mask]
                staked = g["USD"].sum()
                payout = g["Payout_USD"].sum()
                roi = (payout - staked) / staked * 100
                wr = (g["Status"] == "won").mean()
                logger.info(
                    "  Odds [%.1f,%.1f): ROI=%.2f%%, n=%d, WR=%.3f", lo, hi, roi, len(g), wr
                )
                mlflow.log_metric(f"seg_roi_odds_{lo}_{hi}", roi)

            # Segment by Market (top markets only)
            logger.info("=== ROI by Market (top 10) ===")
            market_groups = selected.groupby("Market")
            market_results = []
            for market, group in market_groups:
                if len(group) < 20:
                    continue
                staked = group["USD"].sum()
                payout = group["Payout_USD"].sum()
                roi = (payout - staked) / staked * 100
                market_results.append((market, roi, len(group)))

            for market, roi, n in sorted(market_results, key=lambda x: -x[1])[:10]:
                logger.info("  %s: ROI=%.2f%%, n=%d", market, roi, n)

            # Find best segment combination for filtering
            logger.info("=== Best segment filter ===")
            # Try: only singles, only low odds favorites
            singles = test_df[(test_df["Is_Parlay"] == "f") & test_df["selected"]]
            if len(singles) > 0:
                s_roi = (
                    (singles["Payout_USD"].sum() - singles["USD"].sum())
                    / singles["USD"].sum()
                    * 100
                )
                logger.info("Singles only: ROI=%.2f%%, n=%d", s_roi, len(singles))
                mlflow.log_metric("roi_singles_only", s_roi)

            # Try: odds < 2.5 (favorites)
            fav = test_df[(test_df["Odds"] < 2.5) & test_df["selected"]]
            if len(fav) > 0:
                f_roi = (fav["Payout_USD"].sum() - fav["USD"].sum()) / fav["USD"].sum() * 100
                logger.info("Favorites (odds<2.5): ROI=%.2f%%, n=%d", f_roi, len(fav))
                mlflow.log_metric("roi_favorites_only", f_roi)

            # Singles + favorites
            sf = test_df[
                (test_df["Odds"] < 2.5) & (test_df["Is_Parlay"] == "f") & test_df["selected"]
            ]
            if len(sf) > 0:
                sf_roi = (sf["Payout_USD"].sum() - sf["USD"].sum()) / sf["USD"].sum() * 100
                logger.info("Singles+favorites: ROI=%.2f%%, n=%d", sf_roi, len(sf))
                mlflow.log_metric("roi_singles_favorites", sf_roi)

            mlflow.log_metrics({"roi": overall["roi"], "n_bets_selected": overall["n_bets"]})
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.3")
            raise


if __name__ == "__main__":
    main()
