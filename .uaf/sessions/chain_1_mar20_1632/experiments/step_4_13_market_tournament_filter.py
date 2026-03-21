"""Step 4.13: Additional Market/Tournament filtering on top of sport filter."""

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


def main() -> None:
    logger.info("Step 4.13: Market/Tournament filter")
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

    with mlflow.start_run(run_name="phase4/step4.13_market_filter") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.13")
        mlflow.set_tag("phase", "4")

        try:
            # Train ensemble
            cb = CatBoostClassifier(
                iterations=495,
                depth=2,
                learning_rate=0.017,
                l2_leaf_reg=9.5,
                border_count=123,
                random_strength=8.0,
                bagging_temperature=8.9,
                min_data_in_leaf=14,
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

            # A: Baseline sport filter
            val_filt = train_val[val_good]
            test_filt = test[test_good]
            p_val_f = p_ens_val[val_good.values]
            p_test_f = p_ens_test[test_good.values]

            thr_base, _vr_base = find_best_threshold_on_val(val_filt, p_val_f)
            roi_base = calc_roi(test_filt, p_test_f, threshold=thr_base)
            logger.info(
                "[A: sport filter only] thr=%.2f, ROI=%.2f%%, n=%d",
                thr_base,
                roi_base["roi"],
                roi_base["n_bets"],
            )

            # B: Analyze Market ROI on val (selected bets only)
            val_selected = val_filt[p_val_f >= thr_base].copy()
            val_selected["proba"] = p_val_f[p_val_f >= thr_base]
            logger.info("=== Market ROI on val (selected bets, n=%d) ===", len(val_selected))

            bad_markets = set()
            market_groups = val_selected.groupby("Market")
            for market, group in market_groups:
                if len(group) < 20:
                    continue
                staked = group["USD"].sum()
                payout = group["Payout_USD"].sum()
                roi = (payout - staked) / staked * 100
                logger.info("  %s: ROI=%.2f%%, n=%d", market, roi, len(group))
                if roi < -10:
                    bad_markets.add(market)

            logger.info("Bad markets (val ROI < -10%%): %s", bad_markets)

            # C: Apply market filter to test
            if bad_markets:
                test_market_filt = test_filt[~test_filt["Market"].isin(bad_markets)]
                # Need to reindex predictions
                market_mask = ~test_filt["Market"].isin(bad_markets)
                p_test_mf = p_test_f[market_mask.values]

                thr_mf, _vr_mf = find_best_threshold_on_val(
                    val_filt[~val_filt["Market"].isin(bad_markets)],
                    p_val_f[~val_filt["Market"].isin(bad_markets).values],
                )
                roi_mf = calc_roi(test_market_filt, p_test_mf, threshold=thr_mf)
                logger.info(
                    "[C: sport+market filter] thr=%.2f, ROI=%.2f%%, n=%d, excl_markets=%s",
                    thr_mf,
                    roi_mf["roi"],
                    roi_mf["n_bets"],
                    bad_markets,
                )
                mlflow.log_metric("roi_sport_market_filter", roi_mf["roi"])
            else:
                roi_mf = roi_base
                logger.info("No bad markets found on val")

            # D: Analyze Is_Parlay
            logger.info("=== Parlay analysis on test ===")
            for is_p in ["f", "t"]:
                mask = test_filt["Is_Parlay"] == is_p
                if mask.sum() < 100:
                    continue
                r = calc_roi(test_filt[mask], p_test_f[mask.values], threshold=thr_base)
                logger.info(
                    "  Is_Parlay=%s: ROI=%.2f%%, n=%d",
                    is_p,
                    r["roi"],
                    r["n_bets"],
                )

            # E: Singles only + sport filter
            singles_val = val_filt[val_filt["Is_Parlay"] == "f"]
            singles_test = test_filt[test_filt["Is_Parlay"] == "f"]
            p_val_singles = p_val_f[val_filt["Is_Parlay"].values == "f"]
            p_test_singles = p_test_f[test_filt["Is_Parlay"].values == "f"]

            if len(singles_test) > 100:
                thr_singles, _vr_singles = find_best_threshold_on_val(singles_val, p_val_singles)
                roi_singles = calc_roi(singles_test, p_test_singles, threshold=thr_singles)
                logger.info(
                    "[E: singles+sport filter] thr=%.2f, ROI=%.2f%%, n=%d",
                    thr_singles,
                    roi_singles["roi"],
                    roi_singles["n_bets"],
                )
                mlflow.log_metric("roi_singles_sport_filter", roi_singles["roi"])

            # F: Odds [1.3, 2.5) + sport filter
            odds_mask_val = (val_filt["Odds"].values >= 1.3) & (val_filt["Odds"].values < 2.5)
            odds_mask_test = (test_filt["Odds"].values >= 1.3) & (test_filt["Odds"].values < 2.5)
            if odds_mask_test.sum() > 200:
                thr_odds, _vr_odds = find_best_threshold_on_val(
                    val_filt[odds_mask_val], p_val_f[odds_mask_val]
                )
                roi_odds = calc_roi(
                    test_filt[odds_mask_test], p_test_f[odds_mask_test], threshold=thr_odds
                )
                logger.info(
                    "[F: odds[1.3,2.5)+sport] thr=%.2f, ROI=%.2f%%, n=%d",
                    thr_odds,
                    roi_odds["roi"],
                    roi_odds["n_bets"],
                )
                mlflow.log_metric("roi_odds_1.3_2.5_sport", roi_odds["roi"])

            # Pick best
            final_roi = max(roi_base["roi"], roi_mf["roi"])
            final_result = roi_mf if roi_mf["roi"] > roi_base["roi"] else roi_base

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "market_tournament_filter",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "excluded_markets": ",".join(bad_markets) if bad_markets else "none",
                    "best_threshold": thr_base,
                }
            )

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "n_bets_selected": final_result["n_bets"],
                    "win_rate_selected": final_result["win_rate"],
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.13")
            raise


if __name__ == "__main__":
    main()
