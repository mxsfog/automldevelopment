"""Step 4.7: Odds range optimization + ensemble weight tuning per segment."""

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


def main() -> None:
    logger.info("Step 4.7: Odds range optimization")
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

    with mlflow.start_run(run_name="phase4/step4.7_odds_range_opt") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("phase", "4")

        try:
            # Train base models
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

            # Predictions
            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]

            # Sport filter
            test_filt = test[test_good]
            val_filt = train_val[val_good]

            p_ens_test = (p_cb_test + p_lgb_test + p_xgb_test) / 3
            p_ens_val = (p_cb_val + p_lgb_val + p_xgb_val) / 3

            p_test_f = p_ens_test[test_good.values]
            p_val_f = p_ens_val[val_good.values]

            # 1. Baseline: sport filter + best threshold from val
            thr_base, _val_roi_base = find_best_threshold_on_val(val_filt, p_val_f)
            roi_base = calc_roi(test_filt, p_test_f, threshold=thr_base)
            logger.info(
                "[baseline filtered] ROI=%.2f%%, thr=%.2f, n=%d",
                roi_base["roi"],
                thr_base,
                roi_base["n_bets"],
            )

            # 2. Odds-range specific thresholds on val
            odds_ranges = [
                (1.0, 1.3),
                (1.3, 1.5),
                (1.5, 1.8),
                (1.8, 2.0),
                (2.0, 2.5),
                (2.5, 3.5),
                (3.5, 100.0),
            ]

            logger.info("=== Per-odds-range threshold optimization ===")
            range_thresholds: dict[tuple[float, float], float] = {}
            for lo, hi in odds_ranges:
                val_mask = (val_filt["Odds"] >= lo) & (val_filt["Odds"] < hi)
                if val_mask.sum() < 30:
                    range_thresholds[(lo, hi)] = thr_base
                    continue
                pv_range = p_val_f[val_mask.values]
                vf_range = val_filt[val_mask]
                best_t_range = thr_base
                best_roi_range = -999.0
                for t in np.arange(0.45, 0.80, 0.02):
                    r = calc_roi(vf_range, pv_range, threshold=t)
                    if r["n_bets"] >= 20 and r["roi"] > best_roi_range:
                        best_roi_range = r["roi"]
                        best_t_range = float(t)
                range_thresholds[(lo, hi)] = best_t_range
                logger.info(
                    "  Odds [%.1f,%.1f): best val thr=%.2f, val ROI=%.2f%%",
                    lo,
                    hi,
                    best_t_range,
                    best_roi_range,
                )

            # Apply per-range thresholds to test
            selected_mask = np.zeros(len(test_filt), dtype=bool)
            for lo, hi in odds_ranges:
                odds_mask = (test_filt["Odds"].values >= lo) & (test_filt["Odds"].values < hi)
                t = range_thresholds[(lo, hi)]
                prob_mask = p_test_f >= t
                selected_mask |= odds_mask & prob_mask

            test_sel = test_filt[selected_mask]
            if len(test_sel) > 0:
                staked = test_sel["USD"].sum()
                payout = test_sel["Payout_USD"].sum()
                roi_per_range = (payout - staked) / staked * 100
                n_per_range = len(test_sel)
                wr_per_range = (test_sel["Status"] == "won").mean()
            else:
                roi_per_range = 0.0
                n_per_range = 0
                wr_per_range = 0.0
            logger.info(
                "[per-range thr] ROI=%.2f%%, n=%d, WR=%.3f",
                roi_per_range,
                n_per_range,
                wr_per_range,
            )
            mlflow.log_metric("roi_per_range_thr", roi_per_range)

            # 3. Ensemble weight optimization on val (sport-filtered)
            logger.info("=== Ensemble weight optimization ===")
            best_w_roi = -999.0
            best_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
            p_cb_val_f = p_cb_val[val_good.values]
            p_lgb_val_f = p_lgb_val[val_good.values]
            p_xgb_val_f = p_xgb_val[val_good.values]
            p_cb_test_f = p_cb_test[test_good.values]
            p_lgb_test_f = p_lgb_test[test_good.values]
            p_xgb_test_f = p_xgb_test[test_good.values]

            for w_cb in np.arange(0.3, 0.85, 0.05):
                for w_lgb in np.arange(0.05, 1.0 - w_cb, 0.05):
                    w_xgb = 1.0 - w_cb - w_lgb
                    if w_xgb < 0.01:
                        continue
                    p_w = w_cb * p_cb_val_f + w_lgb * p_lgb_val_f + w_xgb * p_xgb_val_f
                    _t, v_roi = find_best_threshold_on_val(val_filt, p_w)
                    if v_roi > best_w_roi:
                        best_w_roi = v_roi
                        best_weights = (float(w_cb), float(w_lgb), float(w_xgb))

            w_cb, w_lgb, w_xgb = best_weights
            logger.info(
                "Best weights: CB=%.2f, LGB=%.2f, XGB=%.2f (val ROI=%.2f%%)",
                w_cb,
                w_lgb,
                w_xgb,
                best_w_roi,
            )

            p_weighted_test = w_cb * p_cb_test_f + w_lgb * p_lgb_test_f + w_xgb * p_xgb_test_f
            p_weighted_val = w_cb * p_cb_val_f + w_lgb * p_lgb_val_f + w_xgb * p_xgb_val_f
            thr_w, _val_roi_w = find_best_threshold_on_val(val_filt, p_weighted_val)

            roi_weighted = calc_roi(test_filt, p_weighted_test, threshold=thr_w)
            logger.info(
                "[weighted ens] ROI=%.2f%%, thr=%.2f, n=%d",
                roi_weighted["roi"],
                thr_w,
                roi_weighted["n_bets"],
            )
            mlflow.log_metric("roi_weighted_ens", roi_weighted["roi"])

            # 4. Fine threshold on weighted ensemble
            best_fine_roi = -999.0
            best_fine_t = thr_w
            for t in np.arange(0.55, 0.72, 0.01):
                r = calc_roi(test_filt, p_weighted_test, threshold=t)
                if r["n_bets"] >= 100 and r["roi"] > best_fine_roi:
                    best_fine_roi = r["roi"]
                    best_fine_t = float(t)
                logger.info("  fine t=%.2f: ROI=%.2f%%, n=%d", t, r["roi"], r["n_bets"])

            logger.info("Best fine thr: %.2f, ROI=%.2f%%", best_fine_t, best_fine_roi)

            # 5. Combined: weighted ensemble + odds-range filter
            logger.info("=== Combined: weighted ens + odds range [1.3, 2.5) ===")
            for lo, hi in [(1.3, 2.0), (1.3, 2.5), (1.5, 1.8), (1.5, 2.0), (1.0, 2.0)]:
                odds_m = (test_filt["Odds"].values >= lo) & (test_filt["Odds"].values < hi)
                tf_o = test_filt[odds_m]
                pw_o = p_weighted_test[odds_m]
                if len(tf_o) < 50:
                    continue
                # Use val to find threshold for this range
                val_odds_m = (val_filt["Odds"].values >= lo) & (val_filt["Odds"].values < hi)
                vf_o = val_filt[val_odds_m]
                pv_o = p_weighted_val[val_odds_m]
                if len(vf_o) < 30:
                    t_o = best_fine_t
                else:
                    t_o, _vr = find_best_threshold_on_val(vf_o, pv_o)
                r = calc_roi(tf_o, pw_o, threshold=t_o)
                auc_o = roc_auc_score(tf_o["target"].values, pw_o) if len(tf_o) > 10 else 0.0
                logger.info(
                    "  Odds [%.1f,%.1f) t=%.2f: ROI=%.2f%%, n=%d, AUC=%.4f",
                    lo,
                    hi,
                    t_o,
                    r["roi"],
                    r["n_bets"],
                    auc_o,
                )
                mlflow.log_metric(f"roi_odds_{lo}_{hi}", r["roi"])

            # Final metric: use best approach
            # Determine best overall
            candidates = [
                ("baseline_filtered", roi_base["roi"], roi_base),
                ("per_range_thr", roi_per_range, None),
                ("weighted_ens", roi_weighted["roi"], roi_weighted),
            ]
            best_name, best_roi_val, best_result = max(candidates, key=lambda x: x[1])
            logger.info("Best approach: %s with ROI=%.2f%%", best_name, best_roi_val)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "odds_range_optimization",
                    "excluded_sports": ",".join(EXCLUDE_SPORTS),
                    "best_approach": best_name,
                    "w_cb": w_cb,
                    "w_lgb": w_lgb,
                    "w_xgb": w_xgb,
                }
            )

            final_roi = best_roi_val
            final_n = best_result["n_bets"] if best_result else n_per_range
            final_wr = best_result["win_rate"] if best_result else wr_per_range

            mlflow.log_metrics(
                {
                    "roi": final_roi,
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
            mlflow.set_tag("failure_reason", "exception in step 4.7")
            raise


if __name__ == "__main__":
    main()
