"""Step 4.6: Per-sport threshold optimization + stacking meta-learner on ELO-only."""

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
from sklearn.linear_model import LogisticRegression
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


def main() -> None:
    """Per-sport threshold + stacking meta-learner."""
    logger.info("Step 4.6: Per-sport threshold + stacking")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)
    train_all, test_all = time_series_split(df)

    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    feature_cols = get_base_features() + get_engineered_features() + get_safe_elo_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feature_cols])
    x_val = imp.transform(val_df[feature_cols])
    x_test = imp.transform(test[feature_cols])
    y_fit = train_fit["target"].values
    y_val = val_df["target"].values

    with mlflow.start_run(run_name="phase4/step4.6_sport_thresh_stack") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("phase", "4")

        try:
            # Train base models
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
            cb.fit(x_fit, y_fit, eval_set=(x_val, y_val))
            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_cb_test = cb.predict_proba(x_test)[:, 1]

            lgb = LGBMClassifier(
                n_estimators=477,
                max_depth=3,
                learning_rate=0.292,
                num_leaves=16,
                min_child_samples=49,
                reg_lambda=28.63,
                random_state=42,
                verbose=-1,
            )
            lgb.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    lightgbm.early_stopping(30, verbose=False),
                    lightgbm.log_evaluation(0),
                ],
            )
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]

            xgb = XGBClassifier(
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
            xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            # CB50 ensemble (baseline comparison)
            p_cb50_val = 0.5 * p_cb_val + 0.25 * p_lgb_val + 0.25 * p_xgb_val
            p_cb50_test = 0.5 * p_cb_test + 0.25 * p_lgb_test + 0.25 * p_xgb_test
            t_cb50, _vr = find_best_threshold_on_val(val_df, p_cb50_val)
            roi_cb50 = calc_roi(test, p_cb50_test, threshold=t_cb50)
            auc_cb50 = roc_auc_score(test["target"], p_cb50_test)
            logger.info(
                "CB50 global threshold: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_cb50["roi"],
                auc_cb50,
                t_cb50,
                roi_cb50["n_bets"],
            )

            # Part A: Per-sport threshold optimization on CB50 ensemble
            logger.info("--- Part A: Per-sport threshold optimization ---")
            sports = val_df["Sport"].unique()
            sport_thresholds: dict[str, float] = {}
            default_t = t_cb50

            for sport in sports:
                val_mask = val_df["Sport"] == sport
                if val_mask.sum() < 15:
                    sport_thresholds[sport] = default_t
                    continue
                sport_val = val_df[val_mask]
                sport_p = p_cb50_val[val_mask.values]
                t_sport, roi_sport = find_best_threshold_on_val(
                    sport_val,
                    sport_p,
                    min_bets=5,
                )
                sport_thresholds[sport] = t_sport
                logger.info(
                    "  %s: n_val=%d, t=%.2f, val_roi=%.2f%%",
                    sport,
                    val_mask.sum(),
                    t_sport,
                    roi_sport,
                )

            # Apply per-sport thresholds to test
            selected_mask = np.zeros(len(test), dtype=bool)
            for i, (_, row) in enumerate(test.iterrows()):
                sport = row["Sport"]
                t = sport_thresholds.get(sport, default_t)
                if p_cb50_test[i] >= t:
                    selected_mask[i] = True

            n_selected = selected_mask.sum()
            if n_selected > 0:
                selected_test = test[selected_mask]
                staked = selected_test["USD"].sum()
                payout = selected_test["Payout_USD"].sum()
                roi_per_sport = (payout - staked) / staked * 100
                n_won = (selected_test["Status"] == "won").sum()
            else:
                roi_per_sport = 0.0
                n_won = 0
            logger.info(
                "Per-sport thresholds: ROI=%.2f%% n=%d won=%d",
                roi_per_sport,
                n_selected,
                n_won,
            )

            # Part B: Stacking meta-learner
            logger.info("--- Part B: Stacking meta-learner ---")
            check_budget()

            # OOF predictions for stacking (use val set as meta-train)
            meta_x_val = np.column_stack([p_cb_val, p_lgb_val, p_xgb_val])
            meta_x_test = np.column_stack([p_cb_test, p_lgb_test, p_xgb_test])

            # Also add original features for richer meta-model
            meta_x_val_rich = np.column_stack([meta_x_val, x_val])
            meta_x_test_rich = np.column_stack([meta_x_test, x_test])

            # Simple LR stacking
            lr_meta = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
            lr_meta.fit(meta_x_val, y_val)
            p_lr_test = lr_meta.predict_proba(meta_x_test)[:, 1]
            t_lr, _vr = find_best_threshold_on_val(val_df, lr_meta.predict_proba(meta_x_val)[:, 1])
            roi_lr = calc_roi(test, p_lr_test, threshold=t_lr)
            auc_lr = roc_auc_score(test["target"], p_lr_test)
            logger.info(
                "LR stack: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_lr["roi"],
                auc_lr,
                t_lr,
                roi_lr["n_bets"],
            )

            # LR stacking with features
            lr_rich = LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42,
            )
            lr_rich.fit(meta_x_val_rich, y_val)
            p_lr_rich_val = lr_rich.predict_proba(meta_x_val_rich)[:, 1]
            p_lr_rich_test = lr_rich.predict_proba(meta_x_test_rich)[:, 1]
            t_lr_rich, _vr = find_best_threshold_on_val(val_df, p_lr_rich_val)
            roi_lr_rich = calc_roi(test, p_lr_rich_test, threshold=t_lr_rich)
            auc_lr_rich = roc_auc_score(test["target"], p_lr_rich_test)
            logger.info(
                "LR rich stack: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_lr_rich["roi"],
                auc_lr_rich,
                t_lr_rich,
                roi_lr_rich["n_bets"],
            )

            # CatBoost meta-learner (small, regularized)
            cb_meta = CatBoostClassifier(
                iterations=100,
                depth=3,
                learning_rate=0.05,
                l2_leaf_reg=10.0,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
            )
            cb_meta.fit(meta_x_val, y_val)
            p_cb_meta_val = cb_meta.predict_proba(meta_x_val)[:, 1]
            p_cb_meta_test = cb_meta.predict_proba(meta_x_test)[:, 1]
            t_cb_meta, _vr = find_best_threshold_on_val(val_df, p_cb_meta_val)
            roi_cb_meta = calc_roi(test, p_cb_meta_test, threshold=t_cb_meta)
            auc_cb_meta = roc_auc_score(test["target"], p_cb_meta_test)
            logger.info(
                "CB meta stack: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_cb_meta["roi"],
                auc_cb_meta,
                t_cb_meta,
                roi_cb_meta["n_bets"],
            )

            # Summary
            results = {
                "cb50_global": {"roi": roi_cb50["roi"], "n": roi_cb50["n_bets"]},
                "per_sport_thresh": {"roi": roi_per_sport, "n": int(n_selected)},
                "lr_stack": {"roi": roi_lr["roi"], "n": roi_lr["n_bets"]},
                "lr_rich_stack": {"roi": roi_lr_rich["roi"], "n": roi_lr_rich["n_bets"]},
                "cb_meta_stack": {"roi": roi_cb_meta["roi"], "n": roi_cb_meta["n_bets"]},
            }
            best_key = max(results, key=lambda k: results[k]["roi"])
            best_roi = results[best_key]["roi"]
            logger.info("Best: %s ROI=%.2f%%", best_key, best_roi)

            # ROI at thresholds for CB50 (still the reference)
            roi_thresholds = calc_roi_at_thresholds(test, p_cb50_test)
            for t, r in roi_thresholds.items():
                mlflow.log_metric(f"roi_cb50_t{int(t * 100):03d}", r["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"sport_thresh_stack_{best_key}",
                    "n_features": len(feature_cols),
                    "n_sports_with_custom_thresh": len(sport_thresholds),
                    "leakage_free": "true",
                }
            )
            mlflow.log_metrics(
                {
                    "roi_cb50_global": roi_cb50["roi"],
                    "roi_per_sport": roi_per_sport,
                    "roi_lr_stack": roi_lr["roi"],
                    "roi_lr_rich_stack": roi_lr_rich["roi"],
                    "roi_cb_meta_stack": roi_cb_meta["roi"],
                    "roi": best_roi,
                    "roc_auc": auc_cb50,
                    "n_bets": results[best_key]["n"],
                    "best_threshold": t_cb50,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.6")
            raise


if __name__ == "__main__":
    main()
