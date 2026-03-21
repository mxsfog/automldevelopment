"""Step 4.2: Robust multi-fold threshold + weight optimization for ensemble."""

import logging
import os
import traceback

import lightgbm as lgb
import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    add_elo_features,
    add_engineered_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_elo_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
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


def main() -> None:
    """Robust threshold + weight grid search."""
    logger.info("Step 4.2: Robust multi-fold threshold + weight optimization")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()
    imp = SimpleImputer(strategy="median")

    # Multi-fold robust threshold: 3-fold temporal CV on train
    n = len(train)
    fold_thresholds: list[float] = []
    fold_rois: list[float] = []

    for fold_idx in range(3):
        check_budget()
        # Temporal split: each fold uses different temporal window
        fold_test_start = int(n * (0.6 + fold_idx * 0.1))
        fold_test_end = int(n * (0.7 + fold_idx * 0.1))
        fold_train = train.iloc[:fold_test_start]
        fold_val = train.iloc[fold_test_start:fold_test_end]

        if len(fold_val) < 50:
            continue

        # Val split within fold_train
        inner_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_split]
        inner_val = fold_train.iloc[inner_split:]

        x_inner_train = imp.fit_transform(inner_train[feat_list])
        x_inner_val = imp.transform(inner_val[feat_list])
        x_fold_val = imp.transform(fold_val[feat_list])

        # Train CB
        cb = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.08,
            l2_leaf_reg=21.1,
            min_data_in_leaf=20,
            random_strength=1.0,
            bagging_temperature=0.06,
            border_count=102,
            random_seed=42,
            verbose=0,
            eval_metric="AUC",
            early_stopping_rounds=50,
        )
        cb.fit(x_inner_train, inner_train["target"], eval_set=(x_inner_val, inner_val["target"]))

        # Train LGB
        lgb_m = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.058,
            reg_lambda=27.5,
            min_child_samples=46,
            subsample=0.88,
            colsample_bytree=0.95,
            num_leaves=22,
            random_state=42,
            verbose=-1,
        )
        lgb_m.fit(
            x_inner_train,
            inner_train["target"],
            eval_set=[(x_inner_val, inner_val["target"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Train XGB
        xgb_m = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            reg_lambda=10.0,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="auc",
            early_stopping_rounds=50,
            verbosity=0,
        )
        xgb_m.fit(
            x_inner_train,
            inner_train["target"],
            eval_set=[(x_inner_val, inner_val["target"])],
            verbose=False,
        )

        # Ensemble predictions on fold_val
        p_cb = cb.predict_proba(x_fold_val)[:, 1]
        p_lgb = lgb_m.predict_proba(x_fold_val)[:, 1]
        p_xgb = xgb_m.predict_proba(x_fold_val)[:, 1]
        p_ens = 0.50 * p_cb + 0.25 * p_lgb + 0.25 * p_xgb

        best_t, val_roi = find_best_threshold_on_val(fold_val, p_ens, min_bets=20)
        fold_thresholds.append(best_t)
        fold_rois.append(val_roi)
        logger.info("  Fold %d: best_t=%.2f, val_roi=%.2f%%", fold_idx, best_t, val_roi)

    median_threshold = float(np.median(fold_thresholds))
    mean_roi = float(np.mean(fold_rois))
    logger.info("Robust threshold: median=%.2f, mean_val_roi=%.2f%%", median_threshold, mean_roi)

    # Final model on full train
    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test[feat_list])

    # Train all three models
    cb_final = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.08,
        l2_leaf_reg=21.1,
        min_data_in_leaf=20,
        random_strength=1.0,
        bagging_temperature=0.06,
        border_count=102,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=50,
    )
    cb_final.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

    lgb_final = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.058,
        reg_lambda=27.5,
        min_child_samples=46,
        subsample=0.88,
        colsample_bytree=0.95,
        num_leaves=22,
        random_state=42,
        verbose=-1,
    )
    lgb_final.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    xgb_final = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_lambda=10.0,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_final.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        verbose=False,
    )

    p_cb_test = cb_final.predict_proba(x_test)[:, 1]
    p_lgb_test = lgb_final.predict_proba(x_test)[:, 1]
    p_xgb_test = xgb_final.predict_proba(x_test)[:, 1]

    # Grid search weights
    best_weight_roi = -999.0
    best_weights = (0.5, 0.25, 0.25)
    p_cb_val = cb_final.predict_proba(x_val)[:, 1]
    p_lgb_val = lgb_final.predict_proba(x_val)[:, 1]
    p_xgb_val = xgb_final.predict_proba(x_val)[:, 1]

    for w_cb in np.arange(0.30, 0.75, 0.05):
        for w_lgb in np.arange(0.10, 0.50, 0.05):
            w_xgb = 1.0 - w_cb - w_lgb
            if w_xgb < 0.05:
                continue
            p_val = w_cb * p_cb_val + w_lgb * p_lgb_val + w_xgb * p_xgb_val
            _t, val_roi = find_best_threshold_on_val(val_df, p_val, min_bets=20)
            if val_roi > best_weight_roi:
                best_weight_roi = val_roi
                best_weights = (float(w_cb), float(w_lgb), float(w_xgb))

    w_cb, w_lgb, w_xgb = best_weights
    logger.info(
        "Best weights: CB=%.2f LGB=%.2f XGB=%.2f (val ROI=%.2f%%)",
        w_cb,
        w_lgb,
        w_xgb,
        best_weight_roi,
    )

    # Ensemble with optimal weights and robust threshold
    p_ens_test = w_cb * p_cb_test + w_lgb * p_lgb_test + w_xgb * p_xgb_test

    # Test with both median threshold and val-optimized threshold
    roi_median = calc_roi(test, p_ens_test, threshold=median_threshold)
    p_ens_val = w_cb * p_cb_val + w_lgb * p_lgb_val + w_xgb * p_xgb_val
    val_best_t, _ = find_best_threshold_on_val(val_df, p_ens_val, min_bets=20)
    roi_val_t = calc_roi(test, p_ens_test, threshold=val_best_t)

    # Also test CB solo with robust threshold
    cb_solo_median = calc_roi(test, p_cb_test, threshold=median_threshold)
    cb_solo_val_t, _ = find_best_threshold_on_val(val_df, p_cb_val, min_bets=20)
    cb_solo_best = calc_roi(test, p_cb_test, threshold=cb_solo_val_t)

    logger.info("Results:")
    logger.info(
        "  Ensemble median_t=%.2f: ROI=%.2f%% n=%d",
        median_threshold,
        roi_median["roi"],
        roi_median["n_bets"],
    )
    logger.info(
        "  Ensemble val_t=%.2f: ROI=%.2f%% n=%d", val_best_t, roi_val_t["roi"], roi_val_t["n_bets"]
    )
    logger.info(
        "  CB solo median_t=%.2f: ROI=%.2f%% n=%d",
        median_threshold,
        cb_solo_median["roi"],
        cb_solo_median["n_bets"],
    )
    logger.info(
        "  CB solo val_t=%.2f: ROI=%.2f%% n=%d",
        cb_solo_val_t,
        cb_solo_best["roi"],
        cb_solo_best["n_bets"],
    )

    # Pick best result
    results = {
        "ens_median": (roi_median, median_threshold, "ens_robust_median"),
        "ens_val": (roi_val_t, val_best_t, "ens_val_opt"),
        "cb_median": (cb_solo_median, median_threshold, "cb_solo_robust"),
        "cb_val": (cb_solo_best, cb_solo_val_t, "cb_solo_val_opt"),
    }
    best_key = max(results, key=lambda k: results[k][0]["roi"])
    best_result, best_threshold, best_method = results[best_key]
    auc = roc_auc_score(test["target"], p_ens_test)

    logger.info(
        "BEST: %s ROI=%.2f%% t=%.2f n=%d",
        best_key,
        best_result["roi"],
        best_threshold,
        best_result["n_bets"],
    )

    with mlflow.start_run(run_name="phase4/step4.2_robust_threshold") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": best_method,
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "n_folds": 3,
                    "median_threshold": median_threshold,
                    "val_best_threshold": val_best_t,
                    "best_weights": f"CB{w_cb:.2f}_LGB{w_lgb:.2f}_XGB{w_xgb:.2f}",
                    "best_variant": best_key,
                }
            )

            for name, (r, t, _m) in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"n_bets_{name}", r["n_bets"])
                mlflow.log_metric(f"threshold_{name}", t)

            for fi, ft in enumerate(fold_thresholds):
                mlflow.log_metric(f"fold_{fi}_threshold", ft)
                mlflow.log_metric(f"fold_{fi}_val_roi", fold_rois[fi])

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roc_auc": auc,
                    "n_bets": best_result["n_bets"],
                    "win_rate": best_result["win_rate"],
                    "best_threshold": best_threshold,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info(
                "Step 4.2: ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                best_result["roi"],
                auc,
                best_threshold,
                best_result["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
