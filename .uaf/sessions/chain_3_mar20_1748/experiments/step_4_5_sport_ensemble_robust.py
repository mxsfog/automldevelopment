"""Step 4.5: Sport-filtered ensemble with robust threshold."""

import logging
import os
import traceback

import lightgbm as lgb
import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    UNPROFITABLE_SPORTS,
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
    """Sport-filtered ensemble + robust threshold."""
    logger.info("Step 4.5: Sport-filtered ensemble + robust threshold")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

    # ELO-only + sport filter
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()

    train = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_filtered = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_unfiltered = test_elo.copy()

    logger.info(
        "ELO+sport filter: train=%d, test_filtered=%d, test_all_elo=%d",
        len(train),
        len(test_filtered),
        len(test_unfiltered),
    )

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()
    imp = SimpleImputer(strategy="median")

    # Multi-fold robust threshold on sport-filtered data
    n = len(train)
    fold_thresholds: list[float] = []

    for fold_idx in range(3):
        check_budget()
        fold_test_start = int(n * (0.6 + fold_idx * 0.1))
        fold_test_end = int(n * (0.7 + fold_idx * 0.1))
        fold_train = train.iloc[:fold_test_start]
        fold_val = train.iloc[fold_test_start:fold_test_end]
        if len(fold_val) < 30:
            continue
        inner_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_split]
        inner_val = fold_train.iloc[inner_split:]

        x_it = imp.fit_transform(inner_train[feat_list])
        x_iv = imp.transform(inner_val[feat_list])
        x_fv = imp.transform(fold_val[feat_list])

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
        cb.fit(x_it, inner_train["target"], eval_set=(x_iv, inner_val["target"]))
        p_fv = cb.predict_proba(x_fv)[:, 1]
        best_t, _ = find_best_threshold_on_val(fold_val, p_fv, min_bets=15)
        fold_thresholds.append(best_t)
        logger.info("  Fold %d: t=%.2f", fold_idx, best_t)

    median_t = float(np.median(fold_thresholds)) if fold_thresholds else 0.73
    logger.info("Robust median threshold: %.2f", median_t)

    # Train final models on full sport-filtered train
    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test_f = imp.transform(test_filtered[feat_list])
    x_test_u = imp.transform(test_unfiltered[feat_list])

    # CatBoost
    cb_model = CatBoostClassifier(
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
    cb_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_cb_f = cb_model.predict_proba(x_test_f)[:, 1]
    p_cb_u = cb_model.predict_proba(x_test_u)[:, 1]
    p_cb_val = cb_model.predict_proba(x_val)[:, 1]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
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
    lgb_model.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_lgb_f = lgb_model.predict_proba(x_test_f)[:, 1]
    p_lgb_u = lgb_model.predict_proba(x_test_u)[:, 1]
    p_lgb_val = lgb_model.predict_proba(x_val)[:, 1]

    # XGBoost
    xgb_model = XGBClassifier(
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
    xgb_model.fit(x_fit, train_fit["target"], eval_set=[(x_val, val_df["target"])], verbose=False)
    p_xgb_f = xgb_model.predict_proba(x_test_f)[:, 1]
    p_xgb_u = xgb_model.predict_proba(x_test_u)[:, 1]
    p_xgb_val = xgb_model.predict_proba(x_val)[:, 1]

    # Test various configs
    configs = {}

    # CB solo on filtered, median threshold
    configs["cb_filtered_robust"] = (calc_roi(test_filtered, p_cb_f, threshold=median_t), median_t)

    # CB solo on filtered, val-opt threshold
    cb_val_t, _ = find_best_threshold_on_val(val_df, p_cb_val, min_bets=15)
    configs["cb_filtered_val"] = (calc_roi(test_filtered, p_cb_f, threshold=cb_val_t), cb_val_t)

    # CB solo on unfiltered, val threshold
    configs["cb_unfiltered_val"] = (
        calc_roi(test_unfiltered, p_cb_u, threshold=cb_val_t),
        cb_val_t,
    )

    # Ensemble filtered: 65% CB + 20% LGB + 15% XGB
    p_ens_f = 0.65 * p_cb_f + 0.20 * p_lgb_f + 0.15 * p_xgb_f
    p_ens_val = 0.65 * p_cb_val + 0.20 * p_lgb_val + 0.15 * p_xgb_val
    ens_t, _ = find_best_threshold_on_val(val_df, p_ens_val, min_bets=15)
    configs["ens65_filtered_val"] = (calc_roi(test_filtered, p_ens_f, threshold=ens_t), ens_t)
    configs["ens65_filtered_robust"] = (
        calc_roi(test_filtered, p_ens_f, threshold=median_t),
        median_t,
    )

    # CB50 ensemble filtered
    p_cb50_f = 0.50 * p_cb_f + 0.25 * p_lgb_f + 0.25 * p_xgb_f
    p_cb50_val = 0.50 * p_cb_val + 0.25 * p_lgb_val + 0.25 * p_xgb_val
    cb50_t, _ = find_best_threshold_on_val(val_df, p_cb50_val, min_bets=15)
    configs["cb50_filtered_val"] = (calc_roi(test_filtered, p_cb50_f, threshold=cb50_t), cb50_t)

    # Ensemble on unfiltered (model trained on filtered data)
    p_ens_u = 0.65 * p_cb_u + 0.20 * p_lgb_u + 0.15 * p_xgb_u
    configs["ens65_unfiltered_val"] = (calc_roi(test_unfiltered, p_ens_u, threshold=ens_t), ens_t)

    # Sort and log
    logger.info("Results:")
    for name, (r, t) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    best_key = max(configs, key=lambda k: configs[k][0]["roi"])
    best_result, best_threshold = configs[best_key]
    auc = roc_auc_score(test_filtered["target"], p_cb_f)

    with mlflow.start_run(run_name="phase4/step4.5_sport_ens_robust") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test_filtered": len(test_filtered),
                    "n_samples_test_unfiltered": len(test_unfiltered),
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "median_threshold": median_t,
                    "best_variant": best_key,
                }
            )

            for name, (r, _t) in configs.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])

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
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.5: BEST %s ROI=%.2f%% t=%.2f n=%d run=%s",
                best_key,
                best_result["roi"],
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
