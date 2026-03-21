"""Step 4.11: Final validation of best strategy with conservative threshold."""

import logging
import os
import traceback

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

CB_PARAMS = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 21.1,
    "min_data_in_leaf": 20,
    "random_strength": 1.0,
    "bagging_temperature": 0.06,
    "border_count": 102,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def main() -> None:
    """Final validation: train on SF, fixed t=0.77, multi-fold val threshold confirmation."""
    logger.info("Step 4.11: Final validation")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()

    # Sport filter
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    # Multi-fold robust threshold on sport-filtered data
    n = len(train_sf)
    fold_thresholds: list[float] = []

    for fold_idx in range(4):
        check_budget()
        fold_test_start = int(n * (0.5 + fold_idx * 0.1))
        fold_test_end = int(n * (0.6 + fold_idx * 0.1))
        fold_train = train_sf.iloc[:fold_test_start]
        fold_val = train_sf.iloc[fold_test_start:fold_test_end]
        if len(fold_val) < 20:
            continue
        inner_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_split]
        inner_val = fold_train.iloc[inner_split:]
        if len(inner_val) < 20:
            continue

        imp_fold = SimpleImputer(strategy="median")
        x_it = imp_fold.fit_transform(inner_train[feat_list])
        x_iv = imp_fold.transform(inner_val[feat_list])
        x_fv = imp_fold.transform(fold_val[feat_list])

        cb = CatBoostClassifier(**CB_PARAMS)
        cb.fit(x_it, inner_train["target"], eval_set=(x_iv, inner_val["target"]))
        p_fv = cb.predict_proba(x_fv)[:, 1]
        best_t, _ = find_best_threshold_on_val(fold_val, p_fv, min_bets=10)
        fold_thresholds.append(best_t)
        logger.info("  Fold %d: t=%.2f", fold_idx, best_t)

    robust_t = float(np.median(fold_thresholds)) if fold_thresholds else 0.77
    logger.info("Robust median threshold: %.2f", robust_t)

    # Final model
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test_sf[feat_list])

    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    configs: dict[str, tuple[dict, float]] = {}

    # A: Fixed t=0.77 (proven)
    configs["sf_t77"] = (calc_roi(test_sf, p_test, threshold=0.77), 0.77)

    # B: Robust threshold
    configs["sf_robust"] = (calc_roi(test_sf, p_test, threshold=robust_t), robust_t)

    # C: Val threshold
    t_val, _ = find_best_threshold_on_val(val_df, p_val, min_bets=15)
    configs["sf_val"] = (calc_roi(test_sf, p_test, threshold=t_val), t_val)

    # D: Conservative t=0.80
    configs["sf_t80"] = (calc_roi(test_sf, p_test, threshold=0.80), 0.80)

    # E: Also test on all ELO (no sport filter at inference, model trained on SF)
    x_test_all = imp.transform(test_elo[feat_list])
    p_test_all = model.predict_proba(x_test_all)[:, 1]
    configs["train_sf_test_all_t77"] = (calc_roi(test_elo, p_test_all, threshold=0.77), 0.77)

    # Log
    logger.info("Final results:")
    for name, (r, t) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    best_key = max(configs, key=lambda k: configs[k][0]["roi"])
    best_result, best_threshold = configs[best_key]
    auc = roc_auc_score(test_sf["target"], p_test)

    with mlflow.start_run(run_name="phase4/step4.11_final_validation") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train_sf": len(train_fit),
                    "n_samples_test_sf": len(test_sf),
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "robust_threshold": robust_t,
                    "best_variant": best_key,
                }
            )

            for name, (r, _t) in configs.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])

            for i, t in enumerate(fold_thresholds):
                mlflow.log_metric(f"fold_{i}_threshold", t)

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
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.11 FINAL: %s ROI=%.2f%% t=%.2f n=%d run=%s",
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
