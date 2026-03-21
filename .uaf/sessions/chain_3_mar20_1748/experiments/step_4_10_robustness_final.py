"""Step 4.10: Robustness validation of best strategy (CB42 SF t=0.77)."""

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
    get_base_features,
    get_elo_features,
    get_engineered_features,
    load_data,
    set_seed,
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
    """4-fold temporal CV to validate CB42+sport filter strategy."""
    logger.info("Step 4.10: Robustness validation (4-fold temporal CV)")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    df = df.sort_values("Created_At").reset_index(drop=True)
    elo_df = df[df["has_elo"] == 1.0].copy()
    n = len(elo_df)

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()
    imp = SimpleImputer(strategy="median")

    # 4-fold expanding window temporal CV
    fold_results_all: list[dict] = []
    fold_results_sf: list[dict] = []

    for fold_idx in range(4):
        check_budget()
        # Each fold: train on first X%, test on next 10%
        # fold 0: train 0-60%, test 60-70%
        # fold 1: train 0-70%, test 70-80%
        # fold 2: train 0-80%, test 80-90%
        # fold 3: train 0-90%, test 90-100%
        train_end = int(n * (0.6 + fold_idx * 0.1))
        test_end = int(n * (0.7 + fold_idx * 0.1))

        fold_train = elo_df.iloc[:train_end]
        fold_test = elo_df.iloc[train_end:test_end]

        if len(fold_test) < 50:
            logger.info("  Fold %d: skipped (only %d test samples)", fold_idx, len(fold_test))
            continue

        # Val split within train
        val_split = int(len(fold_train) * 0.8)
        train_fit = fold_train.iloc[:val_split]
        val_df = fold_train.iloc[val_split:]

        x_fit = imp.fit_transform(train_fit[feat_list])
        x_val = imp.transform(val_df[feat_list])
        x_test = imp.transform(fold_test[feat_list])

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        p_test = model.predict_proba(x_test)[:, 1]

        # All ELO, t=0.77
        r_all = calc_roi(fold_test, p_test, threshold=0.77)
        auc_all = roc_auc_score(fold_test["target"], p_test)
        fold_results_all.append({"roi": r_all["roi"], "n_bets": r_all["n_bets"], "auc": auc_all})

        # Sport-filtered, t=0.77
        mask_sf = ~fold_test["Sport"].isin(UNPROFITABLE_SPORTS)
        if mask_sf.sum() >= 20:
            r_sf = calc_roi(fold_test[mask_sf], p_test[mask_sf.values], threshold=0.77)
            auc_sf = roc_auc_score(fold_test[mask_sf]["target"], p_test[mask_sf.values])
            fold_results_sf.append({"roi": r_sf["roi"], "n_bets": r_sf["n_bets"], "auc": auc_sf})
        else:
            fold_results_sf.append({"roi": 0.0, "n_bets": 0, "auc": 0.0})

        logger.info(
            "  Fold %d: ALL ROI=%.2f%% n=%d AUC=%.4f | SF ROI=%.2f%% n=%d AUC=%.4f",
            fold_idx,
            fold_results_all[-1]["roi"],
            fold_results_all[-1]["n_bets"],
            fold_results_all[-1]["auc"],
            fold_results_sf[-1]["roi"],
            fold_results_sf[-1]["n_bets"],
            fold_results_sf[-1]["auc"],
        )

    # Summary
    rois_all = [r["roi"] for r in fold_results_all]
    rois_sf = [r["roi"] for r in fold_results_sf if r["n_bets"] > 0]

    mean_roi_all = float(np.mean(rois_all))
    std_roi_all = float(np.std(rois_all))
    mean_roi_sf = float(np.mean(rois_sf)) if rois_sf else 0.0
    std_roi_sf = float(np.std(rois_sf)) if rois_sf else 0.0

    logger.info("Summary:")
    logger.info(
        "  ALL ELO: mean ROI=%.2f%% +/- %.2f%%, min=%.2f%%, max=%.2f%%",
        mean_roi_all,
        std_roi_all,
        min(rois_all),
        max(rois_all),
    )
    logger.info(
        "  Sport-filtered: mean ROI=%.2f%% +/- %.2f%%, min=%.2f%%, max=%.2f%%",
        mean_roi_sf,
        std_roi_sf,
        min(rois_sf) if rois_sf else 0.0,
        max(rois_sf) if rois_sf else 0.0,
    )

    # Sport filter improvement
    improvements = [
        sf["roi"] - all_r["roi"]
        for sf, all_r in zip(fold_results_sf, fold_results_all, strict=True)
        if sf["n_bets"] > 0
    ]
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0
    logger.info("  Sport filter avg improvement: %.2f п.п.", mean_improvement)

    with mlflow.start_run(run_name="phase4/step4.10_robustness") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.10")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "4_fold_temporal_cv",
                    "seed": 42,
                    "method": "cb42_sport_filter_t77",
                    "n_features": len(feat_list),
                    "n_folds": 4,
                    "threshold": 0.77,
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                }
            )

            for i, r_all in enumerate(fold_results_all):
                mlflow.log_metric(f"fold_{i}_roi_all", r_all["roi"])
                mlflow.log_metric(f"fold_{i}_auc_all", r_all["auc"])
                mlflow.log_metric(f"fold_{i}_n_bets_all", r_all["n_bets"])

            for i, r_sf in enumerate(fold_results_sf):
                mlflow.log_metric(f"fold_{i}_roi_sf", r_sf["roi"])
                mlflow.log_metric(f"fold_{i}_auc_sf", r_sf["auc"])
                mlflow.log_metric(f"fold_{i}_n_bets_sf", r_sf["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": mean_roi_sf,
                    "roi_std": std_roi_sf,
                    "roi_mean_all": mean_roi_all,
                    "roi_std_all": std_roi_all,
                    "roc_auc": float(np.mean([r["auc"] for r in fold_results_sf if r["auc"] > 0])),
                    "n_bets": float(
                        np.mean([r["n_bets"] for r in fold_results_sf if r["n_bets"] > 0])
                    ),
                    "sport_filter_improvement": mean_improvement,
                    "best_threshold": 0.77,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.10: CV mean ROI=%.2f%% +/- %.2f%% (SF), run=%s",
                mean_roi_sf,
                std_roi_sf,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
