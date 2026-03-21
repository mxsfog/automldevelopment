"""Step 4.6: Final robustness validation (5-fold temporal CV)."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
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
    """5-fold temporal CV for final robustness of full-train model."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    # Full dataset split for final test
    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    # 5-fold temporal CV on the entire ELO+SF dataset
    all_sf = pd.concat([train_sf, test_sf]).sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    logger.info("5-fold temporal CV on ALL ELO+SF data (n=%d, block=%d)", n, block_size)

    fold_results = []
    for fold_idx in range(n_folds):
        check_budget()
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(train_end + block_size, n)

        fold_train = all_sf.iloc[:train_end].copy()
        fold_test = all_sf.iloc[test_start:test_end].copy()

        if len(fold_train) < 100 or len(fold_test) < 20:
            continue

        # Inner val for threshold + iteration detection
        val_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:val_split]
        inner_val = fold_train.iloc[val_split:]

        imp = SimpleImputer(strategy="median")
        x_inner_train = imp.fit_transform(inner_train[feat_list])
        x_inner_val = imp.transform(inner_val[feat_list])

        cb_iter = CatBoostClassifier(**CB_PARAMS)
        cb_iter.fit(
            x_inner_train,
            inner_train["target"],
            eval_set=(x_inner_val, inner_val["target"]),
        )
        best_iter = cb_iter.get_best_iteration()
        p_inner_val = cb_iter.predict_proba(x_inner_val)[:, 1]
        t_val, _ = find_best_threshold_on_val(inner_val, p_inner_val, min_bets=10)

        # Full-train model
        imp_full = SimpleImputer(strategy="median")
        x_full = imp_full.fit_transform(fold_train[feat_list])
        x_test = imp_full.transform(fold_test[feat_list])

        params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
        params_ft["iterations"] = max(best_iter + 10, 50)
        cb_full = CatBoostClassifier(**params_ft)
        cb_full.fit(x_full, fold_train["target"])

        p_test = cb_full.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(fold_test["target"], p_test)
        roi_t77 = calc_roi(fold_test, p_test, threshold=0.77)
        roi_t76 = calc_roi(fold_test, p_test, threshold=0.76)
        roi_val_t = calc_roi(fold_test, p_test, threshold=t_val)

        # 80/20 model for comparison
        p_test_80 = cb_iter.predict_proba(imp.transform(fold_test[feat_list]))[:, 1]
        roi_t77_80 = calc_roi(fold_test, p_test_80, threshold=0.77)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_n": len(fold_train),
                "test_n": len(fold_test),
                "auc": auc,
                "roi_t77_full": roi_t77["roi"],
                "roi_t76_full": roi_t76["roi"],
                "roi_val_t_full": roi_val_t["roi"],
                "n_bets_t77": roi_t77["n_bets"],
                "n_bets_t76": roi_t76["n_bets"],
                "val_threshold": t_val,
                "best_iter": best_iter,
                "roi_t77_80": roi_t77_80["roi"],
                "test_dates": (
                    f"{fold_test['Created_At'].min()} to {fold_test['Created_At'].max()}"
                ),
            }
        )

        logger.info(
            "  Fold %d: full77=%.2f%% n=%d | full76=%.2f%% | 80/20=%.2f%% | AUC=%.4f | %s",
            fold_idx,
            roi_t77["roi"],
            roi_t77["n_bets"],
            roi_t76["roi"],
            roi_t77_80["roi"],
            auc,
            fold_results[-1]["test_dates"],
        )

    # Summary stats
    rois_full_77 = [r["roi_t77_full"] for r in fold_results]
    rois_full_76 = [r["roi_t76_full"] for r in fold_results]
    rois_80 = [r["roi_t77_80"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]

    avg_roi_full_77 = np.mean(rois_full_77)
    std_roi_full_77 = np.std(rois_full_77)
    avg_roi_full_76 = np.mean(rois_full_76)
    avg_roi_80 = np.mean(rois_80)
    avg_auc = np.mean(aucs)
    positive_folds = sum(1 for r in rois_full_77 if r > 0)

    logger.info("CV Summary:")
    logger.info(
        "  Full-train t=0.77: avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_roi_full_77,
        std_roi_full_77,
        positive_folds,
        len(fold_results),
    )
    logger.info("  Full-train t=0.76: avg=%.2f%%", avg_roi_full_76)
    logger.info("  80/20 t=0.77: avg=%.2f%%", avg_roi_80)
    logger.info("  AUC: avg=%.4f", avg_auc)

    # Final test result
    check_budget()
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp_final = SimpleImputer(strategy="median")
    x_fit_final = imp_final.fit_transform(train_fit[feat_list])
    x_val_final = imp_final.transform(val_df[feat_list])

    final_ref = CatBoostClassifier(**CB_PARAMS)
    final_ref.fit(x_fit_final, train_fit["target"], eval_set=(x_val_final, val_df["target"]))
    final_best_iter = final_ref.get_best_iteration()

    imp_full_final = SimpleImputer(strategy="median")
    x_full_final = imp_full_final.fit_transform(train_sf[feat_list])
    x_test_final = imp_full_final.transform(test_sf[feat_list])

    params_final = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_final["iterations"] = final_best_iter + 10
    final_model = CatBoostClassifier(**params_final)
    final_model.fit(x_full_final, train_sf["target"])
    p_final = final_model.predict_proba(x_test_final)[:, 1]
    auc_final = roc_auc_score(test_sf["target"], p_final)
    roi_final_77 = calc_roi(test_sf, p_final, threshold=0.77)
    roi_final_76 = calc_roi(test_sf, p_final, threshold=0.76)

    logger.info(
        "Final test: ROI(0.77)=%.2f%% n=%d | ROI(0.76)=%.2f%% n=%d | AUC=%.4f",
        roi_final_77["roi"],
        roi_final_77["n_bets"],
        roi_final_76["roi"],
        roi_final_76["n_bets"],
        auc_final,
    )

    with mlflow.start_run(run_name="phase4/step4.6_robustness_final") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "full_train_5fold_cv",
                    "n_features": len(feat_list),
                    "n_folds": len(fold_results),
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "final_best_iter": final_best_iter,
                }
            )

            for fr in fold_results:
                mlflow.log_metric(f"cv_roi_full77_fold{fr['fold']}", fr["roi_t77_full"])
                mlflow.log_metric(f"cv_roi_full76_fold{fr['fold']}", fr["roi_t76_full"])
                mlflow.log_metric(f"cv_roi_80_fold{fr['fold']}", fr["roi_t77_80"])
                mlflow.log_metric(f"cv_auc_fold{fr['fold']}", fr["auc"])

            mlflow.log_metrics(
                {
                    "roi": roi_final_77["roi"],
                    "roc_auc": auc_final,
                    "n_bets": roi_final_77["n_bets"],
                    "win_rate": roi_final_77["win_rate"],
                    "best_threshold": 0.77,
                    "cv_avg_roi_full77": avg_roi_full_77,
                    "cv_std_roi_full77": std_roi_full_77,
                    "cv_avg_roi_full76": avg_roi_full_76,
                    "cv_avg_roi_80": avg_roi_80,
                    "cv_avg_auc": avg_auc,
                    "cv_positive_folds": positive_folds,
                    "roi_final_76": roi_final_76["roi"],
                    "roi_final_77": roi_final_77["roi"],
                    "auc_final": auc_final,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.6: Final ROI(0.77)=%.2f%% AUC=%.4f, CV avg=%.2f%% run=%s",
                roi_final_77["roi"],
                auc_final,
                avg_roi_full_77,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
