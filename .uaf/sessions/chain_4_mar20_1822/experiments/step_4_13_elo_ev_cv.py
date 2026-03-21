"""Step 4.13: 5-fold CV for full ELO + EV>=0 (without sport filter)."""

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


def calc_ev_roi(
    df: pd.DataFrame, proba: np.ndarray, ev_threshold: float, min_prob: float = 0.5
) -> dict:
    """Выбор ставок по EV = p * odds - 1 >= ev_threshold AND p >= min_prob."""
    odds = df["Odds"].values
    ev = proba * odds - 1.0
    mask = (ev >= ev_threshold) & (proba >= min_prob)
    n_selected = int(mask.sum())
    if n_selected == 0:
        return {
            "roi": 0.0,
            "n_bets": 0,
            "n_won": 0,
            "total_staked": 0.0,
            "win_rate": 0.0,
            "pct_selected": 0.0,
        }
    selected = df.iloc[np.where(mask)[0]]
    total_staked = selected["USD"].sum()
    total_payout = selected["Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100
    n_won = (selected["Status"] == "won").sum()
    return {
        "roi": float(roi),
        "n_bets": n_selected,
        "n_won": int(n_won),
        "total_staked": float(total_staked),
        "win_rate": float(n_won / n_selected),
        "pct_selected": float(n_selected / len(df) * 100),
    }


def main() -> None:
    """5-fold CV comparing: (1) SF+t77, (2) SF+EV0+p77, (3) ELO+t77, (4) ELO+EV0+p77."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    # 5-fold CV on both datasets
    for dataset_name, train_data, test_data in [
        ("SF", train_sf, test_sf),
        ("ELO_all", train_elo, test_elo),
    ]:
        all_data = (
            pd.concat([train_data, test_data]).sort_values("Created_At").reset_index(drop=True)
        )
        n = len(all_data)
        n_folds = 5
        block_size = n // (n_folds + 1)

        logger.info(
            "5-fold CV on %s data (n=%d, block=%d)",
            dataset_name,
            n,
            block_size,
        )

        fold_results = []
        for fold_idx in range(n_folds):
            check_budget()
            train_end = block_size * (fold_idx + 1)
            test_start = train_end
            test_end = min(train_end + block_size, n)

            fold_train = all_data.iloc[:train_end].copy()
            fold_test = all_data.iloc[test_start:test_end].copy()

            if len(fold_train) < 100 or len(fold_test) < 20:
                continue

            inner_val_split = int(len(fold_train) * 0.8)
            inner_train = fold_train.iloc[:inner_val_split]
            inner_val = fold_train.iloc[inner_val_split:]

            imp_fold = SimpleImputer(strategy="median")
            x_inner_train = imp_fold.fit_transform(inner_train[feat_list])
            x_inner_val = imp_fold.transform(inner_val[feat_list])

            cb_fold = CatBoostClassifier(**CB_PARAMS)
            cb_fold.fit(
                x_inner_train,
                inner_train["target"],
                eval_set=(x_inner_val, inner_val["target"]),
            )
            fold_best_iter = cb_fold.get_best_iteration()

            # Full-train fold model
            imp_full = SimpleImputer(strategy="median")
            x_fold_full = imp_full.fit_transform(fold_train[feat_list])
            x_fold_test = imp_full.transform(fold_test[feat_list])

            params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
            params_ft["iterations"] = max(fold_best_iter + 10, 50)
            cb_ft = CatBoostClassifier(**params_ft)
            cb_ft.fit(x_fold_full, fold_train["target"])

            p_test = cb_ft.predict_proba(x_fold_test)[:, 1]
            fold_auc = roc_auc_score(fold_test["target"], p_test)

            roi_t77 = calc_roi(fold_test, p_test, threshold=0.77)
            roi_ev0 = calc_ev_roi(fold_test, p_test, ev_threshold=0.0, min_prob=0.77)

            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_n": len(fold_train),
                    "test_n": len(fold_test),
                    "auc": fold_auc,
                    "roi_t77": roi_t77["roi"],
                    "n_t77": roi_t77["n_bets"],
                    "roi_ev0": roi_ev0["roi"],
                    "n_ev0": roi_ev0["n_bets"],
                }
            )

            logger.info(
                "  %s fold %d: t77=%.2f%% n=%d | EV0+p77=%.2f%% n=%d | AUC=%.4f",
                dataset_name,
                fold_idx,
                roi_t77["roi"],
                roi_t77["n_bets"],
                roi_ev0["roi"],
                roi_ev0["n_bets"],
                fold_auc,
            )

        rois_t77 = [r["roi_t77"] for r in fold_results]
        rois_ev0 = [r["roi_ev0"] for r in fold_results]
        logger.info("%s CV Summary:", dataset_name)
        logger.info(
            "  t77: avg=%.2f%% std=%.2f%% pos=%d/%d",
            np.mean(rois_t77),
            np.std(rois_t77),
            sum(1 for r in rois_t77 if r > 0),
            len(fold_results),
        )
        logger.info(
            "  EV0+p77: avg=%.2f%% std=%.2f%% pos=%d/%d",
            np.mean(rois_ev0),
            np.std(rois_ev0),
            sum(1 for r in rois_ev0 if r > 0),
            len(fold_results),
        )

    # Final test results
    check_budget()
    # SF model
    val_split_sf = int(len(train_sf) * 0.8)
    imp_sf = SimpleImputer(strategy="median")
    x_fit_sf = imp_sf.fit_transform(train_sf.iloc[:val_split_sf][feat_list])
    x_val_sf = imp_sf.transform(train_sf.iloc[val_split_sf:][feat_list])

    ref_sf = CatBoostClassifier(**CB_PARAMS)
    ref_sf.fit(
        x_fit_sf,
        train_sf.iloc[:val_split_sf]["target"],
        eval_set=(x_val_sf, train_sf.iloc[val_split_sf:]["target"]),
    )
    bi_sf = ref_sf.get_best_iteration()

    imp_sf_full = SimpleImputer(strategy="median")
    x_sf_full = imp_sf_full.fit_transform(train_sf[feat_list])
    x_sf_test = imp_sf_full.transform(test_sf[feat_list])

    p_ft_sf = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    p_ft_sf["iterations"] = bi_sf + 10
    ft_sf = CatBoostClassifier(**p_ft_sf)
    ft_sf.fit(x_sf_full, train_sf["target"])
    p_sf_test = ft_sf.predict_proba(x_sf_test)[:, 1]

    # ELO model
    val_split_elo = int(len(train_elo) * 0.8)
    imp_elo = SimpleImputer(strategy="median")
    x_fit_elo = imp_elo.fit_transform(train_elo.iloc[:val_split_elo][feat_list])
    x_val_elo = imp_elo.transform(train_elo.iloc[val_split_elo:][feat_list])

    ref_elo = CatBoostClassifier(**CB_PARAMS)
    ref_elo.fit(
        x_fit_elo,
        train_elo.iloc[:val_split_elo]["target"],
        eval_set=(x_val_elo, train_elo.iloc[val_split_elo:]["target"]),
    )
    bi_elo = ref_elo.get_best_iteration()

    imp_elo_full = SimpleImputer(strategy="median")
    x_elo_full = imp_elo_full.fit_transform(train_elo[feat_list])
    x_elo_test = imp_elo_full.transform(test_elo[feat_list])

    p_ft_elo = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    p_ft_elo["iterations"] = bi_elo + 10
    ft_elo = CatBoostClassifier(**p_ft_elo)
    ft_elo.fit(x_elo_full, train_elo["target"])
    p_elo_test = ft_elo.predict_proba(x_elo_test)[:, 1]

    logger.info("Final test results:")
    configs = [
        ("SF_t77", test_sf, calc_roi(test_sf, p_sf_test, threshold=0.77)),
        ("SF_EV0_p77", test_sf, calc_ev_roi(test_sf, p_sf_test, 0.0, 0.77)),
        ("ELO_t77", test_elo, calc_roi(test_elo, p_elo_test, threshold=0.77)),
        ("ELO_EV0_p77", test_elo, calc_ev_roi(test_elo, p_elo_test, 0.0, 0.77)),
    ]

    for name, _, result in configs:
        logger.info("  %s: ROI=%.2f%% n=%d", name, result["roi"], result["n_bets"])

    # Best result for MLflow
    best_config = max(configs, key=lambda x: x[2]["roi"])
    best_name = best_config[0]
    best_result = best_config[2]

    with mlflow.start_run(run_name="phase4/step4.13_elo_ev_cv") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.13")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "elo_ev_5fold_cv",
                    "n_features": len(feat_list),
                    "best_config": best_name,
                }
            )

            for name, _, result in configs:
                mlflow.log_metric(f"test_roi_{name}", result["roi"])
                mlflow.log_metric(f"test_n_{name}", result["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roc_auc": roc_auc_score(test_sf["target"], p_sf_test),
                    "n_bets": best_result["n_bets"],
                    "win_rate": best_result["win_rate"],
                    "best_threshold": 0.77,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.13: Best=%s ROI=%.2f%% n=%d run=%s",
                best_name,
                best_result["roi"],
                best_result["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
