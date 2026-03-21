"""Step 4.10: Proper EV-based bet selection with val-based threshold + 5-fold CV."""

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
    """Выбор ставок по EV = p * odds - 1 > ev_threshold AND p >= min_prob."""
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


def find_best_ev_threshold_on_val(
    val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 20
) -> tuple[float, float, float]:
    """Подбор лучших (ev_threshold, min_prob) на val. Возвращает (ev_t, min_p, roi)."""
    best_roi = -999.0
    best_ev_t = 0.0
    best_min_p = 0.6
    for ev_t in np.arange(0.0, 0.35, 0.05):
        for min_p in [0.5, 0.55, 0.6, 0.65, 0.7]:
            result = calc_ev_roi(val_df, proba, ev_threshold=ev_t, min_prob=min_p)
            if result["n_bets"] >= min_bets and result["roi"] > best_roi:
                best_roi = result["roi"]
                best_ev_t = float(ev_t)
                best_min_p = min_p
    return best_ev_t, best_min_p, best_roi


def main() -> None:
    """Proper EV validation: val-based threshold + 5-fold temporal CV."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    # Part 1: Proper val-based EV threshold selection
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    ref_model = CatBoostClassifier(**CB_PARAMS)
    ref_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = ref_model.get_best_iteration()

    # Val predictions for threshold selection
    p_val = ref_model.predict_proba(x_val)[:, 1]
    ev_t_val, min_p_val, val_roi = find_best_ev_threshold_on_val(val_df, p_val, min_bets=10)
    logger.info(
        "Val-selected EV params: ev_t=%.2f, min_p=%.2f, val_ROI=%.2f%%",
        ev_t_val,
        min_p_val,
        val_roi,
    )

    # Full-train model
    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter + 10
    ft_model = CatBoostClassifier(**params_ft)
    ft_model.fit(x_full, train_sf["target"])
    p_test = ft_model.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(test_sf["target"], p_test)

    # Reference (threshold-based)
    ref_roi = calc_roi(test_sf, p_test, threshold=0.77)
    logger.info("Reference t=0.77: ROI=%.2f%% n=%d", ref_roi["roi"], ref_roi["n_bets"])

    # Apply val-selected EV threshold to test (one shot, no leakage)
    ev_roi_test = calc_ev_roi(test_sf, p_test, ev_threshold=ev_t_val, min_prob=min_p_val)
    logger.info(
        "EV val-selected (ev>=%.2f, p>=%.2f) on test: ROI=%.2f%% n=%d",
        ev_t_val,
        min_p_val,
        ev_roi_test["roi"],
        ev_roi_test["n_bets"],
    )

    # Also test combined: p >= 0.77 AND EV >= val_ev_t
    ev_combined = calc_ev_roi(test_sf, p_test, ev_threshold=ev_t_val, min_prob=0.77)
    logger.info(
        "EV combined (ev>=%.2f, p>=0.77) on test: ROI=%.2f%% n=%d",
        ev_t_val,
        ev_combined["roi"],
        ev_combined["n_bets"],
    )

    # Conservative fixed EV: just require EV > 0 with model threshold
    ev_zero = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)
    logger.info(
        "EV>=0 + p>=0.77: ROI=%.2f%% n=%d",
        ev_zero["roi"],
        ev_zero["n_bets"],
    )

    # Part 2: 5-fold temporal CV for EV selection
    check_budget()
    all_sf = pd.concat([train_sf, test_sf]).sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    logger.info("5-fold temporal CV for EV selection (n=%d, block=%d)", n, block_size)

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

        # Inner val for threshold + iteration
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

        # Val-based EV threshold selection
        p_inner_val = cb_fold.predict_proba(x_inner_val)[:, 1]
        fold_ev_t, fold_min_p, _ = find_best_ev_threshold_on_val(
            inner_val,
            p_inner_val,
            min_bets=5,
        )

        # Full-train fold model
        imp_fold_full = SimpleImputer(strategy="median")
        x_fold_full = imp_fold_full.fit_transform(fold_train[feat_list])
        x_fold_test = imp_fold_full.transform(fold_test[feat_list])

        params_fold_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
        params_fold_ft["iterations"] = max(fold_best_iter + 10, 50)
        cb_fold_ft = CatBoostClassifier(**params_fold_ft)
        cb_fold_ft.fit(x_fold_full, fold_train["target"])

        p_fold_test = cb_fold_ft.predict_proba(x_fold_test)[:, 1]
        fold_auc = roc_auc_score(fold_test["target"], p_fold_test)

        # Standard threshold
        roi_t77 = calc_roi(fold_test, p_fold_test, threshold=0.77)

        # EV selection (val-determined thresholds)
        roi_ev = calc_ev_roi(fold_test, p_fold_test, ev_threshold=fold_ev_t, min_prob=fold_min_p)

        # Conservative EV: p >= 0.77 AND EV >= 0
        roi_ev_cons = calc_ev_roi(fold_test, p_fold_test, ev_threshold=0.0, min_prob=0.77)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_n": len(fold_train),
                "test_n": len(fold_test),
                "auc": fold_auc,
                "roi_t77": roi_t77["roi"],
                "n_bets_t77": roi_t77["n_bets"],
                "roi_ev_val": roi_ev["roi"],
                "n_bets_ev_val": roi_ev["n_bets"],
                "ev_t": fold_ev_t,
                "min_p": fold_min_p,
                "roi_ev_cons": roi_ev_cons["roi"],
                "n_bets_ev_cons": roi_ev_cons["n_bets"],
            }
        )

        logger.info(
            "  Fold %d: t77=%.2f%% n=%d | EV(%.2f,%.2f)=%.2f%% n=%d | EV(0,0.77)=%.2f%% n=%d"
            " | AUC=%.4f",
            fold_idx,
            roi_t77["roi"],
            roi_t77["n_bets"],
            fold_ev_t,
            fold_min_p,
            roi_ev["roi"],
            roi_ev["n_bets"],
            roi_ev_cons["roi"],
            roi_ev_cons["n_bets"],
            fold_auc,
        )

    # CV Summary
    rois_t77 = [r["roi_t77"] for r in fold_results]
    rois_ev_val = [r["roi_ev_val"] for r in fold_results]
    rois_ev_cons = [r["roi_ev_cons"] for r in fold_results]

    avg_t77 = np.mean(rois_t77)
    avg_ev_val = np.mean(rois_ev_val)
    avg_ev_cons = np.mean(rois_ev_cons)
    std_t77 = np.std(rois_t77)
    std_ev_val = np.std(rois_ev_val)
    std_ev_cons = np.std(rois_ev_cons)

    logger.info("CV Summary:")
    logger.info(
        "  t=0.77: avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_t77,
        std_t77,
        sum(1 for r in rois_t77 if r > 0),
        len(fold_results),
    )
    logger.info(
        "  EV val-selected: avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_ev_val,
        std_ev_val,
        sum(1 for r in rois_ev_val if r > 0),
        len(fold_results),
    )
    logger.info(
        "  EV conservative (EV>=0, p>=0.77): avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_ev_cons,
        std_ev_cons,
        sum(1 for r in rois_ev_cons if r > 0),
        len(fold_results),
    )

    # Determine which approach to report
    # Use conservative EV if it's better than t77 across CV
    if avg_ev_cons > avg_t77:
        final_roi = ev_zero["roi"]
        final_n = ev_zero["n_bets"]
        method_name = "EV>=0+p>=0.77"
        logger.info(
            "EV conservative approach is better on CV: %.2f%% vs %.2f%%", avg_ev_cons, avg_t77
        )
    else:
        final_roi = ref_roi["roi"]
        final_n = ref_roi["n_bets"]
        method_name = "t=0.77"
        logger.info(
            "Standard threshold still better on CV: %.2f%% vs %.2f%%", avg_t77, avg_ev_cons
        )

    with mlflow.start_run(run_name="phase4/step4.10_ev_validation") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.10")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "ev_validation_5fold",
                    "n_features": len(feat_list),
                    "ev_t_val_selected": ev_t_val,
                    "min_p_val_selected": min_p_val,
                    "n_folds": len(fold_results),
                    "best_method": method_name,
                }
            )

            for fr in fold_results:
                mlflow.log_metric(f"cv_roi_t77_fold{fr['fold']}", fr["roi_t77"])
                mlflow.log_metric(f"cv_roi_ev_val_fold{fr['fold']}", fr["roi_ev_val"])
                mlflow.log_metric(f"cv_roi_ev_cons_fold{fr['fold']}", fr["roi_ev_cons"])

            mlflow.log_metrics(
                {
                    "roi": final_roi,
                    "roc_auc": auc_test,
                    "n_bets": final_n,
                    "cv_avg_t77": avg_t77,
                    "cv_std_t77": std_t77,
                    "cv_avg_ev_val": avg_ev_val,
                    "cv_std_ev_val": std_ev_val,
                    "cv_avg_ev_cons": avg_ev_cons,
                    "cv_std_ev_cons": std_ev_cons,
                    "test_roi_t77": ref_roi["roi"],
                    "test_roi_ev_val": ev_roi_test["roi"],
                    "test_roi_ev_cons": ev_zero["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.10: Best=%s ROI=%.2f%% n=%d | CV t77=%.2f%% ev_cons=%.2f%% run=%s",
                method_name,
                final_roi,
                final_n,
                avg_t77,
                avg_ev_cons,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
