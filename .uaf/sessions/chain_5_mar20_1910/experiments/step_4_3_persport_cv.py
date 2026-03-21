"""Step 4.3: CV-валидация per-sport EV thresholds + hybrid models.

Sanity check: per-sport EV дал 52% ROI на 132 ставках.
Нужно проверить через 5-fold temporal CV, что это не overfitting.
"""

import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    CB_BEST_PARAMS,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
    calc_roi,
    check_budget,
    get_all_features,
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


def find_sport_ev_thresholds(val_df, p_val: np.ndarray, min_bets: int = 3) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта на val."""
    sports = val_df["Sport"].unique()
    thresholds: dict[str, float] = {}
    for sport in sports:
        sport_mask = (val_df["Sport"] == sport).values
        if sport_mask.sum() < 15:
            thresholds[sport] = 0.0
            continue
        val_sport = val_df[sport_mask]
        p_sport = p_val[sport_mask]
        best_ev_t = 0.0
        best_roi = -999.0
        for ev_t in np.arange(-0.05, 0.20, 0.01):
            r = calc_ev_roi(val_sport, p_sport, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= min_bets and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_ev_t = float(ev_t)
        thresholds[sport] = best_ev_t
    return thresholds


def apply_sport_ev(test_df, p_test: np.ndarray, sport_ev: dict[str, float]) -> dict:
    """Применение per-sport EV thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t = sport_ev.get(sports[i], 0.0)
        if ev[i] >= ev_t and p_test[i] >= 0.77:
            mask[i] = True

    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def main() -> None:
    """5-fold CV for per-sport EV thresholds."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_all_features()

    # 5-fold temporal CV
    all_sf = train_sf.copy().sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    logger.info("5-fold temporal CV (n=%d, block=%d)", n, block_size)

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

        # Inner val
        inner_val_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_val_split]
        inner_val = fold_train.iloc[inner_val_split:]

        imp_fold = SimpleImputer(strategy="median")
        x_inner_train = imp_fold.fit_transform(inner_train[feat_list])
        x_inner_val = imp_fold.transform(inner_val[feat_list])

        cb_fold = CatBoostClassifier(**CB_BEST_PARAMS)
        cb_fold.fit(
            x_inner_train, inner_train["target"], eval_set=(x_inner_val, inner_val["target"])
        )
        fold_best_iter = cb_fold.get_best_iteration()

        # Val predictions for threshold selection
        p_inner_val = cb_fold.predict_proba(x_inner_val)[:, 1]

        # Per-sport EV thresholds on inner val
        sport_ev = find_sport_ev_thresholds(inner_val, p_inner_val)

        # Full-train fold model
        imp_fold_full = SimpleImputer(strategy="median")
        x_fold_full = imp_fold_full.fit_transform(fold_train[feat_list])
        x_fold_test = imp_fold_full.transform(fold_test[feat_list])

        params_fold = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        params_fold["iterations"] = max(fold_best_iter + 10, 50)
        cb_fold_ft = CatBoostClassifier(**params_fold)
        cb_fold_ft.fit(x_fold_full, fold_train["target"])

        p_fold_test = cb_fold_ft.predict_proba(x_fold_test)[:, 1]
        fold_auc = roc_auc_score(fold_test["target"], p_fold_test)

        # Standard: EV>=0 + p>=0.77
        roi_std = calc_ev_roi(fold_test, p_fold_test, ev_threshold=0.0, min_prob=0.77)
        # Standard: t=0.77 only
        roi_t77 = calc_roi(fold_test, p_fold_test, threshold=0.77)
        # Per-sport EV
        roi_ps = apply_sport_ev(fold_test, p_fold_test, sport_ev)

        fold_results.append(
            {
                "fold": fold_idx,
                "auc": fold_auc,
                "roi_t77": roi_t77["roi"],
                "n_bets_t77": roi_t77["n_bets"],
                "roi_ev0": roi_std["roi"],
                "n_bets_ev0": roi_std["n_bets"],
                "roi_ps": roi_ps["roi"],
                "n_bets_ps": roi_ps["n_bets"],
                "sport_ev": sport_ev,
            }
        )

        logger.info(
            "  Fold %d: t77=%.2f%% n=%d | EV0=%.2f%% n=%d | PS_EV=%.2f%% n=%d | AUC=%.4f",
            fold_idx,
            roi_t77["roi"],
            roi_t77["n_bets"],
            roi_std["roi"],
            roi_std["n_bets"],
            roi_ps["roi"],
            roi_ps["n_bets"],
            fold_auc,
        )

    # CV Summary
    avg_t77 = np.mean([r["roi_t77"] for r in fold_results])
    avg_ev0 = np.mean([r["roi_ev0"] for r in fold_results])
    avg_ps = np.mean([r["roi_ps"] for r in fold_results])
    std_t77 = np.std([r["roi_t77"] for r in fold_results])
    std_ev0 = np.std([r["roi_ev0"] for r in fold_results])
    std_ps = np.std([r["roi_ps"] for r in fold_results])

    pos_t77 = sum(1 for r in fold_results if r["roi_t77"] > 0)
    pos_ev0 = sum(1 for r in fold_results if r["roi_ev0"] > 0)
    pos_ps = sum(1 for r in fold_results if r["roi_ps"] > 0)

    logger.info("CV Summary:")
    logger.info(
        "  t=0.77:       avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_t77,
        std_t77,
        pos_t77,
        len(fold_results),
    )
    logger.info(
        "  EV>=0+p77:    avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_ev0,
        std_ev0,
        pos_ev0,
        len(fold_results),
    )
    logger.info(
        "  Per-sport EV: avg=%.2f%% std=%.2f%% positive=%d/%d",
        avg_ps,
        std_ps,
        pos_ps,
        len(fold_results),
    )

    # Also get test-set results for reporting
    check_budget()
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb_ref.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]

    # Get per-sport thresholds from val
    sport_ev_final = find_sport_ev_thresholds(val_df, p_val)
    roi_ps_test = apply_sport_ev(test_sf, p_test, sport_ev_final)
    roi_ev0_test = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)

    logger.info("Test results:")
    logger.info("  EV>=0+p77: ROI=%.2f%% n=%d", roi_ev0_test["roi"], roi_ev0_test["n_bets"])
    logger.info("  Per-sport EV: ROI=%.2f%% n=%d", roi_ps_test["roi"], roi_ps_test["n_bets"])
    logger.info("  Sport EV thresholds: %s", sport_ev_final)

    # Determine best approach based on CV
    best_method = "per_sport_ev" if avg_ps > avg_ev0 + 2.0 and pos_ps >= pos_ev0 else "ev0_p77"
    best_roi = roi_ps_test["roi"] if best_method == "per_sport_ev" else roi_ev0_test["roi"]
    best_n = roi_ps_test["n_bets"] if best_method == "per_sport_ev" else roi_ev0_test["n_bets"]

    with mlflow.start_run(run_name="phase4/step4.3_persport_cv") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.3")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "persport_ev_cv_validation",
                    "n_features": len(feat_list),
                    "n_folds": len(fold_results),
                    "best_method": best_method,
                }
            )

            for fr in fold_results:
                mlflow.log_metric(f"cv_roi_t77_fold{fr['fold']}", fr["roi_t77"])
                mlflow.log_metric(f"cv_roi_ev0_fold{fr['fold']}", fr["roi_ev0"])
                mlflow.log_metric(f"cv_roi_ps_fold{fr['fold']}", fr["roi_ps"])

            mlflow.log_metrics(
                {
                    "roi": best_roi,
                    "n_bets": best_n,
                    "cv_avg_t77": avg_t77,
                    "cv_std_t77": std_t77,
                    "cv_avg_ev0": avg_ev0,
                    "cv_std_ev0": std_ev0,
                    "cv_avg_ps": avg_ps,
                    "cv_std_ps": std_ps,
                    "test_roi_ev0": roi_ev0_test["roi"],
                    "test_roi_ps": roi_ps_test["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")

            logger.info(
                "Step 4.3: Best=%s CV: ev0=%.2f%% ps=%.2f%% | Test: ev0=%.2f%% ps=%.2f%% run=%s",
                best_method,
                avg_ev0,
                avg_ps,
                roi_ev0_test["roi"],
                roi_ps_test["roi"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
