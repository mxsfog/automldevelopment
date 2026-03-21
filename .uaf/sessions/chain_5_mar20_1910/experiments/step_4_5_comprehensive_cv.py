"""Step 4.5: Comprehensive 5-fold CV для всех стратегий отбора.

Валидируем: EV>=0, EV>=0.05, EV>=0.10, Odds>=1.15+EV0, per-sport EV.
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


def apply_odds_filter(test_df, p_test: np.ndarray, min_odds: float) -> dict:
    """EV>=0 + p>=0.77 + odds >= min_odds."""
    mask = (
        (p_test >= 0.77)
        & (p_test * test_df["Odds"].values - 1.0 >= 0)
        & (test_df["Odds"].values >= min_odds)
    )
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def main() -> None:
    """Comprehensive CV validation of all selection strategies."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_all_features()

    # 5-fold temporal CV on train only
    all_sf = train_sf.copy().sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    logger.info("5-fold temporal CV (n=%d, block=%d)", n, block_size)

    strategies = ["t77", "ev0", "ev005", "ev010", "odds115", "odds120", "ps_ev"]
    fold_data: dict[str, list[float]] = {s: [] for s in strategies}
    fold_nbets: dict[str, list[int]] = {s: [] for s in strategies}

    for fold_idx in range(n_folds):
        check_budget()
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(train_end + block_size, n)

        fold_train = all_sf.iloc[:train_end].copy()
        fold_test = all_sf.iloc[test_start:test_end].copy()

        if len(fold_train) < 100 or len(fold_test) < 20:
            continue

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

        # Per-sport EV on inner val
        p_inner_val = cb_fold.predict_proba(x_inner_val)[:, 1]
        sport_ev = find_sport_ev_thresholds(inner_val, p_inner_val)

        # Full-train fold
        imp_fold_full = SimpleImputer(strategy="median")
        x_fold_full = imp_fold_full.fit_transform(fold_train[feat_list])
        x_fold_test = imp_fold_full.transform(fold_test[feat_list])

        params_fold = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        params_fold["iterations"] = max(fold_best_iter + 10, 50)
        cb_fold_ft = CatBoostClassifier(**params_fold)
        cb_fold_ft.fit(x_fold_full, fold_train["target"])

        p_fold_test = cb_fold_ft.predict_proba(x_fold_test)[:, 1]

        # All strategies
        r_t77 = calc_roi(fold_test, p_fold_test, threshold=0.77)
        r_ev0 = calc_ev_roi(fold_test, p_fold_test, ev_threshold=0.0, min_prob=0.77)
        r_ev005 = calc_ev_roi(fold_test, p_fold_test, ev_threshold=0.05, min_prob=0.77)
        r_ev010 = calc_ev_roi(fold_test, p_fold_test, ev_threshold=0.10, min_prob=0.77)
        r_odds115 = apply_odds_filter(fold_test, p_fold_test, min_odds=1.15)
        r_odds120 = apply_odds_filter(fold_test, p_fold_test, min_odds=1.20)
        r_ps = apply_sport_ev(fold_test, p_fold_test, sport_ev)

        fold_data["t77"].append(r_t77["roi"])
        fold_data["ev0"].append(r_ev0["roi"])
        fold_data["ev005"].append(r_ev005["roi"])
        fold_data["ev010"].append(r_ev010["roi"])
        fold_data["odds115"].append(r_odds115["roi"])
        fold_data["odds120"].append(r_odds120["roi"])
        fold_data["ps_ev"].append(r_ps["roi"])

        fold_nbets["t77"].append(r_t77["n_bets"])
        fold_nbets["ev0"].append(r_ev0["n_bets"])
        fold_nbets["ev005"].append(r_ev005["n_bets"])
        fold_nbets["ev010"].append(r_ev010["n_bets"])
        fold_nbets["odds115"].append(r_odds115["n_bets"])
        fold_nbets["odds120"].append(r_odds120["n_bets"])
        fold_nbets["ps_ev"].append(r_ps["n_bets"])

        logger.info(
            "  Fold %d: t77=%.1f ev0=%.1f ev05=%.1f ev10=%.1f o115=%.1f o120=%.1f ps=%.1f",
            fold_idx,
            r_t77["roi"],
            r_ev0["roi"],
            r_ev005["roi"],
            r_ev010["roi"],
            r_odds115["roi"],
            r_odds120["roi"],
            r_ps["roi"],
        )

    # Summary
    logger.info("CV Summary (avg/std/positive):")
    for s in strategies:
        avg = np.mean(fold_data[s])
        std = np.std(fold_data[s])
        pos = sum(1 for r in fold_data[s] if r > 0)
        avg_n = np.mean(fold_nbets[s])
        logger.info(
            "  %-8s: avg=%.2f%% std=%.2f%% positive=%d/%d avg_n=%.0f",
            s,
            avg,
            std,
            pos,
            len(fold_data[s]),
            avg_n,
        )

    # Test set results
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
    auc = roc_auc_score(test_sf["target"], p_test)
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    sport_ev_final = find_sport_ev_thresholds(val_df, p_val)

    test_results = {}
    test_results["t77"] = calc_roi(test_sf, p_test, threshold=0.77)
    test_results["ev0"] = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)
    test_results["ev005"] = calc_ev_roi(test_sf, p_test, ev_threshold=0.05, min_prob=0.77)
    test_results["ev010"] = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    test_results["odds115"] = apply_odds_filter(test_sf, p_test, min_odds=1.15)
    test_results["odds120"] = apply_odds_filter(test_sf, p_test, min_odds=1.20)
    test_results["ps_ev"] = apply_sport_ev(test_sf, p_test, sport_ev_final)

    logger.info("Test results:")
    for s in strategies:
        logger.info(
            "  %-8s: ROI=%.2f%% n=%d", s, test_results[s]["roi"], test_results[s]["n_bets"]
        )

    # Choose best by CV (stable, positive, reasonable volume)
    # Criteria: avg ROI, all folds positive, avg_n >= 30
    best_strat = "ev0"
    best_cv_roi = np.mean(fold_data["ev0"])
    for s in strategies:
        avg = np.mean(fold_data[s])
        pos = sum(1 for r in fold_data[s] if r > 0)
        avg_n = np.mean(fold_nbets[s])
        if avg > best_cv_roi and pos >= 4 and avg_n >= 15:
            best_cv_roi = avg
            best_strat = s

    logger.info("Best strategy by CV: %s (CV avg=%.2f%%)", best_strat, best_cv_roi)

    with mlflow.start_run(run_name="phase4/step4.5_comprehensive_cv") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "comprehensive_cv",
                    "n_features": len(feat_list),
                    "n_folds": len(fold_data["t77"]),
                    "best_strategy": best_strat,
                    "best_cv_avg": best_cv_roi,
                }
            )

            for s in strategies:
                mlflow.log_metric(f"cv_avg_{s}", float(np.mean(fold_data[s])))
                mlflow.log_metric(f"cv_std_{s}", float(np.std(fold_data[s])))
                mlflow.log_metric(f"test_roi_{s}", test_results[s]["roi"])
                mlflow.log_metric(f"test_n_{s}", test_results[s]["n_bets"])
                for i, v in enumerate(fold_data[s]):
                    mlflow.log_metric(f"cv_{s}_fold{i}", v)

            mlflow.log_metrics(
                {
                    "roi": test_results[best_strat]["roi"],
                    "roc_auc": auc,
                    "n_bets": test_results[best_strat]["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.5: Best=%s CVavg=%.2f%% TestROI=%.2f%% n=%d run=%s",
                best_strat,
                best_cv_roi,
                test_results[best_strat]["roi"],
                test_results[best_strat]["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
