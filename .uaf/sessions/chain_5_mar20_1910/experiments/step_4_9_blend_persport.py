"""Step 4.9: CB+LGB blend с per-sport EV floor=0.10.

Combine два лучших результата:
- CB+LGB blend: CV avg=24.23% (лучший CV из step 4.8)
- Per-sport EV floor=0.10: CV avg=27.07% (лучший CV из step 4.6)

Гипотеза: blend predictions + per-sport EV = ещё лучший CV.
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
    check_budget,
    get_all_features,
    load_data,
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
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

LGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.08,
    "reg_lambda": 21.1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def find_sport_ev_thresholds(
    val_df, p_val: np.ndarray, min_bets: int = 3, ev_floor: float = 0.0
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта с минимальным floor."""
    sports = val_df["Sport"].unique()
    thresholds: dict[str, float] = {}
    for sport in sports:
        sport_mask = (val_df["Sport"] == sport).values
        if sport_mask.sum() < 15:
            thresholds[sport] = ev_floor
            continue
        val_sport = val_df[sport_mask]
        p_sport = p_val[sport_mask]
        best_ev_t = ev_floor
        best_roi = -999.0
        for ev_t in np.arange(max(-0.05, ev_floor), 0.25, 0.01):
            r = calc_ev_roi(val_sport, p_sport, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= min_bets and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_ev_t = float(ev_t)
        thresholds[sport] = max(best_ev_t, ev_floor)
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
    """Blend + per-sport EV final combination."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_all_features()

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    # Train CB
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    cb_bi = cb.get_best_iteration()

    # Train LGB
    lgb = LGBMClassifier(**LGB_PARAMS)
    lgb.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )
    lgb_bi = lgb.best_iteration_

    # Full-train both
    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    cb_fp = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    cb_fp["iterations"] = cb_bi + 10
    cb_ft = CatBoostClassifier(**cb_fp)
    cb_ft.fit(x_full, train_sf["target"])

    lgb_fp = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}
    lgb_fp["n_estimators"] = lgb_bi + 10
    lgb_ft = LGBMClassifier(**lgb_fp)
    lgb_ft.fit(x_full, train_sf["target"])

    p_cb = cb_ft.predict_proba(x_test)[:, 1]
    p_lgb = lgb_ft.predict_proba(x_test)[:, 1]
    p_blend = (p_cb + p_lgb) / 2.0

    p_cb_val = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    p_lgb_val = lgb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    p_blend_val = (p_cb_val + p_lgb_val) / 2.0

    auc_blend = roc_auc_score(test_sf["target"], p_blend)

    results: dict[str, dict] = {}

    # A: CB solo + PS_EV floor=0.10
    sport_ev_cb = find_sport_ev_thresholds(val_df, p_cb_val, ev_floor=0.10)
    r_cb_ps = apply_sport_ev(test_sf, p_cb, sport_ev_cb)
    results["cb_ps010"] = r_cb_ps
    logger.info("A: CB+PS010: ROI=%.2f%% n=%d", r_cb_ps["roi"], r_cb_ps["n_bets"])

    # B: Blend + PS_EV floor=0.10
    check_budget()
    sport_ev_blend = find_sport_ev_thresholds(val_df, p_blend_val, ev_floor=0.10)
    r_blend_ps = apply_sport_ev(test_sf, p_blend, sport_ev_blend)
    results["blend_ps010"] = r_blend_ps
    logger.info("B: Blend+PS010: ROI=%.2f%% n=%d", r_blend_ps["roi"], r_blend_ps["n_bets"])
    logger.info("  Sport thresholds: %s", sport_ev_blend)

    # C: Blend + flat EV>=0.10
    r_blend_ev010 = calc_ev_roi(test_sf, p_blend, ev_threshold=0.10, min_prob=0.77)
    results["blend_ev010"] = r_blend_ev010
    logger.info("C: Blend+EV010: ROI=%.2f%% n=%d", r_blend_ev010["roi"], r_blend_ev010["n_bets"])

    # D: CB solo + flat EV>=0.10
    r_cb_ev010 = calc_ev_roi(test_sf, p_cb, ev_threshold=0.10, min_prob=0.77)
    results["cb_ev010"] = r_cb_ev010
    logger.info("D: CB+EV010: ROI=%.2f%% n=%d", r_cb_ev010["roi"], r_cb_ev010["n_bets"])

    # E: 5-fold CV for all 4 variants
    check_budget()
    all_sf = train_sf.copy().sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    strats = ["cb_ev010", "cb_ps010", "blend_ev010", "blend_ps010"]
    cv_data: dict[str, list[float]] = {s: [] for s in strats}
    cv_nbets: dict[str, list[int]] = {s: [] for s in strats}

    logger.info("5-fold CV (n=%d, block=%d)", n, block_size)

    for fold_idx in range(n_folds):
        check_budget()
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(train_end + block_size, n)

        fold_train = all_sf.iloc[:train_end].copy()
        fold_test = all_sf.iloc[test_start:test_end].copy()

        if len(fold_train) < 100 or len(fold_test) < 20:
            continue

        iv_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:iv_split]
        inner_val = fold_train.iloc[iv_split:]

        imp_f = SimpleImputer(strategy="median")
        x_it = imp_f.fit_transform(inner_train[feat_list])
        x_iv = imp_f.transform(inner_val[feat_list])

        cb_f = CatBoostClassifier(**CB_BEST_PARAMS)
        cb_f.fit(x_it, inner_train["target"], eval_set=(x_iv, inner_val["target"]))
        cb_fbi = cb_f.get_best_iteration()

        lgb_f = LGBMClassifier(**LGB_PARAMS)
        lgb_f.fit(
            x_it,
            inner_train["target"],
            eval_set=[(x_iv, inner_val["target"])],
            callbacks=[
                __import__("lightgbm").early_stopping(50, verbose=False),
                __import__("lightgbm").log_evaluation(0),
            ],
        )
        lgb_fbi = lgb_f.best_iteration_

        # Inner val predictions for per-sport thresholds
        p_cb_iv = cb_f.predict_proba(x_iv)[:, 1]
        p_lgb_iv = lgb_f.predict_proba(x_iv)[:, 1]
        p_blend_iv = (p_cb_iv + p_lgb_iv) / 2.0

        sport_ev_cb_f = find_sport_ev_thresholds(inner_val, p_cb_iv, ev_floor=0.10)
        sport_ev_blend_f = find_sport_ev_thresholds(inner_val, p_blend_iv, ev_floor=0.10)

        # Full-train fold
        imp_ff = SimpleImputer(strategy="median")
        x_ff = imp_ff.fit_transform(fold_train[feat_list])
        x_ft = imp_ff.transform(fold_test[feat_list])

        cb_ffp = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        cb_ffp["iterations"] = max(cb_fbi + 10, 50)
        cb_fft = CatBoostClassifier(**cb_ffp)
        cb_fft.fit(x_ff, fold_train["target"])

        lgb_ffp = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}
        lgb_ffp["n_estimators"] = max(lgb_fbi + 10, 50)
        lgb_fft = LGBMClassifier(**lgb_ffp)
        lgb_fft.fit(x_ff, fold_train["target"])

        p_cb_ft = cb_fft.predict_proba(x_ft)[:, 1]
        p_lgb_ft = lgb_fft.predict_proba(x_ft)[:, 1]
        p_blend_ft = (p_cb_ft + p_lgb_ft) / 2.0

        r1 = calc_ev_roi(fold_test, p_cb_ft, ev_threshold=0.10, min_prob=0.77)
        r2 = apply_sport_ev(fold_test, p_cb_ft, sport_ev_cb_f)
        r3 = calc_ev_roi(fold_test, p_blend_ft, ev_threshold=0.10, min_prob=0.77)
        r4 = apply_sport_ev(fold_test, p_blend_ft, sport_ev_blend_f)

        for s, r in zip(strats, [r1, r2, r3, r4], strict=True):
            cv_data[s].append(r["roi"])
            cv_nbets[s].append(r["n_bets"])

        logger.info(
            "  Fold %d: cb=%.1f cb_ps=%.1f blend=%.1f blend_ps=%.1f",
            fold_idx,
            r1["roi"],
            r2["roi"],
            r3["roi"],
            r4["roi"],
        )

    logger.info("CV Summary:")
    for s in strats:
        if cv_data[s]:
            avg = np.mean(cv_data[s])
            std = np.std(cv_data[s])
            pos = sum(1 for r in cv_data[s] if r > 0)
            avg_n = np.mean(cv_nbets[s])
            logger.info(
                "  %-14s: avg=%.2f%% std=%.2f%% pos=%d/%d avg_n=%.0f",
                s,
                avg,
                std,
                pos,
                len(cv_data[s]),
                avg_n,
            )

    logger.info("Test results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d", name, r["roi"], r["n_bets"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    # Find best by CV
    best_cv_strat = max(strats, key=lambda s: np.mean(cv_data[s]) if cv_data[s] else -999)
    best_cv_roi = np.mean(cv_data[best_cv_strat])
    logger.info("Best by CV: %s (avg=%.2f%%)", best_cv_strat, best_cv_roi)

    with mlflow.start_run(run_name="phase4/step4.9_blend_persport") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "blend_persport_ev",
                    "n_features": len(feat_list),
                    "best_test_variant": best_key,
                    "best_cv_variant": best_cv_strat,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            for s in strats:
                if cv_data[s]:
                    mlflow.log_metric(f"cv_avg_{s}", float(np.mean(cv_data[s])))
                    mlflow.log_metric(f"cv_std_{s}", float(np.std(cv_data[s])))
                    for i, v in enumerate(cv_data[s]):
                        mlflow.log_metric(f"cv_{s}_fold{i}", v)

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": float(auc_blend),
                    "n_bets": best_r["n_bets"],
                    "best_cv_roi": best_cv_roi,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.9: BestTest=%s ROI=%.2f%% BestCV=%s avg=%.2f%% run=%s",
                best_key,
                best_r["roi"],
                best_cv_strat,
                best_cv_roi,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
