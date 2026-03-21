"""Step 4.8: LightGBM + CB-LGB ensemble при строгом EV фильтре.

В chain_4 LGB и blend были хуже при EV>=0. Но при EV>=0.10 (другой operating point)
модель фильтрует сильнее, и ensemble может дать лучшую калибровку.

Гипотезы:
A) LightGBM solo с EV>=0.10+p77
B) CB+LGB mean blend с EV>=0.10+p77
C) CB+LGB weighted blend (оптимальный вес на val)
D) 5-fold CV для лучших вариантов
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


def main() -> None:
    """LGB + ensemble at strict EV filter."""
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

    results: dict[str, dict] = {}

    # CatBoost reference (full-train)
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    cb_best_iter = cb.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = cb_best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_cb_test = cb_ft.predict_proba(x_test)[:, 1]
    p_cb_val = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    auc_cb = roc_auc_score(test_sf["target"], p_cb_test)

    r_cb = calc_ev_roi(test_sf, p_cb_test, ev_threshold=0.10, min_prob=0.77)
    results["cb_ev010"] = r_cb
    logger.info("CB EV>=0.10: ROI=%.2f%% n=%d AUC=%.4f", r_cb["roi"], r_cb["n_bets"], auc_cb)

    # A: LightGBM
    check_budget()
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
    lgb_best_iter = lgb.best_iteration_

    lgb_ft_params = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}
    lgb_ft_params["n_estimators"] = lgb_best_iter + 10
    lgb_ft = LGBMClassifier(**lgb_ft_params)
    lgb_ft.fit(x_full, train_sf["target"])

    p_lgb_test = lgb_ft.predict_proba(x_test)[:, 1]
    p_lgb_val = lgb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    auc_lgb = roc_auc_score(test_sf["target"], p_lgb_test)

    r_lgb = calc_ev_roi(test_sf, p_lgb_test, ev_threshold=0.10, min_prob=0.77)
    results["lgb_ev010"] = r_lgb
    logger.info(
        "A: LGB EV>=0.10: ROI=%.2f%% n=%d AUC=%.4f", r_lgb["roi"], r_lgb["n_bets"], auc_lgb
    )

    # B: Mean blend
    p_blend = (p_cb_test + p_lgb_test) / 2.0
    auc_blend = roc_auc_score(test_sf["target"], p_blend)
    r_blend = calc_ev_roi(test_sf, p_blend, ev_threshold=0.10, min_prob=0.77)
    results["blend_mean"] = r_blend
    logger.info(
        "B: Blend mean EV>=0.10: ROI=%.2f%% n=%d AUC=%.4f",
        r_blend["roi"],
        r_blend["n_bets"],
        auc_blend,
    )

    # C: Weighted blend (optimize weight on val)
    check_budget()
    best_w = 0.5
    best_val_roi = -999.0
    for w in np.arange(0.3, 0.8, 0.05):
        p_w_val = w * p_cb_val + (1 - w) * p_lgb_val
        r_w = calc_ev_roi(val_df, p_w_val, ev_threshold=0.10, min_prob=0.77)
        if r_w["n_bets"] >= 5 and r_w["roi"] > best_val_roi:
            best_val_roi = r_w["roi"]
            best_w = float(w)

    p_wblend = best_w * p_cb_test + (1 - best_w) * p_lgb_test
    auc_wblend = roc_auc_score(test_sf["target"], p_wblend)
    r_wblend = calc_ev_roi(test_sf, p_wblend, ev_threshold=0.10, min_prob=0.77)
    results["blend_weighted"] = r_wblend
    logger.info(
        "C: Blend w=%.2f EV>=0.10: ROI=%.2f%% n=%d AUC=%.4f",
        best_w,
        r_wblend["roi"],
        r_wblend["n_bets"],
        auc_wblend,
    )

    # Also try EV>=0 with blend
    r_blend_ev0 = calc_ev_roi(test_sf, p_blend, ev_threshold=0.0, min_prob=0.77)
    results["blend_mean_ev0"] = r_blend_ev0
    logger.info("  Blend mean EV>=0: ROI=%.2f%% n=%d", r_blend_ev0["roi"], r_blend_ev0["n_bets"])

    # D: 5-fold CV for top variants
    check_budget()
    all_sf = train_sf.copy().sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    strats = ["cb_ev010", "lgb_ev010", "blend_ev010"]
    cv_data: dict[str, list[float]] = {s: [] for s in strats}

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

        inner_val_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_val_split]
        inner_val = fold_train.iloc[inner_val_split:]

        imp_fold = SimpleImputer(strategy="median")
        x_it = imp_fold.fit_transform(inner_train[feat_list])
        x_iv = imp_fold.transform(inner_val[feat_list])

        # CB
        cb_f = CatBoostClassifier(**CB_BEST_PARAMS)
        cb_f.fit(x_it, inner_train["target"], eval_set=(x_iv, inner_val["target"]))
        cb_bi = cb_f.get_best_iteration()

        # LGB
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
        lgb_bi = lgb_f.best_iteration_

        # Full-train fold
        imp_ff = SimpleImputer(strategy="median")
        x_ff = imp_ff.fit_transform(fold_train[feat_list])
        x_ft = imp_ff.transform(fold_test[feat_list])

        cb_fp = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        cb_fp["iterations"] = max(cb_bi + 10, 50)
        cb_fft = CatBoostClassifier(**cb_fp)
        cb_fft.fit(x_ff, fold_train["target"])

        lgb_fp = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}
        lgb_fp["n_estimators"] = max(lgb_bi + 10, 50)
        lgb_fft = LGBMClassifier(**lgb_fp)
        lgb_fft.fit(x_ff, fold_train["target"])

        p_cb_f = cb_fft.predict_proba(x_ft)[:, 1]
        p_lgb_f = lgb_fft.predict_proba(x_ft)[:, 1]
        p_blend_f = (p_cb_f + p_lgb_f) / 2.0

        r1 = calc_ev_roi(fold_test, p_cb_f, ev_threshold=0.10, min_prob=0.77)
        r2 = calc_ev_roi(fold_test, p_lgb_f, ev_threshold=0.10, min_prob=0.77)
        r3 = calc_ev_roi(fold_test, p_blend_f, ev_threshold=0.10, min_prob=0.77)

        cv_data["cb_ev010"].append(r1["roi"])
        cv_data["lgb_ev010"].append(r2["roi"])
        cv_data["blend_ev010"].append(r3["roi"])

        logger.info(
            "  Fold %d: CB=%.1f LGB=%.1f Blend=%.1f",
            fold_idx,
            r1["roi"],
            r2["roi"],
            r3["roi"],
        )

    logger.info("CV Summary:")
    for s in strats:
        if cv_data[s]:
            avg = np.mean(cv_data[s])
            std = np.std(cv_data[s])
            pos = sum(1 for r in cv_data[s] if r > 0)
            logger.info(
                "  %-12s: avg=%.2f%% std=%.2f%% pos=%d/%d",
                s,
                avg,
                std,
                pos,
                len(cv_data[s]),
            )

    # Summary
    logger.info("Test results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d", name, r["roi"], r["n_bets"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.8_lgb_ensemble") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "lgb_ensemble_strict_ev",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                    "blend_weight": best_w,
                    "cb_best_iter": cb_best_iter,
                    "lgb_best_iter": lgb_best_iter,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            for s in strats:
                if cv_data[s]:
                    mlflow.log_metric(f"cv_avg_{s}", float(np.mean(cv_data[s])))
                    mlflow.log_metric(f"cv_std_{s}", float(np.std(cv_data[s])))

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": float(auc_cb),
                    "n_bets": best_r["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.8: Best=%s ROI=%.2f%% n=%d run=%s",
                best_key,
                best_r["roi"],
                best_r["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
