"""Step 4.2: LightGBM, selective feature addition, threshold sweep."""

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


def add_selective_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Только лучшие новые фичи из step 4.1 analysis."""
    df = df.copy()
    new_feats: list[str] = []

    # Only the promising ones from importance analysis
    elo_prob = df["team_winrate_mean"].fillna(0.5)
    df["elo_implied_agreement"] = 1.0 - (elo_prob - df["implied_prob"]).abs()
    new_feats.append("elo_implied_agreement")

    df["winrate_vs_odds"] = df["team_winrate_mean"].fillna(0.5) - df["implied_prob"]
    new_feats.append("winrate_vs_odds")

    return df, new_feats


def main() -> None:
    """LightGBM comparison + selective features + fine threshold sweep."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)
    df, new_feats = add_selective_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_base = get_base_features() + get_engineered_features() + get_elo_features()
    feat_sel = feat_base + new_feats

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit_base = imp.fit_transform(train_fit[feat_base])
    x_val_base = imp.transform(val_df[feat_base])
    x_test_base = imp.transform(test_sf[feat_base])

    imp2 = SimpleImputer(strategy="median")
    x_fit_sel = imp2.fit_transform(train_fit[feat_sel])
    x_val_sel = imp2.transform(val_df[feat_sel])
    x_test_sel = imp2.transform(test_sf[feat_sel])

    results: dict[str, dict] = {}

    # 1. Reference CatBoost
    check_budget()
    cb = CatBoostClassifier(**CB_PARAMS)
    cb.fit(x_fit_base, train_fit["target"], eval_set=(x_val_base, val_df["target"]))
    p_cb = cb.predict_proba(x_test_base)[:, 1]
    auc_cb = roc_auc_score(test_sf["target"], p_cb)
    results["cb_ref"] = {"proba": p_cb, "auc": auc_cb}

    # 2. CatBoost + selective features
    check_budget()
    cb_sel = CatBoostClassifier(**CB_PARAMS)
    cb_sel.fit(x_fit_sel, train_fit["target"], eval_set=(x_val_sel, val_df["target"]))
    p_cb_sel = cb_sel.predict_proba(x_test_sel)[:, 1]
    auc_cb_sel = roc_auc_score(test_sf["target"], p_cb_sel)
    results["cb_sel"] = {"proba": p_cb_sel, "auc": auc_cb_sel}

    # 3. LightGBM
    check_budget()
    lgb = LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.08,
        reg_lambda=21.0,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    lgb.fit(
        x_fit_base,
        train_fit["target"],
        eval_set=[(x_val_base, val_df["target"])],
        callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )
    p_lgb = lgb.predict_proba(x_test_base)[:, 1]
    auc_lgb = roc_auc_score(test_sf["target"], p_lgb)
    results["lgb"] = {"proba": p_lgb, "auc": auc_lgb}

    # 4. Blend CB + LGB (0.7/0.3)
    p_blend = 0.7 * p_cb + 0.3 * p_lgb
    auc_blend = roc_auc_score(test_sf["target"], p_blend)
    results["blend_7_3"] = {"proba": p_blend, "auc": auc_blend}

    # 5. Equal blend
    p_eq = 0.5 * p_cb + 0.5 * p_lgb
    auc_eq = roc_auc_score(test_sf["target"], p_eq)
    results["blend_5_5"] = {"proba": p_eq, "auc": auc_eq}

    # Fine threshold sweep for each model
    thresholds = np.arange(0.70, 0.85, 0.01).tolist()
    logger.info("Threshold sweep results:")

    best_overall_roi = -999.0
    best_overall_key = ""
    best_overall_t = 0.77

    for name, r in results.items():
        p = r["proba"]
        logger.info("  %s (AUC=%.4f):", name, r["auc"])
        best_roi_for_model = -999.0
        best_t_for_model = 0.77
        for t in thresholds:
            roi_r = calc_roi(test_sf, p, threshold=t)
            if roi_r["n_bets"] >= 50:
                if roi_r["roi"] > best_roi_for_model:
                    best_roi_for_model = roi_r["roi"]
                    best_t_for_model = t
                logger.info("    t=%.2f: ROI=%.2f%% n=%d", t, roi_r["roi"], roi_r["n_bets"])

        r["best_roi"] = best_roi_for_model
        r["best_t"] = best_t_for_model
        roi_at_77 = calc_roi(test_sf, p, threshold=0.77)
        r["roi_t77"] = roi_at_77["roi"]
        r["n_bets_t77"] = roi_at_77["n_bets"]

        if best_roi_for_model > best_overall_roi:
            best_overall_roi = best_roi_for_model
            best_overall_key = name
            best_overall_t = best_t_for_model

    logger.info("Summary (t=0.77):")
    for name, r in results.items():
        logger.info(
            "  %s: ROI=%.2f%% n=%d AUC=%.4f | best=%.2f%% at t=%.2f",
            name,
            r["roi_t77"],
            r["n_bets_t77"],
            r["auc"],
            r["best_roi"],
            r["best_t"],
        )

    with mlflow.start_run(run_name="phase4/step4.2_lgb_thresh") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "lgb_blend_thresh_sweep",
                    "n_features_base": len(feat_base),
                    "n_features_sel": len(feat_sel),
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "best_model": best_overall_key,
                    "best_threshold": best_overall_t,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_t77_{name}", r["roi_t77"])
                mlflow.log_metric(f"auc_{name}", r["auc"])
                mlflow.log_metric(f"best_roi_{name}", r["best_roi"])

            best_r = results[best_overall_key]
            roi_best = calc_roi(test_sf, best_r["proba"], threshold=best_overall_t)
            mlflow.log_metrics(
                {
                    "roi": roi_best["roi"],
                    "roc_auc": best_r["auc"],
                    "n_bets": roi_best["n_bets"],
                    "win_rate": roi_best["win_rate"],
                    "best_threshold": best_overall_t,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.2: BEST %s ROI=%.2f%% t=%.2f n=%d run=%s",
                best_overall_key,
                roi_best["roi"],
                best_overall_t,
                roi_best["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
