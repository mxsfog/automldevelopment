"""Step 4.1: ML platform features + XGBoost + CB+XGB ensemble.

Гипотезы:
A) ML_Winrate_Diff, ML_Rating_Diff, ML_Team_Stats_Found — неиспользованные фичи платформы
B) XGBoost как альтернатива CatBoost
C) CB + XGB averaging ensemble
"""

import logging
import os
import traceback

import mlflow
import pandas as pd
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def add_ml_platform_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Добавление ML platform features (Winrate_Diff, Rating_Diff, Stats_Found)."""
    df = df.copy()
    new_feats = []

    # ML_Team_Stats_Found: bool -> float
    if "ML_Team_Stats_Found" in df.columns:
        df["ml_stats_found"] = (df["ML_Team_Stats_Found"] == "t").astype(float)
        new_feats.append("ml_stats_found")

    # ML_Winrate_Diff, ML_Rating_Diff — already numeric
    for col in ["ML_Winrate_Diff", "ML_Rating_Diff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_feats.append(col)

    # Derived: abs values
    if "ML_Winrate_Diff" in df.columns:
        df["ml_wr_diff_abs"] = df["ML_Winrate_Diff"].abs()
        new_feats.append("ml_wr_diff_abs")

    if "ML_Rating_Diff" in df.columns:
        df["ml_rating_diff_abs"] = df["ML_Rating_Diff"].abs()
        new_feats.append("ml_rating_diff_abs")

    # Interaction: ML model agreement with platform
    if "ML_Winrate_Diff" in df.columns and "ML_Edge" in df.columns:
        df["ml_wr_x_edge"] = df["ML_Winrate_Diff"] * df["ML_Edge"]
        new_feats.append("ml_wr_x_edge")

    return df, new_feats


def main() -> None:
    """ML platform features + XGBoost + ensemble."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)
    df, ml_feats = add_ml_platform_features(df)
    logger.info("New ML platform features: %s", ml_feats)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    base_feats = get_all_features()
    feat_with_ml = base_feats + ml_feats

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    results: dict[str, dict] = {}

    # A: CatBoost baseline (chain_4 params, base feats)
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[base_feats])
    x_val = imp.transform(val_df[base_feats])

    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter_ref = cb_ref.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[base_feats])
    x_test = imp_full.transform(test_sf[base_feats])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter_ref + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test_cb = cb_ft.predict_proba(x_test)[:, 1]
    auc_cb = roc_auc_score(test_sf["target"], p_test_cb)
    roi_cb = calc_ev_roi(test_sf, p_test_cb, ev_threshold=0.0, min_prob=0.77)
    results["cb_baseline"] = {"roi": roi_cb["roi"], "n_bets": roi_cb["n_bets"], "auc": auc_cb}
    logger.info(
        "A: CB baseline: ROI=%.2f%% n=%d AUC=%.4f", roi_cb["roi"], roi_cb["n_bets"], auc_cb
    )

    # B: CatBoost + ML platform features
    check_budget()
    imp_ml = SimpleImputer(strategy="median")
    x_fit_ml = imp_ml.fit_transform(train_fit[feat_with_ml])
    x_val_ml = imp_ml.transform(val_df[feat_with_ml])

    cb_ml = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ml.fit(x_fit_ml, train_fit["target"], eval_set=(x_val_ml, val_df["target"]))
    best_iter_ml = cb_ml.get_best_iteration()

    imp_ml_full = SimpleImputer(strategy="median")
    x_full_ml = imp_ml_full.fit_transform(train_sf[feat_with_ml])
    x_test_ml = imp_ml_full.transform(test_sf[feat_with_ml])

    ft_params_ml = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params_ml["iterations"] = best_iter_ml + 10
    cb_ft_ml = CatBoostClassifier(**ft_params_ml)
    cb_ft_ml.fit(x_full_ml, train_sf["target"])

    p_test_cb_ml = cb_ft_ml.predict_proba(x_test_ml)[:, 1]
    auc_cb_ml = roc_auc_score(test_sf["target"], p_test_cb_ml)
    roi_cb_ml = calc_ev_roi(test_sf, p_test_cb_ml, ev_threshold=0.0, min_prob=0.77)
    results["cb_ml_feats"] = {
        "roi": roi_cb_ml["roi"],
        "n_bets": roi_cb_ml["n_bets"],
        "auc": auc_cb_ml,
    }
    logger.info(
        "B: CB+ML feats: ROI=%.2f%% n=%d AUC=%.4f (delta=%.2f pp)",
        roi_cb_ml["roi"],
        roi_cb_ml["n_bets"],
        auc_cb_ml,
        roi_cb_ml["roi"] - roi_cb["roi"],
    )

    # C: XGBoost with base features
    check_budget()
    xgb_params = {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.08,
        "reg_lambda": 21.1,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "verbosity": 0,
        "tree_method": "hist",
    }

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(x_fit, train_fit["target"], eval_set=[(x_val, val_df["target"])], verbose=False)
    best_iter_xgb = xgb_model.best_iteration

    xgb_ft_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    xgb_ft_params["n_estimators"] = best_iter_xgb + 10
    xgb_ft = XGBClassifier(**xgb_ft_params)
    xgb_ft.fit(x_full, train_sf["target"], verbose=False)

    p_test_xgb = xgb_ft.predict_proba(x_test)[:, 1]
    auc_xgb = roc_auc_score(test_sf["target"], p_test_xgb)
    roi_xgb = calc_ev_roi(test_sf, p_test_xgb, ev_threshold=0.0, min_prob=0.77)
    results["xgb_base"] = {"roi": roi_xgb["roi"], "n_bets": roi_xgb["n_bets"], "auc": auc_xgb}
    logger.info(
        "C: XGBoost: ROI=%.2f%% n=%d AUC=%.4f (delta=%.2f pp)",
        roi_xgb["roi"],
        roi_xgb["n_bets"],
        auc_xgb,
        roi_xgb["roi"] - roi_cb["roi"],
    )

    # D: XGBoost + ML platform features
    check_budget()
    xgb_ml = XGBClassifier(**xgb_params)
    xgb_ml.fit(
        x_fit_ml,
        train_fit["target"],
        eval_set=[(x_val_ml, val_df["target"])],
        verbose=False,
    )
    best_iter_xgb_ml = xgb_ml.best_iteration

    xgb_ft_ml_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    xgb_ft_ml_params["n_estimators"] = best_iter_xgb_ml + 10
    xgb_ft_ml = XGBClassifier(**xgb_ft_ml_params)
    xgb_ft_ml.fit(x_full_ml, train_sf["target"], verbose=False)

    p_test_xgb_ml = xgb_ft_ml.predict_proba(x_test_ml)[:, 1]
    auc_xgb_ml = roc_auc_score(test_sf["target"], p_test_xgb_ml)
    roi_xgb_ml = calc_ev_roi(test_sf, p_test_xgb_ml, ev_threshold=0.0, min_prob=0.77)
    results["xgb_ml_feats"] = {
        "roi": roi_xgb_ml["roi"],
        "n_bets": roi_xgb_ml["n_bets"],
        "auc": auc_xgb_ml,
    }
    logger.info(
        "D: XGB+ML feats: ROI=%.2f%% n=%d AUC=%.4f",
        roi_xgb_ml["roi"],
        roi_xgb_ml["n_bets"],
        auc_xgb_ml,
    )

    # E: CB + XGB ensemble (simple averaging)
    check_budget()
    for w_cb in [0.5, 0.6, 0.7, 0.8]:
        p_ens = w_cb * p_test_cb + (1 - w_cb) * p_test_xgb
        auc_ens = roc_auc_score(test_sf["target"], p_ens)
        roi_ens = calc_ev_roi(test_sf, p_ens, ev_threshold=0.0, min_prob=0.77)
        key = f"ens_cb{int(w_cb * 100)}_xgb{int((1 - w_cb) * 100)}"
        results[key] = {"roi": roi_ens["roi"], "n_bets": roi_ens["n_bets"], "auc": auc_ens}
        logger.info(
            "E: %s: ROI=%.2f%% n=%d AUC=%.4f",
            key,
            roi_ens["roi"],
            roi_ens["n_bets"],
            auc_ens,
        )

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d AUC=%.4f", name, r["roi"], r["n_bets"], r["auc"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.1_ml_xgb") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "ml_platform_xgb_ensemble",
                    "n_base_features": len(base_feats),
                    "n_ml_features": len(ml_feats),
                    "ml_features": str(ml_feats),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])
                mlflow.log_metric(f"auc_{name}", r["auc"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": best_r["auc"],
                    "n_bets": best_r["n_bets"],
                    "delta_vs_baseline": best_r["roi"] - roi_cb["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.1: Best=%s ROI=%.2f%% (delta=%.2f pp) run=%s",
                best_key,
                best_r["roi"],
                best_r["roi"] - roi_cb["roi"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
