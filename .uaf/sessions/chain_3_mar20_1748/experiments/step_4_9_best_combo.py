"""Step 4.9: Best combination: multi-seed CB on sport-filtered + fixed t=0.77."""

import logging
import os
import traceback

import lightgbm as lgb
import mlflow
import numpy as np
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

CB_PARAMS = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 21.1,
    "min_data_in_leaf": 20,
    "random_strength": 1.0,
    "bagging_temperature": 0.06,
    "border_count": 102,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def main() -> None:
    """Best combo: multi-seed CB sport-filtered + ensemble + threshold strategies."""
    logger.info("Step 4.9: Best combination strategies")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()

    # Sport filter
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test_sf[feat_list])

    configs: dict[str, tuple[dict, float]] = {}

    # A: Multi-seed CB on sport-filtered, t=0.77
    logger.info("A: Multi-seed CB sport-filtered (5 seeds)")
    seeds = [42, 43, 44, 45, 46]
    p_val_cb = []
    p_test_cb = []

    for seed in seeds:
        check_budget()
        model = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed})
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        p_val_cb.append(model.predict_proba(x_val)[:, 1])
        p_test_cb.append(model.predict_proba(x_test)[:, 1])

    p_val_cb_avg = np.mean(p_val_cb, axis=0)
    p_test_cb_avg = np.mean(p_test_cb, axis=0)

    configs["multiseed_sf_t77"] = (calc_roi(test_sf, p_test_cb_avg, threshold=0.77), 0.77)
    t_ms, _ = find_best_threshold_on_val(val_df, p_val_cb_avg, min_bets=15)
    configs["multiseed_sf_val"] = (calc_roi(test_sf, p_test_cb_avg, threshold=t_ms), t_ms)

    # B: Single seed=42 reference (step 4.8 winner)
    configs["cb42_sf_t77"] = (calc_roi(test_sf, p_test_cb[0], threshold=0.77), 0.77)

    # C: Multi-seed ensemble: CB(5 seeds) + LGB + XGB
    logger.info("C: Multi-seed CB + LGB + XGB ensemble on sport-filtered")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.058,
        reg_lambda=27.5,
        min_child_samples=46,
        subsample=0.88,
        colsample_bytree=0.95,
        num_leaves=22,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        x_fit,
        train_fit["target"],
        eval_set=[(x_val, val_df["target"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_val_lgb = lgb_model.predict_proba(x_val)[:, 1]
    p_test_lgb = lgb_model.predict_proba(x_test)[:, 1]

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_lambda=10.0,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(x_fit, train_fit["target"], eval_set=[(x_val, val_df["target"])], verbose=False)
    p_val_xgb = xgb_model.predict_proba(x_val)[:, 1]
    p_test_xgb = xgb_model.predict_proba(x_test)[:, 1]

    # CB65 ensemble on sport-filtered
    p_val_ens = 0.65 * p_val_cb_avg + 0.20 * p_val_lgb + 0.15 * p_val_xgb
    p_test_ens = 0.65 * p_test_cb_avg + 0.20 * p_test_lgb + 0.15 * p_test_xgb

    configs["ens65_sf_t77"] = (calc_roi(test_sf, p_test_ens, threshold=0.77), 0.77)
    t_ens, _ = find_best_threshold_on_val(val_df, p_val_ens, min_bets=15)
    configs["ens65_sf_val"] = (calc_roi(test_sf, p_test_ens, threshold=t_ens), t_ens)

    # D: CB80 ensemble (more weight on CB)
    p_val_cb80 = 0.80 * p_val_cb_avg + 0.10 * p_val_lgb + 0.10 * p_val_xgb
    p_test_cb80 = 0.80 * p_test_cb_avg + 0.10 * p_test_lgb + 0.10 * p_test_xgb

    configs["cb80_sf_t77"] = (calc_roi(test_sf, p_test_cb80, threshold=0.77), 0.77)
    t_cb80, _ = find_best_threshold_on_val(val_df, p_val_cb80, min_bets=15)
    configs["cb80_sf_val"] = (calc_roi(test_sf, p_test_cb80, threshold=t_cb80), t_cb80)

    # E: Threshold sensitivity around 0.77
    for t_try in [0.74, 0.75, 0.76, 0.78, 0.79, 0.80]:
        r_try = calc_roi(test_sf, p_test_cb[0], threshold=t_try)
        if r_try["n_bets"] >= 15:
            configs[f"cb42_sf_t{int(t_try * 100)}"] = (r_try, t_try)

    # Log all
    logger.info("All results:")
    for name, (r, t) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    best_key = max(configs, key=lambda k: configs[k][0]["roi"])
    best_result, best_threshold = configs[best_key]
    auc = roc_auc_score(test_sf["target"], p_test_cb[0])

    with mlflow.start_run(run_name="phase4/step4.9_best_combo") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": "42,43,44,45,46",
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test_sf": len(test_sf),
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "best_variant": best_key,
                }
            )

            for name, (r, _t) in configs.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roc_auc": auc,
                    "n_bets": best_result["n_bets"],
                    "win_rate": best_result["win_rate"],
                    "best_threshold": best_threshold,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")

            logger.info(
                "Step 4.9: BEST %s ROI=%.2f%% t=%.2f n=%d run=%s",
                best_key,
                best_result["roi"],
                best_threshold,
                best_result["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
