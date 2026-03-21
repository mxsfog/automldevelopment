"""Step 2.5: ELO features + Sport Filter CatBoost (воспроизведение chain_2-4 baseline)."""

import logging
import os
import traceback

import mlflow
from catboost import CatBoostClassifier
from common import (
    CB_BEST_PARAMS,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
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


def main() -> None:
    """ELO + SF CatBoost с проверенными параметрами из chain_3."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

    # ELO filter
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info(
        "ELO filter: train %d->%d, test %d->%d",
        len(train_all),
        len(train_elo),
        len(test_all),
        len(test_elo),
    )

    # Sport filter
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    logger.info(
        "Sport filter: train %d->%d, test %d->%d",
        len(train_elo),
        len(train_sf),
        len(test_elo),
        len(test_sf),
    )

    feat_list = get_all_features()

    # Val split для threshold selection и early stopping
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    # Reference model с early stopping
    model_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    model_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = model_ref.get_best_iteration()
    logger.info("Best iteration: %d", best_iter)

    # Val threshold selection
    p_val = model_ref.predict_proba(x_val)[:, 1]
    best_t, val_roi = find_best_threshold_on_val(val_df, p_val, min_bets=15)
    logger.info("Val threshold: %.2f (val ROI=%.2f%%)", best_t, val_roi)

    # Full-train model
    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    params_ft = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter + 10
    model_ft = CatBoostClassifier(**params_ft)
    model_ft.fit(x_full, train_sf["target"])

    p_test = model_ft.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    # Test results
    roi_t77 = calc_roi(test_sf, p_test, threshold=0.77)
    roi_ev0 = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)

    logger.info("Test t=0.77: ROI=%.2f%% n=%d AUC=%.4f", roi_t77["roi"], roi_t77["n_bets"], auc)
    logger.info("Test EV>=0+p77: ROI=%.2f%% n=%d", roi_ev0["roi"], roi_ev0["n_bets"])

    # Feature importance
    importances = model_ft.get_feature_importance()
    feat_imp = sorted(zip(feat_list, importances, strict=True), key=lambda x: x[1], reverse=True)
    logger.info("Top features:")
    for fname, imp_val in feat_imp[:10]:
        logger.info("  %s: %.2f", fname, imp_val)

    with mlflow.start_run(run_name="phase2/step2.5_elo_sf") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "2.5")
        mlflow.set_tag("phase", "2")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "CatBoost_ELO_SF_fulltrain",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_sf),
                    "n_samples_val": len(test_sf),
                    "best_iteration": best_iter,
                    "threshold": 0.77,
                    "ev_threshold": 0.0,
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": roi_ev0["roi"],
                    "roc_auc": auc,
                    "n_bets": roi_ev0["n_bets"],
                    "win_rate": roi_ev0["win_rate"],
                    "roi_t77_only": roi_t77["roi"],
                    "n_bets_t77_only": roi_t77["n_bets"],
                }
            )

            # Log feature importance
            imp_text = "\n".join(f"{f}: {v:.2f}" for f, v in feat_imp)
            mlflow.log_text(imp_text, "feature_importance.txt")
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            logger.info(
                "Step 2.5: ROI=%.2f%% (EV0+p77) / %.2f%% (t77) AUC=%.4f run=%s",
                roi_ev0["roi"],
                roi_t77["roi"],
                auc,
                run.info.run_id,
            )

            return {
                "roi_ev0": roi_ev0["roi"],
                "roi_t77": roi_t77["roi"],
                "auc": auc,
                "run_id": run.info.run_id,
            }

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
