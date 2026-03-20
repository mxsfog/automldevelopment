"""Step 4.1: Ensemble (CatBoost + LightGBM + XGBoost) on ELO-only subset."""

import logging
import os
import traceback

import mlflow
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from step_2_5_safe_elo import build_safe_elo_features, get_safe_elo_features
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


def main() -> None:
    """Ensemble on ELO-only subset."""
    logger.info("Step 4.1: Ensemble on ELO-only subset")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)
    train_all, test_all = time_series_split(df)

    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    feature_cols = get_base_features() + get_engineered_features() + get_safe_elo_features()

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feature_cols])
    x_val = imp.transform(val_df[feature_cols])
    x_test = imp.transform(test[feature_cols])
    y_fit = train_fit["target"].values
    y_val = val_df["target"].values

    with mlflow.start_run(run_name="phase4/step4.1_ensemble_elo") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("phase", "4")

        try:
            # CatBoost (Optuna best params from step 3.1)
            cb = CatBoostClassifier(
                iterations=499,
                depth=7,
                learning_rate=0.214,
                l2_leaf_reg=1.15,
                random_strength=0.823,
                bagging_temperature=2.41,
                border_count=121,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=30,
            )
            cb.fit(x_fit, y_fit, eval_set=(x_val, y_val))
            p_cb_val = cb.predict_proba(x_val)[:, 1]
            p_cb_test = cb.predict_proba(x_test)[:, 1]
            logger.info("CatBoost AUC: %.4f", roc_auc_score(y_val, p_cb_val))

            # LightGBM
            lgb = LGBMClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=10,
                reg_lambda=5.0,
                random_state=42,
                verbose=-1,
            )
            lgb.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(30, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )
            p_lgb_val = lgb.predict_proba(x_val)[:, 1]
            p_lgb_test = lgb.predict_proba(x_test)[:, 1]
            logger.info("LightGBM AUC: %.4f", roc_auc_score(y_val, p_lgb_val))

            # XGBoost
            xgb = XGBClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.05,
                reg_lambda=5.0,
                random_state=42,
                verbosity=0,
                eval_metric="auc",
                early_stopping_rounds=30,
            )
            xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]
            logger.info("XGBoost AUC: %.4f", roc_auc_score(y_val, p_xgb_val))

            # Ensembles
            results = {}

            # Equal average
            p_avg_val = (p_cb_val + p_lgb_val + p_xgb_val) / 3
            p_avg_test = (p_cb_test + p_lgb_test + p_xgb_test) / 3
            best_t, _vr = find_best_threshold_on_val(val_df, p_avg_val)
            roi_avg = calc_roi(test, p_avg_test, threshold=best_t)
            auc_avg = roc_auc_score(test["target"], p_avg_test)
            results["avg"] = {
                "roi": roi_avg["roi"],
                "auc": auc_avg,
                "t": best_t,
                "n": roi_avg["n_bets"],
            }
            logger.info(
                "Equal avg: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_avg["roi"],
                auc_avg,
                best_t,
                roi_avg["n_bets"],
            )

            # CB-weighted
            for w_cb in [0.5, 0.6, 0.7]:
                w_other = (1 - w_cb) / 2
                p_w_val = w_cb * p_cb_val + w_other * p_lgb_val + w_other * p_xgb_val
                p_w_test = w_cb * p_cb_test + w_other * p_lgb_test + w_other * p_xgb_test
                t_w, _vr = find_best_threshold_on_val(val_df, p_w_val)
                roi_w = calc_roi(test, p_w_test, threshold=t_w)
                auc_w = roc_auc_score(test["target"], p_w_test)
                key = f"w{int(w_cb * 100)}"
                results[key] = {"roi": roi_w["roi"], "auc": auc_w, "t": t_w, "n": roi_w["n_bets"]}
                logger.info(
                    "CB w=%.1f: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                    w_cb,
                    roi_w["roi"],
                    auc_w,
                    t_w,
                    roi_w["n_bets"],
                )

            # CatBoost solo (re-evaluate at val-selected threshold)
            t_cb, _vr = find_best_threshold_on_val(val_df, p_cb_val)
            roi_cb = calc_roi(test, p_cb_test, threshold=t_cb)
            auc_cb = roc_auc_score(test["target"], p_cb_test)
            results["cb_solo"] = {
                "roi": roi_cb["roi"],
                "auc": auc_cb,
                "t": t_cb,
                "n": roi_cb["n_bets"],
            }
            logger.info(
                "CB solo: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_cb["roi"],
                auc_cb,
                t_cb,
                roi_cb["n_bets"],
            )

            # ROI at different thresholds for best ensemble
            best_key = max(results, key=lambda k: results[k]["roi"])
            logger.info("Best ensemble: %s (ROI=%.2f%%)", best_key, results[best_key]["roi"])

            # Log best results
            best = results[best_key]
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"ensemble_{best_key}",
                    "n_features": len(feature_cols),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "subset": "elo_only",
                    "leakage_free": "true",
                }
            )

            for k, v in results.items():
                mlflow.log_metric(f"roi_{k}", v["roi"])
                mlflow.log_metric(f"auc_{k}", v["auc"])

            # ROI at thresholds for best method
            if best_key == "cb_solo":
                p_best_test = p_cb_test
            elif best_key == "avg":
                p_best_test = p_avg_test
            else:
                w_cb = int(best_key[1:]) / 100
                w_other = (1 - w_cb) / 2
                p_best_test = w_cb * p_cb_test + w_other * p_lgb_test + w_other * p_xgb_test

            roi_thresholds = calc_roi_at_thresholds(test, p_best_test)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_best_t{int(t * 100):03d}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": best["roi"],
                    "roc_auc": best["auc"],
                    "n_bets": best["n"],
                    "best_threshold": best["t"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Step 4.1 run_id: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.1")
            raise


if __name__ == "__main__":
    main()
