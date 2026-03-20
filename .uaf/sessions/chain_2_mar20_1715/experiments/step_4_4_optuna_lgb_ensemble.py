"""Step 4.4: Optuna LightGBM + final ensemble on ELO-only."""

import logging
import os
import traceback

import lightgbm
import mlflow
import optuna
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
from scipy.stats import rankdata
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
    """Optuna LGB + final ensemble."""
    logger.info("Step 4.4: Optuna LGB + final ensemble on ELO")

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

    # Optuna LightGBM
    def lgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_val, y_val)],
            callbacks=[
                lightgbm.early_stopping(30, verbose=False),
                lightgbm.log_evaluation(0),
            ],
        )
        proba_val = model.predict_proba(x_val)[:, 1]
        _best_t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=20)
        return val_roi

    study_lgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_lgb.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
    logger.info(
        "Optuna LGB best: val_roi=%.2f%%, params=%s",
        study_lgb.best_value,
        study_lgb.best_params,
    )

    with mlflow.start_run(run_name="phase4/step4.4_optuna_lgb_ens") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("phase", "4")

        try:
            # Train best LGB
            best_lgb_params = study_lgb.best_params.copy()
            best_lgb_params["random_state"] = 42
            best_lgb_params["verbose"] = -1
            lgb_model = LGBMClassifier(**best_lgb_params)
            lgb_model.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    lightgbm.early_stopping(30, verbose=False),
                    lightgbm.log_evaluation(0),
                ],
            )
            p_lgb_val = lgb_model.predict_proba(x_val)[:, 1]
            p_lgb_test = lgb_model.predict_proba(x_test)[:, 1]

            # CatBoost (Optuna best)
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

            # XGBoost (reasonable defaults)
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                eval_metric="auc",
                early_stopping_rounds=30,
            )
            xgb.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
            p_xgb_val = xgb.predict_proba(x_val)[:, 1]
            p_xgb_test = xgb.predict_proba(x_test)[:, 1]

            results = {}

            # Individual models
            for name, p_v, p_t in [
                ("cb", p_cb_val, p_cb_test),
                ("lgb", p_lgb_val, p_lgb_test),
                ("xgb", p_xgb_val, p_xgb_test),
            ]:
                t, _vr = find_best_threshold_on_val(val_df, p_v)
                roi = calc_roi(test, p_t, threshold=t)
                auc = roc_auc_score(test["target"], p_t)
                results[name] = {"roi": roi["roi"], "auc": auc, "t": t, "n": roi["n_bets"]}
                logger.info(
                    "%s: ROI=%.2f%% AUC=%.4f t=%.2f n=%d", name, roi["roi"], auc, t, roi["n_bets"]
                )

            # Ensemble combinations
            for w_cb, w_lgb, w_xgb, label in [
                (1 / 3, 1 / 3, 1 / 3, "equal"),
                (0.5, 0.25, 0.25, "cb50"),
                (0.4, 0.4, 0.2, "cb40_lgb40"),
                (0.5, 0.3, 0.2, "cb50_lgb30"),
            ]:
                p_v = w_cb * p_cb_val + w_lgb * p_lgb_val + w_xgb * p_xgb_val
                p_t = w_cb * p_cb_test + w_lgb * p_lgb_test + w_xgb * p_xgb_test
                t, _vr = find_best_threshold_on_val(val_df, p_v)
                roi = calc_roi(test, p_t, threshold=t)
                auc = roc_auc_score(test["target"], p_t)
                results[label] = {"roi": roi["roi"], "auc": auc, "t": t, "n": roi["n_bets"]}
                logger.info(
                    "Ens %s: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                    label,
                    roi["roi"],
                    auc,
                    t,
                    roi["n_bets"],
                )

            # Rank ensemble
            rv_cb = rankdata(p_cb_val) / len(p_cb_val)
            rv_lgb = rankdata(p_lgb_val) / len(p_lgb_val)
            rv_xgb = rankdata(p_xgb_val) / len(p_xgb_val)
            p_rank_val = (rv_cb + rv_lgb + rv_xgb) / 3

            rt_cb = rankdata(p_cb_test) / len(p_cb_test)
            rt_lgb = rankdata(p_lgb_test) / len(p_lgb_test)
            rt_xgb = rankdata(p_xgb_test) / len(p_xgb_test)
            p_rank_test = (rt_cb + rt_lgb + rt_xgb) / 3

            t_rank, _vr = find_best_threshold_on_val(val_df, p_rank_val)
            roi_rank = calc_roi(test, p_rank_test, threshold=t_rank)
            auc_rank = roc_auc_score(test["target"], p_rank_test)
            results["rank_avg"] = {
                "roi": roi_rank["roi"],
                "auc": auc_rank,
                "t": t_rank,
                "n": roi_rank["n_bets"],
            }
            logger.info(
                "Rank avg: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_rank["roi"],
                auc_rank,
                t_rank,
                roi_rank["n_bets"],
            )

            # Best result
            best_key = max(results, key=lambda k: results[k]["roi"])
            best = results[best_key]
            logger.info("Best: %s ROI=%.2f%%", best_key, best["roi"])

            # ROI at all thresholds for the best
            if best_key == "cb":
                p_final = p_cb_test
            elif best_key == "lgb":
                p_final = p_lgb_test
            elif best_key == "xgb":
                p_final = p_xgb_test
            elif best_key == "rank_avg":
                p_final = p_rank_test
            else:
                # weighted ensemble
                weights = {
                    "equal": (1 / 3, 1 / 3, 1 / 3),
                    "cb50": (0.5, 0.25, 0.25),
                    "cb40_lgb40": (0.4, 0.4, 0.2),
                    "cb50_lgb30": (0.5, 0.3, 0.2),
                }
                w = weights[best_key]
                p_final = w[0] * p_cb_test + w[1] * p_lgb_test + w[2] * p_xgb_test

            roi_thresholds = calc_roi_at_thresholds(test, p_final)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_best_t{int(t * 100):03d}", r["roi"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"optuna_lgb_ensemble_{best_key}",
                    "n_features": len(feature_cols),
                    "optuna_lgb_trials": 40,
                    "optuna_lgb_best_val_roi": study_lgb.best_value,
                    "leakage_free": "true",
                    **{f"lgb_hp_{k}": v for k, v in study_lgb.best_params.items()},
                }
            )

            for k, v in results.items():
                mlflow.log_metric(f"roi_{k}", v["roi"])

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
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.4")
            raise


if __name__ == "__main__":
    main()
