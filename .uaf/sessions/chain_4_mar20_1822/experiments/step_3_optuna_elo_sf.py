"""Phase 3: Optuna hyperparameter optimization on ELO + sport-filtered data."""

import logging
import os
import traceback

import mlflow
import optuna
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
    """Optuna TPE на ELO + sport-filtered data."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()

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

    logger.info(
        "Data: train_fit=%d, val=%d, test_sf=%d, features=%d",
        len(train_fit),
        len(val_df),
        len(test_sf),
        len(feat_list),
    )

    def objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 60.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 60),
            "random_strength": trial.suggest_float("random_strength", 0.3, 8.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        p_val = model.predict_proba(x_val)[:, 1]
        _t, val_roi = find_best_threshold_on_val(val_df, p_val, min_bets=15)
        return val_roi

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=40, show_progress_bar=True)
    best_params = study.best_params
    logger.info("Optuna best val ROI: %.2f%%, params: %s", study.best_value, best_params)

    # Train final models
    final_params = {
        "iterations": 1000,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        **best_params,
    }

    # Optuna model
    model_opt = CatBoostClassifier(**final_params)
    model_opt.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val_opt = model_opt.predict_proba(x_val)[:, 1]
    p_test_opt = model_opt.predict_proba(x_test)[:, 1]

    # Reference model (chain_3 best params)
    ref_params = {
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
    model_ref = CatBoostClassifier(**ref_params)
    model_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val_ref = model_ref.predict_proba(x_val)[:, 1]
    p_test_ref = model_ref.predict_proba(x_test)[:, 1]

    # Evaluate all combos
    results: dict[str, dict] = {}

    t_opt, _ = find_best_threshold_on_val(val_df, p_val_opt, min_bets=15)
    results["optuna_val_t"] = {
        **calc_roi(test_sf, p_test_opt, threshold=t_opt),
        "threshold": t_opt,
    }
    results["optuna_t77"] = {
        **calc_roi(test_sf, p_test_opt, threshold=0.77),
        "threshold": 0.77,
    }

    t_ref, _ = find_best_threshold_on_val(val_df, p_val_ref, min_bets=15)
    results["ref_val_t"] = {
        **calc_roi(test_sf, p_test_ref, threshold=t_ref),
        "threshold": t_ref,
    }
    results["ref_t77"] = {
        **calc_roi(test_sf, p_test_ref, threshold=0.77),
        "threshold": 0.77,
    }

    auc_opt = roc_auc_score(test_sf["target"], p_test_opt)
    auc_ref = roc_auc_score(test_sf["target"], p_test_ref)

    logger.info("Results comparison:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], r["threshold"], r["n_bets"])
    logger.info("AUC: optuna=%.4f, ref=%.4f", auc_opt, auc_ref)

    # Feature importance
    fi = model_opt.get_feature_importance()
    fi_pairs = sorted(zip(feat_list, fi, strict=True), key=lambda x: x[1], reverse=True)
    logger.info("Top-10 features (Optuna model):")
    for fname, imp_val in fi_pairs[:10]:
        logger.info("  %s: %.2f", fname, imp_val)

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]
    best_auc = auc_opt if "optuna" in best_key else auc_ref

    with mlflow.start_run(run_name="phase3/step3.1_optuna_sf") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "3.1")
        mlflow.set_tag("phase", "3")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "optuna_catboost_sf",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test_sf": len(test_sf),
                    "n_optuna_trials": 40,
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "best_variant": best_key,
                    **{f"opt_{k}": v for k, v in best_params.items()},
                    **{
                        f"ref_{k}": v
                        for k, v in ref_params.items()
                        if k
                        not in (
                            "iterations",
                            "random_seed",
                            "verbose",
                            "eval_metric",
                            "early_stopping_rounds",
                        )
                    },
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": best_auc,
                    "n_bets": best_r["n_bets"],
                    "win_rate": best_r["win_rate"],
                    "best_threshold": best_r["threshold"],
                    "auc_optuna": auc_opt,
                    "auc_ref": auc_ref,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Phase 3: BEST %s ROI=%.2f%% AUC=%.4f t=%.2f n=%d run=%s",
                best_key,
                best_r["roi"],
                best_auc,
                best_r["threshold"],
                best_r["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
