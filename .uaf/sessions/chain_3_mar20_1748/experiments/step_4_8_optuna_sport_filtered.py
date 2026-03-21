"""Step 4.8: Optuna CB on sport-filtered data + min_bets sensitivity."""

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
    """Optuna on sport-filtered ELO data + threshold/min_bets sensitivity."""
    logger.info("Step 4.8: Optuna CB on sport-filtered ELO data")

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

    # Optuna: conservative search space (avoid overfitting)
    logger.info("Optuna: 30 trials, conservative search space")

    def objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 6, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 10.0, 50.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 50),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 64, 200),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        proba_val = model.predict_proba(x_val)[:, 1]
        _t, val_roi = find_best_threshold_on_val(val_df, proba_val, min_bets=15)
        return val_roi

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    best_params = study.best_params
    logger.info("Optuna best val ROI: %.2f%%, params: %s", study.best_value, best_params)

    # Train final model
    final_params = {
        "iterations": 1000,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        **best_params,
    }
    model_optuna = CatBoostClassifier(**final_params)
    model_optuna.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val_opt = model_optuna.predict_proba(x_val)[:, 1]
    p_test_opt = model_optuna.predict_proba(x_test)[:, 1]

    configs: dict[str, tuple[dict, float]] = {}

    # A: Optuna model, val threshold
    t_opt, _ = find_best_threshold_on_val(val_df, p_val_opt, min_bets=15)
    configs["optuna_sf_val"] = (calc_roi(test_sf, p_test_opt, threshold=t_opt), t_opt)

    # B: Optuna model, fixed t=0.77
    configs["optuna_sf_t77"] = (calc_roi(test_sf, p_test_opt, threshold=0.77), 0.77)

    # C: Reference: step 3.1 params on sport-filtered data
    ref_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.08,
        l2_leaf_reg=21.1,
        min_data_in_leaf=20,
        random_strength=1.0,
        bagging_temperature=0.06,
        border_count=102,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=50,
    )
    ref_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val_ref = ref_model.predict_proba(x_val)[:, 1]
    p_test_ref = ref_model.predict_proba(x_test)[:, 1]

    t_ref, _ = find_best_threshold_on_val(val_df, p_val_ref, min_bets=15)
    configs["ref_sf_val"] = (calc_roi(test_sf, p_test_ref, threshold=t_ref), t_ref)
    configs["ref_sf_t77"] = (calc_roi(test_sf, p_test_ref, threshold=0.77), 0.77)

    # D: Also test ref model trained on ALL ELO + sport filter at inference
    # This is the winning config from step 4.6
    x_fit_all = imp.fit_transform(train_elo.iloc[: int(len(train_elo) * 0.8)][feat_list])
    x_val_all = imp.transform(train_elo.iloc[int(len(train_elo) * 0.8) :][feat_list])
    x_test_all = imp.transform(test_elo[feat_list])

    val_all_df = train_elo.iloc[int(len(train_elo) * 0.8) :]
    ref_all = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.08,
        l2_leaf_reg=21.1,
        min_data_in_leaf=20,
        random_strength=1.0,
        bagging_temperature=0.06,
        border_count=102,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=50,
    )
    ref_all.fit(
        x_fit_all,
        train_elo.iloc[: int(len(train_elo) * 0.8)]["target"],
        eval_set=(x_val_all, val_all_df["target"]),
    )
    p_test_all = ref_all.predict_proba(x_test_all)[:, 1]

    # Sport filter at inference
    mask_sf = ~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)
    configs["train_all_infer_sf_t77"] = (
        calc_roi(test_elo[mask_sf], p_test_all[mask_sf.values], threshold=0.77),
        0.77,
    )

    # E: min_bets sensitivity for best Optuna model
    logger.info("Min-bets sensitivity (Optuna model):")
    for min_b in [10, 15, 20, 30]:
        t_mb, _ = find_best_threshold_on_val(val_df, p_val_opt, min_bets=min_b)
        r_mb = calc_roi(test_sf, p_test_opt, threshold=t_mb)
        logger.info(
            "  min_bets=%d: ROI=%.2f%% t=%.2f n=%d", min_b, r_mb["roi"], t_mb, r_mb["n_bets"]
        )

    # Log all
    logger.info("All results:")
    for name, (r, t) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    best_key = max(configs, key=lambda k: configs[k][0]["roi"])
    best_result, best_threshold = configs[best_key]
    auc = roc_auc_score(test_sf["target"], p_test_opt)

    with mlflow.start_run(run_name="phase4/step4.8_optuna_sf") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test_sf": len(test_sf),
                    "n_optuna_trials": 30,
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "best_variant": best_key,
                    **{f"hp_{k}": v for k, v in best_params.items()},
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
                "Step 4.8: BEST %s ROI=%.2f%% t=%.2f n=%d run=%s",
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
