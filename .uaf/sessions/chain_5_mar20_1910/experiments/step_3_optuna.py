"""Step 3.1: Optuna TPE на ROI с EV-отбором. Расширенное пространство поиска."""

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
    calc_ev_roi,
    calc_roi,
    check_budget,
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
    """Optuna ROI-oriented optimization."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_all_features()

    # Val split: 80% train, 20% val (из train_sf)
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    def objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }

        model = CatBoostClassifier(**params)
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

        p_val = model.predict_proba(x_val)[:, 1]
        # Оптимизируем ROI с EV>=0 + p>=0.77 на val
        roi_ev = calc_ev_roi(val_df, p_val, ev_threshold=0.0, min_prob=0.77)

        if roi_ev["n_bets"] < 10:
            return -100.0

        return roi_ev["roi"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=50, timeout=300)

    best_params = study.best_params
    best_val_roi = study.best_value
    logger.info("Optuna best val ROI: %.2f%% params: %s", best_val_roi, best_params)

    # Full-train с лучшими параметрами
    check_budget()
    best_full_params = {
        "iterations": 1000,
        **best_params,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
    }

    model_es = CatBoostClassifier(**best_full_params)
    model_es.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = model_es.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    ft_params = {k: v for k, v in best_full_params.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    model_ft = CatBoostClassifier(**ft_params)
    model_ft.fit(x_full, train_sf["target"])

    p_test = model_ft.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    roi_t77 = calc_roi(test_sf, p_test, threshold=0.77)
    roi_ev0 = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)

    logger.info("Test t=0.77: ROI=%.2f%% n=%d", roi_t77["roi"], roi_t77["n_bets"])
    logger.info("Test EV>=0+p77: ROI=%.2f%% n=%d", roi_ev0["roi"], roi_ev0["n_bets"])
    logger.info("AUC=%.4f best_iter=%d", auc, best_iter)

    # Сравнение с baseline (chain_4 params)
    from common import CB_BEST_PARAMS

    model_base_es = CatBoostClassifier(**CB_BEST_PARAMS)
    model_base_es.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    base_iter = model_base_es.get_best_iteration()

    base_ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    base_ft_params["iterations"] = base_iter + 10
    model_base_ft = CatBoostClassifier(**base_ft_params)
    model_base_ft.fit(x_full, train_sf["target"])

    p_test_base = model_base_ft.predict_proba(x_test)[:, 1]
    roi_base_ev0 = calc_ev_roi(test_sf, p_test_base, ev_threshold=0.0, min_prob=0.77)
    logger.info("Baseline EV>=0+p77: ROI=%.2f%% n=%d", roi_base_ev0["roi"], roi_base_ev0["n_bets"])
    logger.info(
        "Delta: %.2f pp",
        roi_ev0["roi"] - roi_base_ev0["roi"],
    )

    with mlflow.start_run(run_name="phase3/step3.1_optuna") as run:
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
                    "method": "optuna_tpe_roi",
                    "n_trials": len(study.trials),
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_sf),
                    "best_iteration": best_iter,
                    **{f"opt_{k}": v for k, v in best_params.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "roi": roi_ev0["roi"],
                    "roc_auc": auc,
                    "n_bets": roi_ev0["n_bets"],
                    "win_rate": roi_ev0["win_rate"],
                    "roi_t77_only": roi_t77["roi"],
                    "val_roi_best": best_val_roi,
                    "roi_baseline": roi_base_ev0["roi"],
                    "delta_pp": roi_ev0["roi"] - roi_base_ev0["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 3.1: Optuna ROI=%.2f%% (delta=%.2f pp) run=%s",
                roi_ev0["roi"],
                roi_ev0["roi"] - roi_base_ev0["roi"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
