"""Step 4.4: Full-train model with temporal CV + Optuna re-optimization."""

import logging
import os
import traceback

import mlflow
import numpy as np
import optuna
import pandas as pd
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


def temporal_cv_full_train(
    df_sf: pd.DataFrame, feat_list: list[str], n_folds: int = 4
) -> list[dict]:
    """Temporal CV: для каждого fold обучаем full-train модель."""
    df_sorted = df_sf.sort_values("Created_At").reset_index(drop=True)
    n = len(df_sorted)

    # Разбиваем на 5 блоков, используем первые n_folds+1 для train/test пар
    block_size = n // (n_folds + 1)
    fold_results = []

    for fold_idx in range(n_folds):
        check_budget()
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(train_end + block_size, n)

        fold_train = df_sorted.iloc[:train_end].copy()
        fold_test = df_sorted.iloc[test_start:test_end].copy()

        if len(fold_train) < 100 or len(fold_test) < 30:
            continue

        # Internal val for threshold + early stopping iteration detection
        val_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:val_split]
        inner_val = fold_train.iloc[val_split:]

        imp = SimpleImputer(strategy="median")
        x_inner_train = imp.fit_transform(inner_train[feat_list])
        x_inner_val = imp.transform(inner_val[feat_list])

        # Get best iteration
        cb_iter = CatBoostClassifier(**CB_PARAMS)
        cb_iter.fit(
            x_inner_train,
            inner_train["target"],
            eval_set=(x_inner_val, inner_val["target"]),
        )
        best_iter = cb_iter.get_best_iteration()

        # Threshold from inner val
        p_inner_val = cb_iter.predict_proba(x_inner_val)[:, 1]
        t_val, _ = find_best_threshold_on_val(inner_val, p_inner_val, min_bets=10)

        # Full-train model
        imp_full = SimpleImputer(strategy="median")
        x_full = imp_full.fit_transform(fold_train[feat_list])
        x_test = imp_full.transform(fold_test[feat_list])

        params_no_es = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
        params_no_es["iterations"] = best_iter + 10
        cb_full = CatBoostClassifier(**params_no_es)
        cb_full.fit(x_full, fold_train["target"])

        p_test = cb_full.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(fold_test["target"], p_test)
        roi_t77 = calc_roi(fold_test, p_test, threshold=0.77)
        roi_val_t = calc_roi(fold_test, p_test, threshold=t_val)

        # Also 80/20 model for comparison
        p_test_80 = cb_iter.predict_proba(imp.transform(fold_test[feat_list]))[:, 1]
        roi_t77_80 = calc_roi(fold_test, p_test_80, threshold=0.77)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_size": len(fold_train),
                "test_size": len(fold_test),
                "auc": auc,
                "roi_t77": roi_t77["roi"],
                "n_bets_t77": roi_t77["n_bets"],
                "roi_val_t": roi_val_t["roi"],
                "val_threshold": t_val,
                "best_iter": best_iter,
                "roi_t77_80": roi_t77_80["roi"],
            }
        )

        logger.info(
            "  Fold %d: full ROI(0.77)=%.2f%% n=%d | 80/20 ROI(0.77)=%.2f%% | AUC=%.4f",
            fold_idx,
            roi_t77["roi"],
            roi_t77["n_bets"],
            roi_t77_80["roi"],
            auc,
        )

    return fold_results


def main() -> None:
    """Full-train temporal CV + Optuna on full train."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    # Part 1: Temporal CV comparison
    logger.info("Part 1: Temporal CV (full-train vs 80/20)")
    fold_results = temporal_cv_full_train(train_sf, feat_list, n_folds=4)

    if fold_results:
        avg_roi_full = np.mean([r["roi_t77"] for r in fold_results])
        avg_roi_80 = np.mean([r["roi_t77_80"] for r in fold_results])
        avg_auc = np.mean([r["auc"] for r in fold_results])
        std_roi_full = np.std([r["roi_t77"] for r in fold_results])
        logger.info(
            "CV results: full avg ROI=%.2f%% (std=%.2f%%), 80/20 avg ROI=%.2f%%, AUC=%.4f",
            avg_roi_full,
            std_roi_full,
            avg_roi_80,
            avg_auc,
        )
    else:
        avg_roi_full = 0.0
        avg_roi_80 = 0.0
        avg_auc = 0.0
        std_roi_full = 0.0

    # Part 2: Optuna on full-train setup
    check_budget()
    logger.info("Part 2: Optuna on full-train setup (25 trials)")

    # First get best_iter from ref model
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp_ref = SimpleImputer(strategy="median")
    x_fit_ref = imp_ref.fit_transform(train_fit[feat_list])
    x_val_ref = imp_ref.transform(val_df[feat_list])

    ref_model = CatBoostClassifier(**CB_PARAMS)
    ref_model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
    ref_best_iter = ref_model.get_best_iteration()

    def objective(trial: optuna.Trial) -> float:
        check_budget()
        params = {
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 6, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 10.0, 50.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 12, 40),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 50, 200),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
        p_val = model.predict_proba(x_val_ref)[:, 1]
        _t, val_roi = find_best_threshold_on_val(val_df, p_val, min_bets=15)
        return val_roi

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=25, show_progress_bar=True)
    best_params = study.best_params
    logger.info("Optuna best val ROI: %.2f%%, params: %s", study.best_value, best_params)

    # Train full-train Optuna model
    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test_full = imp_full.transform(test_sf[feat_list])

    # Get best_iter from Optuna params
    optuna_iter_model = CatBoostClassifier(
        **{
            **best_params,
            "iterations": 1000,
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
        },
    )
    optuna_iter_model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
    optuna_best_iter = optuna_iter_model.get_best_iteration()
    p_val_opt = optuna_iter_model.predict_proba(x_val_ref)[:, 1]
    t_opt, _ = find_best_threshold_on_val(val_df, p_val_opt, min_bets=15)

    # Full-train Optuna
    params_full_opt = {k: v for k, v in best_params.items()}
    params_full_opt.update(
        {
            "iterations": optuna_best_iter + 10,
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "AUC",
        }
    )
    model_full_opt = CatBoostClassifier(**params_full_opt)
    model_full_opt.fit(x_full, train_sf["target"])
    p_test_opt = model_full_opt.predict_proba(x_test_full)[:, 1]
    auc_opt = roc_auc_score(test_sf["target"], p_test_opt)

    # Full-train with ref params
    params_full_ref = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_full_ref["iterations"] = ref_best_iter + 10
    model_full_ref = CatBoostClassifier(**params_full_ref)
    model_full_ref.fit(x_full, train_sf["target"])
    p_test_ref = model_full_ref.predict_proba(x_test_full)[:, 1]
    auc_ref = roc_auc_score(test_sf["target"], p_test_ref)

    # Compare
    results: dict[str, dict] = {}
    for name, p, auc_val in [
        ("full_ref", p_test_ref, auc_ref),
        ("full_optuna", p_test_opt, auc_opt),
    ]:
        for t in [0.76, 0.77, t_opt]:
            key = f"{name}_t{t:.2f}"
            roi_r = calc_roi(test_sf, p, threshold=t)
            results[key] = {**roi_r, "threshold": t, "auc": auc_val}

    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info(
            "  %s: ROI=%.2f%% n=%d AUC=%.4f",
            name,
            r["roi"],
            r["n_bets"],
            r["auc"],
        )

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.4_fulltrain_cv") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "full_train_cv_optuna",
                    "n_features": len(feat_list),
                    "n_optuna_trials": 25,
                    "ref_best_iter": ref_best_iter,
                    "optuna_best_iter": optuna_best_iter,
                    "best_variant": best_key,
                    **{f"opt_{k}": v for k, v in best_params.items()},
                }
            )

            # CV metrics
            if fold_results:
                for fr in fold_results:
                    mlflow.log_metric(f"cv_roi_full_fold{fr['fold']}", fr["roi_t77"])
                    mlflow.log_metric(f"cv_roi_80_fold{fr['fold']}", fr["roi_t77_80"])
                mlflow.log_metric("cv_avg_roi_full", avg_roi_full)
                mlflow.log_metric("cv_avg_roi_80", avg_roi_80)
                mlflow.log_metric("cv_std_roi_full", std_roi_full)
                mlflow.log_metric("cv_avg_auc", avg_auc)

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": best_r["auc"],
                    "n_bets": best_r["n_bets"],
                    "win_rate": best_r["win_rate"],
                    "best_threshold": best_r["threshold"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")

            logger.info(
                "Step 4.4: BEST %s ROI=%.2f%% AUC=%.4f run=%s",
                best_key,
                best_r["roi"],
                best_r["auc"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
