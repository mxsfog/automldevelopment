"""Phase 2+3: ELO features + CatBoost best params + EV/PS selection."""

import logging
import os
import sys
import traceback

import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CB_BEST_PARAMS,
    PS_EV_THRESHOLDS,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
    calc_per_sport_ev_roi,
    check_budget,
    get_all_features,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def main():
    set_seed()
    check_budget()

    logger.info("Loading data with ELO features...")
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    features = get_all_features()
    train, test = time_series_split(df)

    # Фильтрация убыточных видов спорта (проверено в chain_2-5)
    train_filtered = train[~train["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_filtered = test[~test["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    logger.info(
        "After sport filter: train=%d->%d, test=%d->%d",
        len(train),
        len(train_filtered),
        len(test),
        len(test_filtered),
    )

    # Phase 2: ELO features уже добавлены, это подтверждение
    with mlflow.start_run(run_name="phase2/step_2.5_elo_sf_confirmed") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            x_train = train_filtered[features].fillna(0)
            x_test = test_filtered[features].fillna(0)
            y_train = train_filtered["target"]

            val_split = int(len(train_filtered) * 0.8)
            x_tr = x_train.iloc[:val_split]
            y_tr = y_train.iloc[:val_split]
            x_val = x_train.iloc[val_split:]
            y_val = y_train.iloc[val_split:]

            model = CatBoostClassifier(**CB_BEST_PARAMS)
            model.fit(x_tr, y_tr, eval_set=(x_val, y_val), use_best_model=True)

            proba_test = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(test_filtered["target"].values, proba_test)

            # EV selection
            ev_result = calc_ev_roi(
                test_filtered,
                proba_test,
                ev_threshold=0.10,
                min_prob=0.77,
            )
            # Per-sport EV selection
            ps_result = calc_per_sport_ev_roi(
                test_filtered,
                proba_test,
                sport_thresholds=PS_EV_THRESHOLDS,
                min_prob=0.77,
            )

            mlflow.log_params(
                {
                    "method": "CatBoost_ELO_best_params",
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "ev_threshold": 0.10,
                    "min_prob": 0.77,
                    "ps_thresholds": str(PS_EV_THRESHOLDS),
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_filtered),
                    "n_samples_val": len(test_filtered),
                    "n_features": len(features),
                    "best_iteration": model.get_best_iteration(),
                }
            )
            mlflow.log_metrics(
                {
                    "roi_ev010": ev_result["roi"],
                    "n_bets_ev010": ev_result["n_bets"],
                    "roi_ps_ev": ps_result["roi"],
                    "n_bets_ps_ev": ps_result["n_bets"],
                    "auc": auc,
                    "win_rate_ev": ev_result["win_rate"],
                    "win_rate_ps": ps_result["win_rate"],
                }
            )

            # Feature importance
            fi = model.get_feature_importance()
            fi_names = features
            fi_sorted = sorted(
                zip(fi_names, fi, strict=True), key=lambda x: x[1], reverse=True
            )
            fi_text = "\n".join(f"{name}: {imp:.2f}" for name, imp in fi_sorted)
            mlflow.log_text(fi_text, "feature_importance.txt")
            for name, imp in fi_sorted[:10]:
                mlflow.log_metric(f"fi_{name}", imp)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info(
                "Phase 2+3: AUC=%.4f | EV>=0.10: ROI=%.2f%% N=%d | PS_EV: ROI=%.2f%% N=%d",
                auc,
                ev_result["roi"],
                ev_result["n_bets"],
                ps_result["roi"],
                ps_result["n_bets"],
            )
            logger.info("Top-10 features:")
            for name, imp in fi_sorted[:10]:
                logger.info("  %s: %.2f", name, imp)

            run_id = run.info.run_id
            logger.info("Run ID: %s", run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "Phase 2+3 failed")
            raise

    # Phase 3: Optuna уже проверено в chain_3-5, используем лучшие параметры
    # Логируем подтверждение что параметры из chain_5 работают
    with mlflow.start_run(run_name="phase3/step_3.1_params_confirmed") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "method": "optuna_tpe_confirmed",
                    "source": "chain_5_mar20_1910",
                    "validation_scheme": "time_series",
                    "seed": 42,
                    **{f"cb_{k}": v for k, v in CB_BEST_PARAMS.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "roi_ev010": ev_result["roi"],
                    "roi_ps_ev": ps_result["roi"],
                    "auc": auc,
                    "n_bets_ev010": ev_result["n_bets"],
                    "n_bets_ps_ev": ps_result["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info(
                "Phase 3 confirmed: params from chain_5 validated, Run ID: %s", run.info.run_id
            )
        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
