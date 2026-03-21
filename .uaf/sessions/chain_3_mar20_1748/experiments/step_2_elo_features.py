"""Phase 2: ELO feature engineering (steps 2.5a, 2.5b, 2.5c)."""

import logging
import os
import traceback

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    add_elo_features,
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
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


def run_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feat_list: list[str],
    run_name: str,
    step_id: str,
    method: str,
) -> dict:
    """Единый runner для CatBoost эксперимента."""
    check_budget()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", step_id)
        mlflow.set_tag("phase", "2")

        try:
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split]
            val_df = train.iloc[val_split:]

            imp = SimpleImputer(strategy="median")
            x_fit = imp.fit_transform(train_fit[feat_list])
            x_val = imp.transform(val_df[feat_list])
            x_test = imp.transform(test[feat_list])

            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

            proba_val = model.predict_proba(x_val)[:, 1]
            proba_test = model.predict_proba(x_test)[:, 1]

            best_t, _val_roi = find_best_threshold_on_val(val_df, proba_val)
            roi_result = calc_roi(test, proba_test, threshold=best_t)
            auc = roc_auc_score(test["target"], proba_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": method,
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test),
                    "best_iteration": model.best_iteration_,
                    "leakage_free": "true",
                }
            )

            roi_thresholds = calc_roi_at_thresholds(test, proba_test)
            for t, r in roi_thresholds.items():
                mlflow.log_metric(f"roi_t{int(t * 100):03d}", r["roi"])

            fi = dict(zip(feat_list, model.feature_importances_, strict=False))
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted[:15]:
                logger.info("  FI: %s = %.3f", fname, fval)
                mlflow.log_metric(f"fi_{fname}", fval)

            mlflow.log_metrics(
                {
                    "roi": roi_result["roi"],
                    "roc_auc": auc,
                    "n_bets": roi_result["n_bets"],
                    "win_rate": roi_result["win_rate"],
                    "best_threshold": best_t,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            logger.info(
                "%s: ROI=%.2f%% AUC=%.4f t=%.2f n=%d iter=%d run=%s",
                method,
                roi_result["roi"],
                auc,
                best_t,
                roi_result["n_bets"],
                model.best_iteration_,
                run.info.run_id,
            )
            return {
                "run_id": run.info.run_id,
                "roi": roi_result["roi"],
                "auc": auc,
                "threshold": best_t,
                "n_bets": roi_result["n_bets"],
            }

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


def main() -> None:
    """Phase 2: ELO feature engineering."""
    logger.info("Phase 2: ELO Feature Engineering")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train, test = time_series_split(df)

    base_feats = get_base_features() + get_engineered_features()
    elo_feats = get_elo_features()
    all_feats = base_feats + elo_feats

    n_elo_train = int((train["has_elo"] == 1.0).sum())
    n_elo_test = int((test["has_elo"] == 1.0).sum())
    logger.info(
        "ELO coverage: train=%d/%.0f, test=%d/%.0f", n_elo_train, len(train), n_elo_test, len(test)
    )

    # Step 2.5a: baseline (no ELO) on all data
    res_a = run_experiment(
        train,
        test,
        base_feats,
        "phase2/step2.5a_baseline",
        "2.5a",
        "catboost_baseline_no_elo",
    )

    # Step 2.5b: with ELO on all data
    res_b = run_experiment(
        train,
        test,
        all_feats,
        "phase2/step2.5b_with_elo",
        "2.5b",
        "catboost_with_safe_elo",
    )

    # Step 2.5c: ELO-only subset
    train_elo = train[train["has_elo"] == 1.0].copy()
    test_elo = test[test["has_elo"] == 1.0].copy()
    logger.info("ELO-only subset: train=%d, test=%d", len(train_elo), len(test_elo))

    res_c = run_experiment(
        train_elo,
        test_elo,
        all_feats,
        "phase2/step2.5c_elo_only",
        "2.5c",
        "catboost_elo_only_safe",
    )

    logger.info("Phase 2 Summary:")
    logger.info("  2.5a baseline: ROI=%.2f%%", res_a["roi"])
    logger.info("  2.5b +ELO all: ROI=%.2f%%", res_b["roi"])
    logger.info("  2.5c ELO-only: ROI=%.2f%%", res_c["roi"])

    delta_elo = res_b["roi"] - res_a["roi"]
    delta_subset = res_c["roi"] - res_b["roi"]
    logger.info("  Delta ELO: %.2f p.p.", delta_elo)
    logger.info("  Delta subset: %.2f p.p.", delta_subset)


if __name__ == "__main__":
    main()
