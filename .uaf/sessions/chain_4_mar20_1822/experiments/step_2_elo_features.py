"""Phase 2: ELO features + sport filter (proven from chain_2/chain_3)."""

import logging
import os
import traceback

import mlflow
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
    """Phase 2: ELO features on CatBoost, three configs."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)

    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO subset: train=%d, test=%d", len(train_elo), len(test_elo))

    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    logger.info("Sport-filtered: train=%d, test=%d", len(train_sf), len(test_sf))

    feat_base = get_base_features() + get_engineered_features()
    feat_elo = feat_base + get_elo_features()

    configs = {
        "2.5a_baseline_no_elo": {"features": feat_base, "train": train_elo, "test": test_elo},
        "2.5b_elo_all": {"features": feat_elo, "train": train_elo, "test": test_elo},
        "2.5c_elo_sf": {"features": feat_elo, "train": train_sf, "test": test_sf},
    }

    results: dict[str, dict] = {}

    for name, cfg in configs.items():
        check_budget()
        logger.info("Running %s (%d features)", name, len(cfg["features"]))

        tr = cfg["train"]
        te = cfg["test"]
        feat = cfg["features"]

        val_split = int(len(tr) * 0.8)
        train_fit = tr.iloc[:val_split]
        val_df = tr.iloc[val_split:]

        imp = SimpleImputer(strategy="median")
        x_fit = imp.fit_transform(train_fit[feat])
        x_val = imp.transform(val_df[feat])
        x_test = imp.transform(te[feat])

        cb = CatBoostClassifier(
            iterations=1000,
            random_seed=42,
            verbose=0,
            eval_metric="AUC",
            early_stopping_rounds=50,
        )
        cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

        p_val = cb.predict_proba(x_val)[:, 1]
        p_test = cb.predict_proba(x_test)[:, 1]

        best_t, _ = find_best_threshold_on_val(val_df, p_val, min_bets=15)
        roi_val = calc_roi(te, p_test, threshold=best_t)
        roi_t77 = calc_roi(te, p_test, threshold=0.77)
        auc = roc_auc_score(te["target"], p_test)

        results[name] = {
            "roi_val": roi_val["roi"],
            "roi_t77": roi_t77["roi"],
            "auc": auc,
            "threshold": best_t,
            "n_bets_val": roi_val["n_bets"],
            "n_bets_t77": roi_t77["n_bets"],
        }
        logger.info(
            "  %s: ROI(val_t)=%.2f%% t=%.2f n=%d | ROI(t=0.77)=%.2f%% n=%d | AUC=%.4f",
            name,
            roi_val["roi"],
            best_t,
            roi_val["n_bets"],
            roi_t77["roi"],
            roi_t77["n_bets"],
            auc,
        )

    # Log best config
    best_name = max(results, key=lambda k: results[k]["roi_t77"])
    best = results[best_name]

    with mlflow.start_run(run_name="phase2/step2.5_elo_features") as run:
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
                    "best_config": best_name,
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "n_features_base": len(feat_base),
                    "n_features_elo": len(feat_elo),
                    "n_elo_train": len(train_elo),
                    "n_elo_test": len(test_elo),
                    "n_sf_train": len(train_sf),
                    "n_sf_test": len(test_sf),
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_val_{name}", r["roi_val"])
                mlflow.log_metric(f"roi_t77_{name}", r["roi_t77"])
                mlflow.log_metric(f"auc_{name}", r["auc"])

            mlflow.log_metrics(
                {
                    "roi": best["roi_t77"],
                    "roc_auc": best["auc"],
                    "n_bets": best["n_bets_t77"],
                    "best_threshold": 0.77,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info(
                "Phase 2 best: %s ROI=%.2f%% AUC=%.4f run=%s",
                best_name,
                best["roi_t77"],
                best["auc"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
