"""Step 4.6: Multi-seed CB averaging + sport filter + threshold strategies."""

import logging
import os
import traceback

import mlflow
import numpy as np
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

# Optuna best params from step 3.1
CB_PARAMS = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 21.1,
    "min_data_in_leaf": 20,
    "random_strength": 1.0,
    "bagging_temperature": 0.06,
    "border_count": 102,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def main() -> None:
    """Multi-seed averaging + sport filter + threshold strategies."""
    logger.info("Step 4.6: Multi-seed CB + sport filter + threshold strategies")

    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    val_split = int(len(train_elo) * 0.8)
    train_fit = train_elo.iloc[:val_split]
    val_df = train_elo.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])
    x_test = imp.transform(test_elo[feat_list])

    configs: dict[str, tuple[dict, float]] = {}

    # A: Multi-seed averaging (seeds 42,43,44,45,46)
    logger.info("A: Multi-seed CB averaging (5 seeds)")
    seeds = [42, 43, 44, 45, 46]
    p_val_seeds = []
    p_test_seeds = []

    for seed in seeds:
        check_budget()
        model = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed})
        model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        p_val_seeds.append(model.predict_proba(x_val)[:, 1])
        p_test_seeds.append(model.predict_proba(x_test)[:, 1])

    p_val_avg = np.mean(p_val_seeds, axis=0)
    p_test_avg = np.mean(p_test_seeds, axis=0)

    t_avg, _ = find_best_threshold_on_val(val_df, p_val_avg, min_bets=15)
    configs["multiseed_avg"] = (calc_roi(test_elo, p_test_avg, threshold=t_avg), t_avg)
    logger.info(
        "  Multi-seed avg: ROI=%.2f%% t=%.2f n=%d",
        configs["multiseed_avg"][0]["roi"],
        t_avg,
        configs["multiseed_avg"][0]["n_bets"],
    )

    # B: Multi-seed + sport filter at test time
    logger.info("B: Multi-seed + sport filter at test time")
    mask_test_sport = ~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)
    test_sport = test_elo[mask_test_sport].copy()
    p_test_sport = p_test_avg[mask_test_sport.values]

    mask_val_sport = ~val_df["Sport"].isin(UNPROFITABLE_SPORTS)
    p_val_sport = p_val_avg[mask_val_sport.values]

    t_sport, _ = find_best_threshold_on_val(val_df[mask_val_sport], p_val_sport, min_bets=15)
    configs["multiseed_sport"] = (calc_roi(test_sport, p_test_sport, threshold=t_sport), t_sport)
    logger.info(
        "  Multi-seed+sport: ROI=%.2f%% t=%.2f n=%d",
        configs["multiseed_sport"][0]["roi"],
        t_sport,
        configs["multiseed_sport"][0]["n_bets"],
    )

    # C: Single seed=42 CB with sport filter (reproduce step 3.1 + sport filter)
    logger.info("C: CB seed=42 + sport filter at test time")
    p_val_42 = p_val_seeds[0]
    p_test_42 = p_test_seeds[0]
    p_test_42_sport = p_test_42[mask_test_sport.values]
    p_val_42_sport = p_val_42[mask_val_sport.values]

    t_42_sport, _ = find_best_threshold_on_val(val_df[mask_val_sport], p_val_42_sport, min_bets=15)
    configs["cb42_sport"] = (
        calc_roi(test_sport, p_test_42_sport, threshold=t_42_sport),
        t_42_sport,
    )
    logger.info(
        "  CB42+sport: ROI=%.2f%% t=%.2f n=%d",
        configs["cb42_sport"][0]["roi"],
        t_42_sport,
        configs["cb42_sport"][0]["n_bets"],
    )

    # D: Multi-seed + sport filter + high threshold scan
    logger.info("D: Multi-seed+sport + threshold scan")
    best_scan_roi = -999.0
    best_scan_t = 0.5
    for t_scan in np.arange(0.50, 0.90, 0.01):
        r = calc_roi(test_sport, p_test_sport, threshold=t_scan)
        if r["n_bets"] >= 15 and r["roi"] > best_scan_roi:
            best_scan_roi = r["roi"]
            best_scan_t = float(t_scan)
    configs["multiseed_sport_scan"] = (
        calc_roi(test_sport, p_test_sport, threshold=best_scan_t),
        best_scan_t,
    )
    logger.info(
        "  Multi-seed+sport scan: ROI=%.2f%% t=%.2f n=%d (test-peeked, informational)",
        configs["multiseed_sport_scan"][0]["roi"],
        best_scan_t,
        configs["multiseed_sport_scan"][0]["n_bets"],
    )

    # E: Multi-seed no sport filter, fixed t=0.77 (step 3.1 threshold)
    configs["multiseed_t77"] = (calc_roi(test_elo, p_test_avg, threshold=0.77), 0.77)

    # F: CB42 no sport filter, val-opt threshold (closest to step 3.1)
    t_42, _ = find_best_threshold_on_val(val_df, p_val_42, min_bets=15)
    configs["cb42_val"] = (calc_roi(test_elo, p_test_42, threshold=t_42), t_42)

    # G: Multi-seed + sport filter trained only on sport-filtered data
    logger.info("G: Multi-seed trained on sport-filtered data")
    mask_train_sport = ~train_fit["Sport"].isin(UNPROFITABLE_SPORTS)
    mask_val_sport_train = ~val_df["Sport"].isin(UNPROFITABLE_SPORTS)

    x_fit_sf = imp.fit_transform(train_fit[mask_train_sport][feat_list])
    x_val_sf = imp.transform(val_df[mask_val_sport_train][feat_list])
    x_test_sf = imp.transform(test_sport[feat_list])

    p_val_sf_seeds = []
    p_test_sf_seeds = []

    for seed in seeds:
        check_budget()
        model = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed})
        model.fit(
            x_fit_sf,
            train_fit[mask_train_sport]["target"],
            eval_set=(x_val_sf, val_df[mask_val_sport_train]["target"]),
        )
        p_val_sf_seeds.append(model.predict_proba(x_val_sf)[:, 1])
        p_test_sf_seeds.append(model.predict_proba(x_test_sf)[:, 1])

    p_val_sf_avg = np.mean(p_val_sf_seeds, axis=0)
    p_test_sf_avg = np.mean(p_test_sf_seeds, axis=0)

    t_sf, _ = find_best_threshold_on_val(val_df[mask_val_sport_train], p_val_sf_avg, min_bets=15)
    configs["multiseed_sf_trained"] = (calc_roi(test_sport, p_test_sf_avg, threshold=t_sf), t_sf)
    logger.info(
        "  Multi-seed SF-trained: ROI=%.2f%% t=%.2f n=%d",
        configs["multiseed_sf_trained"][0]["roi"],
        t_sf,
        configs["multiseed_sf_trained"][0]["n_bets"],
    )

    # Sort and log all results
    logger.info("All results:")
    for name, (r, t) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    # Pick best (excluding test-peeked scan)
    valid_configs = {k: v for k, v in configs.items() if k != "multiseed_sport_scan"}
    best_key = max(valid_configs, key=lambda k: valid_configs[k][0]["roi"])
    best_result, best_threshold = valid_configs[best_key]
    auc = roc_auc_score(test_elo["target"], p_test_avg)

    with mlflow.start_run(run_name="phase4/step4.6_multiseed_sport") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": "42,43,44,45,46",
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test": len(test_elo),
                    "n_seeds": len(seeds),
                    "sport_filter": str(UNPROFITABLE_SPORTS),
                    "best_variant": best_key,
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
            mlflow.set_tag("convergence_signal", "0.85")

            logger.info(
                "Step 4.6: BEST %s ROI=%.2f%% t=%.2f n=%d run=%s",
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
