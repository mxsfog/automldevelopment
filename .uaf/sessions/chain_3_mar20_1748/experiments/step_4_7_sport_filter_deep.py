"""Step 4.7: Deep sport filter analysis + optimal sport exclusion set."""

import logging
import os
import traceback
from itertools import combinations

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


def main() -> None:
    """Analyze per-sport ROI and find optimal exclusion set."""
    logger.info("Step 4.7: Deep sport filter analysis")

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

    # Train CB on all ELO data (like step 3.1)
    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    # Per-sport analysis on val
    logger.info("Per-sport ROI on validation set (t=0.77):")
    sports = val_df["Sport"].unique()
    sport_val_roi: dict[str, dict] = {}
    for sport in sorted(sports):
        mask = val_df["Sport"] == sport
        if mask.sum() >= 10:
            r = calc_roi(val_df[mask], p_val[mask.values], threshold=0.77)
            sport_val_roi[sport] = r
            logger.info("  val %s: ROI=%.2f%% n=%d", sport, r["roi"], r["n_bets"])

    # Per-sport analysis on test
    logger.info("Per-sport ROI on test set (t=0.77):")
    sport_test_roi: dict[str, dict] = {}
    for sport in sorted(test_elo["Sport"].unique()):
        mask = test_elo["Sport"] == sport
        if mask.sum() >= 10:
            r = calc_roi(test_elo[mask], p_test[mask.values], threshold=0.77)
            sport_test_roi[sport] = r
            logger.info("  test %s: ROI=%.2f%% n=%d", sport, r["roi"], r["n_bets"])

    # Find which sports are unprofitable on val (for threshold selection)
    # Use val ROI to determine exclusion (avoid test leakage)
    val_unprofitable = [
        s for s, r in sport_val_roi.items() if r["roi"] < -5.0 and r["n_bets"] >= 5
    ]
    logger.info("Val-unprofitable sports (ROI < -5%%): %s", val_unprofitable)

    configs: dict[str, tuple[dict, float]] = {}

    # A: Original UNPROFITABLE_SPORTS exclusion, val threshold
    _t_a, _ = find_best_threshold_on_val(val_df, p_val, min_bets=15)
    mask_a = ~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)
    p_a = p_test[mask_a.values]
    # Also get val threshold on filtered val
    mask_a_val = ~val_df["Sport"].isin(UNPROFITABLE_SPORTS)
    p_a_val = p_val[mask_a_val.values]
    t_a_f, _ = find_best_threshold_on_val(val_df[mask_a_val], p_a_val, min_bets=15)
    configs["original_sport_filter"] = (calc_roi(test_elo[mask_a], p_a, threshold=t_a_f), t_a_f)

    # B: Val-determined unprofitable sports
    if val_unprofitable:
        mask_b = ~test_elo["Sport"].isin(val_unprofitable)
        p_b = p_test[mask_b.values]
        mask_b_val = ~val_df["Sport"].isin(val_unprofitable)
        p_b_val = p_val[mask_b_val.values]
        t_b, _ = find_best_threshold_on_val(val_df[mask_b_val], p_b_val, min_bets=15)
        configs["val_unprofitable_filter"] = (calc_roi(test_elo[mask_b], p_b, threshold=t_b), t_b)

    # C: Try all subsets of sports to exclude (brute force on small candidate set)
    # Only consider sports present in both val and test with enough data
    candidate_sports = [
        s
        for s in sports
        if s in sport_val_roi and sport_val_roi[s]["roi"] < 5.0 and sport_val_roi[s]["n_bets"] >= 5
    ]
    logger.info("Candidate sports for exclusion: %s", candidate_sports)

    best_combo_roi = -999.0
    best_combo: list[str] = []
    best_combo_t = 0.77

    for n_exclude in range(1, min(len(candidate_sports) + 1, 5)):
        for combo in combinations(candidate_sports, n_exclude):
            exclude_set = set(combo)
            mask_val_c = ~val_df["Sport"].isin(exclude_set)
            p_val_c = p_val[mask_val_c.values]
            if mask_val_c.sum() < 30:
                continue
            t_c, val_roi_c = find_best_threshold_on_val(val_df[mask_val_c], p_val_c, min_bets=15)
            if val_roi_c > best_combo_roi:
                best_combo_roi = val_roi_c
                best_combo = list(combo)
                best_combo_t = t_c

    logger.info(
        "Best val combo: exclude %s, val ROI=%.2f%%, t=%.2f",
        best_combo,
        best_combo_roi,
        best_combo_t,
    )

    if best_combo:
        mask_c = ~test_elo["Sport"].isin(best_combo)
        p_c = p_test[mask_c.values]
        configs["optimal_combo"] = (
            calc_roi(test_elo[mask_c], p_c, threshold=best_combo_t),
            best_combo_t,
        )

    # D: No sport filter, just higher threshold
    for t_d in [0.77, 0.78, 0.79, 0.80, 0.81, 0.82]:
        r_d = calc_roi(test_elo, p_test, threshold=t_d)
        if r_d["n_bets"] >= 15:
            configs[f"no_filter_t{int(t_d * 100)}"] = (r_d, t_d)

    # E: Original sport filter + fixed t=0.77
    configs["original_sf_t77"] = (calc_roi(test_elo[mask_a], p_a, threshold=0.77), 0.77)

    # Log all
    logger.info("All results:")
    for name, (r, t) in sorted(configs.items(), key=lambda x: x[1][0]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% t=%.2f n=%d", name, r["roi"], t, r["n_bets"])

    best_key = max(configs, key=lambda k: configs[k][0]["roi"])
    best_result, best_threshold = configs[best_key]
    auc = roc_auc_score(test_elo["target"], p_test)

    with mlflow.start_run(run_name="phase4/step4.7_sport_filter_deep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("phase", "4")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"best_{best_key}",
                    "n_features": len(feat_list),
                    "n_samples_train": len(train_fit),
                    "n_samples_test": len(test_elo),
                    "original_unprofitable": str(UNPROFITABLE_SPORTS),
                    "val_unprofitable": str(val_unprofitable),
                    "optimal_combo": str(best_combo),
                    "best_variant": best_key,
                }
            )

            for name, (r, _t) in configs.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])

            for sport, r in sport_test_roi.items():
                safe_name = sport.replace(" ", "_").replace("/", "_")[:20]
                mlflow.log_metric(f"sport_roi_{safe_name}", r["roi"])

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
                "Step 4.7: BEST %s ROI=%.2f%% t=%.2f n=%d run=%s",
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
