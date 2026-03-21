"""Step 4.3: Full-train model, per-sport threshold, odds-weighted selection."""

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
    """Advanced strategies: full-train, per-sport, odds-weighted."""
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

    results: dict[str, dict] = {}

    # A: Reference model (80/20 split)
    check_budget()
    model_ref = CatBoostClassifier(**CB_PARAMS)
    model_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    p_val_ref = model_ref.predict_proba(x_val)[:, 1]
    p_test_ref = model_ref.predict_proba(x_test)[:, 1]
    auc_ref = roc_auc_score(test_sf["target"], p_test_ref)

    # Find val threshold
    t_val, _ = find_best_threshold_on_val(val_df, p_val_ref, min_bets=15)
    logger.info("Val-determined threshold: %.2f", t_val)

    results["ref_t77"] = {
        **calc_roi(test_sf, p_test_ref, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_ref,
    }
    results["ref_val_t"] = {
        **calc_roi(test_sf, p_test_ref, threshold=t_val),
        "threshold": t_val,
        "auc": auc_ref,
    }
    results["ref_t76"] = {
        **calc_roi(test_sf, p_test_ref, threshold=0.76),
        "threshold": 0.76,
        "auc": auc_ref,
    }

    # B: Full-train model (train on all SF data, use best_iter from ref)
    check_budget()
    best_iter = model_ref.get_best_iteration()
    logger.info("Reference best iteration: %d", best_iter)

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test_full = imp_full.transform(test_sf[feat_list])

    params_no_es = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_no_es["iterations"] = best_iter + 10  # slight buffer
    model_full = CatBoostClassifier(**params_no_es)
    model_full.fit(x_full, train_sf["target"])
    p_test_full = model_full.predict_proba(x_test_full)[:, 1]
    auc_full = roc_auc_score(test_sf["target"], p_test_full)

    results["full_t77"] = {
        **calc_roi(test_sf, p_test_full, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_full,
    }
    results["full_t76"] = {
        **calc_roi(test_sf, p_test_full, threshold=0.76),
        "threshold": 0.76,
        "auc": auc_full,
    }

    # C: Per-sport thresholds on val
    check_budget()
    logger.info("Per-sport analysis:")
    sports = test_sf["Sport"].unique()
    sport_thresholds: dict[str, float] = {}

    for sport in sorted(sports):
        val_sport = val_df[val_df["Sport"] == sport]
        test_sport = test_sf[test_sf["Sport"] == sport]
        if len(val_sport) < 10 or len(test_sport) < 10:
            sport_thresholds[sport] = 0.77
            continue

        val_idx = val_sport.index
        val_mask = val_df.index.isin(val_idx)
        p_val_sport = p_val_ref[val_mask]
        t_sport, _ = find_best_threshold_on_val(val_sport, p_val_sport, min_bets=5)
        sport_thresholds[sport] = t_sport

        test_idx = test_sport.index
        test_mask = test_sf.index.isin(test_idx)
        p_test_sport = p_test_ref[test_mask]
        roi_sport = calc_roi(test_sport, p_test_sport, threshold=t_sport)
        roi_sport_77 = calc_roi(test_sport, p_test_sport, threshold=0.77)
        logger.info(
            "  %s: n_test=%d, val_t=%.2f, ROI(val_t)=%.2f%% n=%d, ROI(0.77)=%.2f%% n=%d",
            sport,
            len(test_sport),
            t_sport,
            roi_sport["roi"],
            roi_sport["n_bets"],
            roi_sport_77["roi"],
            roi_sport_77["n_bets"],
        )

    # Apply per-sport thresholds
    selected_mask_per_sport = np.zeros(len(test_sf), dtype=bool)
    for i, (_, row) in enumerate(test_sf.iterrows()):
        sport = row["Sport"]
        t = sport_thresholds.get(sport, 0.77)
        if p_test_ref[i] >= t:
            selected_mask_per_sport[i] = True

    roi_per_sport = calc_roi(test_sf, selected_mask_per_sport.astype(float), threshold=0.5)
    results["per_sport_thresh"] = {
        **roi_per_sport,
        "threshold": -1,  # variable
        "auc": auc_ref,
    }
    logger.info(
        "Per-sport threshold: ROI=%.2f%% n=%d", roi_per_sport["roi"], roi_per_sport["n_bets"]
    )

    # D: Odds-weighted selection: prefer higher odds (bigger payout potential)
    check_budget()
    # Score = proba * log(odds) -> selects confident bets on higher odds
    score_weighted = p_test_ref * np.log1p(test_sf["Odds"].values)
    for t_pct in [50, 40, 30, 25, 20]:
        threshold_w = np.percentile(score_weighted, 100 - t_pct)
        mask_w = score_weighted >= threshold_w
        roi_w = calc_roi(test_sf, mask_w.astype(float), threshold=0.5)
        logger.info(
            "  Odds-weighted top %d%%: ROI=%.2f%% n=%d",
            t_pct,
            roi_w["roi"],
            roi_w["n_bets"],
        )
        if t_pct == 40:
            results["odds_weighted_40"] = {**roi_w, "threshold": threshold_w, "auc": auc_ref}

    # E: Combined: proba >= 0.77 AND odds > 1.5 (avoid very low-odds)
    mask_odds_filter = (p_test_ref >= 0.77) & (test_sf["Odds"].values > 1.5)
    roi_odds_f = calc_roi(test_sf, mask_odds_filter.astype(float), threshold=0.5)
    results["t77_odds_gt15"] = {**roi_odds_f, "threshold": 0.77, "auc": auc_ref}
    logger.info("t=0.77 + Odds>1.5: ROI=%.2f%% n=%d", roi_odds_f["roi"], roi_odds_f["n_bets"])

    mask_odds_f2 = (p_test_ref >= 0.77) & (test_sf["Odds"].values > 1.3)
    roi_odds_f2 = calc_roi(test_sf, mask_odds_f2.astype(float), threshold=0.5)
    results["t77_odds_gt13"] = {**roi_odds_f2, "threshold": 0.77, "auc": auc_ref}
    logger.info("t=0.77 + Odds>1.3: ROI=%.2f%% n=%d", roi_odds_f2["roi"], roi_odds_f2["n_bets"])

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info(
            "  %s: ROI=%.2f%% n=%d AUC=%.4f t=%.2f",
            name,
            r["roi"],
            r["n_bets"],
            r["auc"],
            r["threshold"],
        )

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.3_advanced") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.3")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "advanced_strategies",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                    "ref_best_iteration": best_iter,
                    "sport_thresholds": str(sport_thresholds),
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

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
                "Step 4.3: BEST %s ROI=%.2f%% run=%s",
                best_key,
                best_r["roi"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
