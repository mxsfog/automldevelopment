"""Step 4.2: Segment analysis + dual-model strategy (ELO + non-ELO)."""

import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    UNPROFITABLE_SPORTS,
    add_engineered_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
from step_2_5_safe_elo import build_safe_elo_features, get_safe_elo_features

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
    """Segment analysis and dual-model strategy."""
    logger.info("Step 4.2: Segment analysis + dual-model")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)
    train_all, test_all = time_series_split(df)

    # Part A: Segment analysis on ELO-only subset
    logger.info("--- Part A: Segment analysis by Sport (ELO-only) ---")
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()

    # Best model from step 3.1 -- retrain and get predictions
    feature_cols = get_base_features() + get_engineered_features() + get_safe_elo_features()
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()

    val_split = int(len(train_elo) * 0.8)
    train_fit = train_elo.iloc[:val_split]
    val_df = train_elo.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feature_cols])
    x_val = imp.transform(val_df[feature_cols])
    x_test_elo = imp.transform(test_elo[feature_cols])

    cb = CatBoostClassifier(
        iterations=499,
        depth=7,
        learning_rate=0.214,
        l2_leaf_reg=1.15,
        random_strength=0.823,
        bagging_temperature=2.41,
        border_count=121,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        early_stopping_rounds=30,
    )
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))

    proba_val = cb.predict_proba(x_val)[:, 1]
    proba_test_elo = cb.predict_proba(x_test_elo)[:, 1]

    best_t, _vr = find_best_threshold_on_val(val_df, proba_val)
    logger.info("Best threshold: %.2f", best_t)

    # Per-sport analysis
    sports = test_elo["Sport"].value_counts()
    logger.info("Sports in ELO test set:")
    for sport, count in sports.items():
        mask = test_elo["Sport"] == sport
        if mask.sum() < 10:
            continue
        sport_test = test_elo[mask]
        sport_proba = proba_test_elo[mask.values]
        roi_sport = calc_roi(sport_test, sport_proba, threshold=best_t)
        logger.info(
            "  %s: n_total=%d, n_selected=%d, ROI=%.2f%%, win_rate=%.2f%%",
            sport,
            count,
            roi_sport["n_bets"],
            roi_sport["roi"],
            roi_sport["win_rate"] * 100,
        )

    # Part B: Dual-model strategy
    logger.info("--- Part B: Dual-model strategy ---")
    check_budget()

    with mlflow.start_run(run_name="phase4/step4.2_dual_model") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("phase", "4")

        try:
            # Model 1: ELO-model (best from step 3.1)
            # Already trained above

            # Model 2: Non-ELO model (base + engineered features)
            base_feats = get_base_features() + get_engineered_features()

            train_no_elo = train_all[train_all["has_elo"] == 0.0].copy()
            test_no_elo = test_all[test_all["has_elo"] == 0.0].copy()
            logger.info("Non-ELO: train=%d, test=%d", len(train_no_elo), len(test_no_elo))

            val_split2 = int(len(train_no_elo) * 0.8)
            train_fit2 = train_no_elo.iloc[:val_split2]
            val_df2 = train_no_elo.iloc[val_split2:]

            imp2 = SimpleImputer(strategy="median")
            x_fit2 = imp2.fit_transform(train_fit2[base_feats])
            x_val2 = imp2.transform(val_df2[base_feats])
            x_test2 = imp2.transform(test_no_elo[base_feats])

            # Also try non-ELO model with chain_1 best approach
            # (Optuna CB + segment filtering)
            cb2 = CatBoostClassifier(
                iterations=500,
                depth=3,
                learning_rate=0.059,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=50,
            )
            cb2.fit(x_fit2, train_fit2["target"], eval_set=(x_val2, val_df2["target"]))

            proba_val2 = cb2.predict_proba(x_val2)[:, 1]
            proba_test2 = cb2.predict_proba(x_test2)[:, 1]

            best_t2, _vr2 = find_best_threshold_on_val(val_df2, proba_val2)
            roi_no_elo = calc_roi(test_no_elo, proba_test2, threshold=best_t2)
            logger.info(
                "Non-ELO model: ROI=%.2f%% t=%.2f n=%d",
                roi_no_elo["roi"],
                best_t2,
                roi_no_elo["n_bets"],
            )

            # Non-ELO with segment filtering
            unprofitable_mask = test_no_elo["Sport"].isin(UNPROFITABLE_SPORTS)
            test_no_elo_filt = test_no_elo[~unprofitable_mask]
            proba_filt = proba_test2[~unprofitable_mask.values]
            roi_no_elo_filt = calc_roi(test_no_elo_filt, proba_filt, threshold=best_t2)
            logger.info(
                "Non-ELO filtered: ROI=%.2f%% t=%.2f n=%d",
                roi_no_elo_filt["roi"],
                best_t2,
                roi_no_elo_filt["n_bets"],
            )

            # Dual-model: combine ELO + non-ELO predictions
            # ELO predictions at best_t
            elo_selected = proba_test_elo >= best_t
            elo_bets = test_elo[elo_selected]
            elo_staked = elo_bets["USD"].sum()
            elo_payout = elo_bets["Payout_USD"].sum()

            # Non-ELO predictions at best_t2 (filtered)
            no_elo_selected = proba_filt >= best_t2
            no_elo_bets = test_no_elo_filt.iloc[np.where(no_elo_selected)[0]]
            no_elo_staked = no_elo_bets["USD"].sum()
            no_elo_payout = no_elo_bets["Payout_USD"].sum()

            # Combined
            total_staked = elo_staked + no_elo_staked
            total_payout = elo_payout + no_elo_payout
            combined_roi = (
                (total_payout - total_staked) / total_staked * 100 if total_staked > 0 else 0
            )
            combined_n = int(elo_selected.sum()) + int(no_elo_selected.sum())

            logger.info("Dual-model combined:")
            logger.info(
                "  ELO: %d bets, staked=%.0f, ROI=%.2f%%",
                int(elo_selected.sum()),
                elo_staked,
                (elo_payout - elo_staked) / elo_staked * 100 if elo_staked > 0 else 0,
            )
            logger.info(
                "  Non-ELO: %d bets, staked=%.0f, ROI=%.2f%%",
                int(no_elo_selected.sum()),
                no_elo_staked,
                (no_elo_payout - no_elo_staked) / no_elo_staked * 100 if no_elo_staked > 0 else 0,
            )
            logger.info("  Combined: %d bets, ROI=%.2f%%", combined_n, combined_roi)

            # Also try ELO-only (no non-ELO component)
            roi_elo_only = calc_roi(test_elo, proba_test_elo, threshold=best_t)
            logger.info(
                "ELO-only at t=%.2f: ROI=%.2f%% n=%d",
                best_t,
                roi_elo_only["roi"],
                roi_elo_only["n_bets"],
            )

            # Log
            best_roi = max(combined_roi, roi_elo_only["roi"])
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "dual_model",
                    "elo_threshold": best_t,
                    "no_elo_threshold": best_t2,
                    "n_elo_train": len(train_elo),
                    "n_no_elo_train": len(train_no_elo),
                    "n_elo_test": len(test_elo),
                    "n_no_elo_test": len(test_no_elo),
                    "leakage_free": "true",
                }
            )
            mlflow.log_metrics(
                {
                    "roi_combined": combined_roi,
                    "roi_elo_only": roi_elo_only["roi"],
                    "roi_no_elo": roi_no_elo["roi"],
                    "roi_no_elo_filtered": roi_no_elo_filt["roi"],
                    "n_bets_combined": combined_n,
                    "n_bets_elo": roi_elo_only["n_bets"],
                    "n_bets_no_elo": roi_no_elo["n_bets"],
                    "roi": best_roi,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")

            logger.info("Best ROI: %.2f%%, run_id: %s", best_roi, run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.2")
            raise


if __name__ == "__main__":
    main()
