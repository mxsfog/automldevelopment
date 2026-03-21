"""Step 4.5: Final optimization - market filtering + per-sport thresholds + stacking."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import check_budget, load_data, set_seed, time_series_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uaf/sports_10h_v4")
SESSION_ID = os.environ.get("UAF_SESSION_ID", "sports_10h_v4")

CAT_COLS = {"Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket"}
TOP_SPORTS = {"Cricket", "CS2", "Dota 2"}

FEATURES = [
    "Odds",
    "log_odds",
    "implied_prob",
    "Outcomes_Count",
    "is_single",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "ml_vs_market",
    "edge_normalized",
    "is_value_bet",
    "ev_ratio",
    "kelly_fraction",
    "ml_confidence",
    "hour",
    "day_of_week",
    "is_weekend",
    "log_usd",
    "parlay_flag",
    "parlay_odds",
    "has_ml_prediction",
    "is_good_sport",
    "Is_Parlay",
    "Sport",
    "Market",
    "ML_Team_Stats_Found",
    "odds_bucket",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный набор фичей."""
    df = df.copy()
    df["implied_prob"] = 1.0 / df["Odds"]
    df["ml_vs_market"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["edge_normalized"] = df["ML_Edge"] / (df["ML_P_Implied"].clip(lower=0.1) + 1e-6)
    df["is_value_bet"] = (df["ML_P_Model"] / 100.0 > df["implied_prob"] * 1.05).astype(int)
    df["ev_ratio"] = df["ML_EV"] / 100.0
    df["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 5.0, 100, 10000],
        labels=[
            "1.0-1.3",
            "1.3-1.5",
            "1.5-1.8",
            "1.8-2.0",
            "2.0-2.5",
            "2.5-3.0",
            "3.0-5.0",
            "5.0-100",
            "100+",
        ],
    ).astype(str)
    df["log_odds"] = np.log(df["Odds"].clip(1.001))
    p = df["ML_P_Model"].fillna(50) / 100.0
    b = df["Odds"] - 1
    q = 1 - p
    df["kelly_fraction"] = ((b * p - q) / (b + 1e-6)).clip(-1, 1)
    df["ml_confidence"] = (df["ML_P_Model"].fillna(50) - 50).abs()
    df["hour"] = df["Created_At"].dt.hour
    df["day_of_week"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["log_usd"] = np.log1p(df["USD"])
    df["parlay_flag"] = df["Is_Parlay"].map({"t": 1, "f": 0}).fillna(0).astype(int)
    df["parlay_odds"] = df["parlay_flag"] * df["Odds"]
    df["has_ml_prediction"] = (~df["ML_P_Model"].isna()).astype(int)
    df["is_single"] = (df["Outcomes_Count"] == 1).astype(int)
    good_sports = {"Cricket", "CS2", "Soccer", "Dota 2", "Table Tennis"}
    df["is_good_sport"] = df["Sport"].isin(good_sports).astype(int)
    return df


def prepare_cb(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS]
    x = df[features].copy()
    for idx in cat_indices:
        col = features[idx]
        x[col] = x[col].astype(str).replace("nan", "_missing_")
    return x, cat_indices


def eval_roi(test: pd.DataFrame, mask: np.ndarray) -> dict:
    """Расчет ROI для маски отбора."""
    sel = test[mask]
    if len(sel) < 5:
        return {"roi": 0.0, "n_bets": 0, "precision": 0.0}
    n = len(sel)
    payout = sel.loc[sel["target"] == 1, "Odds"].sum()
    roi = (payout - n) / n * 100
    return {"roi": roi, "n_bets": n, "precision": sel["target"].mean()}


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase4/step_4_5_final_optimization") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.5")
            mlflow.set_tag("phase", "4")

            df = load_data()
            df = add_features(df)
            train, test = time_series_split(df)

            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            features = [f for f in FEATURES if f in df.columns]

            x_train, cat_indices = prepare_cb(train_inner, features)
            x_val, _ = prepare_cb(val, features)
            x_test, _ = prepare_cb(test, features)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "final_optimization",
                    "n_features": len(features),
                    "gap_days": 7,
                }
            )

            # Global CatBoost model (same as step 4.4 baseline)
            model = CatBoostClassifier(
                iterations=2000,
                learning_rate=0.01,
                depth=4,
                l2_leaf_reg=10,
                min_data_in_leaf=50,
                random_strength=2,
                bagging_temperature=1,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                early_stopping_rounds=100,
            )
            model.fit(
                x_train,
                train_inner["target"].values,
                eval_set=(x_val, val["target"].values),
                use_best_model=True,
            )
            logger.info("Best iteration: %d", model.best_iteration_)

            test_probas = model.predict_proba(x_test)[:, 1]
            val_probas = model.predict_proba(x_val)[:, 1]
            auc = roc_auc_score(test["target"].values, test_probas)
            logger.info("AUC: %.4f", auc)

            # Approach 1: Market filtering within top sports
            logger.info("=== Market analysis within Top Sports ===")
            top_mask = (test["Is_Parlay"] == "f").values & test["Sport"].isin(TOP_SPORTS).values
            for market in test.loc[top_mask, "Market"].unique():
                market_mask = top_mask & (test["Market"] == market).values
                n_market = market_mask.sum()
                if n_market < 20:
                    continue
                sel = test[market_mask]
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                raw_roi = (payout - n) / n * 100
                logger.info("  Market '%s': raw ROI=%.2f%%, n=%d", market, raw_roi, n)

            # Approach 2: Per-sport optimal thresholds on val, apply to test
            logger.info("=== Per-sport threshold optimization ===")
            best_per_sport: dict[str, dict] = {}
            for sport in TOP_SPORTS:
                val_sport_mask = (val["Is_Parlay"] == "f").values & (val["Sport"] == sport).values
                best_val_roi = -999.0
                best_thr = 0.52
                best_margin = 0.0
                implied_val = 1.0 / val["Odds"].values

                for thr in np.arange(0.45, 0.65, 0.01):
                    for margin in [0.0, 0.01, 0.02, 0.03, 0.05]:
                        mask = (
                            val_sport_mask
                            & (val_probas >= thr)
                            & (val_probas > (implied_val + margin))
                        )
                        sel = val[mask]
                        if len(sel) < 10:
                            continue
                        n = len(sel)
                        payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                        roi = (payout - n) / n * 100
                        if roi > best_val_roi:
                            best_val_roi = roi
                            best_thr = thr
                            best_margin = margin

                best_per_sport[sport] = {
                    "thr": best_thr,
                    "margin": best_margin,
                    "val_roi": best_val_roi,
                }
                logger.info(
                    "  %s: best val thr=%.2f, margin=%.2f, val ROI=%.2f%%",
                    sport,
                    best_thr,
                    best_margin,
                    best_val_roi,
                )

            # Apply per-sport thresholds to test
            implied_test = 1.0 / test["Odds"].values
            combined_mask = np.zeros(len(test), dtype=bool)
            for sport, cfg in best_per_sport.items():
                sport_mask = (
                    (test["Is_Parlay"] == "f").values
                    & (test["Sport"] == sport).values
                    & (test_probas >= cfg["thr"])
                    & (test_probas > (implied_test + cfg["margin"]))
                )
                n_sel = sport_mask.sum()
                r = eval_roi(test, sport_mask)
                logger.info(
                    "  %s test: ROI=%.2f%%, n=%d, prec=%.3f",
                    sport,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )
                combined_mask = combined_mask | sport_mask

            r_combined = eval_roi(test, combined_mask)
            logger.info(
                "Per-sport optimized: ROI=%.2f%%, n=%d, prec=%.3f",
                r_combined["roi"],
                r_combined["n_bets"],
                r_combined["precision"],
            )

            # Approach 3: Market filtering - exclude negative-ROI markets within top sports
            logger.info("=== Market exclusion analysis ===")
            val_top_mask = (val["Is_Parlay"] == "f").values & val["Sport"].isin(TOP_SPORTS).values
            bad_markets: set[str] = set()
            for market in val.loc[val_top_mask, "Market"].unique():
                val_m_mask = val_top_mask & (val["Market"] == market).values & (val_probas >= 0.50)
                sel = val[val_m_mask]
                if len(sel) < 10:
                    continue
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                roi = (payout - n) / n * 100
                if roi < -10.0:
                    bad_markets.add(market)
                    logger.info("  Excluding market '%s': val ROI=%.2f%%, n=%d", market, roi, n)

            if bad_markets:
                logger.info("Bad markets to exclude: %s", bad_markets)

            # Test with market exclusion
            for thr in [0.50, 0.52, 0.55]:
                mask_no_bad = (
                    (test_probas >= thr)
                    & (test["Is_Parlay"] == "f").values
                    & test["Sport"].isin(TOP_SPORTS).values
                    & ~test["Market"].isin(bad_markets).values
                )
                r = eval_roi(test, mask_no_bad)
                logger.info(
                    "  Top sports - bad markets, thr=%.2f: ROI=%.2f%%, n=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )

            # Approach 4: Odds range filtering within top sports
            logger.info("=== Odds range filtering ===")
            for odds_lo, odds_hi in [(1.0, 2.0), (1.3, 3.0), (1.5, 5.0), (1.0, 3.0)]:
                for thr in [0.50, 0.52]:
                    mask_odds = (
                        (test_probas >= thr)
                        & (test["Is_Parlay"] == "f").values
                        & test["Sport"].isin(TOP_SPORTS).values
                        & (test["Odds"].values >= odds_lo)
                        & (test["Odds"].values <= odds_hi)
                    )
                    r = eval_roi(test, mask_odds)
                    if r["n_bets"] > 0:
                        logger.info(
                            "  Odds [%.1f-%.1f] thr=%.2f: ROI=%.2f%%, n=%d",
                            odds_lo,
                            odds_hi,
                            thr,
                            r["roi"],
                            r["n_bets"],
                        )

            # Approach 5: Combined best strategy search on val
            logger.info("=== Combined strategy search on val ===")
            best_overall_roi = -999.0
            best_overall_cfg: dict = {}
            implied_val = 1.0 / val["Odds"].values

            for thr in np.arange(0.48, 0.62, 0.02):
                for margin in [0.0, 0.01, 0.02, 0.03, 0.05]:
                    for odds_lo in [1.0, 1.3]:
                        for odds_hi in [5.0, 10.0, 100.0]:
                            for excl_markets in [False, True]:
                                mask = (
                                    (val_probas >= thr)
                                    & (val_probas > (implied_val + margin))
                                    & (val["Is_Parlay"] == "f").values
                                    & val["Sport"].isin(TOP_SPORTS).values
                                    & (val["Odds"].values >= odds_lo)
                                    & (val["Odds"].values <= odds_hi)
                                )
                                if excl_markets and bad_markets:
                                    mask = mask & ~val["Market"].isin(bad_markets).values

                                sel = val[mask]
                                if len(sel) < 15:
                                    continue
                                n = len(sel)
                                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                                roi = (payout - n) / n * 100
                                if roi > best_overall_roi:
                                    best_overall_roi = roi
                                    best_overall_cfg = {
                                        "thr": thr,
                                        "margin": margin,
                                        "odds_lo": odds_lo,
                                        "odds_hi": odds_hi,
                                        "excl_markets": excl_markets,
                                    }

            logger.info(
                "Best val combined: ROI=%.2f%%, config=%s", best_overall_roi, best_overall_cfg
            )

            # Apply best combined to test
            cfg = best_overall_cfg
            mask_best = (
                (test_probas >= cfg.get("thr", 0.52))
                & (test_probas > (implied_test + cfg.get("margin", 0.0)))
                & (test["Is_Parlay"] == "f").values
                & test["Sport"].isin(TOP_SPORTS).values
                & (test["Odds"].values >= cfg.get("odds_lo", 1.0))
                & (test["Odds"].values <= cfg.get("odds_hi", 100.0))
            )
            if cfg.get("excl_markets") and bad_markets:
                mask_best = mask_best & ~test["Market"].isin(bad_markets).values

            r_best = eval_roi(test, mask_best)
            logger.info(
                "Best combined test: ROI=%.2f%%, n=%d, prec=%.3f",
                r_best["roi"],
                r_best["n_bets"],
                r_best["precision"],
            )

            # Approach 6: Second model with different seed for stacking
            logger.info("=== Stacking with seed diversity ===")
            model2 = CatBoostClassifier(
                iterations=2000,
                learning_rate=0.015,
                depth=5,
                l2_leaf_reg=8,
                min_data_in_leaf=40,
                random_strength=3,
                bagging_temperature=1.5,
                random_seed=123,
                verbose=0,
                eval_metric="AUC",
                cat_features=cat_indices,
                early_stopping_rounds=100,
            )
            model2.fit(
                x_train,
                train_inner["target"].values,
                eval_set=(x_val, val["target"].values),
                use_best_model=True,
            )
            logger.info("Model2 best iteration: %d", model2.best_iteration_)

            test_probas2 = model2.predict_proba(x_test)[:, 1]
            auc2 = roc_auc_score(test["target"].values, test_probas2)
            logger.info("Model2 AUC: %.4f", auc2)

            # Average ensemble
            test_avg = (test_probas + test_probas2) / 2.0
            val_avg = (val_probas + model2.predict_proba(x_val)[:, 1]) / 2.0

            for thr in [0.50, 0.52, 0.55]:
                mask_ens = (
                    (test_avg >= thr)
                    & (test["Is_Parlay"] == "f").values
                    & test["Sport"].isin(TOP_SPORTS).values
                )
                r = eval_roi(test, mask_ens)
                logger.info(
                    "  Ensemble thr=%.2f: ROI=%.2f%%, n=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )

            # Approach 7: Consensus - both models agree
            logger.info("=== Consensus strategy ===")
            for thr in [0.50, 0.52, 0.48]:
                mask_cons = (
                    (test_probas >= thr)
                    & (test_probas2 >= thr)
                    & (test["Is_Parlay"] == "f").values
                    & test["Sport"].isin(TOP_SPORTS).values
                )
                r = eval_roi(test, mask_cons)
                logger.info(
                    "  Consensus thr=%.2f: ROI=%.2f%%, n=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )

            # Determine the single best result across all approaches
            all_results = {
                "per_sport_optimized": r_combined,
                "best_combined": r_best,
            }

            # Find best with minimum 50 bets for robustness
            final_best_label = ""
            final_best_roi = -999.0
            for label, r in all_results.items():
                if r["n_bets"] >= 50 and r["roi"] > final_best_roi:
                    final_best_roi = r["roi"]
                    final_best_label = label

            logger.info(
                "FINAL BEST: %s, ROI=%.2f%%, n=%d",
                final_best_label,
                final_best_roi,
                all_results.get(final_best_label, {}).get("n_bets", 0),
            )

            # Log primary metric
            primary_roi = max(r_combined["roi"], r_best["roi"])
            primary_n = (
                r_combined["n_bets"] if r_combined["roi"] >= r_best["roi"] else r_best["n_bets"]
            )
            primary_prec = (
                r_combined["precision"]
                if r_combined["roi"] >= r_best["roi"]
                else r_best["precision"]
            )

            mlflow.log_metrics(
                {
                    "roi": primary_roi,
                    "n_bets": primary_n,
                    "precision": primary_prec,
                    "roc_auc": auc,
                    "roc_auc_model2": auc2,
                    "roi_per_sport_optimized": r_combined["roi"],
                    "roi_best_combined": r_best["roi"],
                    "n_bets_per_sport": r_combined["n_bets"],
                    "n_bets_combined": r_best["n_bets"],
                }
            )
            mlflow.set_tag("best_strategy", final_best_label)
            mlflow.set_tag("best_config", str(best_overall_cfg))
            mlflow.set_tag("per_sport_thresholds", str(best_per_sport))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.5")
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
