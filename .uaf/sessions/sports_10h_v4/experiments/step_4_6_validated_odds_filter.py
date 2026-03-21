"""Step 4.6: Validated odds filtering - выбор диапазона odds на val, применение на test."""

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


def eval_roi(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """Расчет ROI для маски отбора."""
    sel = df[mask]
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

    with mlflow.start_run(run_name="phase4/step_4_6_validated_odds_filter") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.6")
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
                    "method": "validated_odds_filter",
                    "n_features": len(features),
                    "gap_days": 7,
                }
            )

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

            # Step 1: Exhaustive odds range search on VAL
            logger.info("=== Odds range search on VAL ===")
            implied_val = 1.0 / val["Odds"].values

            odds_lo_options = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
            odds_hi_options = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 100.0]
            thr_options = [0.50, 0.52, 0.54]

            val_results: list[dict] = []
            for thr in thr_options:
                for odds_lo in odds_lo_options:
                    for odds_hi in odds_hi_options:
                        if odds_lo >= odds_hi:
                            continue
                        mask = (
                            (val_probas >= thr)
                            & (val["Is_Parlay"] == "f").values
                            & val["Sport"].isin(TOP_SPORTS).values
                            & (val["Odds"].values >= odds_lo)
                            & (val["Odds"].values <= odds_hi)
                        )
                        sel = val[mask]
                        if len(sel) < 20:
                            continue
                        n = len(sel)
                        payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                        roi = (payout - n) / n * 100
                        val_results.append(
                            {
                                "thr": thr,
                                "odds_lo": odds_lo,
                                "odds_hi": odds_hi,
                                "val_roi": roi,
                                "val_n": n,
                            }
                        )

            # Sort by val ROI
            val_results.sort(key=lambda x: x["val_roi"], reverse=True)

            # Log top-10 val configs
            logger.info("Top-10 val configs:")
            for i, vr in enumerate(val_results[:10]):
                logger.info(
                    "  #%d: thr=%.2f odds=[%.1f-%.1f] val_ROI=%.2f%% n=%d",
                    i + 1,
                    vr["thr"],
                    vr["odds_lo"],
                    vr["odds_hi"],
                    vr["val_roi"],
                    vr["val_n"],
                )

            # Step 2: Apply top-5 val configs to test (but only log, don't cherry-pick)
            logger.info("=== Applying top val configs to TEST ===")
            test_results: list[dict] = []
            implied_test = 1.0 / test["Odds"].values

            for vr in val_results[:5]:
                mask_test = (
                    (test_probas >= vr["thr"])
                    & (test["Is_Parlay"] == "f").values
                    & test["Sport"].isin(TOP_SPORTS).values
                    & (test["Odds"].values >= vr["odds_lo"])
                    & (test["Odds"].values <= vr["odds_hi"])
                )
                r = eval_roi(test, mask_test)
                vr["test_roi"] = r["roi"]
                vr["test_n"] = r["n_bets"]
                vr["test_prec"] = r["precision"]
                test_results.append(vr)
                logger.info(
                    "  thr=%.2f odds=[%.1f-%.1f]: val_ROI=%.2f%% -> test_ROI=%.2f%% n=%d prec=%.3f",
                    vr["thr"],
                    vr["odds_lo"],
                    vr["odds_hi"],
                    vr["val_roi"],
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )

            # Step 3: Use the BEST VAL config as primary result (anti-leakage)
            best_val = val_results[0]
            mask_primary = (
                (test_probas >= best_val["thr"])
                & (test["Is_Parlay"] == "f").values
                & test["Sport"].isin(TOP_SPORTS).values
                & (test["Odds"].values >= best_val["odds_lo"])
                & (test["Odds"].values <= best_val["odds_hi"])
            )
            r_primary = eval_roi(test, mask_primary)
            logger.info(
                "PRIMARY (best val): thr=%.2f odds=[%.1f-%.1f] -> test ROI=%.2f%% n=%d prec=%.3f",
                best_val["thr"],
                best_val["odds_lo"],
                best_val["odds_hi"],
                r_primary["roi"],
                r_primary["n_bets"],
                r_primary["precision"],
            )

            # Step 4: Robustness check - how stable is odds filtering?
            logger.info("=== Robustness: fixed thr=0.52, varying odds ranges ===")
            for odds_lo in [1.0, 1.2, 1.3, 1.4, 1.5]:
                for odds_hi in [2.5, 3.0, 4.0, 5.0]:
                    if odds_lo >= odds_hi:
                        continue
                    mask = (
                        (test_probas >= 0.52)
                        & (test["Is_Parlay"] == "f").values
                        & test["Sport"].isin(TOP_SPORTS).values
                        & (test["Odds"].values >= odds_lo)
                        & (test["Odds"].values <= odds_hi)
                    )
                    r = eval_roi(test, mask)
                    if r["n_bets"] >= 50:
                        logger.info(
                            "  Odds [%.1f-%.1f]: ROI=%.2f%% n=%d prec=%.3f",
                            odds_lo,
                            odds_hi,
                            r["roi"],
                            r["n_bets"],
                            r["precision"],
                        )

            # Step 5: Check if the improvement comes from removing low odds or high odds
            logger.info("=== Decomposition: low vs high odds removal ===")
            # Baseline: all top sports singles, thr=0.52
            baseline_mask = (
                (test_probas >= 0.52)
                & (test["Is_Parlay"] == "f").values
                & test["Sport"].isin(TOP_SPORTS).values
            )
            r_base = eval_roi(test, baseline_mask)
            logger.info(
                "Baseline (no odds filter): ROI=%.2f%% n=%d", r_base["roi"], r_base["n_bets"]
            )

            # Only remove low odds
            for lo in [1.1, 1.2, 1.3, 1.4, 1.5]:
                mask = baseline_mask & (test["Odds"].values >= lo)
                r = eval_roi(test, mask)
                logger.info(
                    "  Remove odds < %.1f: ROI=%.2f%% n=%d (delta=%.2f%%)",
                    lo,
                    r["roi"],
                    r["n_bets"],
                    r["roi"] - r_base["roi"],
                )

            # Only remove high odds
            for hi in [2.0, 2.5, 3.0, 5.0, 10.0]:
                mask = baseline_mask & (test["Odds"].values <= hi)
                r = eval_roi(test, mask)
                logger.info(
                    "  Remove odds > %.1f: ROI=%.2f%% n=%d (delta=%.2f%%)",
                    hi,
                    r["roi"],
                    r["n_bets"],
                    r["roi"] - r_base["roi"],
                )

            # Log results
            mlflow.log_metrics(
                {
                    "roi": r_primary["roi"],
                    "n_bets": r_primary["n_bets"],
                    "precision": r_primary["precision"],
                    "roc_auc": auc,
                    "best_val_roi": best_val["val_roi"],
                    "best_odds_lo": best_val["odds_lo"],
                    "best_odds_hi": best_val["odds_hi"],
                    "best_threshold": best_val["thr"],
                }
            )
            mlflow.set_tag("best_config", str(best_val))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.6")
            logger.exception("Step 4.6 failed")
            raise


if __name__ == "__main__":
    main()
