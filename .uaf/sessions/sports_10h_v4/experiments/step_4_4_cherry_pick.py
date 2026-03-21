"""Step 4.4: Cherry-pick strategy - фокус на Cricket+CS2+Dota2 + threshold calibration."""

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
GOOD_SPORTS = {"Cricket", "CS2", "Soccer", "Dota 2", "Table Tennis"}
TOP_SPORTS = {"Cricket", "CS2", "Dota 2"}  # ROI > 5%

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
    df["is_good_sport"] = df["Sport"].isin(GOOD_SPORTS).astype(int)
    return df


def prepare_cb(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS]
    x = df[features].copy()
    for idx in cat_indices:
        col = features[idx]
        x[col] = x[col].astype(str).replace("nan", "_missing_")
    return x, cat_indices


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase4/step_4_4_cherry_pick") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.4")
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
                    "method": "cherry_pick_strategy",
                    "n_features": len(features),
                    "gap_days": 7,
                }
            )

            # Global CatBoost model
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
            auc = roc_auc_score(test["target"].values, test_probas)
            logger.info("AUC: %.4f", auc)

            # Strategy 1: Singles + Top sports (Cricket, CS2, Dota2)
            logger.info("=== Strategy: Singles + Top Sports (Cricket, CS2, Dota2) ===")
            for thr in [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
                mask = (
                    (test_probas >= thr)
                    & (test["Is_Parlay"] == "f").values
                    & test["Sport"].isin(TOP_SPORTS).values
                )
                sel = test[mask]
                if len(sel) < 5:
                    continue
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                roi = (payout - n) / n * 100
                logger.info(
                    "  thr=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f", thr, roi, n, sel["target"].mean()
                )

            # Strategy 2: Singles + Good sports
            logger.info("=== Strategy: Singles + Good Sports (5 sports) ===")
            for thr in [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
                mask = (
                    (test_probas >= thr)
                    & (test["Is_Parlay"] == "f").values
                    & test["Sport"].isin(GOOD_SPORTS).values
                )
                sel = test[mask]
                if len(sel) < 5:
                    continue
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                roi = (payout - n) / n * 100
                logger.info(
                    "  thr=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f", thr, roi, n, sel["target"].mean()
                )

            # Strategy 3: Top sports + value betting
            logger.info("=== Strategy: Top Sports + Value Betting ===")
            implied = 1.0 / test["Odds"].values
            for thr in [0.45, 0.50, 0.52]:
                for margin in [0.0, 0.01, 0.02, 0.03, 0.05]:
                    mask = (
                        (test_probas >= thr)
                        & (test_probas > (implied + margin))
                        & (test["Is_Parlay"] == "f").values
                        & test["Sport"].isin(TOP_SPORTS).values
                    )
                    sel = test[mask]
                    if len(sel) < 10:
                        continue
                    n = len(sel)
                    payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                    roi = (payout - n) / n * 100
                    logger.info(
                        "  thr=%.2f margin=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f",
                        thr,
                        margin,
                        roi,
                        n,
                        sel["target"].mean(),
                    )

            # Strategy 4: Singles + All sports but remove bad ones
            bad_sports = {"MMA", "Ice Hockey", "League of Legends", "FIFA", "Basketball", "Tennis"}
            logger.info("=== Strategy: All minus bad sports ===")
            for thr in [0.45, 0.50, 0.52, 0.55]:
                mask = (
                    (test_probas >= thr)
                    & (test["Is_Parlay"] == "f").values
                    & ~test["Sport"].isin(bad_sports).values
                )
                sel = test[mask]
                if len(sel) < 5:
                    continue
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                roi = (payout - n) / n * 100
                logger.info(
                    "  thr=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f", thr, roi, n, sel["target"].mean()
                )

            # Best overall: find on val
            val_probas = model.predict_proba(x_val)[:, 1]
            best_roi = -999.0
            best_strategy = {}

            for sport_set_name, sport_set in [("top3", TOP_SPORTS), ("good5", GOOD_SPORTS)]:
                for thr in [0.45, 0.50, 0.52, 0.55]:
                    for margin in [0.0, 0.02, 0.05]:
                        implied_val = 1.0 / val["Odds"].values
                        mask = (
                            (val_probas >= thr)
                            & (val_probas > (implied_val + margin))
                            & (val["Is_Parlay"] == "f").values
                            & val["Sport"].isin(sport_set).values
                        )
                        sel = val[mask]
                        if len(sel) < 15:
                            continue
                        n = len(sel)
                        payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                        roi = (payout - n) / n * 100
                        if roi > best_roi:
                            best_roi = roi
                            best_strategy = {
                                "thr": thr,
                                "margin": margin,
                                "sport_set": sport_set_name,
                            }

            logger.info("Best val strategy: %s, val ROI: %.2f%%", best_strategy, best_roi)

            # Apply best to test
            sport_set = TOP_SPORTS if best_strategy.get("sport_set") == "top3" else GOOD_SPORTS
            best_thr = best_strategy.get("thr", 0.50)
            best_margin = best_strategy.get("margin", 0.0)

            mask_test = (
                (test_probas >= best_thr)
                & (test_probas > (implied + best_margin))
                & (test["Is_Parlay"] == "f").values
                & test["Sport"].isin(sport_set).values
            )
            sel_test = test[mask_test]
            if len(sel_test) > 0:
                n_t = len(sel_test)
                payout_t = sel_test.loc[sel_test["target"] == 1, "Odds"].sum()
                roi_test = (payout_t - n_t) / n_t * 100
                prec_t = sel_test["target"].mean()
            else:
                roi_test, n_t, prec_t = 0.0, 0, 0.0

            logger.info(
                "Best test: ROI=%.2f%%, bets=%d, prec=%.3f, strategy=%s",
                roi_test,
                n_t,
                prec_t,
                best_strategy,
            )

            mlflow.log_metrics(
                {
                    "roi": roi_test,
                    "n_bets": n_t,
                    "precision": prec_t,
                    "roc_auc": auc,
                    "best_threshold": best_thr,
                    "best_margin": best_margin,
                    "best_val_roi": best_roi,
                }
            )
            mlflow.set_tag("best_strategy", str(best_strategy))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.4")
            logger.exception("Step 4.4 failed")
            raise


if __name__ == "__main__":
    main()
