"""Step 4.2: Focused strategy - singles only + sport/market filtering + CatBoost."""

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
from utils import (
    check_budget,
    load_data,
    set_seed,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "uaf/sports_10h_v4")
SESSION_ID = os.environ.get("UAF_SESSION_ID", "sports_10h_v4")

CAT_COLS = {"Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket"}

# Из step 4.1 сегментного анализа: спорты с положительным ROI
GOOD_SPORTS = {"Soccer", "Cricket", "CS2", "Table Tennis", "Dota 2"}


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


def prepare_cb(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS]
    x = df[features].copy()
    for idx in cat_indices:
        col = features[idx]
        x[col] = x[col].astype(str).replace("nan", "_missing_")
    return x, cat_indices


def eval_strategy(
    test: pd.DataFrame,
    probas: np.ndarray,
    singles_only: bool = False,
    good_sports_only: bool = False,
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    """Оценка стратегии с фильтрами."""
    mask = probas >= threshold
    if singles_only:
        mask = mask & (test["Is_Parlay"] == "f").values
    if good_sports_only:
        mask = mask & test["Sport"].isin(GOOD_SPORTS).values

    selected = test[mask]
    if len(selected) < 5:
        return {"roi": 0.0, "n_bets": 0, "label": label}

    n = len(selected)
    n_won = int(selected["target"].sum())
    payout = selected.loc[selected["target"] == 1, "Odds"].sum()
    roi = (payout - n) / n * 100
    prec = n_won / n

    logger.info(
        "%s: ROI=%.2f%%, bets=%d/%d, prec=%.3f, sel=%.3f",
        label,
        roi,
        n,
        len(test),
        prec,
        n / len(test),
    )
    return {
        "roi": roi,
        "n_bets": n,
        "n_won": n_won,
        "precision": prec,
        "selectivity": n / len(test),
        "label": label,
    }


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase4/step_4_2_focused_strategy") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.2")
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
                    "method": "focused_strategy_catboost",
                    "n_features": len(features),
                    "gap_days": 7,
                    "good_sports": str(list(GOOD_SPORTS)),
                }
            )

            # Train CatBoost
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

            # Strategy matrix
            results = []
            for thr in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
                # All bets
                r = eval_strategy(test, test_probas, threshold=thr, label=f"all_thr{thr:.2f}")
                results.append(r)

                # Singles only
                r = eval_strategy(
                    test,
                    test_probas,
                    singles_only=True,
                    threshold=thr,
                    label=f"singles_thr{thr:.2f}",
                )
                results.append(r)

                # Singles + good sports
                r = eval_strategy(
                    test,
                    test_probas,
                    singles_only=True,
                    good_sports_only=True,
                    threshold=thr,
                    label=f"singles_good_thr{thr:.2f}",
                )
                results.append(r)

            # Find best on val
            val_probas = model.predict_proba(x_val)[:, 1]
            best_val_roi = -999.0
            best_val_config = {"thr": 0.5, "singles": False, "good_sports": False}

            for thr in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
                for singles in [False, True]:
                    for good_sp in [False, True]:
                        mask = val_probas >= thr
                        if singles:
                            mask = mask & (val["Is_Parlay"] == "f").values
                        if good_sp:
                            mask = mask & val["Sport"].isin(GOOD_SPORTS).values
                        sel = val[mask]
                        if len(sel) < 15:
                            continue
                        n = len(sel)
                        payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                        roi = (payout - n) / n * 100
                        if roi > best_val_roi:
                            best_val_roi = roi
                            best_val_config = {
                                "thr": thr,
                                "singles": singles,
                                "good_sports": good_sp,
                            }

            logger.info("Best val config: %s, val ROI: %.2f%%", best_val_config, best_val_roi)

            # Apply best val config to test
            r_best = eval_strategy(
                test,
                test_probas,
                singles_only=best_val_config["singles"],
                good_sports_only=best_val_config["good_sports"],
                threshold=best_val_config["thr"],
                label="best_val_config",
            )

            # Per-sport analysis with threshold = 0.50
            logger.info("=== Per-sport analysis (thr=0.50, singles only) ===")
            sport_results = {}
            for sport in test["Sport"].unique():
                sport_mask = (
                    (test_probas >= 0.50)
                    & (test["Is_Parlay"] == "f").values
                    & (test["Sport"] == sport).values
                )
                sel = test[sport_mask]
                if len(sel) < 10:
                    continue
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                roi = (payout - n) / n * 100
                sport_results[sport] = {"roi": roi, "n": n}
                logger.info("  %s: ROI=%.2f%%, n=%d", sport, roi, n)

            # Log best
            best_roi_test = r_best["roi"]
            mlflow.log_metrics(
                {
                    "roi": best_roi_test,
                    "roi_all_50": results[1]["roi"] if len(results) > 1 else 0,
                    "n_bets": r_best["n_bets"],
                    "precision": r_best.get("precision", 0),
                    "selectivity": r_best.get("selectivity", 0),
                    "roc_auc": auc,
                    "best_threshold": best_val_config["thr"],
                    "best_val_roi": best_val_roi,
                }
            )
            mlflow.set_tag("best_config", str(best_val_config))

            # Log per-sport ROI
            for sport, sr in sport_results.items():
                mlflow.log_metric(f"roi_sport_{sport.replace(' ', '_')}", sr["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.2")
            logger.exception("Step 4.2 failed")
            raise


if __name__ == "__main__":
    main()
