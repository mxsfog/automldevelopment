"""Step 4.3: Per-sport CatBoost models + sport-specific thresholds."""

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

CAT_COLS = {"Is_Parlay", "Market", "ML_Team_Stats_Found", "odds_bucket"}

GOOD_SPORTS = ["Cricket", "CS2", "Soccer", "Dota 2", "Table Tennis"]
ALL_FEATURES = [
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
    "Is_Parlay",
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
    return df


def prepare_cb(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS]
    x = df[features].copy()
    for idx in cat_indices:
        col = features[idx]
        x[col] = x[col].astype(str).replace("nan", "_missing_")
    return x, cat_indices


def train_and_eval_sport(
    sport: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict:
    """Обучение и оценка модели для конкретного спорта."""
    # Только singles
    train_s = train[(train["Sport"] == sport) & (train["Is_Parlay"] == "f")]
    val_s = val[(val["Sport"] == sport) & (val["Is_Parlay"] == "f")]
    test_s = test[(test["Sport"] == sport) & (test["Is_Parlay"] == "f")]

    if len(train_s) < 50 or len(test_s) < 10:
        logger.info(
            "  %s: insufficient data (train=%d, test=%d)", sport, len(train_s), len(test_s)
        )
        return {"sport": sport, "roi": None, "n_bets": 0}

    x_train, cat_indices = prepare_cb(train_s, features)
    x_val, _ = prepare_cb(val_s, features)
    x_test, _ = prepare_cb(test_s, features)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.01,
        depth=4,
        l2_leaf_reg=10,
        min_data_in_leaf=20,
        random_strength=2,
        bagging_temperature=1,
        random_seed=42,
        verbose=0,
        eval_metric="AUC",
        cat_features=cat_indices,
        early_stopping_rounds=100,
    )

    if len(val_s) >= 10:
        model.fit(
            x_train,
            train_s["target"].values,
            eval_set=(x_val, val_s["target"].values),
            use_best_model=True,
        )
    else:
        model.fit(x_train, train_s["target"].values)

    test_probas = model.predict_proba(x_test)[:, 1]

    try:
        auc = roc_auc_score(test_s["target"].values, test_probas)
    except ValueError:
        auc = 0.5

    # Best threshold on val
    if len(val_s) >= 10:
        val_probas = model.predict_proba(x_val)[:, 1]
        best_thr = 0.5
        best_val_roi = -999.0
        for thr in np.arange(0.40, 0.70, 0.02):
            mask = val_probas >= thr
            sel = val_s[mask]
            if len(sel) < 5:
                continue
            n = len(sel)
            payout = sel.loc[sel["target"] == 1, "Odds"].sum()
            roi = (payout - n) / n * 100
            if roi > best_val_roi:
                best_val_roi = roi
                best_thr = thr
    else:
        best_thr = 0.50

    # Test ROI
    results = {}
    for thr in [0.45, 0.50, best_thr]:
        mask = test_probas >= thr
        sel = test_s[mask]
        if len(sel) < 5:
            continue
        n = len(sel)
        payout = sel.loc[sel["target"] == 1, "Odds"].sum()
        roi = (payout - n) / n * 100
        results[f"thr_{thr:.2f}"] = {"roi": roi, "n": n, "prec": sel["target"].mean()}
        logger.info(
            "  %s thr=%.2f: ROI=%.2f%%, n=%d, prec=%.3f", sport, thr, roi, n, sel["target"].mean()
        )

    # Default result at thr=0.50
    thr_key = "thr_0.50"
    if thr_key in results:
        main_result = results[thr_key]
    elif results:
        main_result = list(results.values())[0]
    else:
        main_result = {"roi": 0.0, "n": 0, "prec": 0.0}

    logger.info(
        "  %s: AUC=%.4f, best_val_thr=%.2f, iters=%s",
        sport,
        auc,
        best_thr,
        getattr(model, "best_iteration_", "N/A"),
    )

    return {
        "sport": sport,
        "roi": main_result["roi"],
        "n_bets": main_result["n"],
        "precision": main_result["prec"],
        "auc": auc,
        "best_thr": best_thr,
        "model": model,
        "results": results,
    }


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase4/step_4_3_per_sport") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.3")
            mlflow.set_tag("phase", "4")

            df = load_data()
            df = add_features(df)
            train, test = time_series_split(df)

            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            features = [f for f in ALL_FEATURES if f in df.columns]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_test": len(test),
                    "method": "per_sport_catboost",
                    "n_features": len(features),
                    "gap_days": 7,
                    "sports": str(GOOD_SPORTS),
                }
            )

            # Train per-sport models
            sport_results = {}
            total_n_bets = 0
            total_payout = 0.0

            for sport in GOOD_SPORTS:
                logger.info("Training model for %s...", sport)
                res = train_and_eval_sport(sport, train_inner, val, test, features)
                sport_results[sport] = res

                if res["roi"] is not None:
                    mlflow.log_metric(f"roi_{sport.replace(' ', '_')}", res["roi"])
                    mlflow.log_metric(f"auc_{sport.replace(' ', '_')}", res["auc"])
                    mlflow.log_metric(f"n_bets_{sport.replace(' ', '_')}", res["n_bets"])

            # Combined ROI across all per-sport models at thr=0.50
            combined_n = 0
            combined_payout = 0.0
            for sport, res in sport_results.items():
                if res["roi"] is not None and res["n_bets"] > 0:
                    r50 = res.get("results", {}).get("thr_0.50", {})
                    if r50:
                        combined_n += r50["n"]
                        sport_test = test[(test["Sport"] == sport) & (test["Is_Parlay"] == "f")]
                        test_probas_s = res["model"].predict_proba(
                            prepare_cb(sport_test, features)[0]
                        )[:, 1]
                        mask_s = test_probas_s >= 0.50
                        sel_s = sport_test[mask_s]
                        combined_payout += sel_s.loc[sel_s["target"] == 1, "Odds"].sum()

            if combined_n > 0:
                combined_roi = (combined_payout - combined_n) / combined_n * 100
            else:
                combined_roi = 0.0

            logger.info(
                "Combined per-sport ROI (thr=0.50, singles only): %.2f%%, bets=%d",
                combined_roi,
                combined_n,
            )

            # Compare with global model
            # Train global model on all good sports singles
            good_mask_train = train_inner["Sport"].isin(GOOD_SPORTS) & (
                train_inner["Is_Parlay"] == "f"
            )
            good_mask_val = val["Sport"].isin(GOOD_SPORTS) & (val["Is_Parlay"] == "f")
            good_mask_test = test["Sport"].isin(GOOD_SPORTS) & (test["Is_Parlay"] == "f")

            train_good = train_inner[good_mask_train]
            val_good = val[good_mask_val]
            test_good = test[good_mask_test]

            features_with_sport = features + ["Sport"] if "Sport" not in features else features
            features_with_sport = [f for f in features_with_sport if f in df.columns]

            cat_cols_with_sport = CAT_COLS | {"Sport"}
            cat_indices_ws = [
                i for i, f in enumerate(features_with_sport) if f in cat_cols_with_sport
            ]

            x_tr_g = train_good[features_with_sport].copy()
            x_va_g = val_good[features_with_sport].copy()
            x_te_g = test_good[features_with_sport].copy()
            for idx in cat_indices_ws:
                col = features_with_sport[idx]
                x_tr_g[col] = x_tr_g[col].astype(str).replace("nan", "_missing_")
                x_va_g[col] = x_va_g[col].astype(str).replace("nan", "_missing_")
                x_te_g[col] = x_te_g[col].astype(str).replace("nan", "_missing_")

            global_model = CatBoostClassifier(
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
                cat_features=cat_indices_ws,
                early_stopping_rounds=100,
            )
            global_model.fit(
                x_tr_g,
                train_good["target"].values,
                eval_set=(x_va_g, val_good["target"].values),
                use_best_model=True,
            )
            global_probas = global_model.predict_proba(x_te_g)[:, 1]

            for thr in [0.45, 0.50, 0.52, 0.55]:
                mask = global_probas >= thr
                sel = test_good[mask]
                if len(sel) < 5:
                    continue
                n = len(sel)
                payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                roi = (payout - n) / n * 100
                logger.info(
                    "Global model (good sports singles) thr=%.2f: ROI=%.2f%%, n=%d", thr, roi, n
                )

            mlflow.log_metrics(
                {
                    "roi": combined_roi,
                    "roi_per_sport_combined": combined_roi,
                    "n_bets": combined_n,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.3")
            logger.exception("Step 4.3 failed")
            raise


if __name__ == "__main__":
    main()
