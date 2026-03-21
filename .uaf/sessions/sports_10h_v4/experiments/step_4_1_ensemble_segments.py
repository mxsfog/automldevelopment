"""Step 4.1: Ensemble (CatBoost + LightGBM) + сегментный анализ ROI."""

import logging
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    calc_roi,
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

CAT_COLS_CB = {"Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket"}
CAT_COLS_LGB = ["Is_Parlay", "Sport", "Market", "ML_Team_Stats_Found", "odds_bucket"]


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
    "Is_Parlay",
    "Sport",
    "Market",
    "ML_Team_Stats_Found",
    "odds_bucket",
]


def prepare_cb(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Подготовка для CatBoost."""
    cat_indices = [i for i, f in enumerate(features) if f in CAT_COLS_CB]
    x = df[features].copy()
    for idx in cat_indices:
        col = features[idx]
        x[col] = x[col].astype(str).replace("nan", "_missing_")
    return x, cat_indices


def prepare_lgb(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Подготовка для LightGBM."""
    x = df[features].copy()
    for col in CAT_COLS_LGB:
        if col in x.columns:
            x[col] = x[col].astype("category")
    return x


def segment_analysis(df: pd.DataFrame, probas: np.ndarray, threshold: float = 0.5) -> None:
    """Анализ ROI по сегментам."""
    mask = probas >= threshold
    df_selected = df[mask].copy()
    df_selected["pred_correct"] = df_selected["target"].values

    for col in ["Sport", "Market", "Is_Parlay", "odds_bucket"]:
        if col not in df_selected.columns:
            continue
        logger.info("--- Segment: %s ---", col)
        for val in df_selected[col].value_counts().head(10).index:
            seg = df_selected[df_selected[col] == val]
            if len(seg) < 10:
                continue
            n = len(seg)
            payout = seg.loc[seg["target"] == 1, "Odds"].sum()
            roi = (payout - n) / n * 100
            logger.info(
                "  %s=%s: n=%d, ROI=%.2f%%, prec=%.3f", col, val, n, roi, seg["target"].mean()
            )


def main() -> None:
    set_seed()
    if check_budget():
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase4/step_4_1_ensemble_segments") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")
            mlflow.set_tag("step", "4.1")
            mlflow.set_tag("phase", "4")

            df = load_data()
            df = add_features(df)
            train, test = time_series_split(df)

            val_split = int(len(train) * 0.8)
            train_inner = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            features = [f for f in FEATURES if f in df.columns]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_inner),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "ensemble_catboost_lgbm",
                    "n_features": len(features),
                    "gap_days": 7,
                }
            )

            # CatBoost
            x_train_cb, cat_indices = prepare_cb(train_inner, features)
            x_val_cb, _ = prepare_cb(val, features)
            x_test_cb, _ = prepare_cb(test, features)

            cb = CatBoostClassifier(
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
            cb.fit(
                x_train_cb,
                train_inner["target"].values,
                eval_set=(x_val_cb, val["target"].values),
                use_best_model=True,
            )
            logger.info("CatBoost best iter: %d", cb.best_iteration_)

            # LightGBM
            x_train_lgb = prepare_lgb(train_inner, features)
            x_val_lgb = prepare_lgb(val, features)
            x_test_lgb = prepare_lgb(test, features)

            lgb = LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.01,
                max_depth=5,
                reg_lambda=10,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                metric="auc",
                n_jobs=-1,
            )
            lgb.fit(
                x_train_lgb,
                train_inner["target"].values,
                eval_set=[(x_val_lgb, val["target"].values)],
                callbacks=[],
            )
            logger.info("LightGBM best iter: %d", lgb.best_iteration_)

            # Ensemble probabilities (average)
            cb_test_proba = cb.predict_proba(x_test_cb)[:, 1]
            lgb_test_proba = lgb.predict_proba(x_test_lgb)[:, 1]
            ensemble_proba = (cb_test_proba + lgb_test_proba) / 2

            cb_val_proba = cb.predict_proba(x_val_cb)[:, 1]
            lgb_val_proba = lgb.predict_proba(x_val_lgb)[:, 1]
            ensemble_val_proba = (cb_val_proba + lgb_val_proba) / 2

            # AUC comparison
            auc_cb = roc_auc_score(test["target"].values, cb_test_proba)
            auc_lgb = roc_auc_score(test["target"].values, lgb_test_proba)
            auc_ens = roc_auc_score(test["target"].values, ensemble_proba)
            logger.info(
                "AUC - CatBoost: %.4f, LightGBM: %.4f, Ensemble: %.4f", auc_cb, auc_lgb, auc_ens
            )

            # ROI scan
            best_roi = -999.0
            best_config = {}
            for thr in np.arange(0.45, 0.70, 0.01):
                for margin in [0.0, 0.01, 0.02, 0.03, 0.05]:
                    implied = 1.0 / val["Odds"].values
                    mask = (ensemble_val_proba >= thr) & (ensemble_val_proba > (implied + margin))
                    if mask.sum() < 15:
                        continue
                    sel = val[mask]
                    n = len(sel)
                    payout = sel.loc[sel["target"] == 1, "Odds"].sum()
                    roi = (payout - n) / n * 100
                    if roi > best_roi:
                        best_roi = roi
                        best_config = {"thr": thr, "margin": margin}

            logger.info("Best val config: %s, val ROI: %.2f%%", best_config, best_roi)

            # Test with best config
            best_thr = best_config.get("thr", 0.50)
            best_margin = best_config.get("margin", 0.0)
            implied_test = 1.0 / test["Odds"].values
            mask_test = (ensemble_proba >= best_thr) & (
                ensemble_proba > (implied_test + best_margin)
            )
            sel_test = test[mask_test]

            if len(sel_test) > 0:
                n_t = len(sel_test)
                payout_t = sel_test.loc[sel_test["target"] == 1, "Odds"].sum()
                roi_test = (payout_t - n_t) / n_t * 100
            else:
                roi_test = 0.0
                n_t = 0

            logger.info(
                "Ensemble test (thr=%.2f, margin=%.2f): ROI=%.2f%%, bets=%d",
                best_thr,
                best_margin,
                roi_test,
                n_t,
            )

            # Also test with simple threshold
            result_50 = calc_roi(test, ensemble_proba, threshold=0.5)
            logger.info(
                "Ensemble thr=0.50: ROI=%.2f%%, bets=%d", result_50["roi"], result_50["n_bets"]
            )

            for thr in [0.50, 0.52, 0.55, 0.58, 0.60]:
                r = calc_roi(test, ensemble_proba, threshold=thr)
                logger.info(
                    "  Ensemble thr=%.2f: ROI=%.2f%%, bets=%d, prec=%.3f",
                    thr,
                    r["roi"],
                    r["n_bets"],
                    r["precision"],
                )

            # Segment analysis
            logger.info("=== Segment analysis (ensemble thr=0.50) ===")
            segment_analysis(test, ensemble_proba, threshold=0.50)

            # Singles only
            singles = test[test["Is_Parlay"] == "f"]
            singles_mask = ensemble_proba[test["Is_Parlay"] == "f"] >= 0.50
            if singles_mask.sum() > 0:
                singles_sel = singles[singles_mask.values]
                n_s = len(singles_sel)
                payout_s = singles_sel.loc[singles_sel["target"] == 1, "Odds"].sum()
                roi_singles = (payout_s - n_s) / n_s * 100
                logger.info("Singles only thr=0.50: ROI=%.2f%%, bets=%d", roi_singles, n_s)

            mlflow.log_metrics(
                {
                    "roi": roi_test,
                    "roi_thr_50": result_50["roi"],
                    "auc_catboost": auc_cb,
                    "auc_lightgbm": auc_lgb,
                    "auc_ensemble": auc_ens,
                    "best_threshold": best_thr,
                    "best_margin": best_margin,
                    "n_bets": n_t,
                    "roc_auc": auc_ens,
                }
            )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.1")
            logger.exception("Step 4.1 failed")
            raise


if __name__ == "__main__":
    main()
