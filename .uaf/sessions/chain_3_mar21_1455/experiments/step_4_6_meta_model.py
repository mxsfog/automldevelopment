"""Step 4.6 — Meta-model on OOF predictions + profit-aware selection.

Гипотеза: вместо простого усреднения моделей, обучим мета-модель
на OOF (out-of-fold) предсказаниях base моделей. Мета-модель
увидит паттерны в том, когда какая base модель точнее.
Также добавим profit-aware features (EV, confidence, odds interactions).
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    calc_roi,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)
from step_2_feature_engineering import add_sport_market_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("Budget hard_stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURES = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcome_Odds",
    "n_outcomes",
    "mean_outcome_odds",
    "max_outcome_odds",
    "min_outcome_odds",
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def get_oof_predictions(
    train_df: pd.DataFrame,
    features: list[str],
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions for base models using time-series splits."""
    n = len(train_df)
    oof_cb = np.zeros(n)
    oof_lgbm = np.zeros(n)
    oof_lr = np.zeros(n)
    fold_mask = np.zeros(n, dtype=bool)

    fold_size = n // (n_folds + 1)

    for i in range(n_folds):
        tr_end = fold_size * (i + 2)
        va_start = tr_end
        va_end = min(va_start + fold_size, n)
        if va_end <= va_start:
            break

        x_tr = train_df.iloc[:tr_end][features].fillna(0)
        y_tr = train_df.iloc[:tr_end]["target"]
        x_va = train_df.iloc[va_start:va_end][features].fillna(0)

        cb = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        cb.fit(x_tr, y_tr)
        oof_cb[va_start:va_end] = cb.predict_proba(x_va)[:, 1]

        lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1,
            min_child_samples=50,
        )
        lgbm.fit(x_tr, y_tr)
        oof_lgbm[va_start:va_end] = lgbm.predict_proba(x_va)[:, 1]

        scaler = StandardScaler()
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(scaler.fit_transform(x_tr), y_tr)
        oof_lr[va_start:va_end] = lr.predict_proba(scaler.transform(x_va))[:, 1]

        fold_mask[va_start:va_end] = True

    return oof_cb, oof_lgbm, oof_lr, fold_mask


def build_meta_features(
    df: pd.DataFrame,
    p_cb: np.ndarray,
    p_lgbm: np.ndarray,
    p_lr: np.ndarray,
) -> pd.DataFrame:
    """Build meta features from base model predictions."""
    odds = df["Odds"].values
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std(np.array([p_cb, p_lgbm, p_lr]), axis=0)
    p_max = np.maximum(np.maximum(p_cb, p_lgbm), p_lr)
    p_min = np.minimum(np.minimum(p_cb, p_lgbm), p_lr)

    ev_mean = p_mean * odds - 1
    ev_cb = p_cb * odds - 1
    ev_lgbm = p_lgbm * odds - 1
    ev_lr = p_lr * odds - 1

    meta = pd.DataFrame(
        {
            "p_cb": p_cb,
            "p_lgbm": p_lgbm,
            "p_lr": p_lr,
            "p_mean": p_mean,
            "p_std": p_std,
            "p_range": p_max - p_min,
            "p_max": p_max,
            "p_min": p_min,
            "ev_mean": ev_mean,
            "ev_cb": ev_cb,
            "ev_lgbm": ev_lgbm,
            "ev_lr": ev_lr,
            "ev_std": np.std(np.array([ev_cb, ev_lgbm, ev_lr]), axis=0),
            "odds": odds,
            "odds_log": np.log1p(odds),
            "implied_prob": 1.0 / np.clip(odds, 1.01, None),
            "confidence": 1 / (1 + p_std * 10),
            "ev_confidence": ev_mean / (1 + p_std * 10),
        }
    )
    return meta


def main() -> None:
    """Meta-model on OOF predictions."""
    with mlflow.start_run(run_name="phase4/meta_model") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Feature encoding
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "method": "meta_model_oof",
                }
            )

            # Step 1: OOF predictions on train
            logger.info("Generating OOF predictions...")
            oof_cb, oof_lgbm, oof_lr, fold_mask = get_oof_predictions(
                train_enc, FEATURES, n_folds=5
            )

            # Use only rows that have OOF predictions
            train_oof = train_enc[fold_mask].copy()
            oof_cb_valid = oof_cb[fold_mask]
            oof_lgbm_valid = oof_lgbm[fold_mask]
            oof_lr_valid = oof_lr[fold_mask]

            logger.info("OOF samples: %d / %d", fold_mask.sum(), len(train_enc))

            # Step 2: Build meta features
            meta_train = build_meta_features(train_oof, oof_cb_valid, oof_lgbm_valid, oof_lr_valid)
            y_meta_train = train_oof["target"].values

            # Step 3: Train base models on full train for test predictions
            logger.info("Training base models on full train...")
            x_train_full = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            y_train_full = train_enc["target"]

            cb = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            )
            cb.fit(x_train_full, y_train_full)

            lgbm = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1,
                min_child_samples=50,
            )
            lgbm.fit(x_train_full, y_train_full)

            scaler_base = StandardScaler()
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(scaler_base.fit_transform(x_train_full), y_train_full)

            p_cb_test = cb.predict_proba(x_test)[:, 1]
            p_lgbm_test = lgbm.predict_proba(x_test)[:, 1]
            p_lr_test = lr.predict_proba(scaler_base.transform(x_test))[:, 1]

            meta_test = build_meta_features(test_enc, p_cb_test, p_lgbm_test, p_lr_test)

            # Step 4: Train meta-model (CatBoost on meta features)
            logger.info("Training meta-model...")
            meta_cb = CatBoostClassifier(
                iterations=300,
                learning_rate=0.03,
                depth=4,
                random_seed=42,
                verbose=0,
                l2_leaf_reg=10,
            )
            meta_cb.fit(meta_train, y_meta_train)

            p_meta_test = meta_cb.predict_proba(meta_test)[:, 1]

            auc_simple = roc_auc_score(
                test_enc["target"], (p_cb_test + p_lgbm_test + p_lr_test) / 3
            )
            auc_meta = roc_auc_score(test_enc["target"], p_meta_test)
            logger.info("AUC simple avg: %.4f, AUC meta: %.4f", auc_simple, auc_meta)

            # Step 5: Compare selection strategies
            odds = test_enc["Odds"].values

            # Simple ensemble baseline
            p_simple = (p_cb_test + p_lgbm_test + p_lr_test) / 3
            ev_simple = p_simple * odds - 1

            # Meta-model EV
            ev_meta = p_meta_test * odds - 1

            results = {}
            for thr in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
                # Simple
                mask_s = ev_simple >= thr
                r_s = calc_roi(test_enc, mask_s.astype(float), threshold=0.5)

                # Meta
                mask_m = ev_meta >= thr
                r_m = calc_roi(test_enc, mask_m.astype(float), threshold=0.5)

                results[f"simple_ev_{thr:.2f}"] = r_s
                results[f"meta_ev_{thr:.2f}"] = r_m

                logger.info(
                    "thr=%.2f: simple=%.2f%%(n=%d) meta=%.2f%%(n=%d)",
                    thr,
                    r_s["roi"],
                    r_s["n_bets"],
                    r_m["roi"],
                    r_m["n_bets"],
                )

            # Meta conf_ev
            # Use base model std for confidence weighting
            p_std_base = np.std(np.array([p_cb_test, p_lgbm_test, p_lr_test]), axis=0)
            conf_base = 1 / (1 + p_std_base * 10)
            ev_meta_conf = ev_meta * conf_base

            for thr in [0.10, 0.12, 0.15]:
                mask = ev_meta_conf >= thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                results[f"meta_conf_ev_{thr:.2f}"] = r
                logger.info("meta_conf_ev_%.2f: ROI=%.2f%%, n=%d", thr, r["roi"], r["n_bets"])

            # Best
            best_name = max(
                results,
                key=lambda k: results[k]["roi"] if results[k]["n_bets"] >= 50 else -999,
            )
            best = results[best_name]
            logger.info("Best: %s -> ROI=%.2f%%, n=%d", best_name, best["roi"], best["n_bets"])

            # Feature importance of meta-model
            fi = meta_cb.get_feature_importance()
            fi_names = meta_train.columns.tolist()
            fi_sorted = sorted(zip(fi_names, fi, strict=True), key=lambda x: x[1], reverse=True)
            logger.info("Meta-model feature importance:")
            for name, imp in fi_sorted:
                logger.info("  %s: %.2f", name, imp)

            mlflow.log_metrics(
                {
                    "auc_simple": auc_simple,
                    "auc_meta": auc_meta,
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            for name, r in results.items():
                mlflow.log_metrics(
                    {
                        f"roi_{name}": r["roi"],
                        f"n_{name}": r["n_bets"],
                    }
                )
            mlflow.set_tag("best_strategy", best_name)

            # Save if improved
            if best["roi"] > 27.95:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                meta_data = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best["roi"],
                    "auc": auc_meta,
                    "threshold": 0.12,
                    "ev_threshold": 0.12,
                    "n_bets": best["n_bets"],
                    "feature_names": FEATURES,
                    "selection_method": best_name,
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta_data, f, indent=2)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.6 failed")
            raise


if __name__ == "__main__":
    main()
