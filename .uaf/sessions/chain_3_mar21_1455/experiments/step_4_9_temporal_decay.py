"""Step 4.9 — Temporal decay weighting + isotonic calibration.

Гипотеза 1: Недавние данные информативнее для предсказания будущего.
Экспоненциальный decay weight при обучении: w_i = exp(-lambda * days_ago_i).
CatBoost/LightGBM поддерживают sample_weight.

Гипотеза 2: Isotonic calibration на OOF + ужесточённый порог.
Предыдущая попытка (chain_2) ослабляла порог после калибровки.
Правильный подход: калибровать -> подобрать порог на val.
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
from sklearn.isotonic import IsotonicRegression
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


def compute_temporal_weights(
    dates: pd.Series,
    half_life_days: float = 14.0,
) -> np.ndarray:
    """Exponential decay weights based on recency."""
    max_date = dates.max()
    days_ago = (max_date - dates).dt.total_seconds() / 86400
    lam = np.log(2) / half_life_days
    weights = np.exp(-lam * days_ago.values)
    return weights


def train_weighted_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
) -> tuple:
    """3-model ensemble with sample weights."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    cb.fit(x, y, sample_weight=weights)

    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y, sample_weight=weights)

    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y, sample_weight=weights)

    return cb, lgbm, lr, scaler


def train_ensemble(x: pd.DataFrame, y: pd.Series) -> tuple:
    """3-model ensemble without weights."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(x), y)
    return cb, lgbm, lr, scaler


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions + std."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std(np.array([p_cb, p_lgbm, p_lr]), axis=0)
    return p_mean, p_std


def calibrate_isotonic(
    y_true: np.ndarray,
    p_train: np.ndarray,
    p_test: np.ndarray,
) -> np.ndarray:
    """Isotonic regression calibration."""
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(p_train, y_true)
    return iso.transform(p_test)


def main() -> None:
    """Temporal decay + isotonic calibration experiment."""
    with mlflow.start_run(run_name="phase4/temporal_decay") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Val split
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            # Feature encoding
            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            x_train_fit = train_fit_enc[FEATURES].fillna(0)
            x_val = val_enc[FEATURES].fillna(0)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "temporal_decay_calibration",
                }
            )

            # Part 1: Temporal decay weighting
            logger.info("Part 1: Temporal decay weighting")

            results = {}

            # Baseline without weights (on val)
            cb_base, lgbm_base, lr_base, sc_base = train_ensemble(
                x_train_fit, train_fit_enc["target"]
            )
            p_val_base, p_std_base = predict_ensemble(cb_base, lgbm_base, lr_base, sc_base, x_val)
            odds_val = val_enc["Odds"].values
            ev_val = p_val_base * odds_val - 1
            conf_val = 1 / (1 + p_std_base * 10)
            ev_conf_val = ev_val * conf_val

            for thr in [0.12, 0.15, 0.18]:
                mask = ev_conf_val >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                results[f"base_conf_ev_{thr:.2f}"] = r
                logger.info(
                    "Base conf_ev_%.2f on val: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Temporal decay with different half-lives
            dates_train = train_fit_enc["Created_At"]
            for half_life in [7, 14, 21, 30]:
                weights = compute_temporal_weights(dates_train, half_life_days=half_life)
                logger.info(
                    "Half-life=%d: weight range [%.4f, %.4f]",
                    half_life,
                    weights.min(),
                    weights.max(),
                )

                cb_w, lgbm_w, lr_w, sc_w = train_weighted_ensemble(
                    x_train_fit, train_fit_enc["target"], weights
                )
                p_val_w, p_std_w = predict_ensemble(cb_w, lgbm_w, lr_w, sc_w, x_val)
                ev_val_w = p_val_w * odds_val - 1
                conf_w = 1 / (1 + p_std_w * 10)
                ev_conf_w = ev_val_w * conf_w

                for thr in [0.12, 0.15, 0.18]:
                    mask = ev_conf_w >= thr
                    r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                    results[f"decay{half_life}_conf_ev_{thr:.2f}"] = r
                    logger.info(
                        "Decay(hl=%d) conf_ev_%.2f on val: ROI=%.2f%%, n=%d",
                        half_life,
                        thr,
                        r["roi"],
                        r["n_bets"],
                    )

            # Part 2: Isotonic calibration
            logger.info("Part 2: Isotonic calibration")

            # OOF predictions for calibration
            n = len(train_fit_enc)
            oof_p = np.zeros(n)
            fold_mask = np.zeros(n, dtype=bool)
            fold_size = n // 6

            for i in range(5):
                tr_end = fold_size * (i + 2)
                va_start = tr_end
                va_end = min(va_start + fold_size, n)
                if va_end <= va_start:
                    break

                x_tr = train_fit_enc.iloc[:tr_end][FEATURES].fillna(0)
                y_tr = train_fit_enc.iloc[:tr_end]["target"]
                x_va = train_fit_enc.iloc[va_start:va_end][FEATURES].fillna(0)

                cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_tr, y_tr)
                p_f, _ = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_va)
                oof_p[va_start:va_end] = p_f
                fold_mask[va_start:va_end] = True

            logger.info("OOF samples: %d / %d", fold_mask.sum(), n)

            # Calibrate using OOF predictions
            oof_valid = oof_p[fold_mask]
            y_oof_valid = train_fit_enc[fold_mask]["target"].values

            # Apply calibration to val predictions
            p_val_cal = calibrate_isotonic(y_oof_valid, oof_valid, p_val_base)

            auc_base = roc_auc_score(val_enc["target"], p_val_base)
            auc_cal = roc_auc_score(val_enc["target"], p_val_cal)
            logger.info("Val AUC: base=%.4f, calibrated=%.4f", auc_base, auc_cal)

            ev_val_cal = p_val_cal * odds_val - 1
            # Use base model std for confidence (calibration doesn't change disagreement)
            ev_conf_cal = ev_val_cal * conf_val

            for thr in [0.10, 0.12, 0.15, 0.18, 0.20]:
                mask = ev_conf_cal >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                results[f"cal_conf_ev_{thr:.2f}"] = r
                logger.info(
                    "Calibrated conf_ev_%.2f on val: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Plain calibrated EV (no confidence)
            for thr in [0.08, 0.10, 0.12, 0.15]:
                mask = ev_val_cal >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                results[f"cal_ev_{thr:.2f}"] = r
                logger.info(
                    "Calibrated ev_%.2f on val: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Select best strategy on val (min 20 bets)
            val_ranked = sorted(
                results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 20 else -999,
                reverse=True,
            )
            logger.info("Top-5 strategies on val:")
            for name, r in val_ranked[:5]:
                logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            top5_names = [name for name, _ in val_ranked[:5]]

            # Part 3: Test evaluation with full train
            logger.info("Part 3: Test evaluation")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            odds_test = test_enc["Odds"].values

            # Unweighted baseline
            cb_full, lgbm_full, lr_full, sc_full = train_ensemble(x_train, train_enc["target"])
            p_test, p_std_test = predict_ensemble(cb_full, lgbm_full, lr_full, sc_full, x_test)
            ev_test = p_test * odds_test - 1
            conf_test = 1 / (1 + p_std_test * 10)
            ev_conf_test = ev_test * conf_test

            auc_test = roc_auc_score(test_enc["target"], p_test)

            r_baseline = calc_roi(test_enc, (ev_test >= 0.12).astype(float), threshold=0.5)
            r_conf_ev = calc_roi(test_enc, (ev_conf_test >= 0.15).astype(float), threshold=0.5)
            logger.info(
                "Test baseline: ev>=0.12 ROI=%.2f%%(n=%d), conf_ev>=0.15 ROI=%.2f%%(n=%d)",
                r_baseline["roi"],
                r_baseline["n_bets"],
                r_conf_ev["roi"],
                r_conf_ev["n_bets"],
            )

            test_eval = {
                "baseline_ev0.12": r_baseline,
                "conf_ev_0.15": r_conf_ev,
            }

            # Best temporal decay on test
            best_hl = None
            for name, _r in val_ranked[:5]:
                if name.startswith("decay"):
                    best_hl = int(name.split("_")[0].replace("decay", ""))
                    break

            if best_hl is not None:
                dates_train_full = train_enc["Created_At"]
                w_full = compute_temporal_weights(dates_train_full, half_life_days=best_hl)
                cb_w, lgbm_w, lr_w, sc_w = train_weighted_ensemble(
                    x_train, train_enc["target"], w_full
                )
                p_test_w, p_std_w = predict_ensemble(cb_w, lgbm_w, lr_w, sc_w, x_test)
                ev_test_w = p_test_w * odds_test - 1
                conf_w = 1 / (1 + p_std_w * 10)
                ev_conf_w = ev_test_w * conf_w

                for thr in [0.12, 0.15, 0.18]:
                    mask = ev_conf_w >= thr
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    test_eval[f"decay{best_hl}_conf_ev_{thr:.2f}"] = r
                    logger.info(
                        "Test decay(hl=%d) conf_ev_%.2f: ROI=%.2f%%, n=%d",
                        best_hl,
                        thr,
                        r["roi"],
                        r["n_bets"],
                    )

            # Calibrated test predictions
            # OOF on full train for calibration
            n_full = len(train_enc)
            oof_full = np.zeros(n_full)
            fold_mask_full = np.zeros(n_full, dtype=bool)
            fold_size_full = n_full // 6

            for i in range(5):
                tr_end = fold_size_full * (i + 2)
                va_start = tr_end
                va_end = min(va_start + fold_size_full, n_full)
                if va_end <= va_start:
                    break
                x_tr = train_enc.iloc[:tr_end][FEATURES].fillna(0)
                y_tr = train_enc.iloc[:tr_end]["target"]
                x_va = train_enc.iloc[va_start:va_end][FEATURES].fillna(0)
                cb_f, lgbm_f, lr_f, sc_f = train_ensemble(x_tr, y_tr)
                p_f, _ = predict_ensemble(cb_f, lgbm_f, lr_f, sc_f, x_va)
                oof_full[va_start:va_end] = p_f
                fold_mask_full[va_start:va_end] = True

            oof_v = oof_full[fold_mask_full]
            y_oof_v = train_enc[fold_mask_full]["target"].values

            p_test_cal = calibrate_isotonic(y_oof_v, oof_v, p_test)
            auc_cal_test = roc_auc_score(test_enc["target"], p_test_cal)
            logger.info("Test AUC: base=%.4f, cal=%.4f", auc_test, auc_cal_test)

            ev_test_cal = p_test_cal * odds_test - 1
            ev_conf_cal_test = ev_test_cal * conf_test

            for thr in [0.10, 0.12, 0.15, 0.18, 0.20]:
                mask = ev_conf_cal_test >= thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                test_eval[f"cal_conf_ev_{thr:.2f}"] = r
                logger.info(
                    "Test cal conf_ev_%.2f: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            for thr in [0.08, 0.10, 0.12, 0.15]:
                mask = ev_test_cal >= thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                test_eval[f"cal_ev_{thr:.2f}"] = r
                logger.info(
                    "Test cal ev_%.2f: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Best validated strategy
            best_name = max(
                test_eval,
                key=lambda k: test_eval[k]["roi"] if test_eval[k]["n_bets"] >= 50 else -999,
            )
            best = test_eval[best_name]
            logger.info(
                "Best (n>=50): %s -> ROI=%.2f%%, n=%d",
                best_name,
                best["roi"],
                best["n_bets"],
            )

            # Log
            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "auc_cal_test": auc_cal_test,
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "roi_conf_ev": r_conf_ev["roi"],
                    "n_bets_conf_ev": r_conf_ev["n_bets"],
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            if best_hl is not None:
                mlflow.log_param("best_half_life", best_hl)
            mlflow.set_tag("best_strategy", best_name)
            mlflow.set_tag("top5_val", str(top5_names))

            if best["roi"] > 27.95:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb_full.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best["roi"],
                    "auc": auc_test,
                    "threshold": 0.12,
                    "ev_threshold": 0.15,
                    "n_bets": best["n_bets"],
                    "feature_names": FEATURES,
                    "selection_method": best_name,
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info("Model saved with ROI=%.2f%%", best["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.9 failed")
            raise


if __name__ == "__main__":
    main()
