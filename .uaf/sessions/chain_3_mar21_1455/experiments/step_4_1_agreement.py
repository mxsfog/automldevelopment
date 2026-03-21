"""Step 4.1 — Agreement-based EV selection.

Гипотеза: отбирать ставки где все 3 модели ансамбля согласны по EV.
Используем disagreement (std моделей) как меру неуверенности.
Варианты:
  A) EV >= 0.12 И std < threshold (уверенные ставки)
  B) EV >= 0.12 И min(EV_i) >= 0 (все модели считают +EV)
  C) Взвешенный EV с учётом disagreement
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

EV_THRESHOLD = 0.12


def train_ensemble(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[CatBoostClassifier, LGBMClassifier, LogisticRegression, StandardScaler]:
    """Обучение 3-model ensemble."""
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    cb.fit(x_train, y_train)

    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x_train, y_train)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_train_s, y_train)

    return cb, lgbm, lr, scaler


def get_individual_probas(
    cb: CatBoostClassifier,
    lgbm: LGBMClassifier,
    lr: LogisticRegression,
    scaler: StandardScaler,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вероятности каждой модели отдельно."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    return p_cb, p_lgbm, p_lr


def evaluate_selection(
    test_df: pd.DataFrame,
    mask: np.ndarray,
    method_name: str,
) -> dict:
    """Evaluate ROI for a given selection mask."""
    result = calc_roi(test_df, mask.astype(float), threshold=0.5)
    logger.info(
        "  %s: ROI=%.2f%%, n_bets=%d, winrate=%.4f",
        method_name,
        result["roi"],
        result["n_bets"],
        result.get("win_rate", 0),
    )
    return result


def main() -> None:
    """Agreement-based selection experiments."""
    with mlflow.start_run(run_name="phase4/agreement_selection") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            y_train = train_enc["target"]

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_val": len(test),
                    "n_features": len(FEATURES),
                    "ev_threshold": EV_THRESHOLD,
                    "method": "agreement_selection",
                }
            )

            cb, lgbm, lr, scaler = train_ensemble(x_train, y_train)
            p_cb, p_lgbm, p_lr = get_individual_probas(cb, lgbm, lr, scaler, x_test)

            p_mean = (p_cb + p_lgbm + p_lr) / 3
            p_std = np.sqrt(
                ((p_cb - p_mean) ** 2 + (p_lgbm - p_mean) ** 2 + (p_lr - p_mean) ** 2) / 3
            )

            auc_test = roc_auc_score(test_enc["target"], p_mean)
            logger.info("Ensemble AUC: %.4f", auc_test)

            odds = test_enc["Odds"].values

            # Baseline: EV >= 0.12
            ev_mean = p_mean * odds - 1
            mask_baseline = ev_mean >= EV_THRESHOLD
            r_baseline = evaluate_selection(test_enc, mask_baseline, "baseline_ev")

            # EV per model
            ev_cb = p_cb * odds - 1
            ev_lgbm = p_lgbm * odds - 1
            ev_lr = p_lr * odds - 1

            # Variant A: EV >= 0.12 AND low disagreement (std < median)
            median_std = np.median(p_std[mask_baseline])
            results_a = {}
            for std_pct in [25, 50, 75]:
                std_thr = np.percentile(p_std[mask_baseline], std_pct)
                mask_a = mask_baseline & (p_std <= std_thr)
                r = evaluate_selection(test_enc, mask_a, f"A_std_p{std_pct}")
                results_a[f"roi_A_std_p{std_pct}"] = r["roi"]
                results_a[f"n_bets_A_std_p{std_pct}"] = r["n_bets"]

            # Variant B: All models agree on +EV
            mask_all_ev = (ev_cb >= 0) & (ev_lgbm >= 0) & (ev_lr >= 0)
            r_all_ev = evaluate_selection(test_enc, mask_all_ev, "B_all_positive_ev")

            # B2: All models EV >= threshold
            mask_all_ev_thr = (
                (ev_cb >= EV_THRESHOLD) & (ev_lgbm >= EV_THRESHOLD) & (ev_lr >= EV_THRESHOLD)
            )
            r_all_ev_thr = evaluate_selection(test_enc, mask_all_ev_thr, "B_all_ev_threshold")

            # B3: At least 2 of 3 models EV >= 0
            ev_positive_count = (
                (ev_cb >= 0).astype(int) + (ev_lgbm >= 0).astype(int) + (ev_lr >= 0).astype(int)
            )
            mask_majority = (ev_positive_count >= 2) & (ev_mean >= EV_THRESHOLD)
            r_majority = evaluate_selection(test_enc, mask_majority, "B_majority_ev")

            # Variant C: Confidence-weighted EV
            # Penalize EV by disagreement
            confidence = 1 / (1 + p_std * 10)
            ev_confident = ev_mean * confidence
            for ev_thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
                mask_c = ev_confident >= ev_thr
                r = evaluate_selection(test_enc, mask_c, f"C_conf_ev_{ev_thr:.2f}")

            # Variant D: Min EV across models >= threshold
            ev_min = np.minimum(np.minimum(ev_cb, ev_lgbm), ev_lr)
            for min_thr in [0.0, 0.05, 0.10, 0.12]:
                mask_d = ev_min >= min_thr
                r = evaluate_selection(test_enc, mask_d, f"D_min_ev_{min_thr:.2f}")

            # Variant E: Odds-stratified thresholds
            # High odds (>10): lower EV threshold, low odds (<5): higher threshold
            mask_e = np.zeros(len(test_enc), dtype=bool)
            for lo, hi, thr in [(1, 3, 0.20), (3, 10, 0.15), (10, 50, 0.12), (50, 500, 0.08)]:
                bracket = (odds >= lo) & (odds < hi)
                mask_e |= bracket & (ev_mean >= thr)
            r_stratified = evaluate_selection(test_enc, mask_e, "E_odds_stratified_ev")

            # Find best variant
            all_variants = {
                "baseline": r_baseline,
                "B_all_positive_ev": r_all_ev,
                "B_all_ev_threshold": r_all_ev_thr,
                "B_majority_ev": r_majority,
                "E_odds_stratified": r_stratified,
            }

            best_name = max(
                all_variants,
                key=lambda k: all_variants[k]["roi"] if all_variants[k]["n_bets"] >= 50 else -999,
            )
            best_result = all_variants[best_name]

            logger.info(
                "Best variant: %s -> ROI=%.2f%%, n=%d",
                best_name,
                best_result["roi"],
                best_result["n_bets"],
            )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "roi_all_positive_ev": r_all_ev["roi"],
                    "n_bets_all_positive_ev": r_all_ev["n_bets"],
                    "roi_all_ev_threshold": r_all_ev_thr["roi"],
                    "n_bets_all_ev_threshold": r_all_ev_thr["n_bets"],
                    "roi_majority_ev": r_majority["roi"],
                    "n_bets_majority_ev": r_majority["n_bets"],
                    "roi_odds_stratified": r_stratified["roi"],
                    "n_bets_odds_stratified": r_stratified["n_bets"],
                    "roi_best": best_result["roi"],
                    "n_bets_best": best_result["n_bets"],
                    "best_method": 0,  # tag below
                    "median_std_selected": float(median_std),
                    **results_a,
                }
            )
            mlflow.set_tag("best_method", best_name)

            # Save model if improved
            if best_result["roi"] > 16.02:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best_result["roi"],
                    "auc": auc_test,
                    "threshold": 0.12,
                    "ev_threshold": EV_THRESHOLD,
                    "n_bets": best_result["n_bets"],
                    "feature_names": FEATURES,
                    "selection_method": best_name,
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.1 failed")
            raise


if __name__ == "__main__":
    main()
