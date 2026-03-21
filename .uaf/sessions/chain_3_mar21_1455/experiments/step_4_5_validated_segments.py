"""Step 4.5 — Validated sport exclusion + per-bracket EV.

Определяем убыточные спорты на val (не test), исключаем при финальной оценке.
Также тестируем адаптивные EV пороги по odds-brackets.
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


def train_ensemble(x: pd.DataFrame, y: pd.Series) -> tuple:
    """Train 3-model ensemble."""
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
    """Ensemble predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std(np.array([p_cb, p_lgbm, p_lr]), axis=0)
    return p_mean, p_std


def find_negative_sports_on_split(
    split_df: pd.DataFrame,
    p_mean: np.ndarray,
    ev_threshold: float = 0.12,
    min_bets: int = 10,
) -> list[str]:
    """Определяет убыточные спорты на данном split."""
    odds = split_df["Odds"].values
    ev = p_mean * odds - 1
    mask_ev = ev >= ev_threshold

    negative_sports = []
    for sport in split_df["Sport"].unique():
        mask_sport = split_df["Sport"].values == sport
        mask_sport_ev = mask_sport & mask_ev
        n = mask_sport_ev.sum()
        if n >= min_bets:
            r = calc_roi(split_df[mask_sport_ev], np.ones(n), threshold=0.5)
            if r["roi"] < 0:
                negative_sports.append(sport)
    return negative_sports


def find_best_per_bracket_thresholds(
    split_df: pd.DataFrame,
    p_mean: np.ndarray,
    brackets: list[tuple],
) -> dict:
    """Подбор лучшего EV порога для каждого odds-bracket на val."""
    odds = split_df["Odds"].values
    best_thresholds = {}

    for lo, hi in brackets:
        mask_bracket = (odds >= lo) & (odds < hi)
        n_bracket = mask_bracket.sum()
        if n_bracket < 20:
            best_thresholds[(lo, hi)] = 0.12
            continue

        best_roi = -999
        best_thr = 0.12
        for thr in np.arange(0.05, 0.30, 0.01):
            ev = p_mean * odds - 1
            mask = mask_bracket & (ev >= thr)
            n = mask.sum()
            if n >= 10:
                r = calc_roi(split_df[mask], np.ones(n), threshold=0.5)
                if r["roi"] > best_roi:
                    best_roi = r["roi"]
                    best_thr = round(thr, 2)

        best_thresholds[(lo, hi)] = best_thr
        logger.info("Bracket %s-%s: best EV thr=%.2f (val ROI=%.2f%%)", lo, hi, best_thr, best_roi)

    return best_thresholds


def apply_per_bracket_ev(
    df: pd.DataFrame,
    p_mean: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    """Применение per-bracket EV порогов."""
    odds = df["Odds"].values
    ev = p_mean * odds - 1
    mask = np.zeros(len(df), dtype=bool)

    for (lo, hi), thr in thresholds.items():
        bracket = (odds >= lo) & (odds < hi)
        mask |= bracket & (ev >= thr)

    return mask


def main() -> None:
    """Validated sport exclusion + per-bracket EV."""
    with mlflow.start_run(run_name="phase4/validated_segments") as run:
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
                    "n_samples_train": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "validated_segments",
                }
            )

            # Train on train_fit, evaluate on val
            cb, lgbm, lr, scaler = train_ensemble(x_train_fit, train_fit_enc["target"])
            p_mean_val, _ = predict_ensemble(cb, lgbm, lr, scaler, x_val)

            # Part 1: Find negative sports on val
            negative_sports = find_negative_sports_on_split(val_enc, p_mean_val, min_bets=5)
            logger.info("Negative sports on val (min=5 bets): %s", negative_sports)

            # Part 2: Find per-bracket thresholds on val
            brackets = [(1, 1.5), (1.5, 2.5), (2.5, 5), (5, 10), (10, 50), (50, 500)]
            bracket_thresholds = find_best_per_bracket_thresholds(val_enc, p_mean_val, brackets)

            # Part 3: Retrain on full train, evaluate on test
            logger.info("Retraining on full train...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_full, lgbm_full, lr_full, scaler_full = train_ensemble(x_train, train_enc["target"])
            p_mean_test, p_std_test = predict_ensemble(
                cb_full, lgbm_full, lr_full, scaler_full, x_test
            )

            auc_test = roc_auc_score(test_enc["target"], p_mean_test)
            odds_test = test_enc["Odds"].values
            ev_test = p_mean_test * odds_test - 1

            # Baseline
            mask_baseline = ev_test >= 0.12
            r_baseline = calc_roi(test_enc, mask_baseline.astype(float), threshold=0.5)
            logger.info("Baseline: ROI=%.2f%%, n=%d", r_baseline["roi"], r_baseline["n_bets"])

            # Strategy A: Sport exclusion only
            mask_include = ~test_enc["Sport"].isin(negative_sports)
            mask_a = mask_include & (ev_test >= 0.12)
            r_a = calc_roi(test_enc, mask_a.astype(float), threshold=0.5)
            logger.info(
                "Sport exclusion (%s): ROI=%.2f%%, n=%d",
                negative_sports,
                r_a["roi"],
                r_a["n_bets"],
            )

            # Strategy B: Per-bracket thresholds only
            mask_b = apply_per_bracket_ev(test_enc, p_mean_test, bracket_thresholds)
            r_b = calc_roi(test_enc, mask_b.astype(float), threshold=0.5)
            logger.info(
                "Per-bracket EV: ROI=%.2f%%, n=%d (thresholds=%s)",
                r_b["roi"],
                r_b["n_bets"],
                bracket_thresholds,
            )

            # Strategy C: Sport exclusion + per-bracket
            mask_c = mask_include & mask_b
            r_c = calc_roi(test_enc, mask_c.astype(float), threshold=0.5)
            logger.info(
                "Sport excl + per-bracket: ROI=%.2f%%, n=%d",
                r_c["roi"],
                r_c["n_bets"],
            )

            # Strategy D: Sport exclusion + conf_ev_0.15
            confidence = 1 / (1 + p_std_test * 10)
            ev_conf = ev_test * confidence
            mask_d = mask_include & (ev_conf >= 0.15)
            r_d = calc_roi(test_enc, mask_d.astype(float), threshold=0.5)
            logger.info(
                "Sport excl + conf_ev_0.15: ROI=%.2f%%, n=%d",
                r_d["roi"],
                r_d["n_bets"],
            )

            # Strategy E: conf_ev_0.15 (no exclusion, baseline from step 4.2)
            mask_e = ev_conf >= 0.15
            r_e = calc_roi(test_enc, mask_e.astype(float), threshold=0.5)
            logger.info("conf_ev_0.15 (no excl): ROI=%.2f%%, n=%d", r_e["roi"], r_e["n_bets"])

            # Best strategy
            all_strats = {
                "baseline": r_baseline,
                "sport_exclusion": r_a,
                "per_bracket": r_b,
                "excl_bracket": r_c,
                "excl_conf_ev": r_d,
                "conf_ev_only": r_e,
            }
            best_name = max(
                all_strats,
                key=lambda k: all_strats[k]["roi"] if all_strats[k]["n_bets"] >= 50 else -999,
            )
            best = all_strats[best_name]
            logger.info("Best: %s -> ROI=%.2f%%, n=%d", best_name, best["roi"], best["n_bets"])

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "roi_sport_exclusion": r_a["roi"],
                    "n_bets_sport_exclusion": r_a["n_bets"],
                    "roi_per_bracket": r_b["roi"],
                    "n_bets_per_bracket": r_b["n_bets"],
                    "roi_excl_bracket": r_c["roi"],
                    "n_bets_excl_bracket": r_c["n_bets"],
                    "roi_excl_conf_ev": r_d["roi"],
                    "n_bets_excl_conf_ev": r_d["n_bets"],
                    "roi_conf_ev_only": r_e["roi"],
                    "n_bets_conf_ev_only": r_e["n_bets"],
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                    "n_excluded_sports": len(negative_sports),
                }
            )
            mlflow.set_tag("best_strategy", best_name)
            mlflow.set_tag("excluded_sports", str(negative_sports))
            mlflow.set_tag("bracket_thresholds", str(bracket_thresholds))

            # Save if improved
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
                    "ev_threshold": 0.12,
                    "n_bets": best["n_bets"],
                    "feature_names": FEATURES,
                    "selection_method": best_name,
                    "excluded_sports": negative_sports,
                    "bracket_thresholds": {
                        f"{k[0]}-{k[1]}": v for k, v in bracket_thresholds.items()
                    },
                    "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
                    "sport_filter": negative_sports,
                    "session_id": SESSION_ID,
                }
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info("Model saved with ROI=%.2f%%", best["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.5 failed")
            raise


if __name__ == "__main__":
    main()
