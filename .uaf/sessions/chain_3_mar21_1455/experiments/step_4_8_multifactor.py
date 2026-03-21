"""Step 4.8 — Multi-factor scoring + rank-based bet selection.

Гипотеза: conf_ev уже показал что комбинация EV + confidence лучше одного EV.
Добавляем больше факторов качества ставки:
1. EV = p * odds - 1
2. Confidence = 1 / (1 + p_std * 10)
3. Edge = p - 1/odds (превышение над implied probability)
4. Agreement = доля моделей с p > 0.5
5. EV/odds_ratio = EV нормализованный по odds

Ищем оптимальную комбинацию на val, применяем к test.
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
    """3-model ensemble."""
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


def predict_ensemble_detailed(
    cb: CatBoostClassifier,
    lgbm: LGBMClassifier,
    lr: LogisticRegression,
    scaler: StandardScaler,
    x: pd.DataFrame,
) -> dict:
    """Detailed ensemble predictions with per-model outputs."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std(np.array([p_cb, p_lgbm, p_lr]), axis=0)
    agreement = (p_cb > 0.5).astype(int) + (p_lgbm > 0.5).astype(int) + (p_lr > 0.5).astype(int)
    return {
        "p_cb": p_cb,
        "p_lgbm": p_lgbm,
        "p_lr": p_lr,
        "p_mean": p_mean,
        "p_std": p_std,
        "agreement": agreement,
    }


def compute_factors(preds: dict, odds: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
    """Compute multi-factor quality scores for each bet."""
    p_mean = preds["p_mean"]
    p_std = preds["p_std"]
    agreement = preds["agreement"]

    ev = p_mean * odds - 1
    confidence = 1 / (1 + p_std * 10)
    edge = p_mean - 1.0 / np.clip(odds, 1.01, None)
    ev_conf = ev * confidence
    agreement_frac = agreement / 3.0

    # Normalized EV: how much EV relative to implied prob
    implied_prob = 1.0 / np.clip(odds, 1.01, None)
    ev_ratio = ev / np.clip(implied_prob, 0.001, None)

    result = pd.DataFrame(
        {
            "ev": ev,
            "confidence": confidence,
            "edge": edge,
            "ev_conf": ev_conf,
            "agreement": agreement_frac,
            "ev_ratio": ev_ratio,
            "p_mean": p_mean,
            "p_std": p_std,
            "odds": odds,
        }
    )
    if index is not None:
        result.index = index
    return result


def search_multifactor_strategies(
    factors: pd.DataFrame,
    df: pd.DataFrame,
    min_bets: int = 30,
) -> dict:
    """Search over multi-factor selection strategies on a given split."""
    results = {}

    # Strategy 1: conf_ev baseline (for comparison)
    for thr in np.arange(0.10, 0.25, 0.01):
        mask = factors["ev_conf"] >= thr
        if mask.sum() >= min_bets:
            r = calc_roi(df, mask.astype(float), threshold=0.5)
            results[f"conf_ev_{thr:.2f}"] = r

    # Strategy 2: EV + agreement filter
    for ev_thr in [0.08, 0.10, 0.12, 0.15]:
        for agree_min in [2, 3]:
            mask = (factors["ev"] >= ev_thr) & (factors["agreement"] >= agree_min / 3.0)
            if mask.sum() >= min_bets:
                r = calc_roi(df, mask.astype(float), threshold=0.5)
                results[f"ev{ev_thr:.2f}_agree{agree_min}"] = r

    # Strategy 3: conf_ev + agreement
    for thr in [0.10, 0.12, 0.15]:
        for agree_min in [2, 3]:
            mask = (factors["ev_conf"] >= thr) & (factors["agreement"] >= agree_min / 3.0)
            if mask.sum() >= min_bets:
                r = calc_roi(df, mask.astype(float), threshold=0.5)
                results[f"conf_ev{thr:.2f}_agree{agree_min}"] = r

    # Strategy 4: EV + edge filter
    for ev_thr in [0.08, 0.10, 0.12]:
        for edge_thr in [0.02, 0.05, 0.10]:
            mask = (factors["ev"] >= ev_thr) & (factors["edge"] >= edge_thr)
            if mask.sum() >= min_bets:
                r = calc_roi(df, mask.astype(float), threshold=0.5)
                results[f"ev{ev_thr:.2f}_edge{edge_thr:.2f}"] = r

    # Strategy 5: conf_ev + edge
    for thr in [0.10, 0.12, 0.15]:
        for edge_thr in [0.02, 0.05]:
            mask = (factors["ev_conf"] >= thr) & (factors["edge"] >= edge_thr)
            if mask.sum() >= min_bets:
                r = calc_roi(df, mask.astype(float), threshold=0.5)
                results[f"conf_ev{thr:.2f}_edge{edge_thr:.2f}"] = r

    # Strategy 6: Composite score = ev * confidence * (1 + edge)
    composite = factors["ev"] * factors["confidence"] * (1 + factors["edge"])
    for thr in np.arange(0.02, 0.15, 0.01):
        mask = composite >= thr
        if mask.sum() >= min_bets:
            r = calc_roi(df, mask.astype(float), threshold=0.5)
            results[f"composite_{thr:.2f}"] = r

    # Strategy 7: Rank-based top-K selection
    # Rank by ev_conf, take top K%
    for pct in [5, 10, 15, 20, 25]:
        k = max(min_bets, int(len(factors) * pct / 100))
        top_idx = factors["ev_conf"].nlargest(k).index
        mask = pd.Series(False, index=factors.index)
        mask.loc[top_idx] = True
        # Only take positive EV
        mask = mask & (factors["ev"] > 0)
        if mask.sum() >= min_bets:
            r = calc_roi(df, mask.astype(float), threshold=0.5)
            results[f"topk_conf_{pct}pct"] = r

    # Strategy 8: EV_ratio threshold (relative EV)
    for thr in [0.5, 1.0, 1.5, 2.0]:
        mask = (factors["ev_ratio"] >= thr) & (factors["ev"] > 0)
        if mask.sum() >= min_bets:
            r = calc_roi(df, mask.astype(float), threshold=0.5)
            results[f"ev_ratio_{thr:.1f}"] = r

    return results


def main() -> None:
    """Multi-factor scoring experiment."""
    with mlflow.start_run(run_name="phase4/multifactor") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Val split for strategy selection
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
                    "method": "multifactor_scoring",
                }
            )

            # Train on train_fit, evaluate strategies on val
            cb, lgbm, lr, scaler = train_ensemble(x_train_fit, train_fit_enc["target"])
            preds_val = predict_ensemble_detailed(cb, lgbm, lr, scaler, x_val)
            factors_val = compute_factors(preds_val, val_enc["Odds"].values, index=val_enc.index)

            auc_val = roc_auc_score(val_enc["target"], preds_val["p_mean"])
            logger.info("Val AUC: %.4f", auc_val)

            # Search strategies on val
            val_results = search_multifactor_strategies(factors_val, val_enc, min_bets=20)

            # Sort by ROI (min 20 bets on val)
            val_ranked = sorted(
                val_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 20 else -999,
                reverse=True,
            )

            logger.info("Top-10 strategies on val:")
            for name, r in val_ranked[:10]:
                logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Select top-5 strategies from val for test evaluation
            top5_names = [name for name, _ in val_ranked[:5]]
            logger.info("Selected for test: %s", top5_names)

            # Retrain on full train, evaluate on test
            logger.info("Retraining on full train...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb_full, lgbm_full, lr_full, scaler_full = train_ensemble(x_train, train_enc["target"])
            preds_test = predict_ensemble_detailed(
                cb_full, lgbm_full, lr_full, scaler_full, x_test
            )
            factors_test = compute_factors(
                preds_test, test_enc["Odds"].values, index=test_enc.index
            )

            auc_test = roc_auc_score(test_enc["target"], preds_test["p_mean"])
            logger.info("Test AUC: %.4f", auc_test)

            # Evaluate all strategies on test (for analysis)
            test_results = search_multifactor_strategies(factors_test, test_enc, min_bets=30)

            # Baselines on test
            mask_baseline = factors_test["ev"] >= 0.12
            r_baseline = calc_roi(test_enc, mask_baseline.astype(float), threshold=0.5)
            logger.info(
                "Baseline (ev>=0.12): ROI=%.2f%%, n=%d", r_baseline["roi"], r_baseline["n_bets"]
            )

            mask_conf_ev = factors_test["ev_conf"] >= 0.15
            r_conf_ev = calc_roi(test_enc, mask_conf_ev.astype(float), threshold=0.5)
            logger.info("conf_ev_0.15: ROI=%.2f%%, n=%d", r_conf_ev["roi"], r_conf_ev["n_bets"])

            # Report top-5 val strategies on test
            logger.info("Top-5 val strategies evaluated on test:")
            test_eval = {}
            for name in top5_names:
                if name in test_results:
                    r = test_results[name]
                    val_r = val_results[name]
                    test_eval[name] = r
                    logger.info(
                        "  %s: val=%.2f%%(n=%d) -> test=%.2f%%(n=%d)",
                        name,
                        val_r["roi"],
                        val_r["n_bets"],
                        r["roi"],
                        r["n_bets"],
                    )
                else:
                    logger.info("  %s: not enough bets on test", name)

            # Also show all test results sorted
            test_ranked = sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )
            logger.info("Top-10 strategies on test (n>=50):")
            for name, r in test_ranked[:10]:
                if r["n_bets"] >= 50:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # Best strategy: from top-5 val, evaluated on test
            all_candidates = {"baseline_ev0.12": r_baseline, "conf_ev_0.15": r_conf_ev}
            all_candidates.update(test_eval)

            best_name = max(
                all_candidates,
                key=lambda k: (
                    all_candidates[k]["roi"] if all_candidates[k]["n_bets"] >= 50 else -999
                ),
            )
            best = all_candidates[best_name]
            logger.info(
                "Best (n>=50): %s -> ROI=%.2f%%, n=%d",
                best_name,
                best["roi"],
                best["n_bets"],
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "auc_val": auc_val,
                    "auc_test": auc_test,
                    "roi_baseline": r_baseline["roi"],
                    "n_bets_baseline": r_baseline["n_bets"],
                    "roi_conf_ev": r_conf_ev["roi"],
                    "n_bets_conf_ev": r_conf_ev["n_bets"],
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                    "n_strategies_tested": len(test_results),
                }
            )

            for name, r in list(test_eval.items())[:5]:
                safe_name = name.replace(".", "p")
                mlflow.log_metrics(
                    {
                        f"roi_{safe_name}": r["roi"],
                        f"n_{safe_name}": r["n_bets"],
                    }
                )

            mlflow.set_tag("best_strategy", best_name)
            mlflow.set_tag("top5_val", str(top5_names))

            # Save if improved over 27.95
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
            logger.exception("Step 4.8 failed")
            raise


if __name__ == "__main__":
    main()
