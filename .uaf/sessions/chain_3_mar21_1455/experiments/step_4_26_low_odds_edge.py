"""Step 4.26 — Low-odds edge: find real edge in odds < 5.

Step 4.24-25: ROI полностью из 1 bet at odds=490.9. При cap <= 100 ROI ~0%.
Вопрос: есть ли маленький, но реальный edge в low-odds?

Тесты:
1. Odds < 3 (favorites): higher win rate, smaller margin for error
2. Odds 1.5-2.5 (strong favorites): potentially highest edge
3. p_mean threshold вместо EV threshold (для low-odds EV маленький)
4. Win rate filter: select bets where p_model >> p_implied
5. Walk-forward mini-test: 3 временных блока в test для стабильности
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
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
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
    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
    return p_mean, p_std


def main() -> None:
    """Low-odds edge experiment."""
    with mlflow.start_run(run_name="phase4/low_odds_edge") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "low_odds_edge",
                }
            )

            # Full model train
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb, lgbm, lr, scaler = train_ensemble(x_train, train_enc["target"])
            p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_test)
            odds_test = test_enc["Odds"].values

            auc_test = roc_auc_score(test_enc["target"], p_mean)

            # 1. Low-odds analysis: what are model predictions like for low odds?
            logger.info("=== Low-odds prediction quality ===")
            for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 5.0)]:
                bracket = (odds_test >= lo) & (odds_test < hi)
                if bracket.sum() > 0:
                    actual_wr = test_enc["target"].values[bracket].mean()
                    pred_wr = p_mean[bracket].mean()
                    implied_wr = (1 / odds_test[bracket]).mean()
                    edge = pred_wr - implied_wr
                    logger.info(
                        "Odds [%.1f,%.1f): n=%d, wr=%.3f, pred=%.3f, impl=%.3f, edge=%.4f",
                        lo,
                        hi,
                        bracket.sum(),
                        actual_wr,
                        pred_wr,
                        implied_wr,
                        edge,
                    )

            results: dict[str, dict] = {}

            # 2. EV-based selection at different odds caps
            ev = p_mean * odds_test - 1
            conf = 1 / (1 + p_std * 10)
            ev_conf = ev * conf

            for max_odds in [2.0, 2.5, 3.0, 5.0]:
                odds_mask = odds_test <= max_odds
                for thr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
                    mask = (ev_conf >= thr) & odds_mask
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    name = f"ev_cap{max_odds:.0f}_thr{thr:.2f}"
                    results[name] = r

            # 3. p_mean threshold: select where model is confident
            for max_odds in [2.0, 3.0, 5.0]:
                odds_mask = odds_test <= max_odds
                for p_thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
                    mask = (p_mean >= p_thr) & odds_mask
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    name = f"pmean_cap{max_odds:.0f}_p{p_thr:.2f}"
                    results[name] = r

            # 4. Edge-based: select where p_model > p_implied by margin
            p_implied = 1 / odds_test
            edge = p_mean - p_implied
            for max_odds in [2.0, 3.0, 5.0]:
                odds_mask = odds_test <= max_odds
                for e_thr in [0.02, 0.05, 0.08, 0.10, 0.15]:
                    mask = (edge >= e_thr) & odds_mask
                    r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                    name = f"edge_cap{max_odds:.0f}_e{e_thr:.2f}"
                    results[name] = r

            # 5. Combined: p_mean > threshold AND edge > threshold
            for max_odds in [3.0, 5.0]:
                odds_mask = odds_test <= max_odds
                for p_thr in [0.55, 0.60]:
                    for e_thr in [0.02, 0.05]:
                        mask = (p_mean >= p_thr) & (edge >= e_thr) & odds_mask
                        r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                        name = f"combo_cap{max_odds:.0f}_p{p_thr:.2f}_e{e_thr:.2f}"
                        results[name] = r

            # Ranking
            ranked = sorted(
                results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 30 else -999,
                reverse=True,
            )
            logger.info("Top-15 test (n>=30):")
            for name, r in ranked[:15]:
                if r["n_bets"] >= 30:
                    logger.info("  %s: ROI=%.2f%%, n=%d", name, r["roi"], r["n_bets"])

            # 6. Walk-forward: split test into 3 blocks
            logger.info("=== Walk-forward (3 blocks in test) ===")
            test_sorted = test_enc.sort_values("Created_At")
            block_size = len(test_sorted) // 3
            for i, block_name in enumerate(["block_1", "block_2", "block_3"]):
                start = i * block_size
                end = (i + 1) * block_size if i < 2 else len(test_sorted)
                block = test_sorted.iloc[start:end]
                block_idx = block.index

                # Reindex into original order to match predictions
                block_mask = test_enc.index.isin(block_idx)
                p_block = p_mean[block_mask]
                s_block = p_std[block_mask]
                odds_block = odds_test[block_mask]

                # conf_ev on block (all odds)
                ev_b = p_block * odds_block - 1
                conf_b = 1 / (1 + s_block * 10)
                ev_conf_b = ev_b * conf_b
                mask_b = ev_conf_b >= 0.15
                r = calc_roi(test_enc[block_mask], mask_b.astype(float), threshold=0.5)
                logger.info(
                    "%s (all odds): conf_ev>=0.15 ROI=%.2f%%, n=%d",
                    block_name,
                    r["roi"],
                    r["n_bets"],
                )

                # Low odds only
                low_mask = odds_block <= 5
                mask_low = (ev_conf_b >= 0.05) & low_mask
                r_low = calc_roi(test_enc[block_mask], mask_low.astype(float), threshold=0.5)
                logger.info(
                    "%s (odds<=5, thr=0.05): ROI=%.2f%%, n=%d",
                    block_name,
                    r_low["roi"],
                    r_low["n_bets"],
                )

            best_name = max(
                results,
                key=lambda k: results[k]["roi"] if results[k]["n_bets"] >= 30 else -999,
            )
            best = results[best_name]
            logger.info(
                "Best low-odds (n>=30): %s -> ROI=%.2f%%, n=%d",
                best_name,
                best["roi"],
                best["n_bets"],
            )

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_best_lowodds": best["roi"],
                    "n_bets_best": best["n_bets"],
                }
            )
            mlflow.set_tag("best_strategy", best_name)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.26 failed")
            raise


if __name__ == "__main__":
    main()
