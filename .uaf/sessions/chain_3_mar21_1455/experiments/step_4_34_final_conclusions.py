"""Step 4.34 — Final conclusions and summary metrics.

Consolidate all findings: compute definitive metrics for report.
This is the closing experiment of the research session.
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
mlflow.set_experiment("uaf/chain_3_mar21_1455")

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
    """Final conclusions experiment."""
    with mlflow.start_run(run_name="phase4/final_conclusions") as run:
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
                    "method": "final_conclusions",
                    "total_experiments": 34,
                }
            )

            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)
            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)

            cb, lgbm, lr, scaler = train_ensemble(x_train, train_enc["target"])
            p_t, s_t = predict_ensemble(cb, lgbm, lr, scaler, x_test)
            odds_test = test_enc["Odds"].values

            auc_test = roc_auc_score(test_enc["target"], p_t)

            ev_t = p_t * odds_test - 1
            conf_t = 1 / (1 + s_t * 10)
            ev_conf_t = ev_t * conf_t

            # Definitive strategy metrics
            logger.info("=== DEFINITIVE RESULTS (34 experiments) ===")

            # 1. Best headline ROI
            mask_confev = ev_conf_t >= 0.15
            r_confev = calc_roi(test_enc, mask_confev.astype(float), threshold=0.5)
            logger.info(
                "conf_ev_0.15: ROI=%.2f%%, n=%d (inflated by 1 extreme bet)",
                r_confev["roi"],
                r_confev["n_bets"],
            )

            # 2. Best robust
            mask_pmean = p_t >= 0.55
            r_pmean = calc_roi(test_enc, mask_pmean.astype(float), threshold=0.5)
            logger.info(
                "pmean_0.55: ROI=%.2f%%, n=%d (CV min=-3.46%%)",
                r_pmean["roi"],
                r_pmean["n_bets"],
            )

            # 3. Best Sharpe
            mask_ev005 = ev_t >= 0.05
            r_ev005 = calc_roi(test_enc, mask_ev005.astype(float), threshold=0.5)
            logger.info(
                "ev_0.05: ROI=%.2f%%, n=%d (CV sharpe=0.44)",
                r_ev005["roi"],
                r_ev005["n_bets"],
            )

            # 4. Capped (no extreme outliers)
            mask_capped = (ev_conf_t >= 0.15) & (odds_test <= 100)
            r_capped = calc_roi(test_enc, mask_capped.astype(float), threshold=0.5)
            logger.info(
                "conf_ev_0.15 capped 100: ROI=%.2f%%, n=%d",
                r_capped["roi"],
                r_capped["n_bets"],
            )

            # Profit concentration
            target = test_enc["target"].values
            pnl = np.where(
                target == 1,
                test_enc["USD"].values * (odds_test - 1),
                -test_enc["USD"].values,
            )
            selected_pnl = pnl[mask_confev]
            if len(selected_pnl) > 0:
                total_profit = selected_pnl.sum()
                sorted_pnl = np.sort(selected_pnl)[::-1]
                top1_pct = sorted_pnl[0] / total_profit * 100 if total_profit > 0 else 0
                top5_pct = sorted_pnl[:5].sum() / total_profit * 100 if total_profit > 0 else 0
                n_positive = (selected_pnl > 0).sum()
                n_negative = (selected_pnl < 0).sum()
                logger.info(
                    "Profit: top1=%.1f%%, top5=%.1f%%, positive=%d, negative=%d",
                    top1_pct,
                    top5_pct,
                    n_positive,
                    n_negative,
                )

            logger.info("=== KEY CONCLUSIONS ===")
            logger.info("1. AUC=%.4f stable across all configs", auc_test)
            logger.info("2. ROI 27.95%% driven by 1 bet at odds=490.9")
            logger.info("3. Realistic ROI: 0-2%% (CV-validated)")
            logger.info("4. No strategy has positive min fold ROI in CV")
            logger.info("5. Model discriminates (AUC>0.78) but edge too small for profit")

            mlflow.log_metrics(
                {
                    "auc_test": auc_test,
                    "roi_confev_015": r_confev["roi"],
                    "roi_pmean_055": r_pmean["roi"],
                    "roi_ev_005": r_ev005["roi"],
                    "roi_confev_015_cap100": r_capped["roi"],
                }
            )
            mlflow.set_tag("conclusion", "no_systematic_edge")
            mlflow.set_tag("realistic_roi", "0-2%")
            mlflow.set_tag("best_auc", f"{auc_test:.4f}")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.34 failed")
            raise


if __name__ == "__main__":
    main()
