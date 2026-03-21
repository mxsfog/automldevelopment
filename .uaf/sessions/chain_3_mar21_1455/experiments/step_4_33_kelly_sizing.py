"""Step 4.33 — Kelly criterion bet sizing with fractional Kelly.

Instead of flat betting, use Kelly fraction to size bets proportional to edge.
Full Kelly is aggressive; test 1/4, 1/2, 3/4 Kelly.
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


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Compute Kelly criterion fraction: f = (p*b - q) / b where b = odds-1."""
    b = odds - 1
    q = 1 - p
    f = (p * b - q) / b
    return np.clip(f, 0, 1)


def simulate_kelly(
    df: pd.DataFrame,
    p: np.ndarray,
    odds: np.ndarray,
    mask: np.ndarray,
    kelly_frac: float,
    initial_bankroll: float = 100000.0,
) -> dict:
    """Simulate Kelly-sized betting on selected bets."""
    bankroll = initial_bankroll
    n_bets = 0
    max_drawdown = 0.0
    peak = initial_bankroll

    target = df["target"].values
    stakes = df["USD"].values

    for i in range(len(df)):
        if not mask[i]:
            continue
        f = kelly_fraction(np.array([p[i]]), np.array([odds[i]]))[0]
        bet_size = bankroll * f * kelly_frac
        bet_size = min(bet_size, stakes[i])
        if bet_size < 1:
            continue

        n_bets += 1
        if target[i] == 1:
            bankroll += bet_size * (odds[i] - 1)
        else:
            bankroll -= bet_size

        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak
        max_drawdown = max(max_drawdown, drawdown)

    roi = (bankroll - initial_bankroll) / initial_bankroll * 100
    return {
        "roi": round(roi, 2),
        "n_bets": n_bets,
        "final_bankroll": round(bankroll, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
    }


def main() -> None:
    """Kelly criterion sizing experiment."""
    with mlflow.start_run(run_name="phase4/kelly_sizing") as run:
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
                    "method": "kelly_sizing",
                    "kelly_fractions": "0.25,0.50,0.75,1.00",
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
            logger.info("AUC test: %.4f", auc_test)

            ev_t = p_t * odds_test - 1
            conf_t = 1 / (1 + s_t * 10)
            ev_conf_t = ev_t * conf_t

            # Selection strategies
            selections = {
                "confev_0.15": ev_conf_t >= 0.15,
                "ev_0.05": ev_t >= 0.05,
                "pmean_0.55": p_t >= 0.55,
            }

            logger.info("=== Kelly sizing results ===")
            for sel_name, mask in selections.items():
                for kf in [0.25, 0.50, 0.75, 1.00]:
                    r = simulate_kelly(test_enc, p_t, odds_test, mask, kf)
                    name = f"{sel_name}_kelly{kf:.2f}"
                    logger.info(
                        "  %s: ROI=%.2f%%, n=%d, drawdown=%.1f%%",
                        name,
                        r["roi"],
                        r["n_bets"],
                        r["max_drawdown"],
                    )
                    mlflow.log_metrics(
                        {
                            f"kelly_roi_{sel_name}_{kf}": r["roi"],
                            f"kelly_dd_{sel_name}_{kf}": r["max_drawdown"],
                        }
                    )

            mlflow.log_metrics({"auc_test": auc_test})
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.33 failed")
            raise


if __name__ == "__main__":
    main()
