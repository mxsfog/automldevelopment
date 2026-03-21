"""Step 4.39 -- Alpha=0.4 variants: 4m vs 5m, risk analysis, odds cap impact.

Alpha=0.4 is the new best Sharpe (16.82). Test with 5-model and analyze risk profile.
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

FEATURES_ENC = [
    "Odds", "USD", "Is_Parlay", "Outcomes_Count",
    "ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV",
    "ML_Winrate_Diff", "ML_Rating_Diff",
    "Outcome_Odds", "n_outcomes", "mean_outcome_odds",
    "max_outcome_odds", "min_outcome_odds",
    "Sport_target_enc", "Sport_count_enc",
    "Market_target_enc", "Market_count_enc",
]

SEEDS = [42, 123, 777, 2024, 31337]


def run_wf(
    df: pd.DataFrame,
    retrain_schedule: list,
    seed: int,
    n_models: int,
    alpha: float,
    odds_cap: float | None = None,
    return_details: bool = False,
) -> dict:
    """Single WF pass."""
    random.seed(seed)
    np.random.seed(seed)

    cum_n = 0
    cum_profit = 0.0
    n_pos = 0
    n_blocks = 0
    block_details: list[dict] = []

    for test_week in retrain_schedule:
        train_mask = df["year_week"] < test_week
        test_mask = df["year_week"] == test_week

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        if len(test_df) == 0:
            continue

        train_enc, _ = add_sport_market_features(train_df, train_df)
        test_enc, _ = add_sport_market_features(test_df, train_enc)

        x_train = train_enc[FEATURES_ENC].fillna(0)
        y_train = train_enc["target"]
        x_test = test_enc[FEATURES_ENC].fillna(0)

        cb = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            random_seed=seed, verbose=0,
        )
        cb.fit(x_train, y_train)

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=seed, verbose=-1, min_child_samples=50,
        )
        lgbm.fit(x_train, y_train)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        lr.fit(x_train_s, y_train)

        preds = [
            cb.predict_proba(x_test)[:, 1],
            lgbm.predict_proba(x_test)[:, 1],
            lr.predict_proba(x_test_s)[:, 1],
        ]

        if n_models >= 4:
            xgb = XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                random_state=seed, verbosity=0, min_child_weight=50,
                use_label_encoder=False, eval_metric="logloss",
            )
            xgb.fit(x_train, y_train)
            preds.append(xgb.predict_proba(x_test)[:, 1])

        if n_models >= 5:
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=500, random_state=seed, early_stopping=True,
                validation_fraction=0.1,
            )
            mlp.fit(x_train_s, y_train)
            preds.append(mlp.predict_proba(x_test_s)[:, 1])

        p_model = np.mean(preds, axis=0)
        p_std = np.std(preds, axis=0)

        odds = test_enc["Odds"].values
        p_implied = 1.0 / odds
        p_final = alpha * p_model + (1 - alpha) * p_implied

        ev = p_final * odds - 1
        mask = (ev >= 0.05) & (p_std <= 0.02)

        if odds_cap is not None:
            mask = mask & (odds <= odds_cap)

        n = mask.sum()
        n_blocks += 1
        block_roi = 0.0
        if n > 0:
            sel = test_enc[mask]
            payout = (sel["target"] * sel["Odds"]).sum()
            block_roi = (payout - n) / n * 100
            cum_n += n
            cum_profit += n * block_roi / 100
            if block_roi > 0:
                n_pos += 1

            if return_details:
                sel_odds = sel["Odds"].values
                block_details.append({
                    "week": test_week,
                    "n": n,
                    "roi": round(block_roi, 2),
                    "mean_odds": round(float(sel_odds.mean()), 2),
                    "max_odds": round(float(sel_odds.max()), 2),
                    "pct_odds_gt50": round(float((sel_odds > 50).mean() * 100), 1),
                })

    overall_roi = (cum_profit / cum_n * 100) if cum_n > 0 else 0.0
    result = {
        "roi": round(overall_roi, 2),
        "n": cum_n,
        "n_pos": n_pos,
        "n_blocks": n_blocks,
    }
    if return_details:
        result["blocks"] = block_details
    return result


def main() -> None:
    """Alpha=0.4 variants and risk analysis."""
    with mlflow.start_run(run_name="phase4/alpha04_variants") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            df["year_week"] = df["Created_At"].dt.year * 100 + df[
                "Created_At"
            ].dt.isocalendar().week.astype(int)
            unique_weeks = sorted(df["year_week"].unique())
            n_weeks = len(unique_weeks)
            min_train_weeks = int(n_weeks * 0.6)
            retrain_schedule = unique_weeks[min_train_weeks:]

            mlflow.log_params({
                "validation_scheme": "walk_forward",
                "method": "alpha04_variants",
            })

            # 1. 5-model alpha=0.4 seed stability
            logger.info("=== 5m alpha=0.4 seed stability ===")
            rois_5m = []
            for seed in SEEDS:
                res = run_wf(df, retrain_schedule, seed, 5, 0.4)
                rois_5m.append(res["roi"])
                logger.info(
                    "  5m_a04 seed=%d: roi=%.2f%% n=%d pos=%d/%d",
                    seed, res["roi"], res["n"], res["n_pos"], res["n_blocks"],
                )

            mean_5m = float(np.mean(rois_5m))
            std_5m = float(np.std(rois_5m))
            sharpe_5m = mean_5m / std_5m if std_5m > 0 else 0
            logger.info(
                "\n5m_a04: mean=%.2f%% std=%.2f%% [%.1f-%.1f] Sharpe=%.2f\n",
                mean_5m, std_5m, min(rois_5m), max(rois_5m), sharpe_5m,
            )
            mlflow.log_metrics({
                "5m_a04_mean": round(mean_5m, 2),
                "5m_a04_std": round(std_5m, 2),
                "5m_a04_sharpe": round(sharpe_5m, 2),
            })

            # 2. 5-model alpha=0.3 seed stability
            logger.info("=== 5m alpha=0.3 seed stability ===")
            rois_5m3 = []
            for seed in SEEDS:
                res = run_wf(df, retrain_schedule, seed, 5, 0.3)
                rois_5m3.append(res["roi"])
                logger.info(
                    "  5m_a03 seed=%d: roi=%.2f%% n=%d pos=%d/%d",
                    seed, res["roi"], res["n"], res["n_pos"], res["n_blocks"],
                )

            mean_5m3 = float(np.mean(rois_5m3))
            std_5m3 = float(np.std(rois_5m3))
            sharpe_5m3 = mean_5m3 / std_5m3 if std_5m3 > 0 else 0
            logger.info(
                "\n5m_a03: mean=%.2f%% std=%.2f%% [%.1f-%.1f] Sharpe=%.2f\n",
                mean_5m3, std_5m3, min(rois_5m3), max(rois_5m3), sharpe_5m3,
            )
            mlflow.log_metrics({
                "5m_a03_mean": round(mean_5m3, 2),
                "5m_a03_std": round(std_5m3, 2),
                "5m_a03_sharpe": round(sharpe_5m3, 2),
            })

            # 3. Risk analysis for 4m alpha=0.4 (seed=42, detailed)
            logger.info("=== 4m alpha=0.4 risk analysis (seed=42) ===")
            res_detail = run_wf(df, retrain_schedule, 42, 4, 0.4, return_details=True)
            for blk in res_detail.get("blocks", []):
                logger.info(
                    "  week=%d n=%d roi=%.1f%% mean_odds=%.1f max=%.1f pct_gt50=%.1f%%",
                    blk["week"], blk["n"], blk["roi"],
                    blk["mean_odds"], blk["max_odds"], blk["pct_odds_gt50"],
                )

            # 4. Odds caps for alpha=0.4
            logger.info("\n=== 4m alpha=0.4 odds caps ===")
            for cap in [5, 10, 20, 50, 100, None]:
                res_cap = run_wf(df, retrain_schedule, 42, 4, 0.4, odds_cap=cap)
                cap_str = str(cap) if cap else "none"
                logger.info(
                    "  cap=%5s: roi=%.2f%% n=%d pos=%d/%d",
                    cap_str, res_cap["roi"], res_cap["n"],
                    res_cap["n_pos"], res_cap["n_blocks"],
                )
                mlflow.log_metrics({
                    f"4m_a04_cap{cap_str}_roi": round(res_cap["roi"], 2),
                    f"4m_a04_cap{cap_str}_n": res_cap["n"],
                })

            # 5. Comprehensive comparison table
            logger.info("\n=== UPDATED STRATEGY TABLE ===")
            configs = [
                ("4m_a10", 4, 1.0),
                ("4m_a05", 4, 0.5),
                ("4m_a04", 4, 0.4),
                ("4m_a03", 4, 0.3),
                ("5m_a05", 5, 0.5),
                ("5m_a04", 5, 0.4),
                ("5m_a03", 5, 0.3),
            ]
            logger.info(
                "%-10s | %8s | %6s | %8s | %6s",
                "Strategy", "Mean ROI", "Std", "Sharpe", "N(s42)",
            )
            for name, n_m, alpha in configs:
                rois_c = []
                for seed in SEEDS:
                    r = run_wf(df, retrain_schedule, seed, n_m, alpha)
                    rois_c.append(r["roi"])
                mc = float(np.mean(rois_c))
                sc = float(np.std(rois_c))
                sh = mc / sc if sc > 0 else 0
                r42 = run_wf(df, retrain_schedule, 42, n_m, alpha)
                logger.info(
                    "%-10s | %7.2f%% | %5.2f%% | %7.2f | %5d",
                    name, mc, sc, sh, r42["n"],
                )

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.39 failed")
            raise


if __name__ == "__main__":
    main()
