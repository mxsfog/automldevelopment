"""Step 4.1 -- Walk-forward validation with expanding window retrain.

Симуляция реального deployment: модель переобучается каждые N дней
на всех доступных данных к этому моменту, предсказывает следующий блок.
Это единственный честный способ оценить edge в production.

Ни одна предыдущая сессия не реализовала walk-forward retrain.
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
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
)

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

FEATURES_BASE = [
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
]

FEATURES_ENC = [
    *FEATURES_BASE,
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


def train_ensemble(x_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Обучить 3-model ensemble (CB + LGBM + LR)."""
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


def predict_ensemble(
    cb: CatBoostClassifier,
    lgbm: LGBMClassifier,
    lr: LogisticRegression,
    scaler: StandardScaler,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Предсказания ensemble: (p_mean, p_std)."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
    return p_mean, p_std


def calc_strategies(df: pd.DataFrame, p_mean: np.ndarray, p_std: np.ndarray) -> dict[str, dict]:
    """Вычислить ROI для нескольких стратегий."""
    odds = df["Odds"].values
    target = df["target"].values
    results = {}

    # Strategy 1: pmean_0.55
    mask_pm = p_mean >= 0.55
    if mask_pm.sum() > 0:
        sel = df[mask_pm]
        n = mask_pm.sum()
        payout = (sel["target"] * sel["Odds"]).sum()
        roi = (payout - n) / n * 100
        results["pmean_0.55"] = {"roi": round(roi, 2), "n": int(n)}
    else:
        results["pmean_0.55"] = {"roi": 0.0, "n": 0}

    # Strategy 2: conf_ev_0.15
    ev = p_mean * odds - 1
    conf = 1.0 / (1.0 + p_std * 10)
    conf_ev = ev * conf
    mask_ce = conf_ev >= 0.15
    if mask_ce.sum() > 0:
        sel = df[mask_ce]
        n = mask_ce.sum()
        payout = (sel["target"] * sel["Odds"]).sum()
        roi = (payout - n) / n * 100
        results["conf_ev_0.15"] = {"roi": round(roi, 2), "n": int(n)}
    else:
        results["conf_ev_0.15"] = {"roi": 0.0, "n": 0}

    # Strategy 3: simple EV >= 0.05
    mask_ev = ev >= 0.05
    if mask_ev.sum() > 0:
        sel = df[mask_ev]
        n = mask_ev.sum()
        payout = (sel["target"] * sel["Odds"]).sum()
        roi = (payout - n) / n * 100
        results["ev_0.05"] = {"roi": round(roi, 2), "n": int(n)}
    else:
        results["ev_0.05"] = {"roi": 0.0, "n": 0}

    # Strategy 4: all bets (baseline)
    n_all = len(target)
    payout_all = (target * odds).sum()
    roi_all = (payout_all - n_all) / n_all * 100
    results["all_bets"] = {"roi": round(roi_all, 2), "n": n_all}

    return results


def main() -> None:
    """Walk-forward validation с expanding window."""
    with mlflow.start_run(run_name="phase4/walkforward_retrain") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            # Walk-forward: разбиваем по неделям
            df["week"] = df["Created_At"].dt.isocalendar().week.astype(int)
            df["year_week"] = df["Created_At"].dt.year * 100 + df[
                "Created_At"
            ].dt.isocalendar().week.astype(int)

            unique_weeks = sorted(df["year_week"].unique())
            n_weeks = len(unique_weeks)
            logger.info(
                "Total weeks: %d, range: %s..%s", n_weeks, unique_weeks[0], unique_weeks[-1]
            )

            # Минимум 60% данных для первого обучения, далее retrain каждую неделю
            min_train_weeks = int(n_weeks * 0.6)
            retrain_schedule = unique_weeks[min_train_weeks:]

            mlflow.log_params(
                {
                    "validation_scheme": "walk_forward",
                    "seed": 42,
                    "n_total_weeks": n_weeks,
                    "min_train_weeks": min_train_weeks,
                    "n_test_blocks": len(retrain_schedule),
                    "retrain_frequency": "weekly",
                    "method": "expanding_window_retrain",
                }
            )

            all_block_results = []
            cumulative_pnl = {s: [] for s in ["pmean_0.55", "conf_ev_0.15", "ev_0.05", "all_bets"]}
            cumulative_n = {s: 0 for s in cumulative_pnl}
            cumulative_profit = {s: 0.0 for s in cumulative_pnl}

            for i, test_week in enumerate(retrain_schedule):
                # Train: все недели до test_week
                train_mask = df["year_week"] < test_week
                test_mask = df["year_week"] == test_week

                train_df = df[train_mask].copy()
                test_df = df[test_mask].copy()

                if len(test_df) == 0:
                    continue

                # Feature engineering (fit on train)
                train_enc, _ = add_sport_market_features(train_df, train_df)
                test_enc, _ = add_sport_market_features(test_df, train_enc)

                x_train = train_enc[FEATURES_ENC].fillna(0)
                y_train = train_enc["target"]
                x_test = test_enc[FEATURES_ENC].fillna(0)

                # Train ensemble
                cb, lgbm, lr, scaler = train_ensemble(x_train, y_train)
                p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_test)

                # AUC
                try:
                    auc = roc_auc_score(test_enc["target"], p_mean)
                except ValueError:
                    auc = 0.5

                # Strategies
                strat_results = calc_strategies(test_enc, p_mean, p_std)

                block_info = {
                    "block": i,
                    "week": test_week,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "auc": round(auc, 4),
                }
                for s_name, s_res in strat_results.items():
                    block_info[f"{s_name}_roi"] = s_res["roi"]
                    block_info[f"{s_name}_n"] = s_res["n"]

                    # Cumulative tracking
                    n = s_res["n"]
                    if n > 0:
                        profit = n * s_res["roi"] / 100
                        cumulative_n[s_name] += n
                        cumulative_profit[s_name] += profit

                all_block_results.append(block_info)
                logger.info(
                    "Block %d (week %d): train=%d test=%d AUC=%.3f "
                    "pmean_0.55=%.1f%%(n=%d) conf_ev=%.1f%%(n=%d)",
                    i,
                    test_week,
                    len(train_df),
                    len(test_df),
                    auc,
                    strat_results["pmean_0.55"]["roi"],
                    strat_results["pmean_0.55"]["n"],
                    strat_results["conf_ev_0.15"]["roi"],
                    strat_results["conf_ev_0.15"]["n"],
                )

            # Summary
            results_df = pd.DataFrame(all_block_results)
            logger.info("Walk-forward results:\n%s", results_df.to_string())

            # Per-strategy aggregated ROI
            for s_name in ["pmean_0.55", "conf_ev_0.15", "ev_0.05", "all_bets"]:
                col_roi = f"{s_name}_roi"
                if col_roi in results_df.columns:
                    # Weighted ROI across blocks
                    total_n = cumulative_n[s_name]
                    total_profit = cumulative_profit[s_name]
                    overall_roi = (total_profit / total_n * 100) if total_n > 0 else 0.0

                    # Per-block stats
                    block_rois = results_df[col_roi].values
                    mean_roi = block_rois.mean()
                    std_roi = block_rois.std()
                    n_positive = (block_rois > 0).sum()
                    n_blocks = len(block_rois)
                    min_roi = block_rois.min()
                    max_roi = block_rois.max()

                    logger.info(
                        "%s: overall_ROI=%.2f%% (n=%d), mean=%.2f%% std=%.2f%%, "
                        "positive=%d/%d, min=%.2f%% max=%.2f%%",
                        s_name,
                        overall_roi,
                        total_n,
                        mean_roi,
                        std_roi,
                        n_positive,
                        n_blocks,
                        min_roi,
                        max_roi,
                    )

                    mlflow.log_metrics(
                        {
                            f"wf_{s_name}_overall_roi": round(overall_roi, 2),
                            f"wf_{s_name}_total_bets": total_n,
                            f"wf_{s_name}_mean_roi": round(mean_roi, 2),
                            f"wf_{s_name}_std_roi": round(std_roi, 2),
                            f"wf_{s_name}_positive_blocks": n_positive,
                            f"wf_{s_name}_total_blocks": n_blocks,
                            f"wf_{s_name}_min_roi": round(min_roi, 2),
                            f"wf_{s_name}_max_roi": round(max_roi, 2),
                        }
                    )

            # Save results
            results_path = str(SESSION_DIR / "experiments" / "walkforward_results.csv")
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(__file__)

            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.1 walk-forward failed")
            raise


if __name__ == "__main__":
    main()
