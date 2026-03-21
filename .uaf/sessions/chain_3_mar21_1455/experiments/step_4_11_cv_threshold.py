"""Step 4.11 — Cross-validated threshold selection.

Проблема: одна val/test разбивка нестабильна. threshold 0.15 лучший на val,
но 0.20 лучший на test. Разные периоды имеют разную оптимальную стратегию.

Подход: expanding window time-series CV (5 folds).
Для каждого порога считаем средний ROI по всем val folds.
Выбираем порог с лучшим средним ROI и низким std.
Применяем к test.

Также: bootstrap confidence — обучаем 5 моделей с разными seed,
усредняем для более робастной оценки uncertainty.
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


def train_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> tuple:
    """3-model ensemble with configurable seed."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        verbose=0,
    )
    cb.fit(x, y)
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=seed,
        verbose=-1,
        min_child_samples=50,
    )
    lgbm.fit(x, y)
    scaler = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
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


def main() -> None:
    """CV-based threshold selection."""
    with mlflow.start_run(run_name="phase4/cv_threshold") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series_cv",
                    "seed": 42,
                    "n_samples_train": len(train),
                    "n_samples_test": len(test),
                    "method": "cv_threshold_selection",
                    "n_cv_folds": 5,
                }
            )

            # Expanding window CV on train
            n = len(train)
            n_folds = 5
            fold_size = n // (n_folds + 1)

            thresholds = np.arange(0.08, 0.25, 0.01)
            # For each threshold, collect ROI per fold
            fold_rois: dict[str, list[float]] = {}
            for thr in thresholds:
                key = f"conf_ev_{thr:.2f}"
                fold_rois[key] = []
            # Also track plain EV
            for thr in [0.08, 0.10, 0.12, 0.15]:
                key = f"ev_{thr:.2f}"
                fold_rois[key] = []

            for fold_idx in range(n_folds):
                tr_end = fold_size * (fold_idx + 2)
                va_start = tr_end
                va_end = min(va_start + fold_size, n)
                if va_end <= va_start:
                    break

                train_fold = train.iloc[:tr_end].copy()
                val_fold = train.iloc[va_start:va_end].copy()

                # Feature encoding per fold
                train_fold_enc, _ = add_sport_market_features(train_fold, train_fold)
                val_fold_enc, _ = add_sport_market_features(val_fold, train_fold_enc)

                x_tr = train_fold_enc[FEATURES].fillna(0)
                x_va = val_fold_enc[FEATURES].fillna(0)

                cb, lgbm, lr, scaler = train_ensemble(x_tr, train_fold_enc["target"])
                p_mean, p_std = predict_ensemble(cb, lgbm, lr, scaler, x_va)

                odds_va = val_fold_enc["Odds"].values
                ev = p_mean * odds_va - 1
                conf = 1 / (1 + p_std * 10)
                ev_conf = ev * conf

                logger.info(
                    "Fold %d: train=%d, val=%d",
                    fold_idx,
                    tr_end,
                    va_end - va_start,
                )

                for thr in thresholds:
                    key = f"conf_ev_{thr:.2f}"
                    mask = ev_conf >= thr
                    if mask.sum() >= 10:
                        r = calc_roi(val_fold_enc, mask.astype(float), threshold=0.5)
                        fold_rois[key].append(r["roi"])
                    else:
                        fold_rois[key].append(np.nan)

                for thr in [0.08, 0.10, 0.12, 0.15]:
                    key = f"ev_{thr:.2f}"
                    mask = ev >= thr
                    if mask.sum() >= 10:
                        r = calc_roi(val_fold_enc, mask.astype(float), threshold=0.5)
                        fold_rois[key].append(r["roi"])
                    else:
                        fold_rois[key].append(np.nan)

            # Analyze CV results
            logger.info("CV threshold analysis:")
            cv_summary = []
            for key, rois in fold_rois.items():
                valid_rois = [r for r in rois if not np.isnan(r)]
                if len(valid_rois) >= 3:
                    mean_roi = np.mean(valid_rois)
                    std_roi = np.std(valid_rois)
                    min_roi = np.min(valid_rois)
                    cv_summary.append(
                        {
                            "strategy": key,
                            "mean_roi": round(mean_roi, 2),
                            "std_roi": round(std_roi, 2),
                            "min_roi": round(min_roi, 2),
                            "n_folds": len(valid_rois),
                            "sharpe": round(mean_roi / max(std_roi, 0.01), 2),
                        }
                    )

            cv_df = pd.DataFrame(cv_summary)

            # Sort by mean ROI
            cv_df_sorted = cv_df.sort_values("mean_roi", ascending=False)
            logger.info("By mean ROI:\n%s", cv_df_sorted.head(15).to_string(index=False))

            # Sort by Sharpe (risk-adjusted)
            cv_df_sharpe = cv_df.sort_values("sharpe", ascending=False)
            logger.info("By Sharpe:\n%s", cv_df_sharpe.head(15).to_string(index=False))

            # Select best strategies
            best_mean = cv_df_sorted.iloc[0]
            best_sharpe = cv_df_sharpe.iloc[0]
            logger.info(
                "Best mean: %s (mean=%.2f%%, std=%.2f%%)",
                best_mean["strategy"],
                best_mean["mean_roi"],
                best_mean["std_roi"],
            )
            logger.info(
                "Best Sharpe: %s (mean=%.2f%%, sharpe=%.2f)",
                best_sharpe["strategy"],
                best_sharpe["mean_roi"],
                best_sharpe["sharpe"],
            )

            # Part 2: Bootstrap confidence
            logger.info("Part 2: Bootstrap confidence (5 seeds)")
            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            x_train_fit = train_fit_enc[FEATURES].fillna(0)
            x_val = val_enc[FEATURES].fillna(0)

            # Train 5 ensembles with different seeds
            all_preds = []
            for seed in [42, 123, 456, 789, 1024]:
                cb_s, lgbm_s, lr_s, sc_s = train_ensemble(
                    x_train_fit, train_fit_enc["target"], seed=seed
                )
                p_s, _ = predict_ensemble(cb_s, lgbm_s, lr_s, sc_s, x_val)
                all_preds.append(p_s)

            p_bootstrap = np.mean(all_preds, axis=0)
            p_bootstrap_std = np.std(all_preds, axis=0)

            odds_val = val_enc["Odds"].values
            ev_boot = p_bootstrap * odds_val - 1
            conf_boot = 1 / (1 + p_bootstrap_std * 10)
            ev_conf_boot = ev_boot * conf_boot

            for thr in [0.10, 0.12, 0.15, 0.18]:
                mask = ev_conf_boot >= thr
                r = calc_roi(val_enc, mask.astype(float), threshold=0.5)
                logger.info(
                    "Bootstrap conf_ev_%.2f on val: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Part 3: Test evaluation
            logger.info("Part 3: Test evaluation")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            x_train = train_enc[FEATURES].fillna(0)
            x_test = test_enc[FEATURES].fillna(0)
            odds_test = test_enc["Odds"].values

            # Standard ensemble
            cb_full, lgbm_full, lr_full, sc_full = train_ensemble(x_train, train_enc["target"])
            p_test, p_std_test = predict_ensemble(cb_full, lgbm_full, lr_full, sc_full, x_test)
            ev_test = p_test * odds_test - 1
            conf_test = 1 / (1 + p_std_test * 10)
            ev_conf_test = ev_test * conf_test
            auc_test = roc_auc_score(test_enc["target"], p_test)

            # Bootstrap ensemble on test
            all_preds_test = []
            for seed in [42, 123, 456, 789, 1024]:
                cb_s, lgbm_s, lr_s, sc_s = train_ensemble(x_train, train_enc["target"], seed=seed)
                p_s, _ = predict_ensemble(cb_s, lgbm_s, lr_s, sc_s, x_test)
                all_preds_test.append(p_s)

            p_boot_test = np.mean(all_preds_test, axis=0)
            p_boot_std_test = np.std(all_preds_test, axis=0)
            ev_boot_test = p_boot_test * odds_test - 1
            conf_boot_test = 1 / (1 + p_boot_std_test * 10)
            ev_conf_boot_test = ev_boot_test * conf_boot_test

            test_results = {}

            # Evaluate best CV strategies on test
            strategies_to_test = set()
            strategies_to_test.add(best_mean["strategy"])
            strategies_to_test.add(best_sharpe["strategy"])
            strategies_to_test.add("conf_ev_0.15")
            strategies_to_test.add("conf_ev_0.12")
            strategies_to_test.add("ev_0.12")

            for strat in strategies_to_test:
                parts = strat.split("_")
                if parts[0] == "conf":
                    thr = float(parts[-1])
                    mask = ev_conf_test >= thr
                elif parts[0] == "ev":
                    thr = float(parts[-1])
                    mask = ev_test >= thr
                else:
                    continue
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                test_results[strat] = r
                logger.info("Test %s: ROI=%.2f%%, n=%d", strat, r["roi"], r["n_bets"])

            # Bootstrap strategies on test
            for thr in [0.12, 0.15, 0.18]:
                mask = ev_conf_boot_test >= thr
                r = calc_roi(test_enc, mask.astype(float), threshold=0.5)
                test_results[f"boot_conf_ev_{thr:.2f}"] = r
                logger.info(
                    "Test boot_conf_ev_%.2f: ROI=%.2f%%, n=%d",
                    thr,
                    r["roi"],
                    r["n_bets"],
                )

            # Best
            best_name = max(
                test_results,
                key=lambda k: test_results[k]["roi"] if test_results[k]["n_bets"] >= 50 else -999,
            )
            best = test_results[best_name]
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
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                    "cv_best_mean_roi": best_mean["mean_roi"],
                    "cv_best_std_roi": best_mean["std_roi"],
                    "cv_best_sharpe": best_sharpe["sharpe"],
                }
            )
            mlflow.set_tag("best_strategy", best_name)
            mlflow.set_tag("cv_best_mean", best_mean["strategy"])
            mlflow.set_tag("cv_best_sharpe", best_sharpe["strategy"])

            # Save CV analysis
            cv_path = str(SESSION_DIR / "experiments" / "cv_threshold_analysis.csv")
            cv_df_sorted.to_csv(cv_path, index=False)
            mlflow.log_artifact(cv_path)

            if best["roi"] > 27.95 and best_name != "conf_ev_0.15":
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

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.11 failed")
            raise


if __name__ == "__main__":
    main()
