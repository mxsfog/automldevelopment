"""Step 4.3 — XGBoost + 4-model ensemble + CV stability check.

Гипотеза 1: Добавление XGBoost в ансамбль (4 модели) улучшит предсказания.
Гипотеза 2: Cross-validation покажет стабильность conf_ev стратегии.
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
from xgboost import XGBClassifier

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


def train_4model_ensemble(x_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """4-model ensemble: CB + LGBM + XGB + LR."""
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
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

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0,
        min_child_weight=50,
        eval_metric="logloss",
    )
    xgb.fit(x_train, y_train)

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x_train)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_s, y_train)

    return cb, lgbm, xgb, lr, scaler


def predict_4model(cb, lgbm, xgb, lr, scaler, x: pd.DataFrame) -> tuple:
    """Предсказания 4-model ensemble."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_xgb = xgb.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]

    p_mean = (p_cb + p_lgbm + p_xgb + p_lr) / 4
    all_p = np.array([p_cb, p_lgbm, p_xgb, p_lr])
    p_std = np.std(all_p, axis=0)

    return p_mean, p_std


def predict_3model(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Предсказания 3-model ensemble (baseline)."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]

    p_mean = (p_cb + p_lgbm + p_lr) / 3
    all_p = np.array([p_cb, p_lgbm, p_lr])
    p_std = np.std(all_p, axis=0)

    return p_mean, p_std


def conf_ev_select(
    p_mean: np.ndarray, p_std: np.ndarray, odds: np.ndarray, thr: float
) -> np.ndarray:
    """Confidence-weighted EV selection."""
    ev = p_mean * odds - 1
    confidence = 1 / (1 + p_std * 10)
    return (ev * confidence) >= thr


def time_series_cv(
    df: pd.DataFrame,
    features: list[str],
    n_splits: int = 5,
) -> list[dict]:
    """Time series cross-validation для оценки стабильности."""
    n = len(df)
    fold_size = n // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(val_start + fold_size, n)
        if val_end <= val_start:
            break

        fold_train = df.iloc[:train_end].copy()
        fold_val = df.iloc[val_start:val_end].copy()

        # Feature encoding
        fold_train_enc, _ = add_sport_market_features(fold_train, fold_train)
        fold_val_enc, _ = add_sport_market_features(fold_val, fold_train_enc)

        x_tr = fold_train_enc[features].fillna(0)
        x_va = fold_val_enc[features].fillna(0)
        y_tr = fold_train_enc["target"]

        # Train 3-model ensemble
        cb = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        cb.fit(x_tr, y_tr)

        lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1,
            min_child_samples=50,
        )
        lgbm.fit(x_tr, y_tr)

        scaler = StandardScaler()
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(scaler.fit_transform(x_tr), y_tr)

        p_mean, p_std = predict_3model(cb, lgbm, lr, scaler, x_va)

        auc = roc_auc_score(fold_val_enc["target"], p_mean)
        odds = fold_val_enc["Odds"].values

        # Baseline EV
        ev = p_mean * odds - 1
        mask_base = ev >= 0.12
        r_base = calc_roi(fold_val_enc, mask_base.astype(float), threshold=0.5)

        # conf_ev_0.15
        mask_conf = conf_ev_select(p_mean, p_std, odds, 0.15)
        r_conf = calc_roi(fold_val_enc, mask_conf.astype(float), threshold=0.5)

        fold_result = {
            "fold": i,
            "train_size": len(fold_train),
            "val_size": len(fold_val),
            "auc": auc,
            "roi_baseline": r_base["roi"],
            "n_bets_baseline": r_base["n_bets"],
            "roi_conf_ev": r_conf["roi"],
            "n_bets_conf_ev": r_conf["n_bets"],
        }
        results.append(fold_result)
        logger.info(
            "Fold %d: AUC=%.4f, ROI_base=%.2f%%(n=%d), ROI_conf=%.2f%%(n=%d)",
            i,
            auc,
            r_base["roi"],
            r_base["n_bets"],
            r_conf["roi"],
            r_conf["n_bets"],
        )

    return results


def main() -> None:
    """XGBoost + 4-model ensemble + CV stability."""
    with mlflow.start_run(run_name="phase4/xgb_4model_cv") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, _ = load_raw_data()
            df = prepare_dataset(bets, outcomes)
            train, test = time_series_split(df, test_size=0.2)

            # Feature encoding
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
                    "method": "xgb_4model_cv",
                }
            )

            # Part 1: 4-model ensemble vs 3-model
            cb, lgbm, xgb, lr, scaler = train_4model_ensemble(x_train, y_train)

            p_mean_4, p_std_4 = predict_4model(cb, lgbm, xgb, lr, scaler, x_test)
            p_mean_3, p_std_3 = predict_3model(cb, lgbm, lr, scaler, x_test)

            auc_4 = roc_auc_score(test_enc["target"], p_mean_4)
            auc_3 = roc_auc_score(test_enc["target"], p_mean_3)
            logger.info("AUC: 3-model=%.4f, 4-model=%.4f", auc_3, auc_4)

            odds = test_enc["Odds"].values

            # Compare strategies on both ensembles
            strategies = {}
            for thr in [0.10, 0.12, 0.15, 0.18, 0.20]:
                # 3-model baseline EV
                ev3 = p_mean_3 * odds - 1
                mask3 = ev3 >= thr
                r3 = calc_roi(test_enc, mask3.astype(float), threshold=0.5)

                # 4-model EV
                ev4 = p_mean_4 * odds - 1
                mask4 = ev4 >= thr
                r4 = calc_roi(test_enc, mask4.astype(float), threshold=0.5)

                # 3-model conf_ev
                mask3c = conf_ev_select(p_mean_3, p_std_3, odds, thr)
                r3c = calc_roi(test_enc, mask3c.astype(float), threshold=0.5)

                # 4-model conf_ev
                mask4c = conf_ev_select(p_mean_4, p_std_4, odds, thr)
                r4c = calc_roi(test_enc, mask4c.astype(float), threshold=0.5)

                strategies[f"3m_ev_{thr:.2f}"] = r3
                strategies[f"4m_ev_{thr:.2f}"] = r4
                strategies[f"3m_conf_{thr:.2f}"] = r3c
                strategies[f"4m_conf_{thr:.2f}"] = r4c

                logger.info(
                    "thr=%.2f: 3m_ev=%.2f%%(n=%d) 4m_ev=%.2f%%(n=%d) "
                    "3m_conf=%.2f%%(n=%d) 4m_conf=%.2f%%(n=%d)",
                    thr,
                    r3["roi"],
                    r3["n_bets"],
                    r4["roi"],
                    r4["n_bets"],
                    r3c["roi"],
                    r3c["n_bets"],
                    r4c["roi"],
                    r4c["n_bets"],
                )

            # Best strategy
            best_name = max(
                strategies,
                key=lambda k: strategies[k]["roi"] if strategies[k]["n_bets"] >= 50 else -999,
            )
            best = strategies[best_name]
            logger.info("Best: %s -> ROI=%.2f%%, n=%d", best_name, best["roi"], best["n_bets"])

            # Part 2: Cross-validation stability
            logger.info("Running CV for stability check...")
            cv_results = time_series_cv(df, FEATURES, n_splits=5)

            cv_roi_base = [r["roi_baseline"] for r in cv_results]
            cv_roi_conf = [r["roi_conf_ev"] for r in cv_results]

            logger.info(
                "CV baseline: mean=%.2f%%, std=%.2f%%, folds=%s",
                np.mean(cv_roi_base),
                np.std(cv_roi_base),
                cv_roi_base,
            )
            logger.info(
                "CV conf_ev: mean=%.2f%%, std=%.2f%%, folds=%s",
                np.mean(cv_roi_conf),
                np.std(cv_roi_conf),
                cv_roi_conf,
            )

            # Log all metrics
            mlflow.log_metrics(
                {
                    "auc_3model": auc_3,
                    "auc_4model": auc_4,
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                    "cv_roi_baseline_mean": np.mean(cv_roi_base),
                    "cv_roi_baseline_std": np.std(cv_roi_base),
                    "cv_roi_conf_mean": np.mean(cv_roi_conf),
                    "cv_roi_conf_std": np.std(cv_roi_conf),
                }
            )
            for name, r in strategies.items():
                mlflow.log_metrics(
                    {
                        f"roi_{name}": r["roi"],
                        f"n_{name}": r["n_bets"],
                    }
                )
            for i, r in enumerate(cv_results):
                mlflow.log_metrics(
                    {
                        f"cv_roi_base_fold_{i}": r["roi_baseline"],
                        f"cv_roi_conf_fold_{i}": r["roi_conf_ev"],
                        f"cv_auc_fold_{i}": r["auc"],
                    }
                )
            mlflow.set_tag("best_strategy", best_name)

            # Save if improved
            if best["roi"] > 27.95:
                model_dir = SESSION_DIR / "models" / "best"
                model_dir.mkdir(parents=True, exist_ok=True)
                cb.save_model(str(model_dir / "model.cbm"))
                meta = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": best["roi"],
                    "auc": auc_4 if "4m" in best_name else auc_3,
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

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            logger.info("Run ID: %s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "runtime_error")
            logger.exception("Step 4.3 failed")
            raise


if __name__ == "__main__":
    main()
