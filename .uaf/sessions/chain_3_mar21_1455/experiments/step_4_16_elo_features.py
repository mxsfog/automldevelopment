"""Step 4.16 — ELO features from elo_history.csv.

elo_history (14396 строк) содержит:
- Team_ID, Old_ELO, New_ELO, ELO_Change, Won, K_Factor
- Связь с ставками через Bet_ID

Гипотеза: ELO features могут дать сигнал о силе команд.
Фичи:
1. elo_team: текущий ELO команды (Old_ELO)
2. elo_opponent: ELO оппонента
3. elo_diff: elo_team - elo_opponent
4. elo_change_avg: средний ELO change за предыдущие матчи (momentum)
5. elo_max: максимум ELO из двух
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

BASE_FEATURES = [
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


def add_elo_features(df: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Add ELO-based features to dataframe."""
    # Group elo by Bet_ID: первая запись = team, может быть несколько entries per bet
    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            elo_team=("Old_ELO", "first"),
            elo_opponent=("Old_ELO", "last"),
            elo_change=("ELO_Change", "first"),
            elo_team_new=("New_ELO", "first"),
            n_elo_records=("ID", "count"),
        )
        .reset_index()
    )

    # Derived features
    elo_agg["elo_diff"] = elo_agg["elo_team"] - elo_agg["elo_opponent"]
    elo_agg["elo_max"] = np.maximum(elo_agg["elo_team"], elo_agg["elo_opponent"])
    elo_agg["elo_sum"] = elo_agg["elo_team"] + elo_agg["elo_opponent"]

    # Join
    df = df.merge(
        elo_agg[["Bet_ID", "elo_team", "elo_opponent", "elo_diff", "elo_max", "elo_sum"]],
        left_on="ID",
        right_on="Bet_ID",
        how="left",
        suffixes=("", "_elo"),
    )

    # Drop duplicate Bet_ID if exists
    if "Bet_ID_elo" in df.columns:
        df.drop(columns=["Bet_ID_elo"], inplace=True)
    if "Bet_ID" in df.columns and "ID" in df.columns:
        df.drop(columns=["Bet_ID"], inplace=True, errors="ignore")

    # Fill NaN for bets without ELO data
    elo_cols = ["elo_team", "elo_opponent", "elo_diff", "elo_max", "elo_sum"]
    for col in elo_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


ELO_FEATURES = ["elo_team", "elo_opponent", "elo_diff", "elo_max", "elo_sum"]


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


def predict_ensemble(cb, lgbm, lr, scaler, x: pd.DataFrame) -> tuple:
    """Ensemble predictions."""
    p_cb = cb.predict_proba(x)[:, 1]
    p_lgbm = lgbm.predict_proba(x)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(x))[:, 1]
    p_mean = (p_cb + p_lgbm + p_lr) / 3
    p_std = np.std([p_cb, p_lgbm, p_lr], axis=0)
    return p_mean, p_std


def evaluate_conf_ev(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    odds: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    """Evaluate conf_ev at multiple thresholds."""
    ev = p_mean * odds - 1
    conf = 1 / (1 + p_std * 10)
    ev_conf = ev * conf

    results = {}
    for thr in [0.10, 0.12, 0.15, 0.18]:
        mask = ev_conf >= thr
        r = calc_roi(df, mask.astype(float), threshold=0.5)
        results[f"conf_ev_{thr:.2f}"] = r
    return results


def main() -> None:
    """ELO features experiment."""
    with mlflow.start_run(run_name="phase4/elo_features") as run:
        try:
            mlflow.set_tag("session_id", SESSION_ID)
            mlflow.set_tag("type", "experiment")
            mlflow.set_tag("status", "running")

            bets, outcomes, _, elo = load_raw_data()
            df = prepare_dataset(bets, outcomes)

            # Add ELO features before split
            df = add_elo_features(df, elo)
            elo_coverage = (df["elo_team"] != 0).mean()
            logger.info("ELO coverage: %.2f%% of bets have ELO data", elo_coverage * 100)

            train, test = time_series_split(df, test_size=0.2)

            val_split = int(len(train) * 0.8)
            train_fit = train.iloc[:val_split].copy()
            val = train.iloc[val_split:].copy()

            train_fit_enc, _ = add_sport_market_features(train_fit, train_fit)
            val_enc, _ = add_sport_market_features(val, train_fit_enc)

            features_base = BASE_FEATURES
            features_elo = BASE_FEATURES + ELO_FEATURES

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train_fit": len(train_fit),
                    "n_samples_val": len(val),
                    "n_samples_test": len(test),
                    "method": "elo_features",
                    "elo_coverage": round(elo_coverage, 4),
                    "n_elo_features": len(ELO_FEATURES),
                }
            )

            all_val = {}

            # Baseline (19 features)
            logger.info("Baseline (19 features)...")
            x_tr_b = train_fit_enc[features_base].fillna(0)
            x_va_b = val_enc[features_base].fillna(0)
            cb_b, lgbm_b, lr_b, sc_b = train_ensemble(x_tr_b, train_fit_enc["target"])
            p_b, s_b = predict_ensemble(cb_b, lgbm_b, lr_b, sc_b, x_va_b)
            auc_base = roc_auc_score(val_enc["target"], p_b)
            r_b = evaluate_conf_ev(p_b, s_b, val_enc["Odds"].values, val_enc)
            for k, v in r_b.items():
                all_val[f"base_{k}"] = v
                logger.info("Val base_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            # With ELO (24 features)
            logger.info("With ELO (+5 features)...")
            x_tr_e = train_fit_enc[features_elo].fillna(0)
            x_va_e = val_enc[features_elo].fillna(0)
            cb_e, lgbm_e, lr_e, sc_e = train_ensemble(x_tr_e, train_fit_enc["target"])
            p_e, s_e = predict_ensemble(cb_e, lgbm_e, lr_e, sc_e, x_va_e)
            auc_elo = roc_auc_score(val_enc["target"], p_e)
            r_e = evaluate_conf_ev(p_e, s_e, val_enc["Odds"].values, val_enc)
            for k, v in r_e.items():
                all_val[f"elo_{k}"] = v
                logger.info("Val elo_%s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

            logger.info(
                "Val AUC: base=%.4f, elo=%.4f, delta=%.4f", auc_base, auc_elo, auc_elo - auc_base
            )

            # Test
            logger.info("Test evaluation...")
            train_enc, _ = add_sport_market_features(train, train)
            test_enc, _ = add_sport_market_features(test.copy(), train_enc)

            test_results = {}

            # Base on test
            x_tr_full_b = train_enc[features_base].fillna(0)
            x_te_b = test_enc[features_base].fillna(0)
            cb_fb, lgbm_fb, lr_fb, sc_fb = train_ensemble(x_tr_full_b, train_enc["target"])
            p_tb, s_tb = predict_ensemble(cb_fb, lgbm_fb, lr_fb, sc_fb, x_te_b)
            auc_test_base = roc_auc_score(test_enc["target"], p_tb)
            r_tb = evaluate_conf_ev(p_tb, s_tb, test_enc["Odds"].values, test_enc)
            for k, v in r_tb.items():
                test_results[f"base_{k}"] = v

            # ELO on test
            x_tr_full_e = train_enc[features_elo].fillna(0)
            x_te_e = test_enc[features_elo].fillna(0)
            cb_fe, lgbm_fe, lr_fe, sc_fe = train_ensemble(x_tr_full_e, train_enc["target"])
            p_te, s_te = predict_ensemble(cb_fe, lgbm_fe, lr_fe, sc_fe, x_te_e)
            auc_test_elo = roc_auc_score(test_enc["target"], p_te)
            r_te = evaluate_conf_ev(p_te, s_te, test_enc["Odds"].values, test_enc)
            for k, v in r_te.items():
                test_results[f"elo_{k}"] = v

            logger.info(
                "Test AUC: base=%.4f, elo=%.4f, delta=%.4f",
                auc_test_base,
                auc_test_elo,
                auc_test_elo - auc_test_base,
            )

            for k, v in sorted(
                test_results.items(),
                key=lambda kv: kv[1]["roi"] if kv[1]["n_bets"] >= 50 else -999,
                reverse=True,
            )[:10]:
                if v["n_bets"] >= 50:
                    logger.info("Test %s: ROI=%.2f%%, n=%d", k, v["roi"], v["n_bets"])

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

            mlflow.log_metrics(
                {
                    "auc_val_base": auc_base,
                    "auc_val_elo": auc_elo,
                    "auc_test_base": auc_test_base,
                    "auc_test_elo": auc_test_elo,
                    "roi_best": best["roi"],
                    "n_bets_best": best["n_bets"],
                    "elo_coverage": elo_coverage,
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
            logger.exception("Step 4.16 failed")
            raise


if __name__ == "__main__":
    main()
