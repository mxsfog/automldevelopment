"""Step 4.5 — Dual-agreement filter: CatBoost + platform model (ML_P_Model).

Гипотеза: когда две независимые модели (CatBoost и платформенная ML_P_Model) одновременно
дают высокий edge, вероятность победы ставки выше. Двойное согласие = более надёжный сигнал.

Метод:
- catboost_kelly = Kelly из CatBoost вероятностей
- platform_edge = ML_P_Model - ML_P_Implied (предсказанная вероятность - подразумеваемая)
- Select: catboost_kelly >= t1 AND platform_edge >= t2
- Оба порога ищем на val (без test leakage)
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_6_mar21_2236/models/best")
LEAKAGE_THRESHOLD = 35.0

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Стандартный feature set (chain_1 compatible)."""
    feats = pd.DataFrame(index=df.index)
    feats["Odds"] = df["Odds"]
    feats["USD"] = df["USD"]
    feats["log_odds"] = np.log(df["Odds"].clip(1.001))
    feats["log_usd"] = np.log1p(df["USD"].clip(0))
    feats["implied_prob"] = 1.0 / df["Odds"].clip(1.001)
    feats["is_parlay"] = (df["Is_Parlay"] == "t").astype(int)
    feats["outcomes_count"] = df["Outcomes_Count"].fillna(1)
    feats["ml_p_model"] = df["ML_P_Model"].fillna(-1)
    feats["ml_p_implied"] = df["ML_P_Implied"].fillna(-1)
    feats["ml_edge"] = df["ML_Edge"].fillna(0.0)
    feats["ml_ev"] = df["ML_EV"].clip(-100, 1000).fillna(0.0)
    feats["ml_team_stats_found"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
    feats["ml_winrate_diff"] = df["ML_Winrate_Diff"].fillna(0.0)
    feats["ml_rating_diff"] = df["ML_Rating_Diff"].fillna(0.0)
    feats["hour"] = df["Created_At"].dt.hour
    feats["day_of_week"] = df["Created_At"].dt.dayofweek
    feats["month"] = df["Created_At"].dt.month
    feats["odds_times_stake"] = feats["Odds"] * feats["USD"]
    feats["ml_edge_pos"] = feats["ml_edge"].clip(0)
    feats["ml_ev_pos"] = feats["ml_ev"].clip(0)
    feats["elo_max"] = df["elo_max"].fillna(-1)
    feats["elo_min"] = df["elo_min"].fillna(-1)
    feats["elo_diff"] = df["elo_diff"].fillna(0.0)
    feats["elo_ratio"] = df["elo_ratio"].fillna(1.0)
    feats["elo_mean"] = df["elo_mean"].fillna(-1)
    feats["elo_std"] = df["elo_std"].fillna(0.0)
    feats["k_factor_mean"] = df["k_factor_mean"].fillna(-1)
    feats["has_elo"] = df["elo_count"].notna().astype(int)
    feats["elo_count"] = df["elo_count"].fillna(0)
    feats["ml_edge_x_elo_diff"] = feats["ml_edge"] * feats["elo_diff"].clip(0, 500) / 500
    feats["elo_implied_agree"] = (
        feats["implied_prob"] - 1.0 / feats["elo_ratio"].clip(0.5, 2.0)
    ).abs()
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """ROI на выбранных ставках."""
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def load_raw_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            elo_max=("Old_ELO", "max"),
            elo_min=("Old_ELO", "min"),
            elo_mean=("Old_ELO", "mean"),
            elo_std=("Old_ELO", "std"),
            elo_count=("Old_ELO", "count"),
            k_factor_mean=("K_Factor", "mean"),
        )
        .reset_index()
    )
    elo_agg["elo_diff"] = elo_agg["elo_max"] - elo_agg["elo_min"]
    elo_agg["elo_ratio"] = elo_agg["elo_max"] / elo_agg["elo_min"].clip(1.0)
    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def find_dual_threshold(
    df: pd.DataFrame,
    kelly: np.ndarray,
    platform_edge: np.ndarray,
    min_bets: int = 50,
) -> tuple[float, float, float]:
    """Grid search по двум порогам (Kelly + platform_edge) на val."""
    best_roi, best_t1, best_t2 = -999.0, 0.455, 0.0

    for t1 in np.arange(0.20, 0.65, 0.010):
        for t2 in np.arange(-0.05, 0.20, 0.010):
            mask = (kelly >= t1) & (platform_edge >= t2)
            if mask.sum() < min_bets:
                continue
            roi, _ = calc_roi(df, mask)
            if roi > best_roi:
                best_roi = roi
                best_t1, best_t2 = t1, t2

    return best_t1, best_t2, best_roi


with mlflow.start_run(run_name="phase4/step4.5_dual_agreement") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.5")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Val: %d, Test: %d", len(val_df), len(test_df))

        # Анализ платформенного edge
        val_platform_edge = val_df["ML_P_Model"].fillna(-1) - val_df["ML_P_Implied"].fillna(-1)
        test_platform_edge = test_df["ML_P_Model"].fillna(-1) - test_df["ML_P_Implied"].fillna(-1)

        logger.info(
            "Val platform_edge: mean=%.3f, std=%.3f, >0: %.1f%%",
            val_platform_edge.mean(),
            val_platform_edge.std(),
            (val_platform_edge > 0).mean() * 100,
        )

        # Загружаем baseline модель
        model = CatBoostClassifier()
        model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_vl, _ = build_features(val_df)
        x_te, _ = build_features(test_df)

        proba_val = model.predict_proba(x_vl)[:, 1]
        proba_test = model.predict_proba(x_te)[:, 1]

        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, proba_test)
        logger.info("Test AUC: %.4f", auc_test)

        # Kelly на val и test
        kelly_val = compute_kelly(proba_val, val_df["Odds"].values)
        kelly_val[val_df["lead_hours"].values <= 0] = -999

        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_test[test_df["lead_hours"].values <= 0] = -999

        # Baseline (single threshold)
        baseline_mask = kelly_test >= 0.455
        baseline_roi, baseline_n = calc_roi(test_df, baseline_mask)
        logger.info("Baseline: roi=%.4f%%, n=%d", baseline_roi, baseline_n)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": train_end,
                "n_samples_val": len(val_df),
                "baseline_threshold": 0.455,
                "approach": "dual_agreement_catboost_platform",
            }
        )

        # Grid search на val
        logger.info("Grid search dual thresholds на val (min_bets=50)...")
        t1, t2, val_roi = find_dual_threshold(
            val_df, kelly_val, val_platform_edge.values, min_bets=50
        )
        logger.info(
            "Best dual: t1(kelly)=%.3f, t2(platform_edge)=%.3f, val_roi=%.2f%%", t1, t2, val_roi
        )

        # Применяем на test
        dual_mask = (kelly_test >= t1) & (test_platform_edge.values >= t2)
        dual_roi, dual_n = calc_roi(test_df, dual_mask)
        delta = dual_roi - baseline_roi

        logger.info(
            "Dual test: roi=%.4f%% (n=%d) vs baseline %.4f%% (n=%d), delta=%.4f%%",
            dual_roi,
            dual_n,
            baseline_roi,
            baseline_n,
            delta,
        )

        # Анализ по вариантам: только platform edge (без Kelly)
        for pe_thresh in [0.0, 0.02, 0.05, 0.10, 0.15]:
            pe_mask = (test_platform_edge.values >= pe_thresh) & (kelly_test >= 0.01)
            pe_roi, pe_n = calc_roi(test_df, pe_mask)
            mlflow.log_metrics(
                {
                    f"roi_pe{int(pe_thresh * 100)}": pe_roi,
                    f"n_pe{int(pe_thresh * 100)}": pe_n,
                }
            )
            logger.info("platform_edge>=%.2f: roi=%.2f%%, n=%d", pe_thresh, pe_roi, pe_n)

        if dual_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", dual_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_metrics(
            {
                "roi": dual_roi,
                "n_selected": dual_n,
                "auc_test": auc_test,
                "kelly_threshold": t1,
                "platform_edge_threshold": t2,
                "val_roi": val_roi,
                "baseline_roi": baseline_roi,
                "delta_vs_baseline": delta,
            }
        )

        current_best_roi = 25.8347
        if dual_roi > current_best_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", dual_roi, current_best_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(best_dir / "model.cbm"))
            meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
            new_meta = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": dual_roi,
                "auc": float(auc_test),
                "threshold": float(t1),
                "platform_edge_threshold": float(t2),
                "n_bets": dual_n,
                "feature_names": meta["feature_names"],
                "params": meta["params"],
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.5",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(
            f"step4.5 DONE: dual_roi={dual_roi:.4f}%, "
            f"t1={t1:.3f}, t2={t2:.3f}, n={dual_n}, delta={delta:.4f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
