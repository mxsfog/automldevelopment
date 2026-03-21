"""Step 4.1 — Odds-bucket segment Kelly thresholds.

Гипотеза: разные диапазоны коэффициентов имеют разный паттерн прибыльности.
Оптимизируем отдельный Kelly-порог для каждого bucket: low (<1.8), mid (1.8-3.0), high (3+).
Ожидаем: +1-5% ROI за счёт более точной фильтрации в каждом сегменте.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import joblib
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


def make_weights(n: int, half_life: float = 0.5) -> np.ndarray:
    """Temporal decay weights."""
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


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


def get_odds_bucket(odds: pd.Series) -> pd.Series:
    """Разбивка коэффициентов на 3 сегмента."""
    return pd.cut(
        odds,
        bins=[0, 1.8, 3.0, np.inf],
        labels=["low", "mid", "high"],
    )


def find_segment_thresholds(
    df: pd.DataFrame,
    kelly: np.ndarray,
    min_bets_per_segment: int = 50,
) -> dict[str, float]:
    """Поиск оптимального Kelly-порога для каждого сегмента по val ROI."""
    df = df.copy()
    df["_kelly"] = kelly
    df["_bucket"] = get_odds_bucket(df["Odds"])

    thresholds: dict[str, float] = {}
    for bucket in ["low", "mid", "high"]:
        seg = df[df["_bucket"] == bucket]
        kelly_seg = seg["_kelly"].values
        best_roi, best_t = -999.0, 0.455  # fallback = baseline threshold

        for t in np.arange(0.01, 0.70, 0.005):
            mask = kelly_seg >= t
            if mask.sum() < min_bets_per_segment:
                break
            seg_mask = (df["_bucket"] == bucket).values & (df["_kelly"].values >= t)
            roi, _ = calc_roi(df, seg_mask)
            if roi > best_roi:
                best_roi = roi
                best_t = t

        thresholds[bucket] = best_t
        logger.info("Bucket %s: threshold=%.3f, val_roi=%.2f%%", bucket, best_t, best_roi)

    return thresholds


def apply_segment_thresholds(
    df: pd.DataFrame, kelly: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    """Применение индивидуальных порогов по сегментам."""
    df = df.copy()
    df["_kelly"] = kelly
    df["_bucket"] = get_odds_bucket(df["Odds"])

    mask = np.zeros(len(df), dtype=bool)
    for bucket, threshold in thresholds.items():
        seg_mask = (df["_bucket"] == bucket).values & (df["_kelly"].values >= threshold)
        mask |= seg_mask
    return mask


with mlflow.start_run(run_name="phase4/step4.1_odds_segments") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.1")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        # Time-series split: train=0-80%, val=64-80%, test=80-100%
        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        train_df = df_raw.iloc[:train_end].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "baseline_threshold": 0.455,
                "approach": "odds_bucket_segment_thresholds",
                "buckets": "low<1.8, mid=1.8-3.0, high>=3.0",
            }
        )

        x_tr, cat_f = build_features(train_df)
        x_vl, _ = build_features(val_df)
        x_te, _ = build_features(test_df)
        y_tr = (train_df["Status"] == "won").astype(int)
        y_vl = (val_df["Status"] == "won").astype(int)

        w = make_weights(len(train_df))

        # Загружаем baseline модель через нативный CatBoost loader
        meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        model = CatBoostClassifier()
        model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        logger.info("Baseline model loaded. Computing probabilities...")

        proba_val = model.predict_proba(x_vl)[:, 1]
        proba_test = model.predict_proba(x_te)[:, 1]

        auc_val = roc_auc_score(y_vl, proba_val)
        logger.info("Val AUC: %.4f", auc_val)

        # Kelly на val
        kelly_val = compute_kelly(proba_val, val_df["Odds"].values)
        # Фильтр pre-match
        kelly_val[val_df["lead_hours"].values <= 0] = -999

        # Kelly на test
        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        lead_test = (
            pd.to_datetime(test_df["Start_Time"], utc=True, errors="coerce")
            - pd.to_datetime(test_df["Created_At"], utc=True)
        ).dt.total_seconds() / 3600.0
        kelly_test[lead_test.values <= 0] = -999

        # Baseline (единый порог)
        baseline_mask_test = kelly_test >= 0.455
        baseline_roi, baseline_n = calc_roi(test_df, baseline_mask_test)
        logger.info("Baseline test ROI: %.4f%%, n=%d", baseline_roi, baseline_n)

        # Поиск segment thresholds на val
        logger.info("Поиск segment thresholds на val...")
        seg_thresholds = find_segment_thresholds(val_df, kelly_val, min_bets_per_segment=30)

        mlflow.log_params(
            {
                "threshold_low": seg_thresholds.get("low", 0.455),
                "threshold_mid": seg_thresholds.get("mid", 0.455),
                "threshold_high": seg_thresholds.get("high", 0.455),
            }
        )

        # Применяем на test
        seg_mask_test = apply_segment_thresholds(test_df, kelly_test, seg_thresholds)
        seg_roi, seg_n = calc_roi(test_df, seg_mask_test)

        delta = seg_roi - baseline_roi
        logger.info(
            "Segment ROI: %.4f%% (n=%d) vs baseline %.4f%% (n=%d), delta=%.4f%%",
            seg_roi,
            seg_n,
            baseline_roi,
            baseline_n,
            delta,
        )

        if seg_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", seg_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_metrics(
            {
                "roi": seg_roi,
                "n_selected": seg_n,
                "baseline_roi": baseline_roi,
                "baseline_n": baseline_n,
                "delta_vs_baseline": delta,
                "auc_val": auc_val,
            }
        )

        # Сегментный анализ на test
        test_df2 = test_df.copy()
        test_df2["_kelly"] = kelly_test
        test_df2["_bucket"] = get_odds_bucket(test_df2["Odds"])
        for bucket in ["low", "mid", "high"]:
            seg_t = seg_thresholds.get(bucket, 0.455)
            b_mask = (test_df2["_bucket"] == bucket).values & (test_df2["_kelly"].values >= seg_t)
            if b_mask.sum() > 0:
                b_roi, b_n = calc_roi(test_df2, b_mask)
                mlflow.log_metrics({f"roi_{bucket}": b_roi, f"n_{bucket}": b_n})
                logger.info("Bucket %s test: ROI=%.2f%%, n=%d", bucket, b_roi, b_n)

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(
            f"step4.1 DONE: seg_roi={seg_roi:.4f}%, baseline={baseline_roi:.4f}%, "
            f"delta={delta:.4f}%, n={seg_n}"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
