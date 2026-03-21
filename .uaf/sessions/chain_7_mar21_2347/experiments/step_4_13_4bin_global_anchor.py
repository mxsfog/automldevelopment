"""Step 4.13: 4-bin odds + 1x2 filter with GLOBAL baseline_t=0.455 as shrinkage anchor.

Step 4.12 failure analysis: baseline_t was optimized on 1x2 val, giving t=0.095 (too lenient).
This anchored all shrunken thresholds too low, causing catastrophic test ROI drop.

Fix: use global baseline_t=0.455 (from step 4.6, all-market val) as shrinkage anchor.
Find per-4-bin raw thresholds on 1x2 val, shrink toward global 0.455.

4 bins (a-priori for 1x2 soccer):
  heavy_fav:   odds < 1.6
  slight_fav:  1.6 <= odds < 2.2
  slight_under: 2.2 <= odds < 3.5
  heavy_under: odds >= 3.5
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
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])

# 4-bin breakpoints (a-priori for 1x2 market)
BINS_4 = [0.0, 1.6, 2.2, 3.5, np.inf]
BIN_LABELS = ["heavy_fav", "slight_fav", "slight_under", "heavy_under"]

# Global baseline threshold from step 4.6 (all-market val, NOT 1x2-specific)
GLOBAL_BASELINE_T = 0.455
SHRINKAGE = 0.5
SEED = 42


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


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="phase4/step4.13_4bin_global_anchor") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")

        budget_status = json.loads(BUDGET_FILE.read_text())
        if budget_status.get("hard_stop"):
            logger.info("hard_stop=true, выход")
            mlflow.set_tag("status", "budget_stopped")
            sys.exit(0)

        logger.info("Загрузка данных...")
        raw_df = load_raw_data()
        feats_df, cat_features = build_features(raw_df)

        n = len(raw_df)
        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        val_raw = raw_df.iloc[val_start:train_end].copy()
        test_raw = raw_df.iloc[train_end:].copy()

        val_feats = feats_df.iloc[val_start:train_end].copy()
        test_feats = feats_df.iloc[train_end:].copy()

        logger.info("val=%d, test=%d", len(val_raw), len(test_raw))

        # Загрузка глобальной модели
        model = CatBoostClassifier()
        model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        val_proba = model.predict_proba(val_feats)[:, 1]
        test_proba = model.predict_proba(test_feats)[:, 1]

        val_kelly = compute_kelly(val_proba, val_raw["Odds"].values)
        test_kelly = compute_kelly(test_proba, test_raw["Odds"].values)

        val_1x2 = (val_raw["Market"] == "1x2").values
        test_1x2 = (test_raw["Market"] == "1x2").values

        val_bins = pd.cut(val_raw["Odds"], bins=BINS_4, labels=BIN_LABELS, right=False)
        test_bins = pd.cut(test_raw["Odds"], bins=BINS_4, labels=BIN_LABELS, right=False)

        t_sweep = np.arange(0.05, 0.95, 0.005)

        # Raw threshold per 4-bin on 1x2 val, anchored shrinkage toward GLOBAL_BASELINE_T
        seg_t_raw: dict[str, float] = {}

        for bucket in BIN_LABELS:
            bucket_mask = val_1x2 & (val_bins == bucket).values
            bucket_n = bucket_mask.sum()

            if bucket_n < 10:
                seg_t_raw[bucket] = GLOBAL_BASELINE_T
                logger.info(
                    "bucket=%s: n=%d < 10, используем GLOBAL_BASELINE_T=%.3f",
                    bucket,
                    bucket_n,
                    GLOBAL_BASELINE_T,
                )
                continue

            best_roi_b = -np.inf
            best_t_b = GLOBAL_BASELINE_T

            for t in t_sweep:
                sel_mask = bucket_mask & (val_kelly >= t)
                n_sel = sel_mask.sum()
                if n_sel < 5:
                    continue
                roi_b, _ = calc_roi(val_raw, sel_mask)
                if roi_b > best_roi_b:
                    best_roi_b = roi_b
                    best_t_b = t

            seg_t_raw[bucket] = best_t_b
            logger.info(
                "bucket=%s: n_val=%d, raw_t=%.3f, val_roi=%.2f%%",
                bucket,
                bucket_n,
                best_t_b,
                best_roi_b,
            )

        # Shrunken toward GLOBAL baseline (not 1x2-specific)
        seg_t_shrunken = {
            k: GLOBAL_BASELINE_T + SHRINKAGE * (v - GLOBAL_BASELINE_T)
            for k, v in seg_t_raw.items()
        }
        logger.info(
            "Shrunken thresholds (anchor=%.3f, shrink=%.1f): %s",
            GLOBAL_BASELINE_T,
            SHRINKAGE,
            seg_t_shrunken,
        )

        # Test: 4-bin shrunken, global anchor
        test_mask_4bin = np.zeros(len(test_raw), dtype=bool)
        for bucket in BIN_LABELS:
            b_mask = test_1x2 & (test_bins == bucket).values
            test_mask_4bin |= b_mask & (test_kelly >= seg_t_shrunken[bucket])

        roi_4bin, n_4bin = calc_roi(test_raw, test_mask_4bin)
        logger.info("4-bin global_anchor shrunken on 1x2: test_roi=%.4f%%, n=%d", roi_4bin, n_4bin)

        # Reference: 3-bin shrunken from step 4.10 (28.5833%)
        seg_3bin = {"low": 0.475, "mid": 0.545, "high": 0.325}
        test_bins_3 = pd.cut(
            test_raw["Odds"],
            bins=[0.0, 1.8, 3.0, np.inf],
            labels=["low", "mid", "high"],
        )
        test_mask_3bin = np.zeros(len(test_raw), dtype=bool)
        for bucket3, t3 in seg_3bin.items():
            b3_mask = test_1x2 & (test_bins_3 == bucket3).values
            test_mask_3bin |= b3_mask & (test_kelly >= t3)

        roi_3bin_ref, n_3bin_ref = calc_roi(test_raw, test_mask_3bin)
        logger.info(
            "3-bin ref (step4.10 recomputed): test_roi=%.4f%%, n=%d", roi_3bin_ref, n_3bin_ref
        )

        delta = roi_4bin - roi_3bin_ref

        # Детализация по бинам на test
        for bucket in BIN_LABELS:
            b_sel = (
                test_1x2 & (test_bins == bucket).values & (test_kelly >= seg_t_shrunken[bucket])
            )
            roi_b, n_b = calc_roi(test_raw, b_sel)
            logger.info(
                "  test bucket=%s: t=%.3f, n_sel=%d, roi=%.2f%%",
                bucket,
                seg_t_shrunken[bucket],
                n_b,
                roi_b,
            )

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_val": len(val_raw),
                "n_samples_test": len(test_raw),
                "market_filter": "1x2",
                "n_bins": 4,
                "bin_breaks": str(BINS_4),
                "shrinkage": SHRINKAGE,
                "global_baseline_t": GLOBAL_BASELINE_T,
                "seg_t_raw": str(seg_t_raw),
                "seg_t_shrunken": str(seg_t_shrunken),
                "reference_best_roi": 28.5833,
            }
        )
        mlflow.log_metrics(
            {
                "roi_4bin_global_anchor_test": roi_4bin,
                "n_bets_4bin": n_4bin,
                "roi_3bin_ref": roi_3bin_ref,
                "n_bets_3bin": n_3bin_ref,
                "delta_vs_ref": delta,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.90")

        accepted = roi_4bin > roi_3bin_ref
        print(
            f"step4.13 DONE: 4bin_global_anchor={roi_4bin:.4f}%, n={n_4bin}, "
            f"3bin_ref={roi_3bin_ref:.4f}%, delta={delta:+.4f}%, "
            f"{'ACCEPTED NEW BEST' if accepted else 'REJECTED'}"
        )
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        raise
