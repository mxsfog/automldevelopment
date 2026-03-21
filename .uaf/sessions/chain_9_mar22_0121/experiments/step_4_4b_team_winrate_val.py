"""Step 4.4b — Team win rate filter, threshold из val (anti-leakage fixed).

Исправление step 4.4: threshold выбирается на VAL (последние 20% train),
применяется к test один раз. Это единственный валидный подход.

Если val имеет мало бетов → используем самый консервативный из
val-valid вариантов или pre-specified threshold.

Baseline: ROI=28.5833% (n=233)
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

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_8_mar22_0035/models/best")

SEGMENT_THRESHOLDS = {"low": 0.475, "mid": 0.545, "high": 0.325}
FEATURE_NAMES = [
    "Odds",
    "USD",
    "log_odds",
    "log_usd",
    "implied_prob",
    "is_parlay",
    "outcomes_count",
    "ml_p_model",
    "ml_p_implied",
    "ml_edge",
    "ml_ev",
    "ml_team_stats_found",
    "ml_winrate_diff",
    "ml_rating_diff",
    "hour",
    "day_of_week",
    "month",
    "odds_times_stake",
    "ml_edge_pos",
    "ml_ev_pos",
    "elo_max",
    "elo_min",
    "elo_diff",
    "elo_ratio",
    "elo_mean",
    "elo_std",
    "k_factor_mean",
    "has_elo",
    "elo_count",
    "ml_edge_x_elo_diff",
    "elo_implied_agree",
    "Sport",
    "Market",
    "Currency",
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def check_budget() -> bool:
    try:
        status = json.loads(BUDGET_FILE.read_text())
        return bool(status.get("hard_stop"))
    except FileNotFoundError:
        return False


def load_raw_data() -> pd.DataFrame:
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time", "Selection"]
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
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return feats


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def apply_shrunken_segments(
    df: pd.DataFrame, kelly: np.ndarray, seg_thresholds: dict[str, float]
) -> np.ndarray:
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in seg_thresholds.items():
        mask |= (buckets == bucket).values & (kelly >= t)
    return mask


def get_baseline_mask(model: CatBoostClassifier, df: pd.DataFrame) -> np.ndarray:
    x = build_features(df)[FEATURE_NAMES]
    proba = model.predict_proba(x)[:, 1]
    kelly = compute_kelly(proba, df["Odds"].values)
    lead_hours = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    kelly[lead_hours.values <= 0] = -999
    mkt_mask = df["Market"].values == "1x2"
    seg_mask = apply_shrunken_segments(df, kelly, SEGMENT_THRESHOLDS)
    return mkt_mask & seg_mask


def compute_team_winrates(
    subtrain_df: pd.DataFrame, kappa: float = 5.0
) -> tuple[dict[str, float], float]:
    """Win rate из Soccer 1x2 sub-train ставок (строго до val периода)."""
    soccer_1x2 = subtrain_df[
        (subtrain_df["Market"] == "1x2")
        & (subtrain_df["Status"].isin({"won", "lost"}))
        & subtrain_df["Selection"].notna()
    ].copy()

    global_mean = float(soccer_1x2["Status"].eq("won").mean())
    stats = soccer_1x2.groupby("Selection").agg(
        n=("Status", "count"),
        n_won=("Status", lambda x: (x == "won").sum()),
    )
    stats["winrate"] = (stats["n_won"] + kappa * global_mean) / (stats["n"] + kappa)
    return stats["winrate"].to_dict(), global_mean


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    model = CatBoostClassifier()
    model.load_model(str(PREV_BEST_DIR / "model.cbm"))

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    # Split: sub_train (0-64%), val (64-80%), test (80-100%)
    val_start = int(train_end * 0.80)  # = 64% of total
    sub_train = df_raw.iloc[:val_start].copy()
    val_df = df_raw.iloc[val_start:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    logger.info(
        "Sub-train: %d (0-%.0f%%), Val: %d (%.0f-80%%), Test: %d (80-100%%)",
        len(sub_train),
        val_start / n * 100,
        len(val_df),
        val_start / n * 100,
        len(test_df),
    )

    # Team win rates из sub_train (строго до val)
    team_winrates, global_mean = compute_team_winrates(sub_train, kappa=5.0)
    logger.info(
        "Team win rates: %d teams, global=%.3f (from sub_train)",
        len(team_winrates),
        global_mean,
    )

    # Добавить team_winrate в val и test
    val_df["team_winrate"] = val_df["Selection"].map(team_winrates).fillna(global_mean)
    test_df["team_winrate"] = test_df["Selection"].map(team_winrates).fillna(global_mean)

    # Baseline masks
    val_baseline_mask = get_baseline_mask(model, val_df)
    test_baseline_mask = get_baseline_mask(model, test_df)

    val_roi, val_n = calc_roi(val_df, val_baseline_mask)
    test_roi_baseline, test_n_baseline = calc_roi(test_df, test_baseline_mask)

    logger.info(
        "Val baseline: ROI=%.4f%% n=%d | Test baseline: ROI=%.4f%% n=%d",
        val_roi,
        val_n,
        test_roi_baseline,
        test_n_baseline,
    )

    # Threshold search на VAL
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    val_results: list[dict] = []
    for t in thresholds:
        team_filter = val_df["team_winrate"].values >= t
        combined_mask = val_baseline_mask & team_filter
        roi, n_bets = calc_roi(val_df, combined_mask)
        val_results.append(
            {
                "threshold": t,
                "val_roi": roi,
                "val_n": n_bets,
                "val_delta": roi - val_roi if n_bets > 0 else float("nan"),
            }
        )
        logger.info("VAL t=%.2f: ROI=%.4f%% n=%d", t, roi, n_bets)

    # Выбрать threshold на val (max ROI, n >= 20 минимум)
    valid_val = [r for r in val_results if r["val_n"] >= 20]
    if valid_val:
        best_val = max(valid_val, key=lambda r: r["val_roi"])
    else:
        # Если нет достаточно бетов — нет threshold
        best_val = {"threshold": None}
        logger.warning("Val baseline n=%d: недостаточно бетов для threshold selection", val_n)

    logger.info(
        "VAL best threshold: %s (val_roi=%.4f%%, n=%d)",
        best_val.get("threshold"),
        best_val.get("val_roi", -999),
        best_val.get("val_n", 0),
    )

    # Применить threshold к test ОДИН РАЗ
    if best_val["threshold"] is not None:
        t_chosen = best_val["threshold"]
        team_filter_test = test_df["team_winrate"].values >= t_chosen
        final_test_mask = test_baseline_mask & team_filter_test
        final_roi, final_n = calc_roi(test_df, final_test_mask)
    else:
        t_chosen = None
        final_roi = test_roi_baseline
        final_n = test_n_baseline

    delta = final_roi - 28.5833  # vs historical baseline

    logger.info(
        "TEST RESULT: threshold=%.2f ROI=%.4f%% n=%d delta=%.4f",
        t_chosen or -1,
        final_roi,
        final_n,
        delta,
    )

    with mlflow.start_run(run_name="phase4/step_4_4b_team_winrate_val") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.4b")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_sub_train": len(sub_train),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "kappa_smoothing": 5.0,
                "threshold_chosen_on_val": t_chosen,
                "val_baseline_n": val_n,
                "val_baseline_roi": round(val_roi, 4),
            }
        )
        for r in val_results:
            mlflow.log_metrics(
                {
                    f"val_roi_t{int(r['threshold'] * 100)}": r["val_roi"],
                    f"val_n_t{int(r['threshold'] * 100)}": r["val_n"],
                }
            )

        mlflow.log_metric("roi", final_roi)
        mlflow.log_metric("n_selected", final_n)
        mlflow.log_metric("roi_delta", delta)

        if final_roi > 35.0 and final_n > 100:
            mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
            mlflow.set_tag("status", "failed")
            sys.exit(1)

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

        print(f"STEP_4_4B_ROI={final_roi:.6f}")
        print(f"STEP_4_4B_N={final_n}")
        print(f"STEP_4_4B_THRESHOLD={t_chosen}")
        print(f"STEP_4_4B_DELTA={delta:.4f}")
        print(f"MLFLOW_RUN_ID={run.info.run_id}")
        print(f"\nVal baseline: ROI={val_roi:.2f}% n={val_n}")
        print("Val threshold sweep:")
        for r in val_results:
            print(f"  t={r['threshold']:.2f}: ROI={r['val_roi']:.2f}% n={r['val_n']}")


if __name__ == "__main__":
    main()
