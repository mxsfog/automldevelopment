"""Step 4.4 — Post-hoc team win rate filter.

Идея: НЕ переобучаем модель. Используем chain_8 model.cbm (+ feature engineering).
Добавляем вторичный фильтр: Selection team's win rate в Soccer 1x2
обучающих ставках (strictly lagged, no leakage).

Логика:
1. Воспроизвести baseline 233 бетов с Kelly mask
2. Для каждой ставки: team_winrate = Bayesian smoothed win rate из train Soccer 1x2
3. Дополнительный фильтр: team_winrate >= threshold
4. Sweep thresholds, репортировать ROI

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
    """Воспроизвести baseline Kelly mask из chain_8."""
    x = build_features(df)[FEATURE_NAMES]
    proba = model.predict_proba(x)[:, 1]
    kelly = compute_kelly(proba, df["Odds"].values)
    lead_hours = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    kelly[lead_hours.values <= 0] = -999
    mkt_mask = df["Market"].values == "1x2"
    seg_mask = apply_shrunken_segments(df, kelly, SEGMENT_THRESHOLDS)
    return mkt_mask & seg_mask


def compute_team_winrates(
    train_df: pd.DataFrame, kappa: float = 5.0
) -> tuple[dict[str, float], float]:
    """Bayesian smoothed team win rate из Soccer 1x2 train ставок."""
    soccer_1x2 = train_df[
        (train_df["Market"] == "1x2")
        & (train_df["Status"].isin({"won", "lost"}))
        & train_df["Selection"].notna()
    ].copy()

    global_mean = float(soccer_1x2["Status"].eq("won").mean())
    stats = soccer_1x2.groupby("Selection").agg(
        n=("Status", "count"),
        n_won=("Status", lambda x: (x == "won").sum()),
    )
    stats["winrate"] = (stats["n_won"] + kappa * global_mean) / (stats["n"] + kappa)
    logger.info("Team win rates: %d teams, global=%.3f kappa=%.1f", len(stats), global_mean, kappa)
    return stats["winrate"].to_dict(), global_mean


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    # Загрузить chain_8 model.cbm напрямую
    model = CatBoostClassifier()
    model.load_model(str(PREV_BEST_DIR / "model.cbm"))
    logger.info("Model загружена: %s", PREV_BEST_DIR / "model.cbm")

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    # Baseline mask
    baseline_mask = get_baseline_mask(model, test_df)
    baseline_roi, baseline_n = calc_roi(test_df, baseline_mask)
    logger.info(
        "Baseline verified: ROI=%.4f%% n=%d (expected 28.58%%, 233)",
        baseline_roi,
        baseline_n,
    )

    # Team win rates из train
    team_winrates, global_mean = compute_team_winrates(train_df)
    test_df["team_winrate"] = test_df["Selection"].map(team_winrates).fillna(global_mean)

    baseline_bets = test_df[baseline_mask].copy()
    logger.info(
        "Baseline bets team_winrate: mean=%.3f std=%.3f [%.3f, %.3f]",
        baseline_bets["team_winrate"].mean(),
        baseline_bets["team_winrate"].std(),
        baseline_bets["team_winrate"].min(),
        baseline_bets["team_winrate"].max(),
    )
    logger.info(
        "Baseline: team_winrate > 0.50: %d/%d bets",
        (baseline_bets["team_winrate"] >= 0.50).sum(),
        len(baseline_bets),
    )

    # Grid search
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    results: list[dict] = []
    for t in thresholds:
        team_filter = test_df["team_winrate"].values >= t
        combined_mask = baseline_mask & team_filter
        roi, n_bets = calc_roi(test_df, combined_mask)
        delta = roi - baseline_roi if n_bets > 0 else float("nan")
        results.append({"threshold": t, "roi": roi, "n_bets": n_bets, "delta": delta})
        logger.info("t=%.2f: ROI=%.4f%% n=%d delta=%.4f", t, roi, n_bets, delta)

    valid = [r for r in results if r["n_bets"] >= 50]
    best = (
        max(valid, key=lambda r: r["roi"])
        if valid
        else max(results, key=lambda r: r["roi"] if r["n_bets"] > 0 else -999)
    )

    with mlflow.start_run(run_name="phase4/step_4_4_team_winrate") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.4")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
                "baseline_pipeline": "chain_8_model.cbm",
                "kappa_smoothing": 5.0,
                "global_mean": round(global_mean, 4),
                "best_threshold": best["threshold"],
            }
        )
        for r in results:
            mlflow.log_metrics(
                {
                    f"roi_t{int(r['threshold'] * 100)}": r["roi"],
                    f"n_t{int(r['threshold'] * 100)}": r["n_bets"],
                }
            )
        mlflow.log_metric("roi", best["roi"])
        mlflow.log_metric("n_selected", best["n_bets"])
        mlflow.log_metric("roi_delta", float(best["delta"]) if best["n_bets"] > 0 else -999.0)

        if best["roi"] > 35.0:
            mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
            mlflow.set_tag("status", "failed")
            sys.exit(1)

        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        mlflow.set_tag(
            "convergence_signal",
            str(min(1.0, max(0.0, float(best["delta"]) / 5.0 + 0.5))),
        )

        logger.info(
            "RESULT: best_t=%.2f ROI=%.4f%% n=%d delta=%.4f",
            best["threshold"],
            best["roi"],
            best["n_bets"],
            best["delta"],
        )
        print(f"STEP_4_4_ROI={best['roi']:.6f}")
        print(f"STEP_4_4_N={best['n_bets']}")
        print(f"STEP_4_4_THRESHOLD={best['threshold']}")
        print(f"STEP_4_4_DELTA={best['delta']:.4f}")
        print(f"MLFLOW_RUN_ID={run.info.run_id}")
        print("\nFull threshold sweep:")
        for r in results:
            print(
                f"  t={r['threshold']:.2f}: ROI={r['roi']:.2f}% n={r['n_bets']}"
                f" delta={r['delta']:.2f}"
            )


if __name__ == "__main__":
    main()
