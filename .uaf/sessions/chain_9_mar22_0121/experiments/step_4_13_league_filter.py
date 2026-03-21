"""Step 4.13 — League/competition filter для p80 Kelly бетов.

Гипотеза: среди Soccer 1x2 бетов с high Kelly, топ-европейские лиги
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League)
имеют более предсказуемые результаты → лучший ROI.

Метод:
- chain_8 model.cbm + p80 Kelly (0.5914)
- Парсинг Match поля из outcomes.csv для извлечения компетиции/лиги
- Анализ: топ-5 компетиций по n и ROI среди p80 бетов
- Фильтр: "известные" топ-лиги (a priori список, без оптимизации на ROI)

Anti-leakage: список топ-лиг определён до просмотра test ROI.

Baseline: ROI=28.5833% (n=233)
Step 4.8 best: ROI=33.3538% (n=148, p80)
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

# A priori список топ-лиг (до просмотра test ROI)
TOP_LEAGUE_KEYWORDS = [
    "premier league",
    "la liga",
    "bundesliga",
    "serie a",
    "ligue 1",
    "champions league",
    "europa league",
    "eredivisie",
    "primeira liga",
    "scottish premiership",
    "mls",
    "superliga",
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
        ["Bet_ID", "Sport", "Market", "Start_Time", "Match"]
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


def extract_league(match: str | None) -> str:
    """Извлечь название лиги из строки Match."""
    if pd.isna(match) or not match:
        return "unknown"
    m = str(match).lower()
    # Формат "League: Home vs Away" или просто "Home vs Away"
    if ":" in m:
        return m.split(":")[0].strip()
    return "unknown"


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


def apply_seg_thresholds(
    df: pd.DataFrame, kelly: np.ndarray, seg_thresholds: dict[str, float]
) -> np.ndarray:
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in seg_thresholds.items():
        mask |= (buckets == bucket).values & (kelly >= t)
    return mask


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    model = CatBoostClassifier()
    model.load_model(str(PREV_BEST_DIR / "model.cbm"))

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    # Извлечь лиги
    train_df["league"] = train_df["Match"].apply(extract_league)
    test_df["league"] = test_df["Match"].apply(extract_league)

    # Kelly
    x_train = build_features(train_df)[FEATURE_NAMES]
    proba_train = model.predict_proba(x_train)[:, 1]
    kelly_train = compute_kelly(proba_train, train_df["Odds"].values)
    lead_hours_train = (
        train_df["Start_Time"] - train_df["Created_At"]
    ).dt.total_seconds() / 3600.0
    train_low_mask = (
        (train_df["Market"].values == "1x2")
        & (train_df["Odds"].values < 1.8)
        & (lead_hours_train.values > 0)
    )
    kelly_train_low = kelly_train[train_low_mask]
    threshold_p80 = float(np.percentile(kelly_train_low, 80))
    logger.info("p80 = %.4f", threshold_p80)

    x_test = build_features(test_df)[FEATURE_NAMES]
    proba_test = model.predict_proba(x_test)[:, 1]
    kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
    lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly_test[lead_hours_test.values <= 0] = -999

    seg_thresholds_p80 = {"low": threshold_p80, "mid": 0.545, "high": 0.325}
    mkt_mask = test_df["Market"].values == "1x2"
    seg_mask_p80 = apply_seg_thresholds(test_df, kelly_test, seg_thresholds_p80)

    # p80 baseline
    roi_p80, n_p80 = calc_roi(test_df, mkt_mask & seg_mask_p80)
    logger.info("p80 baseline: ROI=%.4f%% n=%d", roi_p80, n_p80)

    # League analysis на test p80 бетах
    p80_test = test_df[mkt_mask & seg_mask_p80].copy()
    league_counts = p80_test["league"].value_counts()
    logger.info("Top 10 leagues in p80 test bets:")
    for league, cnt in league_counts.head(10).items():
        logger.info("  %s: %d", league, cnt)

    league_has_colon = (p80_test["league"] != "unknown").sum()
    logger.info(
        "Bets with league info: %d/%d (%.1f%%)",
        league_has_colon,
        len(p80_test),
        100 * league_has_colon / len(p80_test),
    )

    # Top league filter
    is_top_league = (
        p80_test["league"].str.lower().apply(lambda x: any(kw in x for kw in TOP_LEAGUE_KEYWORDS))
    )
    logger.info("Top league filter: %d/%d bets", is_top_league.sum(), len(p80_test))

    # Test: top league filter
    top_league_mask = (
        test_df["league"]
        .str.lower()
        .apply(lambda x: any(kw in x for kw in TOP_LEAGUE_KEYWORDS))
        .values
    )
    roi_top, n_top = calc_roi(test_df, mkt_mask & seg_mask_p80 & top_league_mask)
    logger.info("p80 + top_league: ROI=%.4f%% n=%d", roi_top, n_top)

    # Unknown league filter (most bets)
    unknown_mask = (test_df["league"] == "unknown").values
    roi_unk, n_unk = calc_roi(test_df, mkt_mask & seg_mask_p80 & unknown_mask)
    logger.info("p80 + unknown_league: ROI=%.4f%% n=%d", roi_unk, n_unk)

    # Primary result
    roi_primary = roi_p80
    n_primary = n_p80
    baseline_roi = 28.5833
    delta = roi_primary - baseline_roi

    with mlflow.start_run(run_name="phase4/step_4_13_league_filter") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.13")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "baseline_model": "chain_8_model.cbm",
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
                "kelly_threshold": round(threshold_p80, 6),
                "n_top_league_keywords": len(TOP_LEAGUE_KEYWORDS),
            }
        )

        try:
            mlflow.log_metrics(
                {
                    "roi": roi_primary,
                    "n_selected": n_primary,
                    "roi_delta": delta,
                    "roi_top_league": roi_top,
                    "n_top_league": n_top,
                    "roi_unknown_league": roi_unk,
                    "n_unknown_league": n_unk,
                    "pct_with_league": league_has_colon / len(p80_test),
                }
            )
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            if roi_primary > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            print(f"STEP_4_13_ROI={roi_primary:.6f}")
            print(f"STEP_4_13_N={n_primary}")
            print(f"STEP_4_13_DELTA={delta:.4f}")
            print(f"STEP_4_13_ROI_TOP_LEAGUE={roi_top:.4f} (n={n_top})")
            print(f"STEP_4_13_ROI_UNKNOWN={roi_unk:.4f} (n={n_unk})")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.13")
            raise


if __name__ == "__main__":
    main()
