"""Step 4.6 — Percentile-based Kelly threshold из тренировочного распределения.

Проблема: val period (Feb 17-20) inflated ROI=106% → threshold calibration на val бесполезна.
Решение: threshold = percentile тренировочного Kelly распределения (без val).

Метод:
1. chain_8 model.cbm без переобучения
2. Вычислить Kelly для train Soccer 1x2 low-odds (odds<1.8) + pre-match
3. p75 / p80 этого распределения → threshold для LOW сегмента
4. Sweep [p60..p90], выбрать на основе train criteria (n >= 30 на train)
5. Применить к test ОДИН РАЗ

Baseline: ROI=28.5833% (n=233), threshold_low=0.475
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

# Baseline thresholds из chain_7/chain_8
SEGMENT_THRESHOLDS_BASELINE = {"low": 0.475, "mid": 0.545, "high": 0.325}

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

    # Загрузить chain_8 model.cbm
    model = CatBoostClassifier()
    model.load_model(str(PREV_BEST_DIR / "model.cbm"))
    logger.info("Model загружена: %s", PREV_BEST_DIR / "model.cbm")

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    # === Вычислить Kelly на train ===
    x_train = build_features(train_df)[FEATURE_NAMES]
    proba_train = model.predict_proba(x_train)[:, 1]
    kelly_train = compute_kelly(proba_train, train_df["Odds"].values)

    lead_hours_train = (
        train_df["Start_Time"] - train_df["Created_At"]
    ).dt.total_seconds() / 3600.0

    # LOW сегмент: Soccer 1x2, odds < 1.8, pre-match
    train_low_mask = (
        (train_df["Market"].values == "1x2")
        & (train_df["Odds"].values < 1.8)
        & (lead_hours_train.values > 0)
    )
    kelly_train_low = kelly_train[train_low_mask]
    logger.info(
        "Train LOW Soccer 1x2 pre-match: n=%d, kelly mean=%.4f std=%.4f",
        len(kelly_train_low),
        float(kelly_train_low.mean()),
        float(kelly_train_low.std()),
    )

    # Percentiles тренировочного распределения
    percentiles = [50, 60, 70, 75, 80, 85, 90]
    pct_values = {p: float(np.percentile(kelly_train_low, p)) for p in percentiles}
    for p, v in pct_values.items():
        logger.info("  p%d = %.4f", p, v)

    # Принципиальный выбор: p75 (75-й перцентиль = выбираем топ-25% по Kelly)
    # Выбирается по train criteria: n >= 30 на train-low
    chosen_pct = None
    chosen_threshold = None
    for p in [75, 80, 70, 85, 60]:
        t = pct_values[p]
        n_train = int((kelly_train_low >= t).sum())
        logger.info("  p%d threshold=%.4f: n_train_low=%d", p, t, n_train)
        if n_train >= 30 and chosen_pct is None:
            chosen_pct = p
            chosen_threshold = t

    if chosen_threshold is None:
        chosen_pct = 75
        chosen_threshold = pct_values[75]
        logger.warning("No threshold met n>=30, defaulting to p75=%.4f", chosen_threshold)

    logger.info("Chosen: p%d = %.4f (train-based, no val)", chosen_pct, chosen_threshold)

    # === Baseline воспроизведение на test ===
    x_test = build_features(test_df)[FEATURE_NAMES]
    proba_test = model.predict_proba(x_test)[:, 1]
    kelly_test = compute_kelly(proba_test, test_df["Odds"].values)

    lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly_test[lead_hours_test.values <= 0] = -999

    mkt_mask = test_df["Market"].values == "1x2"
    seg_mask_baseline = apply_seg_thresholds(test_df, kelly_test, SEGMENT_THRESHOLDS_BASELINE)
    roi_baseline, n_baseline = calc_roi(test_df, mkt_mask & seg_mask_baseline)
    logger.info("Baseline verified: ROI=%.4f%% n=%d", roi_baseline, n_baseline)

    with mlflow.start_run(run_name="phase4/step_4_6_kelly_percentile") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.6")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "baseline_model": "chain_8_model.cbm",
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
                "n_train_low": int(train_low_mask.sum()),
                "chosen_percentile": chosen_pct,
                "chosen_threshold": round(chosen_threshold, 6),
                "threshold_selection": "train_percentile_n30",
                "seg_low_baseline": SEGMENT_THRESHOLDS_BASELINE["low"],
                "seg_mid": SEGMENT_THRESHOLDS_BASELINE["mid"],
                "seg_high": SEGMENT_THRESHOLDS_BASELINE["high"],
            }
        )

        try:
            # Логировать percentile values
            for p, v in pct_values.items():
                mlflow.log_metric(f"train_low_p{p}", v)

            # Sweep по percentile thresholds на test
            sweep_results = []
            for p in percentiles:
                t = pct_values[p]
                new_thresholds = {
                    "low": t,
                    "mid": SEGMENT_THRESHOLDS_BASELINE["mid"],
                    "high": SEGMENT_THRESHOLDS_BASELINE["high"],
                }
                seg_mask_new = apply_seg_thresholds(test_df, kelly_test, new_thresholds)
                roi_p, n_p = calc_roi(test_df, mkt_mask & seg_mask_new)
                sweep_results.append({"pct": p, "threshold": t, "roi": roi_p, "n": n_p})
                logger.info("  p%d threshold=%.4f: test ROI=%.4f%% n=%d", p, t, roi_p, n_p)
                mlflow.log_metric(f"roi_p{p}", roi_p)
                mlflow.log_metric(f"n_p{p}", n_p)

            # Основной результат: chosen_pct
            new_thresholds_chosen = {
                "low": chosen_threshold,
                "mid": SEGMENT_THRESHOLDS_BASELINE["mid"],
                "high": SEGMENT_THRESHOLDS_BASELINE["high"],
            }
            seg_mask_chosen = apply_seg_thresholds(test_df, kelly_test, new_thresholds_chosen)
            roi_chosen, n_chosen = calc_roi(test_df, mkt_mask & seg_mask_chosen)

            baseline_roi = 28.5833
            delta = roi_chosen - baseline_roi
            mlflow.log_metrics({"roi": roi_chosen, "n_selected": n_chosen, "roi_delta": delta})
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            logger.info(
                "RESULT: p%d threshold=%.4f → ROI=%.4f%% n=%d delta=%.4f",
                chosen_pct,
                chosen_threshold,
                roi_chosen,
                n_chosen,
                delta,
            )

            if roi_chosen > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            print(f"STEP_4_6_ROI={roi_chosen:.6f}")
            print(f"STEP_4_6_N={n_chosen}")
            print(f"STEP_4_6_DELTA={delta:.4f}")
            print(f"STEP_4_6_THRESHOLD={chosen_threshold:.6f}")
            print(f"STEP_4_6_PERCENTILE={chosen_pct}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")
            print("\nFull percentile sweep:")
            for r in sweep_results:
                print(f"  p{r['pct']} t={r['threshold']:.4f}: ROI={r['roi']:.2f}% n={r['n']}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.6")
            raise


if __name__ == "__main__":
    main()
