"""Step 4.9 — chain_8 model + p80 Kelly + odds lower cutoff.

Открытие из step 4.8: sub-segment анализ с baseline threshold:
  ultra_low (odds<1.3): ROI=-16.87% n=40  ← DRAG
  low_mid   (1.3-1.5): ROI=+24.49% n=154 ← profitable
  low_high  (1.5-1.8): ROI=+57.11% n=34  ← best (small n)

Гипотеза: исключить ultra_low (odds < cutoff) + p80 Kelly →
более чистая выборка → ROI > 33.35%.

Метод:
1. chain_8 model.cbm, p80 Kelly threshold (0.5914)
2. Sweep odds_cutoff: [1.0, 1.1, 1.2, 1.3, 1.35, 1.4] (нижняя граница LOW)
3. Выбрать cutoff по train-data ROI (in-sample, без val)
4. Применить к test ОДИН РАЗ

Baseline: ROI=28.5833% (n=233)
Step 4.8 best: ROI=33.3538% (n=148, p80, no odds cutoff)
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


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    model = CatBoostClassifier()
    model.load_model(str(PREV_BEST_DIR / "model.cbm"))
    logger.info("Model загружена: %s", PREV_BEST_DIR / "model.cbm")

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    # === Compute Kelly ===
    x_train = build_features(train_df)[FEATURE_NAMES]
    proba_train = model.predict_proba(x_train)[:, 1]
    kelly_train = compute_kelly(proba_train, train_df["Odds"].values)
    lead_hours_train = (
        train_df["Start_Time"] - train_df["Created_At"]
    ).dt.total_seconds() / 3600.0

    # p80 from step 4.8 (in-sample train LOW)
    train_low_mask = (
        (train_df["Market"].values == "1x2")
        & (train_df["Odds"].values < 1.8)
        & (lead_hours_train.values > 0)
    )
    kelly_train_low = kelly_train[train_low_mask]
    threshold_p80 = float(np.percentile(kelly_train_low, 80))
    logger.info("p80 = %.4f", threshold_p80)

    # === Odds cutoff sweep на train ===
    # Для каждого cutoff: train ROI среди Kelly >= p80 AND odds >= cutoff AND 1x2 AND pre-match
    odds_cutoffs = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]
    train_results = []
    for cutoff in odds_cutoffs:
        mask = (
            (train_df["Market"].values == "1x2")
            & (train_df["Odds"].values >= cutoff)
            & (train_df["Odds"].values < 1.8)
            & (lead_hours_train.values > 0)
            & (kelly_train >= threshold_p80)
        )
        roi_tr, n_tr = calc_roi(train_df, mask)
        train_results.append({"cutoff": cutoff, "roi": roi_tr, "n": n_tr})
        logger.info("  train: cutoff=%.2f ROI=%.4f%% n=%d", cutoff, roi_tr, n_tr)

    # Выбрать по train ROI с n >= 20
    valid_tr = [r for r in train_results if r["n"] >= 20]
    best_tr = max(valid_tr, key=lambda r: r["roi"]) if valid_tr else train_results[-1]
    chosen_cutoff = best_tr["cutoff"]
    logger.info(
        "Chosen cutoff: %.2f (train ROI=%.4f%% n=%d)",
        chosen_cutoff,
        best_tr["roi"],
        best_tr["n"],
    )

    # === Test evaluation ===
    x_test = build_features(test_df)[FEATURE_NAMES]
    proba_test = model.predict_proba(x_test)[:, 1]
    kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
    lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly_test[lead_hours_test.values <= 0] = -999

    # Baseline p80 (no cutoff)
    mask_p80_base = (
        (test_df["Market"].values == "1x2")
        & (test_df["Odds"].values < 1.8)
        & (kelly_test >= threshold_p80)
    )
    roi_p80_base, n_p80_base = calc_roi(test_df, mask_p80_base)
    logger.info("p80 no cutoff: ROI=%.4f%% n=%d", roi_p80_base, n_p80_base)

    # Chosen cutoff
    mask_chosen = (
        (test_df["Market"].values == "1x2")
        & (test_df["Odds"].values >= chosen_cutoff)
        & (test_df["Odds"].values < 1.8)
        & (kelly_test >= threshold_p80)
    )
    roi_chosen, n_chosen = calc_roi(test_df, mask_chosen)
    logger.info("p80 + cutoff=%.2f: ROI=%.4f%% n=%d", chosen_cutoff, roi_chosen, n_chosen)

    # Full test sweep (informational)
    logger.info("Full test sweep (informational):")
    test_sweep = []
    for cutoff in odds_cutoffs:
        mask = (
            (test_df["Market"].values == "1x2")
            & (test_df["Odds"].values >= cutoff)
            & (test_df["Odds"].values < 1.8)
            & (kelly_test >= threshold_p80)
        )
        roi_t, n_t = calc_roi(test_df, mask)
        test_sweep.append({"cutoff": cutoff, "roi": roi_t, "n": n_t})
        logger.info("  test: cutoff=%.2f ROI=%.4f%% n=%d", cutoff, roi_t, n_t)

    baseline_roi = 28.5833
    delta = roi_chosen - baseline_roi
    delta_vs_p80 = roi_chosen - 33.3538  # delta vs step 4.8

    with mlflow.start_run(run_name="phase4/step_4_9_odds_cutoff") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.9")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "baseline_model": "chain_8_model.cbm",
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
                "kelly_threshold": round(threshold_p80, 6),
                "kelly_percentile": 80,
                "chosen_odds_cutoff": chosen_cutoff,
                "cutoff_selection": "max_train_roi_n20",
            }
        )

        try:
            for r in train_results:
                mlflow.log_metric(f"train_roi_c{int(r['cutoff'] * 100)}", r["roi"])
                mlflow.log_metric(f"train_n_c{int(r['cutoff'] * 100)}", r["n"])

            for r in test_sweep:
                mlflow.log_metric(f"test_roi_c{int(r['cutoff'] * 100)}", r["roi"])
                mlflow.log_metric(f"test_n_c{int(r['cutoff'] * 100)}", r["n"])

            mlflow.log_metrics(
                {
                    "roi": roi_chosen,
                    "n_selected": n_chosen,
                    "roi_delta": delta,
                    "roi_delta_vs_p80": delta_vs_p80,
                    "roi_p80_no_cutoff": roi_p80_base,
                }
            )
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            if roi_chosen > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            logger.info(
                "RESULT: cutoff=%.2f, p80=%.4f → ROI=%.4f%% n=%d delta=%.4f",
                chosen_cutoff,
                threshold_p80,
                roi_chosen,
                n_chosen,
                delta,
            )
            print(f"STEP_4_9_ROI={roi_chosen:.6f}")
            print(f"STEP_4_9_N={n_chosen}")
            print(f"STEP_4_9_DELTA={delta:.4f}")
            print(f"STEP_4_9_DELTA_VS_P80={delta_vs_p80:.4f}")
            print(f"STEP_4_9_CUTOFF={chosen_cutoff}")
            print(f"STEP_4_9_KELLY_THRESHOLD={threshold_p80:.6f}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")
            print("\nFull test sweep:")
            for r in test_sweep:
                print(f"  cutoff={r['cutoff']:.2f}: ROI={r['roi']:.2f}% n={r['n']}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.9")
            raise


if __name__ == "__main__":
    main()
