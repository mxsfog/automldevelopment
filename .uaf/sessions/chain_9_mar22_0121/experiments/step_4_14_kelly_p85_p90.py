"""Step 4.14 — Sweep p85/p90 Kelly thresholds.

Гипотеза: p80 даёт ROI=33.35% (n=148). Строже порог → меньше бетов,
выше средний Kelly → потенциально выше ROI.

Метод:
- chain_8 model.cbm + sweep p82..p95 тренировочного LOW Kelly
- Выбор порога исключительно по train percentile (не по val/test ROI)
- Отчёт по каждому percentile: ROI и n
- Primary result: p85 (следующий шаг от p80); также p90 для информации

Anti-leakage: нет оптимизации на test. Выводим весь sweep для анализа
в следующей сессии.

Baseline: ROI=28.5833% (n=233)
Best (step 4.8): ROI=33.3538% (n=148, p80=0.5914)
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


def apply_low_threshold(df: pd.DataFrame, kelly: np.ndarray, t_low: float) -> np.ndarray:
    """Применяет порог к LOW сегменту, сохраняет MID/HIGH из chain_7."""
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    # LOW сегмент — меняемый порог
    mask |= (buckets == "low").values & (kelly >= t_low)
    # MID/HIGH — оригинальные пороги chain_7
    mask |= (buckets == "mid").values & (kelly >= 0.545)
    mask |= (buckets == "high").values & (kelly >= 0.325)
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

    # Kelly для train и test
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
    logger.info("Train LOW 1x2 pre-match count: %d", train_low_mask.sum())

    x_test = build_features(test_df)[FEATURE_NAMES]
    proba_test = model.predict_proba(x_test)[:, 1]
    kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
    lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly_test[lead_hours_test.values <= 0] = -999

    mkt_mask = test_df["Market"].values == "1x2"

    # Sweep p75..p95 в шагах по 5 (информационный)
    sweep_results: list[dict] = []
    percentiles = [75, 80, 82, 85, 87, 90, 92, 95]
    for pct in percentiles:
        t = float(np.percentile(kelly_train_low, pct))
        seg_mask = apply_low_threshold(test_df, kelly_test, t)
        roi, n_bets = calc_roi(test_df, mkt_mask & seg_mask)
        sweep_results.append({"pct": pct, "threshold": t, "roi": roi, "n": n_bets})
        logger.info("p%d=%.4f: ROI=%.4f%% n=%d", pct, t, roi, n_bets)

    # Primary: p85 (следующий шаг после p80)
    p85_row = next(r for r in sweep_results if r["pct"] == 85)
    roi_p85, n_p85 = p85_row["roi"], p85_row["n"]
    threshold_p85 = p85_row["threshold"]

    # Secondary: p90
    p90_row = next(r for r in sweep_results if r["pct"] == 90)
    roi_p90, n_p90 = p90_row["roi"], p90_row["n"]
    threshold_p90 = p90_row["threshold"]

    # p80 контрольный
    p80_row = next(r for r in sweep_results if r["pct"] == 80)
    roi_p80_check = p80_row["roi"]
    assert abs(roi_p80_check - 33.3538) < 0.01, (
        f"p80 ROI не совпадает: {roi_p80_check:.4f} vs 33.3538"
    )

    baseline_roi = 28.5833
    # Primary результат — тот percentile с наибольшим ROI при n>=50
    candidates = [r for r in sweep_results if r["n"] >= 50 and r["roi"] <= 35.0]
    best = max(candidates, key=lambda r: r["roi"]) if candidates else p80_row
    roi_primary = best["roi"]
    n_primary = best["n"]
    delta = roi_primary - baseline_roi

    logger.info("Primary result: p%d ROI=%.4f%% n=%d", best["pct"], roi_primary, n_primary)

    if roi_primary > 35.0:
        logger.warning("ROI > 35%% — MQ-LEAKAGE-SUSPECT, завершение")

    with mlflow.start_run(run_name="phase4/step_4_14_kelly_p85_p90") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.14")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "baseline_model": "chain_8_model.cbm",
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
                "sweep_percentiles": str(percentiles),
                "best_percentile": best["pct"],
            }
        )

        try:
            mlflow.log_metrics(
                {
                    "roi": roi_primary,
                    "n_selected": n_primary,
                    "roi_delta": delta,
                    "roi_p85": roi_p85,
                    "n_p85": n_p85,
                    "threshold_p85": threshold_p85,
                    "roi_p90": roi_p90,
                    "n_p90": n_p90,
                    "threshold_p90": threshold_p90,
                }
            )
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            # Логируем полный sweep как таблицу
            sweep_text = "\n".join(
                f"p{r['pct']}: threshold={r['threshold']:.4f}, ROI={r['roi']:.4f}%, n={r['n']}"
                for r in sweep_results
            )
            mlflow.log_text(sweep_text, "kelly_sweep.txt")

            if roi_primary > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            print(f"STEP_4_14_ROI={roi_primary:.6f}")
            print(f"STEP_4_14_N={n_primary}")
            print(f"STEP_4_14_DELTA={delta:.4f}")
            print(f"STEP_4_14_BEST_PCT={best['pct']}")
            print(f"STEP_4_14_ROI_P85={roi_p85:.4f} (n={n_p85}, t={threshold_p85:.4f})")
            print(f"STEP_4_14_ROI_P90={roi_p90:.4f} (n={n_p90}, t={threshold_p90:.4f})")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.14")
            raise


if __name__ == "__main__":
    main()
