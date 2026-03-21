"""Step 4.15 — Temporal analysis: объяснить рост ROI с Kelly threshold.

Гипотеза: ROI монотонно растёт с Kelly percentile (p75=30.9% → p95=66.4%).
Возможны два объяснения:
  A) Высокий Kelly действительно выбирает лучшие ставки (generalizable signal)
  B) Высокий Kelly концентрирует ставки в "горячем" Q3-Q4 периоде (Feb 21-22)
     → temporal overfitting, не generalizable

Метод:
- Для каждого percentile p80..p90: проверить концентрацию бетов в test Q1-Q4
- Q1=Feb 20 00:00-12:00, Q2=Feb 20 12:00-Feb 21 00:00,
  Q3=Feb 21 00:00-12:00, Q4=Feb 21 12:00-Feb 22
- Если p85+ ставки непропорционально концентрируются в Q3-Q4 → объяснение B
- Если распределение стабильно → объяснение A, сигнал может быть generalizable

Anti-leakage: только descriptive analysis, не выбираем threshold по ROI.
Результат используется для понимания, не для trade.

Baseline: ROI=28.5833% (n=233)
Best: ROI=33.3538% (n=148, p80)
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


def temporal_breakdown(df: pd.DataFrame, mask: np.ndarray, label: str) -> dict:
    """ROI breakdown по quartile тестового периода."""
    test_start = df["Created_At"].min()
    test_end = df["Created_At"].max()
    period = test_end - test_start
    q1_end = test_start + period * 0.25
    q2_end = test_start + period * 0.50
    q3_end = test_start + period * 0.75

    results = {}
    for q_name, q_mask in [
        ("Q1", df["Created_At"] < q1_end),
        ("Q2", (df["Created_At"] >= q1_end) & (df["Created_At"] < q2_end)),
        ("Q3", (df["Created_At"] >= q2_end) & (df["Created_At"] < q3_end)),
        ("Q4", df["Created_At"] >= q3_end),
    ]:
        combined = mask & q_mask.values
        roi, n_q = calc_roi(df, combined)
        pct_of_total = n_q / mask.sum() * 100 if mask.sum() > 0 else 0.0
        results[q_name] = {"roi": roi, "n": n_q, "pct_of_total": pct_of_total}
        logger.info("  %s %s: ROI=%.2f%% n=%d (%.1f%%)", label, q_name, roi, n_q, pct_of_total)
    return results


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

    x_test = build_features(test_df)[FEATURE_NAMES]
    proba_test = model.predict_proba(x_test)[:, 1]
    kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
    lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly_test[lead_hours_test.values <= 0] = -999

    mkt_mask = test_df["Market"].values == "1x2"

    # Test period info
    logger.info("Test period: %s – %s", test_df["Created_At"].min(), test_df["Created_At"].max())

    # Temporal breakdown для p80, p85, p90
    analysis_results: dict[int, dict] = {}
    for pct in [80, 85, 90]:
        t = float(np.percentile(kelly_train_low, pct))
        buckets = pd.cut(
            test_df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"]
        )
        seg_mask = (
            ((buckets == "low").values & (kelly_test >= t))
            | ((buckets == "mid").values & (kelly_test >= 0.545))
            | ((buckets == "high").values & (kelly_test >= 0.325))
        )
        final_mask = mkt_mask & seg_mask
        roi, n_total = calc_roi(test_df, final_mask)
        logger.info("p%d (t=%.4f): ROI=%.4f%% n=%d", pct, t, roi, n_total)
        q_results = temporal_breakdown(test_df, final_mask, f"p{pct}")
        analysis_results[pct] = {
            "threshold": t,
            "roi": roi,
            "n": n_total,
            "quartiles": q_results,
        }

    # Вычислить Q3+Q4 concentration
    p80_q34_pct = (
        analysis_results[80]["quartiles"]["Q3"]["pct_of_total"]
        + analysis_results[80]["quartiles"]["Q4"]["pct_of_total"]
    )
    p85_q34_pct = (
        analysis_results[85]["quartiles"]["Q3"]["pct_of_total"]
        + analysis_results[85]["quartiles"]["Q4"]["pct_of_total"]
    )
    p90_q34_pct = (
        analysis_results[90]["quartiles"]["Q3"]["pct_of_total"]
        + analysis_results[90]["quartiles"]["Q4"]["pct_of_total"]
    )

    logger.info(
        "Q3+Q4 concentration: p80=%.1f%%, p85=%.1f%%, p90=%.1f%%",
        p80_q34_pct,
        p85_q34_pct,
        p90_q34_pct,
    )

    if p85_q34_pct > p80_q34_pct + 5.0:
        temporal_explanation = "B_temporal_overfitting"
        logger.info("Вывод B: p85+ концентрируется в Q3-Q4 → temporal overfitting")
    else:
        temporal_explanation = "A_genuine_signal"
        logger.info("Вывод A: концентрация стабильна → Kelly signal generalizable")

    # Train Q3+Q4 analysis для сравнения
    lead_hours_train_ser = pd.Series(lead_hours_train.values, index=train_df.index)
    train_mkt = (train_df["Market"].values == "1x2") & (lead_hours_train_ser.values > 0)
    t_p80 = float(np.percentile(kelly_train_low, 80))
    train_buckets = pd.cut(
        train_df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"]
    )
    train_seg = (
        ((train_buckets == "low").values & (kelly_train >= t_p80))
        | ((train_buckets == "mid").values & (kelly_train >= 0.545))
        | ((train_buckets == "high").values & (kelly_train >= 0.325))
    )
    train_final = train_mkt & train_seg
    train_roi, train_n = calc_roi(train_df, train_final)
    logger.info("Train p80 ROI=%.4f%% n=%d (in-sample reference)", train_roi, train_n)

    train_q = temporal_breakdown(train_df, train_final, "train_p80")
    train_q34_pct = train_q["Q3"]["pct_of_total"] + train_q["Q4"]["pct_of_total"]
    logger.info("Train Q3+Q4 concentration: %.1f%%", train_q34_pct)

    roi_primary = analysis_results[80]["roi"]
    baseline_roi = 28.5833
    delta = roi_primary - baseline_roi

    with mlflow.start_run(run_name="phase4/step_4_15_temporal_kelly") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "analysis")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.15")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "baseline_model": "chain_8_model.cbm",
                "analysis_type": "temporal_kelly_breakdown",
                "temporal_explanation": temporal_explanation,
            }
        )

        try:
            mlflow.log_metrics(
                {
                    "roi": roi_primary,
                    "n_selected": analysis_results[80]["n"],
                    "roi_delta": delta,
                    "p80_q34_pct": p80_q34_pct,
                    "p85_q34_pct": p85_q34_pct,
                    "p90_q34_pct": p90_q34_pct,
                    "train_p80_roi": train_roi,
                    "train_p80_q34_pct": train_q34_pct,
                }
            )
            mlflow.set_tag("temporal_explanation", temporal_explanation)
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            # Полный отчёт
            report_lines = ["Temporal Kelly Analysis Report", "=" * 40]
            for pct in [80, 85, 90]:
                r = analysis_results[pct]
                report_lines.append(
                    f"\np{pct} (t={r['threshold']:.4f}): ROI={r['roi']:.2f}% n={r['n']}"
                )
                for q, qr in r["quartiles"].items():
                    pct_str = f"{qr['pct_of_total']:.1f}%"
                    report_lines.append(
                        f"  {q}: ROI={qr['roi']:.2f}% n={qr['n']} ({pct_str} of total)"
                    )
            q34_line = (
                f"\nQ3+Q4 concentration: "
                f"p80={p80_q34_pct:.1f}%, p85={p85_q34_pct:.1f}%, p90={p90_q34_pct:.1f}%"
            )
            report_lines.append(q34_line)
            report_lines.append(f"Train p80 Q3+Q4: {train_q34_pct:.1f}%")
            report_lines.append(f"\nConclusion: {temporal_explanation}")
            mlflow.log_text("\n".join(report_lines), "temporal_analysis.txt")

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            print(f"STEP_4_15_ROI={roi_primary:.6f}")
            print(f"STEP_4_15_N={analysis_results[80]['n']}")
            print(f"STEP_4_15_TEMPORAL_EXPLANATION={temporal_explanation}")
            print(f"STEP_4_15_P80_Q34_PCT={p80_q34_pct:.1f}")
            print(f"STEP_4_15_P85_Q34_PCT={p85_q34_pct:.1f}")
            print(f"STEP_4_15_P90_Q34_PCT={p90_q34_pct:.1f}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.15")
            raise


if __name__ == "__main__":
    main()
