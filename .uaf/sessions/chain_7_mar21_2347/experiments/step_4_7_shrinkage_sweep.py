"""Step 4.7 — Shrinkage sweep: найти оптимальный shrinkage на val, применить на test.

Метод: НЕ выбираем shrinkage по результатам на test (это leakage).
Оцениваем разные shrinkage на val через cross-val по последним 16% (64-80%).
Выбираем лучший shrinkage по средней валидационной метрике.
Применяем на test ОДИН раз.
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


def apply_shrunken_segments(
    df: pd.DataFrame,
    kelly: np.ndarray,
    raw_thresholds: dict[str, float],
    baseline_t: float,
    shrink: float,
) -> np.ndarray:
    """Применение shrunken segment thresholds."""
    shrunken = {k: baseline_t + shrink * (v - baseline_t) for k, v in raw_thresholds.items()}
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in shrunken.items():
        seg_m = (buckets == bucket).values & (kelly >= t)
        mask |= seg_m
    return mask


def find_segment_thresholds_on_val(
    df: pd.DataFrame,
    kelly: np.ndarray,
    min_bets: int = 30,
) -> dict[str, float]:
    """Поиск segment thresholds на конкретном val-сплите."""
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    thresholds = {}
    for bucket in ["low", "mid", "high"]:
        b_mask = (buckets == bucket).values
        b_kelly = kelly.copy()
        b_kelly[~b_mask] = -999
        best_roi, best_t = -999.0, 0.455
        for t in np.arange(0.01, 0.70, 0.005):
            mask = b_kelly >= t
            if mask.sum() < min_bets:
                break
            roi, _ = calc_roi(df, mask)
            if roi > best_roi:
                best_roi = roi
                best_t = t
        thresholds[bucket] = best_t
    return thresholds


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


with mlflow.start_run(run_name="phase4/step4.7_shrinkage_sweep") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.7")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        # Val window: 64-80% (полный val для поиска raw thresholds)
        val_start = int(n * 0.64)
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        # Sub-splits val для оценки shrinkage:
        # val_train: 64-72% (для segment thresholds)
        # val_eval: 72-80% (для оценки shrinkage)
        val_train_end = int(n * 0.72)
        val_train_df = df_raw.iloc[val_start:val_train_end].copy()
        val_eval_df = df_raw.iloc[val_train_end:train_end].copy()

        logger.info(
            "Val_train: %d, Val_eval: %d, Test: %d",
            len(val_train_df),
            len(val_eval_df),
            len(test_df),
        )

        model = CatBoostClassifier()
        model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_vt, _ = build_features(val_train_df)
        x_ve, _ = build_features(val_eval_df)
        x_te, _ = build_features(test_df)

        proba_vt = model.predict_proba(x_vt)[:, 1]
        proba_ve = model.predict_proba(x_ve)[:, 1]
        proba_test = model.predict_proba(x_te)[:, 1]

        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, proba_test)

        kelly_vt = compute_kelly(proba_vt, val_train_df["Odds"].values)
        kelly_vt[val_train_df["lead_hours"].values <= 0] = -999

        kelly_ve = compute_kelly(proba_ve, val_eval_df["Odds"].values)
        kelly_ve[val_eval_df["lead_hours"].values <= 0] = -999

        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_test[test_df["lead_hours"].values <= 0] = -999

        # Step 1: найти raw segment thresholds на val_train (64-72%)
        logger.info("Поиск raw segment thresholds на val_train (64-72%%)...")
        raw_thresholds = find_segment_thresholds_on_val(val_train_df, kelly_vt, min_bets=30)
        logger.info("Raw thresholds: %s", raw_thresholds)

        # Step 2: sweep shrinkage на val_eval (72-80%) — без test!
        baseline_t = 0.455
        shrink_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        val_eval_results = []

        for shrink in shrink_values:
            mask = apply_shrunken_segments(
                val_eval_df, kelly_ve, raw_thresholds, baseline_t, shrink
            )
            roi, n_sel = calc_roi(val_eval_df, mask)
            val_eval_results.append((shrink, roi, n_sel))
            logger.info("Shrink=%.1f: val_eval_roi=%.2f%%, n=%d", shrink, roi, n_sel)
            mlflow.log_metrics({f"val_eval_roi_s{int(shrink * 10)}": roi})

        # Step 3: выбор лучшего shrinkage по val_eval
        best_shrink, best_val_roi, _ = max(val_eval_results, key=lambda x: x[1])
        logger.info("Лучший shrink по val_eval: %.1f (roi=%.2f%%)", best_shrink, best_val_roi)

        # Step 4: теперь пересчитываем thresholds на ПОЛНОМ val (64-80%) для теста
        x_full_val, _ = build_features(val_df)
        proba_full_val = model.predict_proba(x_full_val)[:, 1]
        kelly_full_val = compute_kelly(proba_full_val, val_df["Odds"].values)
        kelly_full_val[val_df["lead_hours"].values <= 0] = -999

        full_val_thresholds = find_segment_thresholds_on_val(val_df, kelly_full_val, min_bets=30)
        logger.info("Full val thresholds: %s", full_val_thresholds)

        # Step 5: применяем на test с best_shrink
        test_mask = apply_shrunken_segments(
            test_df, kelly_test, full_val_thresholds, baseline_t, best_shrink
        )
        test_roi, test_n = calc_roi(test_df, test_mask)
        baseline_roi = 24.9088
        delta = test_roi - baseline_roi

        logger.info(
            "Test ROI: %.4f%% (n=%d), shrink=%.1f, delta=%.4f%%",
            test_roi,
            test_n,
            best_shrink,
            delta,
        )

        if test_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", test_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": train_end,
                "n_samples_val_train": len(val_train_df),
                "n_samples_val_eval": len(val_eval_df),
                "best_shrink": best_shrink,
                "baseline_threshold": baseline_t,
            }
        )
        mlflow.log_metrics(
            {
                "roi": test_roi,
                "n_selected": test_n,
                "auc_test": auc_test,
                "best_shrink": best_shrink,
                "best_val_eval_roi": best_val_roi,
                "baseline_roi": baseline_roi,
                "delta_vs_baseline": delta,
            }
        )

        current_best_roi = 26.9345
        if test_roi > current_best_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", test_roi, current_best_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(best_dir / "model.cbm"))
            meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
            new_meta = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": test_roi,
                "auc": float(auc_test),
                "threshold": None,
                "segment_thresholds": {
                    k: baseline_t + best_shrink * (v - baseline_t)
                    for k, v in full_val_thresholds.items()
                },
                "shrinkage": best_shrink,
                "n_bets": test_n,
                "feature_names": meta["feature_names"],
                "params": meta["params"],
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.7",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(
            f"step4.7 DONE: test_roi={test_roi:.4f}%, shrink={best_shrink:.1f}, "
            f"n={test_n}, delta={delta:.4f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
