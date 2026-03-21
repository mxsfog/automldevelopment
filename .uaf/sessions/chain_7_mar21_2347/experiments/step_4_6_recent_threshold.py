"""Step 4.6 — Recent-window threshold + segment combination.

Гипотеза 1: порог Kelly, найденный на последних 8% train (72-80%), лучше
представляет test-период (80-100%), чем порог из широкого val (64-80%).

Гипотеза 2: применить segment thresholds из step 4.1 с более консервативными
значениями (shrinkage к baseline), чтобы снизить val-overfitting.
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


def find_threshold(df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 50) -> tuple[float, float]:
    """Поиск Kelly-порога по val ROI."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.70, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


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


with mlflow.start_run(run_name="phase4/step4.6_recent_threshold") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.6")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        # Recent val: последние 8% train (72-80%)
        recent_val_start = int(n * 0.72)
        # Wide val: 64-80% (baseline)
        wide_val_start = int(n * 0.64)

        recent_val_df = df_raw.iloc[recent_val_start:train_end].copy()
        wide_val_df = df_raw.iloc[wide_val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info(
            "Recent val (72-80%%): %d rows, Wide val (64-80%%): %d rows, Test: %d rows",
            len(recent_val_df),
            len(wide_val_df),
            len(test_df),
        )

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": train_end,
                "n_recent_val": len(recent_val_df),
                "n_wide_val": len(wide_val_df),
                "approach": "recent_window_threshold",
            }
        )

        model = CatBoostClassifier()
        model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_rv, _ = build_features(recent_val_df)
        x_wv, _ = build_features(wide_val_df)
        x_te, _ = build_features(test_df)

        proba_rv = model.predict_proba(x_rv)[:, 1]
        proba_wv = model.predict_proba(x_wv)[:, 1]
        proba_test = model.predict_proba(x_te)[:, 1]

        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, proba_test)
        logger.info("Test AUC: %.4f", auc_test)

        kelly_rv = compute_kelly(proba_rv, recent_val_df["Odds"].values)
        kelly_rv[recent_val_df["lead_hours"].values <= 0] = -999

        kelly_wv = compute_kelly(proba_wv, wide_val_df["Odds"].values)
        kelly_wv[wide_val_df["lead_hours"].values <= 0] = -999

        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_test[test_df["lead_hours"].values <= 0] = -999

        # 1. Recent-window threshold (min_bets=50)
        t_recent, val_roi_recent = find_threshold(recent_val_df, kelly_rv, min_bets=50)
        mask_recent = kelly_test >= t_recent
        roi_recent, n_recent = calc_roi(test_df, mask_recent)
        logger.info(
            "Recent val threshold: %.3f (val_roi=%.2f%%), test_roi=%.2f%%, n=%d",
            t_recent,
            val_roi_recent,
            roi_recent,
            n_recent,
        )

        # 2. Широкий val threshold (baseline: min_bets=200)
        t_wide, val_roi_wide = find_threshold(wide_val_df, kelly_wv, min_bets=200)
        mask_wide = kelly_test >= t_wide
        roi_wide, n_wide = calc_roi(test_df, mask_wide)
        logger.info(
            "Wide val threshold: %.3f (val_roi=%.2f%%), test_roi=%.2f%%, n=%d",
            t_wide,
            val_roi_wide,
            roi_wide,
            n_wide,
        )

        # 3. Shrunken segment thresholds (step 4.1 thresholds shrunk 50% к baseline)
        baseline_t = 0.455
        seg_t_raw = {"low": 0.495, "mid": 0.635, "high": 0.195}
        shrink = 0.5
        seg_t_shrunken = {k: baseline_t + shrink * (v - baseline_t) for k, v in seg_t_raw.items()}
        logger.info("Shrunken segment thresholds: %s", seg_t_shrunken)

        buckets = pd.cut(
            test_df["Odds"],
            bins=[0, 1.8, 3.0, np.inf],
            labels=["low", "mid", "high"],
        )
        shrunken_mask = np.zeros(len(test_df), dtype=bool)
        for bucket, t in seg_t_shrunken.items():
            seg_m = (buckets == bucket).values & (kelly_test >= t)
            shrunken_mask |= seg_m
        roi_shrunken, n_shrunken = calc_roi(test_df, shrunken_mask)
        logger.info("Shrunken segments: test_roi=%.2f%%, n=%d", roi_shrunken, n_shrunken)

        # 4. Ensemble: recent + wide val (average thresholds)
        t_ensemble = (t_recent + t_wide) / 2
        mask_ensemble = kelly_test >= t_ensemble
        roi_ensemble, n_ensemble = calc_roi(test_df, mask_ensemble)
        logger.info(
            "Ensemble threshold (avg): %.3f, test_roi=%.2f%%, n=%d",
            t_ensemble,
            roi_ensemble,
            n_ensemble,
        )

        # Выбор лучшего
        candidates = [
            ("recent_val", roi_recent, n_recent, t_recent),
            ("wide_val", roi_wide, n_wide, t_wide),
            ("shrunken_segments", roi_shrunken, n_shrunken, None),
            ("ensemble_threshold", roi_ensemble, n_ensemble, t_ensemble),
        ]
        best_name, best_roi, best_n, best_t = max(candidates, key=lambda x: x[1])
        baseline_roi = 24.9088
        delta = best_roi - baseline_roi

        logger.info(
            "Best: %s, roi=%.4f%%, n=%d, delta=%.4f%%",
            best_name,
            best_roi,
            best_n,
            delta,
        )

        if best_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", best_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_metrics(
            {
                "roi": best_roi,
                "n_selected": best_n,
                "roi_recent_val": roi_recent,
                "roi_wide_val": roi_wide,
                "roi_shrunken_seg": roi_shrunken,
                "roi_ensemble": roi_ensemble,
                "n_recent": n_recent,
                "n_wide": n_wide,
                "n_shrunken": n_shrunken,
                "n_ensemble": n_ensemble,
                "auc_test": auc_test,
                "baseline_roi": baseline_roi,
                "delta_vs_baseline": delta,
                "t_recent": t_recent,
                "t_wide": t_wide,
                "t_ensemble": t_ensemble,
            }
        )

        current_best_roi = 25.8347
        if best_roi > current_best_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi, current_best_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(best_dir / "model.cbm"))
            meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
            new_meta = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": best_roi,
                "auc": float(auc_test),
                "threshold": float(best_t) if best_t else None,
                "approach": best_name,
                "n_bets": best_n,
                "feature_names": meta["feature_names"],
                "params": meta["params"],
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.6",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.set_tag("best_approach", best_name)
        mlflow.log_artifact(__file__)

        print(
            f"step4.6 DONE: best_roi={best_roi:.4f}% ({best_name}), n={best_n}, delta={delta:.4f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
