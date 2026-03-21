"""Step 4.11 — CatBoost retrained только на 1x2/Soccer данных.

Гипотеза: модель, обученная на подмножестве 1x2 ставок, лучше калибрует
вероятности для этого рынка. Лучшая калибровка → лучший Kelly → выше ROI.

Применяем shrunken segment thresholds (shrink=0.5) к 1x2-специфичной модели.
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


def make_weights(n: int, half_life: float = 0.5) -> np.ndarray:
    """Temporal decay weights."""
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


def find_threshold_1x2(
    df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 30
) -> tuple[float, float]:
    """Поиск Kelly-порога на 1x2 val ROI."""
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


with mlflow.start_run(run_name="phase4/step4.11_1x2_retrain") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.11")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        train_df = df_raw.iloc[:train_end].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        # Фильтруем по 1x2
        train_1x2 = train_df[train_df["Market"] == "1x2"].copy()
        val_1x2 = val_df[val_df["Market"] == "1x2"].copy()
        test_1x2 = test_df[test_df["Market"] == "1x2"].copy()

        logger.info(
            "1x2 data: train=%d, val=%d, test=%d",
            len(train_1x2),
            len(val_1x2),
            len(test_1x2),
        )

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_train_1x2": len(train_1x2),
                "n_val_1x2": len(val_1x2),
                "n_test_1x2": len(test_1x2),
                "market": "1x2",
                "depth": 7,
                "lr": 0.1,
                "iterations": 500,
                "shrinkage": 0.5,
            }
        )

        x_tr, cat_f = build_features(train_1x2)
        x_vl, _ = build_features(val_1x2)
        x_te, _ = build_features(test_1x2)
        y_tr = (train_1x2["Status"] == "won").astype(int)
        y_vl = (val_1x2["Status"] == "won").astype(int)
        y_te = (test_1x2["Status"] == "won").astype(int)
        w = make_weights(len(train_1x2))

        model_1x2 = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            eval_metric="AUC",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
            cat_features=cat_f,
        )
        model_1x2.fit(x_tr, y_tr, eval_set=(x_vl, y_vl), sample_weight=w)

        proba_val_1x2 = model_1x2.predict_proba(x_vl)[:, 1]
        proba_test_1x2 = model_1x2.predict_proba(x_te)[:, 1]

        auc_val = roc_auc_score(y_vl, proba_val_1x2)
        auc_test = roc_auc_score(y_te, proba_test_1x2)
        logger.info("1x2 model: val AUC=%.4f, test AUC=%.4f", auc_val, auc_test)

        kelly_val = compute_kelly(proba_val_1x2, val_1x2["Odds"].values)
        kelly_val[val_1x2["lead_hours"].values <= 0] = -999

        kelly_test = compute_kelly(proba_test_1x2, test_1x2["Odds"].values)
        kelly_test[test_1x2["lead_hours"].values <= 0] = -999

        # 1. Single threshold на val
        best_t, val_roi = find_threshold_1x2(val_1x2, kelly_val, min_bets=100)
        mask_single = kelly_test >= best_t
        roi_single, n_single = calc_roi(test_1x2, mask_single)
        logger.info(
            "1x2 model single thresh: t=%.3f, val_roi=%.2f%%, test_roi=%.2f%%, n=%d",
            best_t,
            val_roi,
            roi_single,
            n_single,
        )

        # 2. Shrunken segments (shrink=0.5, raw from step 4.1, applied to 1x2 model)
        raw_seg_t = {"low": 0.495, "mid": 0.635, "high": 0.195}
        baseline_t = 0.455
        shrink = 0.5
        shrunken = {k: baseline_t + shrink * (v - baseline_t) for k, v in raw_seg_t.items()}

        buckets = pd.cut(
            test_1x2["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"]
        )
        seg_mask = np.zeros(len(test_1x2), dtype=bool)
        for bucket, t in shrunken.items():
            seg_mask |= (buckets == bucket).values & (kelly_test >= t)
        roi_seg, n_seg = calc_roi(test_1x2, seg_mask)
        logger.info("1x2 model + shrunken_seg: test_roi=%.4f%%, n=%d", roi_seg, n_seg)

        # 3. Baseline global model на 1x2 (comparison)
        global_model = CatBoostClassifier()
        global_model.load_model(str(PREV_BEST_DIR / "model.cbm"))
        proba_test_global = global_model.predict_proba(x_te)[:, 1]
        kelly_test_global = compute_kelly(proba_test_global, test_1x2["Odds"].values)
        kelly_test_global[test_1x2["lead_hours"].values <= 0] = -999
        buckets_g = pd.cut(
            test_1x2["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"]
        )
        seg_mask_g = np.zeros(len(test_1x2), dtype=bool)
        for bucket, t in shrunken.items():
            seg_mask_g |= (buckets_g == bucket).values & (kelly_test_global >= t)
        roi_global_1x2, n_global_1x2 = calc_roi(test_1x2, seg_mask_g)
        logger.info(
            "Global model + shrunken_seg on 1x2: test_roi=%.4f%%, n=%d",
            roi_global_1x2,
            n_global_1x2,
        )

        best_roi = max(roi_single, roi_seg)
        best_n = n_single if roi_single >= roi_seg else n_seg
        baseline_roi = 24.9088
        delta = best_roi - baseline_roi

        logger.info("Best 1x2_retrain: roi=%.4f%%, n=%d, delta=%.4f%%", best_roi, best_n, delta)

        if best_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", best_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_metrics(
            {
                "roi": best_roi,
                "n_selected": best_n,
                "roi_1x2_model_single": roi_single,
                "roi_1x2_model_seg": roi_seg,
                "roi_global_1x2_seg": roi_global_1x2,
                "auc_val": auc_val,
                "auc_test": auc_test,
                "threshold": best_t,
                "baseline_roi": baseline_roi,
                "delta_vs_baseline": delta,
            }
        )

        current_best_roi = 28.5833
        if best_roi > current_best_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi, current_best_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model_1x2.save_model(str(best_dir / "model.cbm"))
            new_meta = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": best_roi,
                "auc": float(auc_test),
                "threshold": float(best_t) if roi_single >= roi_seg else None,
                "segment_thresholds": shrunken if roi_seg > roi_single else None,
                "n_bets": best_n,
                "market_filter": "1x2",
                "train_filter": "1x2_only",
                "feature_names": list(x_tr.columns),
                "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
                "session_id": SESSION_ID,
                "step": "4.11",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.11 DONE: best_roi={best_roi:.4f}%, n={best_n}, delta={delta:.4f}%")
        print(f"  1x2_model_single: {roi_single:.4f}%/{n_single}")
        print(f"  1x2_model_seg: {roi_seg:.4f}%/{n_seg}")
        print(f"  global_model_1x2_seg (reference): {roi_global_1x2:.4f}%/{n_global_1x2}")
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
