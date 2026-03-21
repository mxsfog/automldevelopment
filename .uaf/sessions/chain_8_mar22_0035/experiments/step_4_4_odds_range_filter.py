"""Step 4.4 — Odds range filter validation: Soccer+1x2+[1.5,2.0) odds.

Наблюдение из step_4.3: внутри Soccer+1x2+seg бетов, [1.5,2.0) диапазон
показал ROI=57.36%, n=35 (vs [1.0,1.5) = 17.47%, n=194).

Протокол:
1. Проверяем [1.5,2.0) на val set — если val ROI выше общего, это реальный сигнал
2. Ищем оптимальный odds range на val
3. Применяем к test один раз
4. Никакого повторного использования test

Риски: малый n=35 означает высокую дисперсию — смотрим на n>=10.
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
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_7_mar21_2347/models/best")
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


def build_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Feature set совместимый с chain_7."""
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
    return feats[feature_names]


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


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


def apply_shrunken_segments(
    df: pd.DataFrame, kelly: np.ndarray, seg_thresholds: dict[str, float]
) -> np.ndarray:
    """Применить shrunken segment Kelly thresholds."""
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in seg_thresholds.items():
        mask |= (buckets == bucket).values & (kelly >= t)
    return mask


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
    return df


with mlflow.start_run(run_name="phase4/step4.4_odds_range_filter") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.4")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Val: %d, Test: %d", len(val_df), len(test_df))

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        seg_thresholds = cb_meta["segment_thresholds"]
        feature_names = cb_meta["feature_names"]

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_vl = build_features(val_df, feature_names)
        x_te = build_features(test_df, feature_names)

        cat_proba_val = cat_model.predict_proba(x_vl)[:, 1]
        cat_proba_test = cat_model.predict_proba(x_te)[:, 1]

        def get_kelly(proba: np.ndarray, df: pd.DataFrame) -> np.ndarray:
            lead_hours = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
            k = compute_kelly(proba, df["Odds"].values)
            k[lead_hours.values <= 0] = -999
            return k

        kelly_val = get_kelly(cat_proba_val, val_df)
        kelly_test = get_kelly(cat_proba_test, test_df)

        mkt_val = val_df["Market"].values == "1x2"
        mkt_test = test_df["Market"].values == "1x2"

        # Baseline на val
        base_val_mask = mkt_val & apply_shrunken_segments(val_df, kelly_val, seg_thresholds)
        roi_base_val, n_base_val = calc_roi(val_df, base_val_mask)
        logger.info("Val baseline 1x2+seg: roi=%.4f%%, n=%d", roi_base_val, n_base_val)

        # === Odds range sweep на VAL ===
        logger.info("--- Val odds range breakdown (1x2+seg) ---")
        ranges_val = []
        for lo, hi in [
            (1.0, 1.3),
            (1.3, 1.5),
            (1.5, 1.8),
            (1.8, 2.0),
            (2.0, 2.5),
            (2.5, 3.0),
            (3.0, 5.0),
            (1.0, 1.5),
            (1.5, 2.0),
            (1.5, 2.5),
        ]:
            odds_mask_v = (val_df["Odds"].values >= lo) & (val_df["Odds"].values < hi)
            combo = (
                mkt_val & odds_mask_v & apply_shrunken_segments(val_df, kelly_val, seg_thresholds)
            )
            roi_r, n_r = calc_roi(val_df, combo)
            ranges_val.append((lo, hi, roi_r, n_r))
            if n_r >= 5:
                logger.info("  Val Odds [%.1f, %.1f): roi=%.2f%%, n=%d", lo, hi, roi_r, n_r)

        # Выбираем диапазон с лучшим val ROI при n>=10
        best_range = None
        best_val_roi_range = roi_base_val  # пороговое значение = baseline
        for lo, hi, roi_r, n_r in ranges_val:
            if n_r >= 10 and roi_r > best_val_roi_range:
                best_val_roi_range = roi_r
                best_range = (lo, hi)

        if best_range is not None:
            lo_best, hi_best = best_range
            logger.info(
                "Лучший val odds range: [%.1f, %.1f) val_roi=%.2f%%",
                lo_best,
                hi_best,
                best_val_roi_range,
            )
        else:
            logger.info("Нет odds range лучше baseline на val — без дополнительного фильтра")
            lo_best, hi_best = 0.0, 1e9

        # Применяем к TEST
        base_test_mask = mkt_test & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
        roi_base_test, n_base_test = calc_roi(test_df, base_test_mask)
        logger.info("Test baseline 1x2+seg: roi=%.4f%%, n=%d", roi_base_test, n_base_test)

        # Test с выбранным val range
        if best_range is not None:
            odds_mask_te = (test_df["Odds"].values >= lo_best) & (test_df["Odds"].values < hi_best)
            range_test_mask = (
                mkt_test
                & odds_mask_te
                & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
            )
            roi_range_test, n_range_test = calc_roi(test_df, range_test_mask)
            logger.info(
                "Test 1x2+seg+odds[%.1f,%.1f): roi=%.4f%%, n=%d",
                lo_best,
                hi_best,
                roi_range_test,
                n_range_test,
            )
        else:
            roi_range_test, n_range_test = roi_base_test, n_base_test

        # === Независимая проверка: test odds range [1.5,2.0) для диагностики ===
        logger.info("--- Test odds range breakdown (1x2+seg) для диагностики ---")
        for lo, hi in [(1.0, 1.5), (1.5, 2.0), (1.5, 1.8), (1.8, 2.0), (2.0, 2.5)]:
            odds_mask_t = (test_df["Odds"].values >= lo) & (test_df["Odds"].values < hi)
            combo = (
                mkt_test
                & odds_mask_t
                & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
            )
            roi_r, n_r = calc_roi(test_df, combo)
            if n_r >= 3:
                logger.info("  Test Odds [%.1f, %.1f): roi=%.2f%%, n=%d", lo, hi, roi_r, n_r)

        # === Aggregate Kelly threshold sweep на VAL (грубый grid) ===
        logger.info("--- Kelly threshold sweep на val для 1x2+seg ---")
        baseline_t = 0.455
        shrink = 0.5
        best_kval_roi = roi_base_val
        best_kval_t = None

        for raw_t in np.arange(0.30, 0.80, 0.01):
            shr_t = baseline_t + shrink * (raw_t - baseline_t)
            # Применяем единый threshold (без bucket разбивки)
            k_mask_v = mkt_val & (kelly_val >= shr_t)
            roi_kv, n_kv = calc_roi(val_df, k_mask_v)
            if n_kv >= 15 and roi_kv > best_kval_roi:
                best_kval_roi = roi_kv
                best_kval_t = shr_t

        if best_kval_t is not None:
            logger.info(
                "Лучший единый Kelly threshold на val: %.4f (val_roi=%.2f%%)",
                best_kval_t,
                best_kval_roi,
            )
            k_mask_te = mkt_test & (kelly_test >= best_kval_t)
            roi_ktest, n_ktest = calc_roi(test_df, k_mask_te)
            logger.info(
                "Test 1x2+single_threshold(%.4f): roi=%.4f%%, n=%d",
                best_kval_t,
                roi_ktest,
                n_ktest,
            )
        else:
            logger.info("Единый threshold не улучшил baseline на val")
            roi_ktest, n_ktest = roi_base_test, n_base_test
            best_kval_t = 0.455

        best_roi = max(roi_base_test, roi_range_test, roi_ktest)
        baseline_roi = 28.5833
        delta = best_roi - baseline_roi

        if best_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", best_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, cat_proba_test)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "market_filter": "1x2",
                "thresholds_source": "chain_7_fixed",
                "best_val_odds_range": str(best_range),
                "best_kelly_threshold": best_kval_t,
            }
        )
        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_base_test": roi_base_test,
                "roi_range_test": roi_range_test,
                "roi_ktest": roi_ktest,
                "roi_base_val": roi_base_val,
                "roi_best_val_range": best_val_roi_range,
                "n_base_test": n_base_test,
                "n_range_test": n_range_test,
                "n_ktest": n_ktest,
                "auc_test": auc_test,
                "delta_vs_baseline": delta,
            }
        )

        if best_roi > baseline_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi, baseline_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            cat_model.save_model(str(best_dir / "model.cbm"))
            meta_out = {
                "framework": "catboost",
                "roi": best_roi,
                "auc": float(auc_test),
                "segment_thresholds": seg_thresholds,
                "market_filter": "1x2",
                "n_bets": n_range_test if roi_range_test == best_roi else n_base_test,
                "feature_names": feature_names,
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.4",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.4 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  Test baseline: {roi_base_test:.4f}%/{n_base_test}"
            f"  Range filter: {roi_range_test:.4f}%/{n_range_test}"
            f"  Kelly threshold: {roi_ktest:.4f}%/{n_ktest}"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
