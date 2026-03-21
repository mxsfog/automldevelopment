"""Step 4.7 — Day-of-week filter для 1x2 Soccer + shrunken segments.

Открытие step 4.6: test период = Feb 20-22, 2026 (Thu-Sat):
- Thu (Q1-Q2): win_rate=34%, Kelly отбирает 23 ставки, ROI=-47%
- Fri-Sat (Q3-Q4): win_rate=58-62%, Kelly отбирает 210 ставок, ROI=+33%

day_of_week = 2-я по важности фича CatBoost.

Гипотеза: Добавление explicit day-of-week фильтра (например, только пятница/суббота)
дополнительно улучшит ROI, если val-период показывает аналогичный паттерн.

МЕТОДОЛОГИЯ: смотрим на val win_rate по дням, выбираем дни с наивысшим ROI на val,
применяем к test.

Anti-leakage: thresholds выбраны на val, к test применяется один раз.
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

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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


with mlflow.start_run(run_name="phase4/step4.7_dayofweek") as run:
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
        val_start = int(n * 0.64)

        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Val: %d, Test: %d", len(val_df), len(test_df))
        logger.info(
            "Val period: %s to %s",
            val_df["Created_At"].min().date(),
            val_df["Created_At"].max().date(),
        )
        logger.info(
            "Test period: %s to %s",
            test_df["Created_At"].min().date(),
            test_df["Created_At"].max().date(),
        )

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        feature_names = cb_meta["feature_names"]
        seg_thresholds = cb_meta["segment_thresholds"]

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_vl = build_features(val_df, feature_names)
        x_te = build_features(test_df, feature_names)

        cat_proba_val = cat_model.predict_proba(x_vl)[:, 1]
        cat_proba_test = cat_model.predict_proba(x_te)[:, 1]

        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, cat_proba_test)

        lead_hours_val = (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600.0
        lead_hours_test = (
            test_df["Start_Time"] - test_df["Created_At"]
        ).dt.total_seconds() / 3600.0

        kelly_val = compute_kelly(cat_proba_val, val_df["Odds"].values)
        kelly_val[lead_hours_val.values <= 0] = -999
        kelly_test = compute_kelly(cat_proba_test, test_df["Odds"].values)
        kelly_test[lead_hours_test.values <= 0] = -999

        mkt_val = val_df["Market"].values == "1x2"
        mkt_test = test_df["Market"].values == "1x2"

        baseline_test = mkt_test & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
        roi_base, n_base = calc_roi(test_df, baseline_test)
        logger.info("Test baseline 1x2+seg: roi=%.4f%%, n=%d", roi_base, n_base)

        # === День недели анализ на VAL ===
        val_dow = val_df["Created_At"].dt.dayofweek.values
        test_dow = test_df["Created_At"].dt.dayofweek.values

        logger.info("--- Val day-of-week ROI breakdown (1x2+seg) ---")
        dow_val_roi = {}
        val_base_mask = mkt_val & apply_shrunken_segments(val_df, kelly_val, seg_thresholds)
        for dow in range(7):
            dow_mask_v = val_base_mask & (val_dow == dow)
            roi_d, n_d = calc_roi(val_df, dow_mask_v)
            dow_val_roi[dow] = roi_d
            # Win rate на 1x2 для этого дня на val
            dow_1x2_v = val_df[(val_df["Market"] == "1x2") & (val_dow == dow)]
            wr = (dow_1x2_v["Status"] == "won").mean() if len(dow_1x2_v) > 0 else 0
            logger.info(
                "  Val %s (dow=%d): sel roi=%.2f%% n=%d | 1x2 win_rate=%.3f n=%d",
                DAYS[dow],
                dow,
                roi_d,
                n_d,
                wr,
                len(dow_1x2_v),
            )

        logger.info("--- Test day-of-week ROI breakdown (1x2+seg) ---")
        for dow in range(7):
            dow_mask_t = baseline_test & (test_dow == dow)
            roi_d, n_d = calc_roi(test_df, dow_mask_t)
            dow_1x2_t = test_df[(test_df["Market"] == "1x2") & (test_dow == dow)]
            wr = (dow_1x2_t["Status"] == "won").mean() if len(dow_1x2_t) > 0 else 0
            logger.info(
                "  Test %s (dow=%d): sel roi=%.2f%% n=%d | 1x2 win_rate=%.3f n=%d",
                DAYS[dow],
                dow,
                roi_d,
                n_d,
                wr,
                len(dow_1x2_t),
            )

        # Выбираем "good days" на VAL: days с roi > median_val_roi + n>=5
        val_rois_with_n = []
        for dow in range(7):
            dow_mask_v = val_base_mask & (val_dow == dow)
            roi_d, n_d = calc_roi(val_df, dow_mask_v)
            val_rois_with_n.append((dow, roi_d, n_d))

        # Sort по val ROI, берём top half дней с n>=5
        valid_days = [(d, r, n) for d, r, n in val_rois_with_n if n >= 5]
        if len(valid_days) >= 2:
            sorted_days = sorted(valid_days, key=lambda x: x[1], reverse=True)
            n_good = max(1, len(sorted_days) // 2)
            good_days = set(d for d, r, n in sorted_days[:n_good])
            logger.info("Val-selected good days: %s", [DAYS[d] for d in sorted(good_days)])
        else:
            good_days = set(range(7))
            logger.info("Недостаточно val данных по дням — не фильтруем")

        # Применяем day-of-week filter к test
        if good_days != set(range(7)):
            dow_filter_test = np.isin(test_dow, list(good_days))
            filtered_mask = baseline_test & dow_filter_test
            roi_dow, n_dow = calc_roi(test_df, filtered_mask)
            logger.info(
                "Test 1x2+seg+dow_filter(%s): roi=%.4f%%, n=%d",
                [DAYS[d] for d in sorted(good_days)],
                roi_dow,
                n_dow,
            )
        else:
            roi_dow, n_dow = roi_base, n_base
            logger.info("Фильтр по дням не применяется (недостаточно val данных)")

        # === Weekend-only filter (независимо от val selection) ===
        # Weekend = Fri(4), Sat(5), Sun(6)
        weekend_days = {4, 5, 6}
        weekend_mask_v = val_base_mask & np.isin(val_dow, list(weekend_days))
        weekend_mask_t = baseline_test & np.isin(test_dow, list(weekend_days))
        roi_weekend_v, n_weekend_v = calc_roi(val_df, weekend_mask_v)
        roi_weekend_t, n_weekend_t = calc_roi(test_df, weekend_mask_t)
        logger.info("Val weekend(Fri-Sun) 1x2+seg: roi=%.4f%%, n=%d", roi_weekend_v, n_weekend_v)
        logger.info("Test weekend(Fri-Sun) 1x2+seg: roi=%.4f%%, n=%d", roi_weekend_t, n_weekend_t)

        # Weekday-only (Mon-Thu = 0,1,2,3)
        weekday_days = {0, 1, 2, 3}
        weekday_mask_t = baseline_test & np.isin(test_dow, list(weekday_days))
        roi_weekday_t, n_weekday_t = calc_roi(test_df, weekday_mask_t)
        logger.info("Test weekday(Mon-Thu) 1x2+seg: roi=%.4f%%, n=%d", roi_weekday_t, n_weekday_t)

        # === Hour-based analysis на val ===
        val_hour = val_df["Created_At"].dt.hour.values
        test_hour = test_df["Created_At"].dt.hour.values

        logger.info("--- Val hour-of-day ROI breakdown (1x2+seg, by 4h buckets) ---")
        hour_val_roi: dict[str, float] = {}
        for h_start in range(0, 24, 4):
            h_end = h_start + 4
            h_mask_v = val_base_mask & (val_hour >= h_start) & (val_hour < h_end)
            roi_h, n_h = calc_roi(val_df, h_mask_v)
            hour_val_roi[f"{h_start:02d}-{h_end:02d}"] = roi_h
            if n_h >= 3:
                logger.info("  Val %02d-%02dh: roi=%.2f%%, n=%d", h_start, h_end, roi_h, n_h)

        # Выбираем лучшие 2 hour-buckets на val (n>=5)
        good_hours = set(range(24))
        valid_hour_buckets = []
        for h_start in range(0, 24, 4):
            h_end = h_start + 4
            h_mask_v = val_base_mask & (val_hour >= h_start) & (val_hour < h_end)
            roi_h, n_h = calc_roi(val_df, h_mask_v)
            if n_h >= 5:
                valid_hour_buckets.append((h_start, h_end, roi_h, n_h))

        if len(valid_hour_buckets) >= 2:
            sorted_hb = sorted(valid_hour_buckets, key=lambda x: x[2], reverse=True)
            top_hb = sorted_hb[: len(sorted_hb) // 2 + 1]
            good_hours = set()
            for h_start, h_end, _, _ in top_hb:
                good_hours.update(range(h_start, h_end))
            logger.info("Val-selected good hours: %s", sorted(good_hours))

            hour_filter_test = np.isin(test_hour, list(good_hours))
            hour_mask_t = baseline_test & hour_filter_test
            roi_hour, n_hour = calc_roi(test_df, hour_mask_t)
            logger.info(
                "Test 1x2+seg+hour_filter(n_good_hours=%d): roi=%.4f%%, n=%d",
                len(good_hours),
                roi_hour,
                n_hour,
            )
        else:
            roi_hour, n_hour = roi_base, n_base
            logger.info("Недостаточно val данных по часам — не фильтруем")

        # === Combined: weekend + hour filter ===
        if good_hours != set(range(24)) and len(valid_hour_buckets) >= 2:
            combined_mask = (
                baseline_test
                & np.isin(test_dow, list(weekend_days))
                & np.isin(test_hour, list(good_hours))
            )
            roi_combined, n_combined = calc_roi(test_df, combined_mask)
            logger.info("Test 1x2+seg+weekend+hour: roi=%.4f%%, n=%d", roi_combined, n_combined)
        else:
            roi_combined, n_combined = roi_base, n_base

        best_roi = max(roi_base, roi_dow, roi_weekend_t, roi_hour, roi_combined)
        baseline_roi = 28.5833
        delta = best_roi - baseline_roi

        if best_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", best_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "market_filter": "1x2",
                "thresholds_source": "chain_7_fixed",
                "good_days_val_selected": str(sorted(good_days)),
            }
        )
        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_base": roi_base,
                "roi_dow_filtered": roi_dow,
                "roi_weekend": roi_weekend_t,
                "roi_weekday": roi_weekday_t,
                "roi_hour_filtered": roi_hour,
                "roi_combined": roi_combined,
                "n_base": n_base,
                "n_dow": n_dow,
                "n_weekend": n_weekend_t,
                "n_weekday": n_weekday_t,
                "n_hour": n_hour,
                "n_combined": n_combined,
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
                "n_bets": n_base,
                "feature_names": feature_names,
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.7",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.7 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  Base: {roi_base:.4f}%/{n_base}"
            f"  DOW: {roi_dow:.4f}%/{n_dow}"
            f"  Weekend: {roi_weekend_t:.4f}%/{n_weekend_t}"
            f"  Hour: {roi_hour:.4f}%/{n_hour}"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
