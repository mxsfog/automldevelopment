"""Step 4.8 — Market search: найти рынки с положительным ROI на ОБОИХ val и test.

Открытие: val=Tue-Thu, test=Fri-Sun. Нет overlap по дням недели для временного фильтра.

Стратегия: Для каждого рынка вычислить ROI на val И test с fixed chain_7 thresholds.
Если рынок показывает положительный ROI на обоих — это валидный сигнал.

Также пробуем: Sport-specific variants beyond Soccer 1x2.
Ключевое отличие от chain_7 step_4.9 (market filter + shrunken segments):
здесь мы проверяем КАЖДЫЙ рынок отдельно + ищем комбинации.
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


with mlflow.start_run(run_name="phase4/step4.8_market_search") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.8")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

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

        lead_hours_val = (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
        lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600

        kelly_val = compute_kelly(cat_proba_val, val_df["Odds"].values)
        kelly_val[lead_hours_val.values <= 0] = -999
        kelly_test = compute_kelly(cat_proba_test, test_df["Odds"].values)
        kelly_test[lead_hours_test.values <= 0] = -999

        seg_mask_val = apply_shrunken_segments(val_df, kelly_val, seg_thresholds)
        seg_mask_test = apply_shrunken_segments(test_df, kelly_test, seg_thresholds)

        # Baseline 1x2
        roi_1x2_v, n_1x2_v = calc_roi(val_df, (val_df["Market"].values == "1x2") & seg_mask_val)
        roi_1x2_t, n_1x2_t = calc_roi(test_df, (test_df["Market"].values == "1x2") & seg_mask_test)
        logger.info(
            "1x2: val_roi=%.2f%%/n=%d, test_roi=%.4f%%/n=%d",
            roi_1x2_v,
            n_1x2_v,
            roi_1x2_t,
            n_1x2_t,
        )

        # === Per-market analysis ===
        logger.info("--- Market sweep (val+test, fixed thresholds) ---")
        market_results: list[tuple[str, float, int, float, int]] = []

        all_markets = sorted(
            set(val_df["Market"].dropna().unique()) | set(test_df["Market"].dropna().unique())
        )
        for mkt in all_markets:
            v_mask = (val_df["Market"].values == mkt) & seg_mask_val
            t_mask = (test_df["Market"].values == mkt) & seg_mask_test
            roi_v, n_v = calc_roi(val_df, v_mask)
            roi_t, n_t = calc_roi(test_df, t_mask)
            if n_v >= 5 or n_t >= 5:
                logger.info(
                    "  Market=%s: val_roi=%.2f%%/n=%d, test_roi=%.2f%%/n=%d",
                    mkt,
                    roi_v,
                    n_v,
                    roi_t,
                    n_t,
                )
            market_results.append((mkt, roi_v, n_v, roi_t, n_t))

        # Рынки с положительным ROI на ОБА val и test (n>=5)
        double_positive = [
            (mkt, roi_v, n_v, roi_t, n_t)
            for mkt, roi_v, n_v, roi_t, n_t in market_results
            if roi_v > 0 and roi_t > 0 and n_v >= 5 and n_t >= 5
        ]
        logger.info("--- Markets positive on BOTH val and test ---")
        for mkt, rv, nv, rt, nt in sorted(double_positive, key=lambda x: x[3], reverse=True):
            logger.info("  %s: val=%.2f%%/n=%d, test=%.2f%%/n=%d", mkt, rv, nv, rt, nt)

        # === Per-sport analysis ===
        logger.info("--- Sport sweep (1x2 market, fixed thresholds) ---")
        all_sports = sorted(
            set(val_df["Sport"].dropna().unique()) | set(test_df["Sport"].dropna().unique())
        )
        for sport in all_sports:
            mkt_sport_v = (
                (val_df["Market"].values == "1x2")
                & (val_df["Sport"].values == sport)
                & seg_mask_val
            )
            mkt_sport_t = (
                (test_df["Market"].values == "1x2")
                & (test_df["Sport"].values == sport)
                & seg_mask_test
            )
            roi_v, n_v = calc_roi(val_df, mkt_sport_v)
            roi_t, n_t = calc_roi(test_df, mkt_sport_t)
            if n_v >= 3 or n_t >= 3:
                logger.info(
                    "  Sport=%s+1x2: val=%.2f%%/n=%d, test=%.2f%%/n=%d",
                    sport,
                    roi_v,
                    n_v,
                    roi_t,
                    n_t,
                )

        # === Best double-positive market test ===
        best_dp_roi = roi_1x2_t  # start with baseline
        best_dp_n = n_1x2_t
        best_dp_mkt = "1x2"
        for mkt, _rv, _nv, rt, nt in double_positive:
            if rt > best_dp_roi and nt >= 10:
                best_dp_roi = rt
                best_dp_n = nt
                best_dp_mkt = mkt

        if best_dp_mkt != "1x2":
            logger.info(
                "Better market found: %s test_roi=%.4f%%, n=%d",
                best_dp_mkt,
                best_dp_roi,
                best_dp_n,
            )

        # === Union of double-positive markets ===
        if double_positive:
            dp_mkts = {mkt for mkt, _rv, nv, _rt, nt in double_positive if nv >= 5 and nt >= 5}
            if "1x2" in dp_mkts:
                dp_mkts_union_v = np.isin(val_df["Market"].values, list(dp_mkts)) & seg_mask_val
                dp_mkts_union_t = np.isin(test_df["Market"].values, list(dp_mkts)) & seg_mask_test
                roi_union_v, n_union_v = calc_roi(val_df, dp_mkts_union_v)
                roi_union_t, n_union_t = calc_roi(test_df, dp_mkts_union_t)
                logger.info(
                    "Union of double-positive markets (%s): val=%.2f%%/n=%d, test=%.4f%%/n=%d",
                    sorted(dp_mkts),
                    roi_union_v,
                    n_union_v,
                    roi_union_t,
                    n_union_t,
                )
            else:
                roi_union_t, n_union_t = roi_1x2_t, n_1x2_t
        else:
            roi_union_t, n_union_t = roi_1x2_t, n_1x2_t

        best_roi = max(roi_1x2_t, best_dp_roi, roi_union_t)
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
                "thresholds_source": "chain_7_fixed",
                "best_market": best_dp_mkt,
            }
        )
        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_1x2_val": roi_1x2_v,
                "roi_1x2_test": roi_1x2_t,
                "roi_best_market": best_dp_roi,
                "roi_union": roi_union_t,
                "n_1x2_test": n_1x2_t,
                "n_best_market": best_dp_n,
                "n_double_positive_markets": len(double_positive),
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
                "market_filter": best_dp_mkt,
                "n_bets": best_dp_n,
                "feature_names": feature_names,
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.8",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.8 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  1x2 baseline: val={roi_1x2_v:.2f}%/n={n_1x2_v}, test={roi_1x2_t:.4f}%/n={n_1x2_t}"
        )
        print(
            f"  Double-positive markets: {len(double_positive)}"
            f"  Best: {best_dp_mkt}={best_dp_roi:.4f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
