"""Step 4.11 — Probability-rank selection вместо Kelly criterion.

Гипотеза: Kelly criterion зависит от абсолютных значений вероятностей.
Отбор по рангу (top-N% по убыванию proba_model) для 1x2 pre-match —
более robust approach, не зависящий от порога.

Методология:
1. Вычислить model probability для всех 1x2 pre-match ставок
2. Ранжировать по убыванию proba
3. Оптимизировать top-N% на val → применить к test
4. Сравнить с Kelly baseline (28.5833%)

Дополнительно: ML_Edge threshold (proba_model - implied_prob > threshold)
как альтернативный критерий отбора.
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


with mlflow.start_run(run_name="phase4/step4.11_probability_rank") as run:
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

        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Val: %d, Test: %d", len(val_df), len(test_df))

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        feature_names = cb_meta["feature_names"]
        seg_thresholds = cb_meta["segment_thresholds"]

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_vl = build_features(val_df, feature_names)
        x_te = build_features(test_df, feature_names)

        proba_val = cat_model.predict_proba(x_vl)[:, 1]
        proba_test = cat_model.predict_proba(x_te)[:, 1]
        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, proba_test)
        logger.info("Test AUC: %.4f", auc_test)

        lead_hours_val = (
            (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
        ).values
        lead_hours_test = (
            (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
        ).values

        # Base filters: 1x2 + pre-match
        market_val = val_df["Market"].values == "1x2"
        prematch_val = lead_hours_val > 0
        base_mask_val = market_val & prematch_val

        market_test = test_df["Market"].values == "1x2"
        prematch_test = lead_hours_test > 0
        base_mask_test = market_test & prematch_test

        # Baseline Kelly (chain_7 thresholds)
        kelly_val = compute_kelly(proba_val, val_df["Odds"].values)
        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_val_f = kelly_val.copy()
        kelly_val_f[~base_mask_val] = -999
        kelly_test_f = kelly_test.copy()
        kelly_test_f[~base_mask_test] = -999

        base_seg_mask = apply_shrunken_segments(test_df, kelly_test_f, seg_thresholds)
        roi_baseline, n_baseline = calc_roi(test_df, base_seg_mask)
        logger.info("Baseline Kelly ROI: %.4f%% (n=%d)", roi_baseline, n_baseline)

        # ---- Метод 1: Probability-rank top-N% ----
        # Для каждого N (процент от всех 1x2 prematch ставок) — оптимизируем на val
        val_1x2 = val_df[base_mask_val].copy()
        val_1x2_proba = proba_val[base_mask_val]
        val_1x2_idx = np.where(base_mask_val)[0]

        test_1x2 = test_df[base_mask_test].copy()
        test_1x2_proba = proba_test[base_mask_test]
        test_1x2_idx = np.where(base_mask_test)[0]

        logger.info("1x2 prematch val: %d, test: %d", len(val_1x2), len(test_1x2))

        best_prank_roi_val = -np.inf
        best_prank_n_pct = 10.0  # процент

        # Sweep N from 1% to 100% с шагом 1%
        for pct in range(1, 101):
            k_top = max(1, int(len(val_1x2) * pct / 100))
            sorted_idx = np.argsort(-val_1x2_proba)[:k_top]
            mask_v = np.zeros(len(val_df), dtype=bool)
            mask_v[val_1x2_idx[sorted_idx]] = True
            rv, nv = calc_roi(val_df, mask_v)
            if nv >= 10 and rv > best_prank_roi_val:
                best_prank_roi_val = rv
                best_prank_n_pct = pct

        logger.info(
            "Probability rank best val: top=%d%% ROI=%.2f%%",
            best_prank_n_pct, best_prank_roi_val,
        )

        # Применить лучший процент к test
        k_top_test = max(1, int(len(test_1x2) * best_prank_n_pct / 100))
        sorted_test_idx = np.argsort(-test_1x2_proba)[:k_top_test]
        prank_mask_test = np.zeros(len(test_df), dtype=bool)
        prank_mask_test[test_1x2_idx[sorted_test_idx]] = True
        roi_prank, n_prank = calc_roi(test_df, prank_mask_test)
        logger.info(
            "Probability rank test ROI: %.4f%% (n=%d, top=%d%%)",
            roi_prank, n_prank, best_prank_n_pct,
        )

        # ---- Метод 2: ML_Edge threshold ----
        # ml_edge = proba_model - implied_prob = proba - 1/odds
        # Отбор: ml_edge > threshold AND 1x2 AND pre-match (без Kelly)
        implied_val = 1.0 / val_df["Odds"].values.clip(1.001)
        ml_edge_val = proba_val - implied_val
        implied_test = 1.0 / test_df["Odds"].values.clip(1.001)
        ml_edge_test = proba_test - implied_test

        best_edge_roi_val = -np.inf
        best_edge_threshold = 0.0

        for t in np.arange(-0.3, 0.5, 0.01):
            edge_mask_v = base_mask_val & (ml_edge_val >= t)
            rv, nv = calc_roi(val_df, edge_mask_v)
            if nv >= 10 and rv > best_edge_roi_val:
                best_edge_roi_val = rv
                best_edge_threshold = t

        logger.info(
            "ML_Edge best val: threshold=%.3f ROI=%.2f%%",
            best_edge_threshold, best_edge_roi_val,
        )

        edge_mask_test = base_mask_test & (ml_edge_test >= best_edge_threshold)
        roi_edge, n_edge = calc_roi(test_df, edge_mask_test)
        logger.info(
            "ML_Edge test ROI: %.4f%% (n=%d, threshold=%.3f)",
            roi_edge, n_edge, best_edge_threshold,
        )

        # ---- Метод 3: Kelly > 0 (все положительные Kelly, без seg thresholds) ----
        # Это нижняя граница — все ставки где Kelly > 0
        kelly_pos_mask_test = base_mask_test & (kelly_test > 0)
        roi_kelly_pos, n_kelly_pos = calc_roi(test_df, kelly_pos_mask_test)
        logger.info(
            "Kelly>0 test ROI: %.4f%% (n=%d)", roi_kelly_pos, n_kelly_pos
        )

        # ---- Итог ----
        best_roi = max(roi_prank, roi_edge, roi_baseline)
        delta = best_roi - roi_baseline

        # Leakage checks
        for label, _roi, n in [("prank", roi_prank, n_prank), ("edge", roi_edge, n_edge)]:
            if n < 10:
                logger.warning("LEAKAGE SUSPECT %s: n=%d < 10", label, n)

        mlflow.log_params({
            "best_prank_pct": best_prank_n_pct,
            "best_edge_threshold": best_edge_threshold,
        })
        mlflow.log_metrics({
            "auc_test": auc_test,
            "roi_baseline": roi_baseline,
            "n_baseline": n_baseline,
            "roi_prank_test": roi_prank,
            "n_prank_test": n_prank,
            "roi_edge_test": roi_edge,
            "n_edge_test": n_edge,
            "roi_kelly_pos_test": roi_kelly_pos,
            "n_kelly_pos_test": n_kelly_pos,
            "roi_val_prank_best": best_prank_roi_val,
            "roi_val_edge_best": best_edge_roi_val,
            "delta_best_vs_baseline": delta,
        })
        mlflow.set_tag("status", "done")
        mlflow.set_tag(
            "result",
            f"prank={roi_prank:.4f}%(n={n_prank}) edge={roi_edge:.4f}%(n={n_edge})"
            f" baseline={roi_baseline:.4f}%",
        )
        logger.info("Run ID: %s", run_id)
        logger.info(
            "ИТОГ: baseline=%.4f%% | prank=%.4f%%(n=%d) | edge=%.4f%%(n=%d) | delta=%.4f%%",
            roi_baseline, roi_prank, n_prank, roi_edge, n_edge, delta,
        )

    except Exception:
        mlflow.set_tag("status", "error")
        logger.exception("Ошибка в step 4.11")
        raise
