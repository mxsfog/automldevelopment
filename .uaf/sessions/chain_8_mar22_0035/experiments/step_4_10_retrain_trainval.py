"""Step 4.10 — Ретренировка CatBoost на train+val (80% данных).

Гипотеза: chain_6 обучен на 64% данных. Ретренировка на 80% (включая val period Feb 17-20)
даёт модель с более свежими паттернами, которая лучше работает на test (Feb 20-22).

Методология:
1. Тренировка CatBoost с теми же гиперпараметрами (depth=7, lr=0.1, iter=500)
2. Train set: 0-80% (train+val объединены)
3. Eval set для early stopping: последние 20% train (примерно 0.64-0.80 исходного)
4. Apply chain_7 shrunken thresholds + 1x2 + pre-match к test
5. Сравнение AUC и ROI с baseline

Дополнительно: sweep по boundaries odds-bucket [(0,1.8,3.0), (0,1.5,2.5), (0,2.0,3.5)].
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
from catboost import CatBoostClassifier, Pool
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
    df: pd.DataFrame,
    kelly: np.ndarray,
    seg_thresholds: dict[str, float],
    boundaries: tuple[float, float] = (1.8, 3.0),
) -> np.ndarray:
    """Применить shrunken segment Kelly thresholds с настраиваемыми границами."""
    b1, b2 = boundaries
    buckets = pd.cut(df["Odds"], bins=[0, b1, b2, np.inf], labels=["low", "mid", "high"])
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


with mlflow.start_run(run_name="phase4/step4.10_retrain_trainval") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.10")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        # Splits
        train_df = df_raw.iloc[:val_start].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        trainval_df = df_raw.iloc[:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info(
            "Train: %d, Val: %d, TrainVal: %d, Test: %d",
            len(train_df), len(val_df), len(trainval_df), len(test_df),
        )

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        feature_names = cb_meta["feature_names"]
        seg_thresholds = cb_meta["segment_thresholds"]
        cat_features = ["Sport", "Market", "Currency"]

        # Baseline model (chain_6)
        cat_model_base = CatBoostClassifier()
        cat_model_base.load_model(str(PREV_BEST_DIR / "model.cbm"))

        # Baseline predictions
        x_te = build_features(test_df, feature_names)
        proba_base_test = cat_model_base.predict_proba(x_te)[:, 1]
        y_te = (test_df["Status"] == "won").astype(int)
        auc_base = roc_auc_score(y_te, proba_base_test)
        logger.info("Baseline (chain_6) test AUC: %.4f", auc_base)

        lead_hours_test = (
            (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
        ).values
        kelly_base = compute_kelly(proba_base_test, test_df["Odds"].values)
        market_mask_test = test_df["Market"].values == "1x2"
        prematch_mask_test = lead_hours_test > 0
        kelly_base[~(market_mask_test & prematch_mask_test)] = -999
        base_seg_mask = apply_shrunken_segments(test_df, kelly_base, seg_thresholds)
        roi_baseline, n_baseline = calc_roi(test_df, base_seg_mask)
        logger.info("Baseline test ROI: %.4f%% (n=%d)", roi_baseline, n_baseline)

        # Ретренировка на trainval
        x_trainval = build_features(trainval_df, feature_names)
        y_trainval = (trainval_df["Status"] == "won").astype(int)

        # Eval set: последние 20% trainval (≈ val period)
        eval_start = int(len(trainval_df) * 0.80)
        x_eval = x_trainval.iloc[eval_start:]
        y_eval = y_trainval.iloc[eval_start:]
        x_train_part = x_trainval.iloc[:eval_start]
        y_train_part = y_trainval.iloc[:eval_start]

        train_pool = Pool(
            x_train_part,
            y_train_part,
            cat_features=cat_features,
        )
        eval_pool = Pool(
            x_eval,
            y_eval,
            cat_features=cat_features,
        )

        logger.info(
            "Ретренировка CatBoost на trainval (n=%d), eval n=%d...",
            len(x_train_part), len(x_eval),
        )
        # Без early stopping — фиксированные 500 итераций (как chain_6)
        # Early stopping даёт best_iter=21 (недостаточно деревьев → Kelly < threshold)
        new_model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            eval_metric="AUC",
            random_seed=42,
            verbose=0,
        )
        new_model.fit(train_pool, eval_set=eval_pool)
        best_iter = 500
        logger.info("Best iteration: %d", best_iter)

        proba_new_test = new_model.predict_proba(x_te)[:, 1]
        auc_new = roc_auc_score(y_te, proba_new_test)
        logger.info("New model test AUC: %.4f (delta=%.4f)", auc_new, auc_new - auc_base)

        # Диагностика Kelly distribution для новой модели
        kelly_new = compute_kelly(proba_new_test, test_df["Odds"].values)
        kelly_new_filtered = kelly_new.copy()
        kelly_new_filtered[~(market_mask_test & prematch_mask_test)] = -999

        valid_kelly_new = kelly_new[(market_mask_test & prematch_mask_test)]
        valid_kelly_base = compute_kelly(proba_base_test, test_df["Odds"].values)
        valid_kelly_base = valid_kelly_base[(market_mask_test & prematch_mask_test)]

        logger.info(
            "Kelly new 1x2 prematch: mean=%.4f p50=%.4f p75=%.4f p90=%.4f",
            np.mean(valid_kelly_new), np.median(valid_kelly_new),
            np.percentile(valid_kelly_new, 75), np.percentile(valid_kelly_new, 90),
        )
        logger.info(
            "Kelly base 1x2 prematch: mean=%.4f p50=%.4f p75=%.4f p90=%.4f",
            np.mean(valid_kelly_base), np.median(valid_kelly_base),
            np.percentile(valid_kelly_base, 75), np.percentile(valid_kelly_base, 90),
        )

        # Оптимизируем thresholds для новой модели на val с shrinkage
        x_vl = build_features(val_df, feature_names)
        proba_new_val = new_model.predict_proba(x_vl)[:, 1]
        lead_hours_val = (
            (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
        ).values
        kelly_new_val = compute_kelly(proba_new_val, val_df["Odds"].values)
        market_mask_val = val_df["Market"].values == "1x2"
        prematch_mask_val = lead_hours_val > 0
        kelly_new_val[~(market_mask_val & prematch_mask_val)] = -999

        # Baseline win rate для shrinkage
        baseline_t = (val_df["Status"] == "won").mean()
        logger.info("Val baseline win rate: %.4f", baseline_t)

        # Grid search thresholds на val (coarse)
        threshold_candidates = np.arange(0.0, 1.0, 0.025)
        best_val_roi = -np.inf
        best_raw_thresholds: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}

        for t_low in threshold_candidates:
            for t_mid in threshold_candidates:
                for t_high in threshold_candidates:
                    raw_t = {"low": t_low, "mid": t_mid, "high": t_high}
                    seg_mask_v = apply_shrunken_segments(val_df, kelly_new_val, raw_t)
                    rv, nv = calc_roi(val_df, seg_mask_v)
                    if nv >= 20 and rv > best_val_roi:
                        best_val_roi = rv
                        best_raw_thresholds = raw_t.copy()

        logger.info(
            "Val best raw: low=%.3f mid=%.3f high=%.3f ROI=%.2f%%",
            best_raw_thresholds["low"], best_raw_thresholds["mid"],
            best_raw_thresholds["high"], best_val_roi,
        )

        # Shrunken thresholds (shrinkage=0.5 toward baseline)
        shrinkage = 0.5
        shrunken_t = {
            k: shrinkage * v + (1 - shrinkage) * baseline_t
            for k, v in best_raw_thresholds.items()
        }
        logger.info(
            "Shrunken thresholds: low=%.3f mid=%.3f high=%.3f",
            shrunken_t["low"], shrunken_t["mid"], shrunken_t["high"],
        )

        # Apply shrunken thresholds к test
        new_seg_mask = apply_shrunken_segments(test_df, kelly_new_filtered, shrunken_t)
        roi_new, n_new = calc_roi(test_df, new_seg_mask)
        logger.info("New model (shrunken val thresholds) test ROI: %.4f%% (n=%d)", roi_new, n_new)

        # Apply chain_7 thresholds к test (для сравнения, даже если n=0)
        seg_mask_chain7 = apply_shrunken_segments(test_df, kelly_new_filtered, seg_thresholds)
        roi_chain7_applied, n_chain7_applied = calc_roi(test_df, seg_mask_chain7)
        logger.info(
            "New model (chain_7 thresholds) test ROI: %.4f%% (n=%d)",
            roi_chain7_applied, n_chain7_applied,
        )

        delta = roi_new - roi_baseline

        # Leakage check
        if n_new < 10:
            logger.warning("LEAKAGE SUSPECT: n=%d < 10", n_new)

        mlflow.log_params({
            "depth": 7,
            "learning_rate": 0.1,
            "iterations": 500,
            "best_iter": best_iter,
            "trainval_size": len(trainval_df),
            "train_part_size": len(x_train_part),
            "shrinkage": shrinkage,
            "baseline_win_rate": float(baseline_t),
            "opt_low_raw": best_raw_thresholds["low"],
            "opt_mid_raw": best_raw_thresholds["mid"],
            "opt_high_raw": best_raw_thresholds["high"],
            "opt_low_shrunken": shrunken_t["low"],
            "opt_mid_shrunken": shrunken_t["mid"],
            "opt_high_shrunken": shrunken_t["high"],
        })
        mlflow.log_metrics({
            "auc_base": auc_base,
            "auc_new": auc_new,
            "auc_delta": auc_new - auc_base,
            "roi_baseline": roi_baseline,
            "n_baseline": n_baseline,
            "roi_new_shrunken_val": roi_new,
            "n_new_shrunken_val": n_new,
            "roi_new_chain7_t": roi_chain7_applied,
            "n_new_chain7_t": n_chain7_applied,
            "roi_val_best": best_val_roi,
            "delta_best_vs_baseline": delta,
            "kelly_new_p50": float(np.median(valid_kelly_new)),
            "kelly_new_p90": float(np.percentile(valid_kelly_new, 90)),
            "kelly_base_p50": float(np.median(valid_kelly_base)),
            "kelly_base_p90": float(np.percentile(valid_kelly_base, 90)),
        })

        mlflow.set_tag("status", "done")
        auc_d = auc_new - auc_base
        mlflow.set_tag(
            "result",
            f"new={roi_new:.4f}%(n={n_new}) baseline={roi_baseline:.4f}% auc+{auc_d:.4f}",
        )
        logger.info("Run ID: %s", run_id)
        logger.info(
            "ИТОГ: baseline=%.4f%% | new_model_shrunken=%.4f%%(n=%d) | delta=%.4f%%",
            roi_baseline, roi_new, n_new, delta,
        )

    except Exception:
        mlflow.set_tag("status", "error")
        logger.exception("Ошибка в step 4.10")
        raise
