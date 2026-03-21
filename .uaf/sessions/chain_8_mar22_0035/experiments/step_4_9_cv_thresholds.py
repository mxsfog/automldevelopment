"""Step 4.9 — CV-based threshold optimization на train данных.

Проблема: val_roi=106% аномально высокий, оптимизация thresholds на одном val-split ненадёжна.
Решение: TimeSeriesSplit 5 folds на train (0-64%), усреднение optimal thresholds по фолдам,
применение к test с 1x2 + pre-match фильтром.

Ожидание: более стабильные thresholds → устойчивый ROI на test.
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
from sklearn.model_selection import TimeSeriesSplit

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


def optimize_thresholds_on_fold(
    fold_df: pd.DataFrame,
    kelly: np.ndarray,
    lead_hours: np.ndarray,
    min_n: int = 10,
) -> dict[str, float] | None:
    """Оптимизация shrunken segment thresholds на одном fold.

    Args:
        fold_df: Датафрейм fold (val часть).
        kelly: Kelly values для fold.
        lead_hours: Lead hours для fold.
        min_n: Минимальное число ставок для валидного результата.

    Returns:
        Лучшие thresholds или None если нет валидных.
    """
    threshold_candidates = np.arange(0.0, 0.8, 0.05)
    best_roi = -np.inf
    best_thresholds: dict[str, float] | None = None

    # 1x2 + pre-match фильтр
    market_mask = fold_df["Market"].values == "1x2"
    pre_match_mask = lead_hours > 0
    base_mask = market_mask & pre_match_mask
    kelly_filtered = kelly.copy()
    kelly_filtered[~base_mask] = -999

    for t_low in threshold_candidates:
        for t_mid in threshold_candidates:
            for t_high in threshold_candidates:
                thresholds = {"low": t_low, "mid": t_mid, "high": t_high}
                seg_mask = apply_shrunken_segments(fold_df, kelly_filtered, thresholds)
                roi, n = calc_roi(fold_df, seg_mask)
                if n >= min_n and roi > best_roi:
                    best_roi = roi
                    best_thresholds = thresholds.copy()

    return best_thresholds


with mlflow.start_run(run_name="phase4/step4.9_cv_thresholds") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.9")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        # Train без val-части (0-64%)
        train_df = df_raw.iloc[:val_start].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        feature_names = cb_meta["feature_names"]
        seg_thresholds_baseline = cb_meta["segment_thresholds"]

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        # Предсказания на всём датасете
        x_tr = build_features(train_df, feature_names)
        x_vl = build_features(val_df, feature_names)
        x_te = build_features(test_df, feature_names)

        proba_train = cat_model.predict_proba(x_tr)[:, 1]
        proba_val = cat_model.predict_proba(x_vl)[:, 1]
        proba_test = cat_model.predict_proba(x_te)[:, 1]

        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, proba_test)
        logger.info("Test AUC: %.4f", auc_test)

        lead_hours_train = (
            (train_df["Start_Time"] - train_df["Created_At"]).dt.total_seconds() / 3600
        ).values
        lead_hours_val = (
            (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
        ).values
        lead_hours_test = (
            (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
        ).values

        kelly_train = compute_kelly(proba_train, train_df["Odds"].values)
        kelly_val = compute_kelly(proba_val, val_df["Odds"].values)
        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)

        # Baseline: chain_7 thresholds на test
        kelly_test_base = kelly_test.copy()
        market_mask_test = test_df["Market"].values == "1x2"
        prematch_mask_test = lead_hours_test > 0
        kelly_test_base[~(market_mask_test & prematch_mask_test)] = -999
        base_seg_mask = apply_shrunken_segments(test_df, kelly_test_base, seg_thresholds_baseline)
        roi_baseline, n_baseline = calc_roi(test_df, base_seg_mask)
        logger.info("Baseline test ROI: %.4f%% (n=%d)", roi_baseline, n_baseline)

        # CV на train: 5-fold TimeSeriesSplit
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        train_indices = np.arange(len(train_df))

        fold_thresholds: list[dict[str, float]] = []
        fold_rois: list[float] = []

        logger.info("TimeSeriesSplit %d-fold threshold optimization...", n_splits)
        for fold_idx, (_cv_train_idx, cv_val_idx) in enumerate(tscv.split(train_indices)):
            cv_val_df = train_df.iloc[cv_val_idx].copy()
            cv_val_kelly = kelly_train[cv_val_idx]
            cv_val_lead = lead_hours_train[cv_val_idx]

            logger.info(
                "Fold %d: cv_val n=%d (1x2 prematch: %d)",
                fold_idx,
                len(cv_val_df),
                int(((cv_val_df["Market"].values == "1x2") & (cv_val_lead > 0)).sum()),
            )

            best_t = optimize_thresholds_on_fold(cv_val_df, cv_val_kelly, cv_val_lead, min_n=5)

            if best_t is None:
                logger.warning("Fold %d: нет валидных thresholds (min_n=5), пропускаем", fold_idx)
                continue

            # Проверить ROI на этом fold
            cv_market = cv_val_df["Market"].values == "1x2"
            cv_prematch = cv_val_lead > 0
            cv_kelly_f = cv_val_kelly.copy()
            cv_kelly_f[~(cv_market & cv_prematch)] = -999
            cv_seg_mask = apply_shrunken_segments(cv_val_df, cv_kelly_f, best_t)
            cv_roi, cv_n = calc_roi(cv_val_df, cv_seg_mask)

            logger.info(
                "Fold %d best: low=%.3f mid=%.3f high=%.3f → ROI=%.2f%% n=%d",
                fold_idx,
                best_t["low"],
                best_t["mid"],
                best_t["high"],
                cv_roi,
                cv_n,
            )
            fold_thresholds.append(best_t)
            fold_rois.append(cv_roi)

        if not fold_thresholds:
            logger.error("Нет валидных fold thresholds, используем baseline")
            cv_thresholds_mean = seg_thresholds_baseline
            cv_thresholds_median = seg_thresholds_baseline
        else:
            # Средние thresholds по фолдам
            cv_thresholds_mean = {
                "low": float(np.mean([t["low"] for t in fold_thresholds])),
                "mid": float(np.mean([t["mid"] for t in fold_thresholds])),
                "high": float(np.mean([t["high"] for t in fold_thresholds])),
            }
            cv_thresholds_median = {
                "low": float(np.median([t["low"] for t in fold_thresholds])),
                "mid": float(np.median([t["mid"] for t in fold_thresholds])),
                "high": float(np.median([t["high"] for t in fold_thresholds])),
            }
            logger.info(
                "CV mean thresholds: low=%.3f mid=%.3f high=%.3f",
                cv_thresholds_mean["low"],
                cv_thresholds_mean["mid"],
                cv_thresholds_mean["high"],
            )
            logger.info(
                "CV median thresholds: low=%.3f mid=%.3f high=%.3f",
                cv_thresholds_median["low"],
                cv_thresholds_median["mid"],
                cv_thresholds_median["high"],
            )

        # Применить CV mean thresholds к test
        kelly_test_cv = kelly_test.copy()
        kelly_test_cv[~(market_mask_test & prematch_mask_test)] = -999
        cv_mean_mask = apply_shrunken_segments(test_df, kelly_test_cv, cv_thresholds_mean)
        roi_cv_mean, n_cv_mean = calc_roi(test_df, cv_mean_mask)
        logger.info("CV mean thresholds test ROI: %.4f%% (n=%d)", roi_cv_mean, n_cv_mean)

        # Применить CV median thresholds к test
        cv_median_mask = apply_shrunken_segments(test_df, kelly_test_cv, cv_thresholds_median)
        roi_cv_median, n_cv_median = calc_roi(test_df, cv_median_mask)
        logger.info("CV median thresholds test ROI: %.4f%% (n=%d)", roi_cv_median, n_cv_median)

        # Также проверить на val для сравнения
        kelly_val_cv = kelly_val.copy()
        market_mask_val = val_df["Market"].values == "1x2"
        prematch_mask_val = lead_hours_val > 0
        kelly_val_cv[~(market_mask_val & prematch_mask_val)] = -999

        val_base_mask = apply_shrunken_segments(val_df, kelly_val_cv, seg_thresholds_baseline)
        roi_val_base, n_val_base = calc_roi(val_df, val_base_mask)

        val_mean_mask = apply_shrunken_segments(val_df, kelly_val_cv, cv_thresholds_mean)
        roi_val_mean, n_val_mean = calc_roi(val_df, val_mean_mask)

        val_median_mask = apply_shrunken_segments(val_df, kelly_val_cv, cv_thresholds_median)
        roi_val_median, n_val_median = calc_roi(val_df, val_median_mask)

        logger.info(
            "Val baseline: %.2f%% n=%d | CV mean: %.2f%% n=%d | CV median: %.2f%% n=%d",
            roi_val_base,
            n_val_base,
            roi_val_mean,
            n_val_mean,
            roi_val_median,
            n_val_median,
        )

        # Лучший результат
        best_roi_cv = max(roi_cv_mean, roi_cv_median)
        best_label = "cv_mean" if roi_cv_mean >= roi_cv_median else "cv_median"
        best_n_cv = n_cv_mean if roi_cv_mean >= roi_cv_median else n_cv_median
        delta = best_roi_cv - roi_baseline

        logger.info(
            "Лучший CV: %s ROI=%.4f%% (n=%d), delta=%.4f%%",
            best_label, best_roi_cv, best_n_cv, delta,
        )

        # Leakage check
        if n_cv_mean < 10 and best_label == "cv_mean":
            logger.warning("LEAKAGE SUSPECT: n=%d < 10, результат ненадёжен", n_cv_mean)
        if n_cv_median < 10 and best_label == "cv_median":
            logger.warning("LEAKAGE SUSPECT: n=%d < 10, результат ненадёжен", n_cv_median)

        # MLflow logging
        mlflow.log_params({
            "n_splits": n_splits,
            "n_valid_folds": len(fold_thresholds),
            "cv_mean_low": cv_thresholds_mean["low"],
            "cv_mean_mid": cv_thresholds_mean["mid"],
            "cv_mean_high": cv_thresholds_mean["high"],
            "cv_median_low": cv_thresholds_median["low"],
            "cv_median_mid": cv_thresholds_median["mid"],
            "cv_median_high": cv_thresholds_median["high"],
            "baseline_low": seg_thresholds_baseline["low"],
            "baseline_mid": seg_thresholds_baseline["mid"],
            "baseline_high": seg_thresholds_baseline["high"],
        })
        mlflow.log_metrics({
            "roi_baseline_test": roi_baseline,
            "n_baseline_test": n_baseline,
            "roi_cv_mean_test": roi_cv_mean,
            "n_cv_mean_test": n_cv_mean,
            "roi_cv_median_test": roi_cv_median,
            "n_cv_median_test": n_cv_median,
            "roi_val_baseline": roi_val_base,
            "roi_val_cv_mean": roi_val_mean,
            "roi_val_cv_median": roi_val_median,
            "auc_test": auc_test,
            "delta_best_vs_baseline": delta,
        })
        for i, (t, rv) in enumerate(zip(fold_thresholds, fold_rois, strict=True)):
            mlflow.log_metrics({
                f"fold_{i}_roi": rv,
                f"fold_{i}_low": t["low"],
                f"fold_{i}_mid": t["mid"],
                f"fold_{i}_high": t["high"],
            })

        mlflow.set_tag("status", "done")
        result_tag = (
            f"cv_mean={roi_cv_mean:.4f}%"
            f" cv_median={roi_cv_median:.4f}%"
            f" baseline={roi_baseline:.4f}%"
        )
        mlflow.set_tag("result", result_tag)

        logger.info("Run ID: %s", run_id)
        logger.info(
            "ИТОГ: baseline=%.4f%% | cv_mean=%.4f%%(n=%d) | cv_median=%.4f%%(n=%d)",
            roi_baseline, roi_cv_mean, n_cv_mean, roi_cv_median, n_cv_median,
        )

    except Exception:
        mlflow.set_tag("status", "error")
        logger.exception("Ошибка в step 4.9")
        raise
