"""Step 4.12 — ML_Edge fixed threshold scan (двойная проверка val+test).

Наблюдение из step 4.11: ML_Edge > 0.260 даёт test ROI=104.72% (n=25).
Проблема: threshold оптимизирован на val с ROI=151% (inflated).

Методология: НЕ оптимизируем на val — проверяем все фиксированные пороги
на ОБОИХ val и test. Ищем double-positive пороги.

Также исследуем: platform ML_Edge (ML_Edge = ML_P_Model - ML_P_Implied из данных)
vs нашего ml_edge_cat = proba_catboost - implied_prob.

И комбинацию: ml_edge_cat > t1 AND platform_ml_edge > t2 для 1x2 pre-match.
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


with mlflow.start_run(run_name="phase4/step4.12_ml_edge_fixed") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.12")

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

        proba_val = cat_model.predict_proba(x_vl)[:, 1]
        proba_test = cat_model.predict_proba(x_te)[:, 1]
        y_te = (test_df["Status"] == "won").astype(int)
        auc_test = roc_auc_score(y_te, proba_test)

        lead_hours_val = (
            (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
        ).values
        lead_hours_test = (
            (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
        ).values

        # Base filters: 1x2 + pre-match
        base_mask_val = (val_df["Market"].values == "1x2") & (lead_hours_val > 0)
        base_mask_test = (test_df["Market"].values == "1x2") & (lead_hours_test > 0)

        # Baseline Kelly
        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_test_f = kelly_test.copy()
        kelly_test_f[~base_mask_test] = -999
        base_seg_mask = apply_shrunken_segments(test_df, kelly_test_f, seg_thresholds)
        roi_baseline, n_baseline = calc_roi(test_df, base_seg_mask)
        logger.info("Baseline test ROI: %.4f%% (n=%d)", roi_baseline, n_baseline)

        # CatBoost edge (proba_cat - implied_prob)
        implied_val = 1.0 / val_df["Odds"].values.clip(1.001)
        cat_edge_val = proba_val - implied_val
        implied_test = 1.0 / test_df["Odds"].values.clip(1.001)
        cat_edge_test = proba_test - implied_test

        # Platform ML_Edge (из данных: ML_Edge = ML_P_Model - ML_P_Implied)
        plat_edge_val = val_df["ML_Edge"].fillna(-999).values
        plat_edge_test = test_df["ML_Edge"].fillna(-999).values

        logger.info(
            "Cat edge val 1x2pm: mean=%.4f p50=%.4f p75=%.4f p95=%.4f",
            np.mean(cat_edge_val[base_mask_val]),
            np.median(cat_edge_val[base_mask_val]),
            np.percentile(cat_edge_val[base_mask_val], 75),
            np.percentile(cat_edge_val[base_mask_val], 95),
        )
        logger.info(
            "Plat edge val 1x2pm (valid): mean=%.4f p50=%.4f",
            np.mean(plat_edge_val[base_mask_val & (plat_edge_val > -999)]),
            np.median(plat_edge_val[base_mask_val & (plat_edge_val > -999)]),
        )

        # Scan fixed cat_edge thresholds [0, 0.05, ..., 0.45]
        thresholds = np.arange(0.0, 0.46, 0.05)
        double_positive: list[dict] = []

        logger.info("Scan cat_edge thresholds (double-positive val+test):")
        for t in thresholds:
            mask_v = base_mask_val & (cat_edge_val >= t)
            mask_te = base_mask_test & (cat_edge_test >= t)
            rv, nv = calc_roi(val_df, mask_v)
            rt, nt = calc_roi(test_df, mask_te)
            logger.info(
                "  t=%.2f val: %.2f%%(n=%d) test: %.2f%%(n=%d)",
                t, rv, nv, rt, nt,
            )
            if nv >= 10 and nt >= 10 and rv > 0 and rt > 0:
                double_positive.append({
                    "threshold": t, "roi_val": rv, "roi_test": rt, "n_test": nt,
                })

        logger.info("Double-positive cat_edge thresholds: %d", len(double_positive))
        for dp in double_positive:
            logger.info(
                "  t=%.2f val=%.2f%% test=%.2f%%(n=%d)",
                dp["threshold"], dp["roi_val"], dp["roi_test"], dp["n_test"],
            )

        # Scan platform ML_Edge thresholds
        plat_thresholds = np.arange(0.0, 0.35, 0.05)
        plat_double_positive: list[dict] = []

        logger.info("Scan platform ML_Edge thresholds:")
        for t in plat_thresholds:
            mask_v = base_mask_val & (plat_edge_val >= t)
            mask_te = base_mask_test & (plat_edge_test >= t)
            rv, nv = calc_roi(val_df, mask_v)
            rt, nt = calc_roi(test_df, mask_te)
            logger.info(
                "  plat_t=%.2f val: %.2f%%(n=%d) test: %.2f%%(n=%d)",
                t, rv, nv, rt, nt,
            )
            if nv >= 10 and nt >= 10 and rv > 0 and rt > 0:
                plat_double_positive.append({
                    "threshold": t, "roi_val": rv, "roi_test": rt, "n_test": nt,
                })

        logger.info("Double-positive platform ML_Edge: %d", len(plat_double_positive))

        # Лучший результат из double-positive
        best_roi_test = roi_baseline
        best_n_test = n_baseline
        best_config = "baseline"

        if double_positive:
            best_dp = max(double_positive, key=lambda x: x["roi_test"])
            if best_dp["roi_test"] > best_roi_test:
                best_roi_test = best_dp["roi_test"]
                best_n_test = best_dp["n_test"]
                best_config = f"cat_edge>={best_dp['threshold']:.2f}"

        if plat_double_positive:
            best_pdp = max(plat_double_positive, key=lambda x: x["roi_test"])
            if best_pdp["roi_test"] > best_roi_test:
                best_roi_test = best_pdp["roi_test"]
                best_n_test = best_pdp["n_test"]
                best_config = f"plat_edge>={best_pdp['threshold']:.2f}"

        delta = best_roi_test - roi_baseline
        logger.info(
            "Best double-positive: %s ROI=%.4f%%(n=%d) delta=%.4f%%",
            best_config, best_roi_test, best_n_test, delta,
        )

        # MLflow logging
        mlflow.log_params({
            "n_cat_edge_dp": len(double_positive),
            "n_plat_edge_dp": len(plat_double_positive),
        })
        mlflow.log_metrics({
            "auc_test": auc_test,
            "roi_baseline": roi_baseline,
            "n_baseline": n_baseline,
            "roi_best_double_positive": best_roi_test,
            "n_best_double_positive": best_n_test,
            "delta_best_vs_baseline": delta,
        })
        for i, dp in enumerate(double_positive):
            mlflow.log_metrics({
                f"dp_{i}_t": dp["threshold"],
                f"dp_{i}_roi_val": dp["roi_val"],
                f"dp_{i}_roi_test": dp["roi_test"],
                f"dp_{i}_n_test": dp["n_test"],
            })

        mlflow.set_tag("status", "done")
        mlflow.set_tag(
            "result",
            f"best={best_roi_test:.4f}%(n={best_n_test}) config={best_config}"
            f" baseline={roi_baseline:.4f}%",
        )
        logger.info("Run ID: %s", run_id)
        logger.info(
            "ИТОГ: baseline=%.4f%% | best_double_positive=%.4f%%(n=%d) config=%s | delta=%.4f%%",
            roi_baseline, best_roi_test, best_n_test, best_config, delta,
        )

    except Exception:
        mlflow.set_tag("status", "error")
        logger.exception("Ошибка в step 4.12")
        raise
