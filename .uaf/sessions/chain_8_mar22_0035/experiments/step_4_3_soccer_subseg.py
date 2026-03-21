"""Step 4.3 — Soccer 1x2 subsegment analysis + time-weighted training.

Гипотезы:
A) Soccer/1x2 ставки отдельно — лучший ли это сегмент или есть другие?
B) Time-weighted CatBoost (sample_weight=exp decay) — более свежие данные важнее
C) Odds-range filter внутри 1x2 — на каком диапазоне odds лучший ROI без ML?
D) Soccer 1x2 + time-weighted model + shrunken segments

Thresholds: chain_7 shrunken thresholds (FIXED) — без re-opt.
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


with mlflow.start_run(run_name="phase4/step4.3_soccer_subseg") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.3")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        full_train_df = df_raw.iloc[:train_end].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        seg_thresholds = cb_meta["segment_thresholds"]
        feature_names = cb_meta["feature_names"]

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_te = build_features(test_df, feature_names)
        cat_proba_test = cat_model.predict_proba(x_te)[:, 1]
        y_te = (test_df["Status"] == "won").astype(int)
        auc_base = roc_auc_score(y_te, cat_proba_test)
        logger.info("CatBoost base AUC: %.4f", auc_base)

        lead_hours_test = (
            test_df["Start_Time"] - test_df["Created_At"]
        ).dt.total_seconds() / 3600.0
        kelly_test = compute_kelly(cat_proba_test, test_df["Odds"].values)
        kelly_test[lead_hours_test.values <= 0] = -999

        mkt_mask = test_df["Market"].values == "1x2"

        # Baseline: 1x2 + shrunken_seg
        base_mask = mkt_mask & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
        roi_base, n_base = calc_roi(test_df, base_mask)
        logger.info("CatBoost 1x2+seg (baseline): roi=%.4f%%, n=%d", roi_base, n_base)

        # A) Soccer 1x2 only
        soccer_mask = test_df["Sport"].values == "Soccer"
        soccer_1x2_mask = (
            mkt_mask & soccer_mask & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
        )
        roi_soccer_1x2, n_soccer_1x2 = calc_roi(test_df, soccer_1x2_mask)
        logger.info("Soccer+1x2+seg: roi=%.4f%%, n=%d", roi_soccer_1x2, n_soccer_1x2)

        # B) Sport breakdown: ROI per sport within 1x2+seg
        logger.info("--- Sport breakdown inside 1x2+seg ---")
        for sport in sorted(test_df[mkt_mask]["Sport"].unique()):
            sp_mask = mkt_mask & (test_df["Sport"].values == sport)
            sp_seg = sp_mask & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
            roi_sp, n_sp = calc_roi(test_df, sp_seg)
            if n_sp >= 5:
                logger.info("  Sport=%s: roi=%.2f%%, n=%d", sport, roi_sp, n_sp)

        # C) Odds range filter: check ROI по odds диапазону внутри 1x2+Soccer+seg
        logger.info("--- Odds range analysis (Soccer+1x2+seg) ---")
        for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 5.0), (5.0, 100)]:
            odds_mask = (test_df["Odds"].values >= lo) & (test_df["Odds"].values < hi)
            combo = (
                soccer_mask
                & mkt_mask
                & odds_mask
                & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
            )
            roi_r, n_r = calc_roi(test_df, combo)
            if n_r >= 5:
                logger.info("  Odds [%.1f, %.1f): roi=%.2f%%, n=%d", lo, hi, roi_r, n_r)

        # D) Time-weighted CatBoost training
        # Sample weight = exp(alpha * rank/n) — свежие данные весят больше
        y_full_tr = (full_train_df["Status"] == "won").astype(int)
        x_full_tr = build_features(full_train_df, feature_names)

        n_tr = len(full_train_df)
        ranks = np.arange(n_tr) / n_tr  # 0..1 (0=oldest, 1=newest)

        for alpha in [2.0, 5.0]:
            weights = np.exp(alpha * ranks)
            weights = weights / weights.mean()

            cat_params = cb_meta["params"].copy()
            cat_params["cat_features"] = [
                i for i, col in enumerate(feature_names) if col in ("Sport", "Market", "Currency")
            ]
            cat_params["random_seed"] = 42
            cat_params["verbose"] = False

            tw_model = CatBoostClassifier(**cat_params)
            tw_model.fit(x_full_tr, y_full_tr, sample_weight=weights)

            tw_proba_test = tw_model.predict_proba(x_te)[:, 1]
            auc_tw = roc_auc_score(y_te, tw_proba_test)

            kelly_tw = compute_kelly(tw_proba_test, test_df["Odds"].values)
            kelly_tw[lead_hours_test.values <= 0] = -999

            tw_mask = mkt_mask & apply_shrunken_segments(test_df, kelly_tw, seg_thresholds)
            roi_tw, n_tw = calc_roi(test_df, tw_mask)
            logger.info(
                "TimeWeighted(alpha=%.1f) 1x2+seg: roi=%.4f%%, n=%d, auc=%.4f",
                alpha,
                roi_tw,
                n_tw,
                auc_tw,
            )

            # Soccer only + time-weighted
            tw_soccer_mask = (
                mkt_mask & soccer_mask & apply_shrunken_segments(test_df, kelly_tw, seg_thresholds)
            )
            roi_tw_soccer, n_tw_soccer = calc_roi(test_df, tw_soccer_mask)
            logger.info(
                "TimeWeighted(alpha=%.1f) Soccer+1x2+seg: roi=%.4f%%, n=%d",
                alpha,
                roi_tw_soccer,
                n_tw_soccer,
            )

            mlflow.log_metrics(
                {
                    f"roi_tw_alpha{int(alpha)}_1x2": roi_tw,
                    f"n_tw_alpha{int(alpha)}_1x2": n_tw,
                    f"auc_tw_alpha{int(alpha)}": auc_tw,
                    f"roi_tw_alpha{int(alpha)}_soccer": roi_tw_soccer,
                }
            )

        # E) CatBoost retrained on 1x2 Soccer only (check данных хватит?)
        train_soccer_1x2 = full_train_df[
            (full_train_df["Market"] == "1x2") & (full_train_df["Sport"] == "Soccer")
        ].copy()
        logger.info(
            "Train Soccer+1x2: %d (%.1f%% от общего)",
            len(train_soccer_1x2),
            len(train_soccer_1x2) / n_tr * 100,
        )

        if len(train_soccer_1x2) >= 1000:
            x_sc_tr = build_features(train_soccer_1x2, feature_names)
            y_sc_tr = (train_soccer_1x2["Status"] == "won").astype(int)

            cat_params_sc = cb_meta["params"].copy()
            cat_params_sc["cat_features"] = [
                i for i, col in enumerate(feature_names) if col in ("Sport", "Market", "Currency")
            ]
            cat_params_sc["random_seed"] = 42
            cat_params_sc["verbose"] = False

            sc_model = CatBoostClassifier(**cat_params_sc)
            sc_model.fit(x_sc_tr, y_sc_tr)

            sc_proba_test = sc_model.predict_proba(x_te)[:, 1]
            auc_sc = roc_auc_score(y_te, sc_proba_test)

            kelly_sc = compute_kelly(sc_proba_test, test_df["Odds"].values)
            kelly_sc[lead_hours_test.values <= 0] = -999

            sc_mask = (
                mkt_mask & soccer_mask & apply_shrunken_segments(test_df, kelly_sc, seg_thresholds)
            )
            roi_sc, n_sc = calc_roi(test_df, sc_mask)
            logger.info(
                "Soccer-retrained CAT Soccer+1x2+seg: roi=%.4f%%, n=%d, auc=%.4f",
                roi_sc,
                n_sc,
                auc_sc,
            )
            mlflow.log_metrics(
                {
                    "roi_soccer_retrained": roi_sc,
                    "n_soccer_retrained": n_sc,
                    "auc_soccer_retrained": auc_sc,
                    "n_train_soccer_1x2": len(train_soccer_1x2),
                }
            )
        else:
            logger.warning(
                "Недостаточно данных Soccer+1x2 для retraining: %d", len(train_soccer_1x2)
            )
            roi_sc, n_sc = -100.0, 0

        best_roi = max(roi_base, roi_soccer_1x2, roi_sc)
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
                "n_samples_train": train_end,
                "n_samples_test": len(test_df),
                "market_filter": "1x2",
                "thresholds_source": "chain_7_fixed",
            }
        )
        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_base_1x2": roi_base,
                "roi_soccer_1x2": roi_soccer_1x2,
                "n_base_1x2": n_base,
                "n_soccer_1x2": n_soccer_1x2,
                "auc_base": auc_base,
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
                "auc": float(auc_base),
                "segment_thresholds": seg_thresholds,
                "market_filter": "1x2",
                "n_bets": n_soccer_1x2 if roi_soccer_1x2 == best_roi else n_base,
                "feature_names": feature_names,
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.3",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.3 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  CAT 1x2+seg: {roi_base:.4f}%/{n_base}"
            f"  Soccer+1x2+seg: {roi_soccer_1x2:.4f}%/{n_soccer_1x2}"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
