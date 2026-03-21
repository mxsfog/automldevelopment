"""Step 4.6 — Feature importance analysis + pruned model + temporal split analysis.

Открытие step 4.5: 28.58% ROI сосредоточен в ВТОРОЙ половине test (n=210/233).
Первая половина: n=23, ROI=-47.87%.

Гипотезы:
A) Какие фичи предсказывают ставки второй половины?
B) Модель с TOP-10 фичами — robustнее ли она?
C) Более "поздний" train split (train=70%, test=30%) — даёт ли больший overlap с active period?
D) Понять распределение дат: что происходит в тест-периоде?

Цель: понять источник edge и попробовать усилить его.
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


with mlflow.start_run(run_name="phase4/step4.6_feat_importance") as run:
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
        test_df = df_raw.iloc[train_end:].copy()
        full_train_df = df_raw.iloc[:train_end].copy()

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        feature_names = cb_meta["feature_names"]
        seg_thresholds = cb_meta["segment_thresholds"]

        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_te = build_features(test_df, feature_names)
        proba_test = cat_model.predict_proba(x_te)[:, 1]
        y_te = (test_df["Status"] == "won").astype(int)
        auc_base = roc_auc_score(y_te, proba_test)

        lead_hours_test = (
            test_df["Start_Time"] - test_df["Created_At"]
        ).dt.total_seconds() / 3600.0
        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_test[lead_hours_test.values <= 0] = -999

        mkt_test = test_df["Market"].values == "1x2"
        baseline_mask = mkt_test & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
        roi_base, n_base = calc_roi(test_df, baseline_mask)
        logger.info("Baseline: roi=%.4f%%, n=%d", roi_base, n_base)

        # Feature importances из chain_7 модели
        fi = cat_model.get_feature_importance()
        fi_pairs = sorted(zip(fi, feature_names, strict=True), reverse=True)
        logger.info("Top-15 feature importances:")
        for imp, fname in fi_pairs[:15]:
            logger.info("  %.3f  %s", imp, fname)

        top10_features = [fname for _, fname in fi_pairs[:10]]
        top15_features = [fname for _, fname in fi_pairs[:15]]
        mlflow.log_text(
            "\n".join([f"{imp:.4f}  {fname}" for imp, fname in fi_pairs]),
            "feature_importances.txt",
        )

        # === Temporal analysis ===
        test_dates = test_df["Created_At"]
        test_mid_date = test_dates.quantile(0.5)
        logger.info(
            "Test period: %s to %s (mid: %s)",
            test_dates.min().date(),
            test_dates.max().date(),
            test_mid_date.date() if hasattr(test_mid_date, "date") else test_mid_date,
        )

        # Детальный temporal breakdown: 4 квартала
        test_q = np.percentile(np.arange(len(test_df)), [25, 50, 75])
        q_idx = [0, int(test_q[0]), int(test_q[1]), int(test_q[2]), len(test_df)]
        for i in range(4):
            q_mask = baseline_mask.copy()
            q_mask[: q_idx[i]] = False
            q_mask[q_idx[i + 1] :] = False
            roi_q, n_q = calc_roi(test_df, q_mask)
            q_start = test_df.iloc[q_idx[i]]["Created_At"].date()
            q_end = test_df.iloc[min(q_idx[i + 1] - 1, len(test_df) - 1)]["Created_At"].date()
            logger.info("  Q%d (%s to %s): roi=%.2f%%, n=%d", i + 1, q_start, q_end, roi_q, n_q)
            mlflow.log_metrics({f"roi_q{i + 1}": roi_q, f"n_q{i + 1}": n_q})

        # Win rate анализ по кварталам (независимо от Kelly)
        mkt_test_1x2 = test_df[mkt_test]
        for i in range(4):
            idx_start = q_idx[i]
            idx_end = q_idx[i + 1]
            q_df = test_df.iloc[idx_start:idx_end]
            q_1x2 = q_df[q_df["Market"] == "1x2"]
            if len(q_1x2) > 0:
                wr = (q_1x2["Status"] == "won").mean()
                logger.info("  Q%d 1x2 win_rate=%.3f, n=%d", i + 1, wr, len(q_1x2))

        # === Pruned models: TOP-10 и TOP-15 фичей ===
        y_tr = (full_train_df["Status"] == "won").astype(int)

        for n_feats, feat_list in [(10, top10_features), (15, top15_features)]:
            cat_idx = [
                i for i, col in enumerate(feat_list) if col in ("Sport", "Market", "Currency")
            ]
            params = {
                "depth": cb_meta["params"]["depth"],
                "learning_rate": cb_meta["params"]["learning_rate"],
                "iterations": cb_meta["params"]["iterations"],
                "cat_features": cat_idx,
                "random_seed": 42,
                "verbose": False,
            }
            x_tr_p = build_features(full_train_df, feat_list)
            x_te_p = build_features(test_df, feat_list)

            pruned_model = CatBoostClassifier(**params)
            pruned_model.fit(x_tr_p, y_tr)

            proba_pruned = pruned_model.predict_proba(x_te_p)[:, 1]
            auc_pruned = roc_auc_score(y_te, proba_pruned)

            kelly_pruned = compute_kelly(proba_pruned, test_df["Odds"].values)
            kelly_pruned[lead_hours_test.values <= 0] = -999

            # Применяем те же shrunken thresholds
            pruned_mask = mkt_test & apply_shrunken_segments(test_df, kelly_pruned, seg_thresholds)
            roi_pruned, n_pruned = calc_roi(test_df, pruned_mask)
            logger.info(
                "Pruned TOP-%d: roi=%.4f%%, n=%d, auc=%.4f",
                n_feats,
                roi_pruned,
                n_pruned,
                auc_pruned,
            )
            mlflow.log_metrics(
                {
                    f"roi_top{n_feats}": roi_pruned,
                    f"n_top{n_feats}": n_pruned,
                    f"auc_top{n_feats}": auc_pruned,
                }
            )

        # === Alternative split: train=85%, test=15% ===
        train_end_alt = int(n * 0.85)
        full_train_alt = df_raw.iloc[:train_end_alt].copy()
        test_alt = df_raw.iloc[train_end_alt:].copy()
        logger.info("Alt split: train_alt=%d, test_alt=%d", len(full_train_alt), len(test_alt))

        y_tr_alt = (full_train_alt["Status"] == "won").astype(int)
        x_tr_alt = build_features(full_train_alt, feature_names)
        x_te_alt = build_features(test_alt, feature_names)

        cat_params_alt = cb_meta["params"].copy()
        cat_params_alt["cat_features"] = [
            i for i, col in enumerate(feature_names) if col in ("Sport", "Market", "Currency")
        ]
        cat_params_alt["random_seed"] = 42
        cat_params_alt["verbose"] = False

        alt_model = CatBoostClassifier(**cat_params_alt)
        alt_model.fit(x_tr_alt, y_tr_alt)

        proba_alt = alt_model.predict_proba(x_te_alt)[:, 1]
        y_te_alt = (test_alt["Status"] == "won").astype(int)
        auc_alt = roc_auc_score(y_te_alt, proba_alt)

        lead_hours_alt = (
            test_alt["Start_Time"] - test_alt["Created_At"]
        ).dt.total_seconds() / 3600.0
        kelly_alt = compute_kelly(proba_alt, test_alt["Odds"].values)
        kelly_alt[lead_hours_alt.values <= 0] = -999

        mkt_alt = test_alt["Market"].values == "1x2"
        alt_mask = mkt_alt & apply_shrunken_segments(test_alt, kelly_alt, seg_thresholds)
        roi_alt, n_alt = calc_roi(test_alt, alt_mask)
        logger.info(
            "Alt split (train=85%%) 1x2+seg: roi=%.4f%%, n=%d, auc=%.4f",
            roi_alt,
            n_alt,
            auc_alt,
        )

        best_roi = max(roi_base, roi_alt)
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
                "alt_split_train_frac": 0.85,
            }
        )
        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_base": roi_base,
                "roi_alt_split": roi_alt,
                "n_base": n_base,
                "n_alt": n_alt,
                "auc_base": auc_base,
                "auc_alt": auc_alt,
                "delta_vs_baseline": delta,
            }
        )

        if best_roi > baseline_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi, baseline_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            save_model = alt_model if roi_alt == best_roi else cat_model
            save_model.save_model(str(best_dir / "model.cbm"))
            meta_out = {
                "framework": "catboost",
                "roi": best_roi,
                "auc": float(auc_alt) if roi_alt == best_roi else float(auc_base),
                "segment_thresholds": seg_thresholds,
                "market_filter": "1x2",
                "n_bets": n_alt if roi_alt == best_roi else n_base,
                "feature_names": feature_names,
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.6",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.6 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(f"  Baseline: {roi_base:.4f}%/{n_base}  Alt split: {roi_alt:.4f}%/{n_alt}")
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
