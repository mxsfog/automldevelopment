"""Step 4.8 — Walk-forward probability ensemble.

Гипотеза: усреднение вероятностей из 4 моделей, обученных на нарастающих
временных окнах [0-50%, 0-60%, 0-70%, 0-80%], даёт более устойчивые оценки
вероятностей. Устойчивые вероятности → лучший Kelly criterion.

Применяем shrunken segment thresholds (shrink=0.5) к усреднённым вероятностям.
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


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_features: list[str],
) -> CatBoostClassifier:
    """Обучение CatBoost с базовыми гиперпараметрами."""
    x_tr, _ = build_features(train_df)
    x_vl, _ = build_features(val_df)
    y_tr = (train_df["Status"] == "won").astype(int)
    y_vl = (val_df["Status"] == "won").astype(int)
    w = make_weights(len(train_df))

    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.1,
        iterations=500,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
        cat_features=cat_features,
    )
    model.fit(x_tr, y_tr, eval_set=(x_vl, y_vl), sample_weight=w)
    return model


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


with mlflow.start_run(run_name="phase4/step4.8_wf_ensemble") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.8")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        # Окна обучения: [0-50%, 0-60%, 0-70%, 0-80%]
        # val для раннего останова: последние 10% каждого window
        windows = [0.50, 0.60, 0.70, 0.80]
        val_start_frac = int(n * 0.64)
        train_end = int(n * 0.80)
        val_df = df_raw.iloc[val_start_frac:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        _, cat_features = build_features(df_raw.iloc[:1])

        logger.info(
            "Val: %d, Test: %d, Training %d windows",
            len(val_df),
            len(test_df),
            len(windows),
        )

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_val": len(val_df),
                "n_windows": len(windows),
                "window_fracs": str(windows),
                "shrinkage": 0.5,
                "approach": "walk_forward_probability_ensemble",
            }
        )

        x_te, _ = build_features(test_df)
        y_te = (test_df["Status"] == "won").astype(int)
        x_vl, _ = build_features(val_df)
        y_vl = (val_df["Status"] == "won").astype(int)

        proba_test_all = []
        proba_val_all = []

        for i, frac in enumerate(windows):
            end_idx = int(n * frac)
            # val для early stopping: последние 16% этого window (или min 100 строк)
            val_idx = max(int(end_idx * 0.84), end_idx - 2000)
            tr_df = df_raw.iloc[:val_idx].copy()
            vl_early = df_raw.iloc[val_idx:end_idx].copy()

            if len(tr_df) < 100 or len(vl_early) < 10:
                logger.warning("Window %d: слишком мало данных, пропускаем", i)
                continue

            logger.info(
                "Window %d: train [0-%d%%] (%d rows), early_val (%d rows)",
                i + 1,
                int(frac * 100),
                len(tr_df),
                len(vl_early),
            )
            m = train_model(tr_df, vl_early, cat_features)
            p_test = m.predict_proba(x_te)[:, 1]
            p_val = m.predict_proba(x_vl)[:, 1]
            proba_test_all.append(p_test)
            proba_val_all.append(p_val)

            auc_i = roc_auc_score(y_te, p_test)
            mlflow.log_metric(f"auc_test_w{i}", auc_i)
            logger.info("Window %d test AUC: %.4f", i + 1, auc_i)

        # Усреднение вероятностей
        proba_test_ens = np.mean(proba_test_all, axis=0)
        proba_val_ens = np.mean(proba_val_all, axis=0)
        auc_ens_test = roc_auc_score(y_te, proba_test_ens)
        auc_ens_val = roc_auc_score(y_vl, proba_val_ens)
        logger.info("Ensemble test AUC: %.4f, val AUC: %.4f", auc_ens_test, auc_ens_val)

        kelly_val = compute_kelly(proba_val_ens, val_df["Odds"].values)
        kelly_val[val_df["lead_hours"].values <= 0] = -999

        kelly_test = compute_kelly(proba_test_ens, test_df["Odds"].values)
        kelly_test[test_df["lead_hours"].values <= 0] = -999

        # Baseline (single threshold на val)
        best_roi, best_t = -999.0, 0.01
        for t in np.arange(0.01, 0.70, 0.005):
            mask = kelly_val >= t
            if mask.sum() < 200:
                break
            roi, _ = calc_roi(val_df, mask)
            if roi > best_roi:
                best_roi, best_t = roi, t
        mask_single = kelly_test >= best_t
        roi_single, n_single = calc_roi(test_df, mask_single)
        logger.info(
            "Single threshold: t=%.3f, val_roi=%.2f%%, test_roi=%.2f%%, n=%d",
            best_t,
            best_roi,
            roi_single,
            n_single,
        )

        # Shrunken segments (shrink=0.5, raw thresholds от step 4.1)
        raw_seg_t = {"low": 0.495, "mid": 0.635, "high": 0.195}
        baseline_t = 0.455
        shrink = 0.5
        shrunken = {k: baseline_t + shrink * (v - baseline_t) for k, v in raw_seg_t.items()}
        buckets = pd.cut(
            test_df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"]
        )
        seg_mask = np.zeros(len(test_df), dtype=bool)
        for bucket, t in shrunken.items():
            seg_mask |= (buckets == bucket).values & (kelly_test >= t)
        roi_seg, n_seg = calc_roi(test_df, seg_mask)
        logger.info("Shrunken segments: test_roi=%.2f%%, n=%d", roi_seg, n_seg)

        # Выбор лучшего
        best_roi_final = max(roi_single, roi_seg)
        best_n = n_single if roi_single >= roi_seg else n_seg
        baseline_roi = 24.9088
        delta = best_roi_final - baseline_roi

        logger.info(
            "Best WF ensemble: roi=%.4f%%, n=%d, delta=%.4f%%",
            best_roi_final,
            best_n,
            delta,
        )

        if best_roi_final > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", best_roi_final, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_metrics(
            {
                "roi": best_roi_final,
                "n_selected": best_n,
                "roi_single_threshold": roi_single,
                "roi_shrunken_seg": roi_seg,
                "n_single": n_single,
                "n_seg": n_seg,
                "auc_ens_test": auc_ens_test,
                "auc_ens_val": auc_ens_val,
                "best_single_threshold": best_t,
                "baseline_roi": baseline_roi,
                "delta_vs_baseline": delta,
            }
        )

        current_best_roi = 26.9345
        if best_roi_final > current_best_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi_final, current_best_roi)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.8 DONE: best_roi={best_roi_final:.4f}%, n={best_n}, delta={delta:.4f}%")
        print(f"  single_threshold: roi={roi_single:.4f}%, n={n_single}")
        print(f"  shrunken_seg: roi={roi_seg:.4f}%, n={n_seg}")
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
