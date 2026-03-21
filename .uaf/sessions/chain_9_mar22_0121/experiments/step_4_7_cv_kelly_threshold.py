"""Step 4.7 — CV-based Kelly threshold через time-series cross-validation.

Проблема step 4.6: p75 тренировочного Kelly использует in-sample значения
(модель обучена на тех же точках → Kelly inflated → порог завышен).

Решение: 5-fold time-series CV на train → OOF Kelly предсказания →
более реалистичная оценка Kelly распределения → лучший порог.

Метод:
1. 5-fold expanding window CV на 80% train (chain_8 feature set)
2. Сохранить OOF Kelly для всех train Soccer 1x2 low-odds бетов
3. Sweep [p60..p85] по OOF Kelly percentile → выбрать threshold через
   CV ROI (n>=30 per fold criterion)
4. Применить к test ОДИН РАЗ

Baseline: ROI=28.5833% (n=233)
Step 4.6 result: ROI=30.9146% (n=196), threshold=0.5220 (in-sample p75)
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

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_8_mar22_0035/models/best")

SEGMENT_THRESHOLDS_BASELINE = {"low": 0.475, "mid": 0.545, "high": 0.325}

FEATURE_NAMES = [
    "Odds",
    "USD",
    "log_odds",
    "log_usd",
    "implied_prob",
    "is_parlay",
    "outcomes_count",
    "ml_p_model",
    "ml_p_implied",
    "ml_edge",
    "ml_ev",
    "ml_team_stats_found",
    "ml_winrate_diff",
    "ml_rating_diff",
    "hour",
    "day_of_week",
    "month",
    "odds_times_stake",
    "ml_edge_pos",
    "ml_ev_pos",
    "elo_max",
    "elo_min",
    "elo_diff",
    "elo_ratio",
    "elo_mean",
    "elo_std",
    "k_factor_mean",
    "has_elo",
    "elo_count",
    "ml_edge_x_elo_diff",
    "elo_implied_agree",
    "Sport",
    "Market",
    "Currency",
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

N_FOLDS = 5
CAT_FEATURES = ["Sport", "Market", "Currency"]


def check_budget() -> bool:
    try:
        status = json.loads(BUDGET_FILE.read_text())
        return bool(status.get("hard_stop"))
    except FileNotFoundError:
        return False


def load_raw_data() -> pd.DataFrame:
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return feats


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    selected = df[mask]
    if len(selected) == 0:
        return -100.0, 0
    won = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def apply_seg_thresholds(
    df: pd.DataFrame, kelly: np.ndarray, seg_thresholds: dict[str, float]
) -> np.ndarray:
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in seg_thresholds.items():
        mask |= (buckets == bucket).values & (kelly >= t)
    return mask


def run_cv_oof(train_df: pd.DataFrame) -> np.ndarray:
    """5-fold expanding window CV → OOF Kelly values для всего train."""
    n = len(train_df)
    oof_kelly = np.full(n, np.nan)
    x_all = build_features(train_df)[FEATURE_NAMES]
    y_all = (train_df["Status"] == "won").astype(int)

    # Expanding window: fold k trains on [0, split_k), validates on [split_k, split_{k+1})
    fold_size = n // (N_FOLDS + 1)
    min_train = fold_size * 1  # минимум 1 fold для обучения

    fold_aucs = []
    for fold_idx in range(N_FOLDS):
        val_start = min_train + fold_idx * fold_size
        val_end = val_start + fold_size
        if val_end > n:
            val_end = n

        if val_start >= n or val_start >= val_end:
            break

        x_tr = x_all.iloc[:val_start]
        y_tr = y_all.iloc[:val_start]
        x_va = x_all.iloc[val_start:val_end]
        y_va = y_all.iloc[val_start:val_end]
        df_va = train_df.iloc[val_start:val_end]

        model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            cat_features=CAT_FEATURES,
            random_seed=42,
            verbose=0,
        )
        model.fit(x_tr, y_tr, cat_features=CAT_FEATURES, verbose=0)

        proba_va = model.predict_proba(x_va)[:, 1]
        auc_va = roc_auc_score(y_va, proba_va)
        fold_aucs.append(auc_va)

        kelly_va = compute_kelly(proba_va, df_va["Odds"].values)
        lead_hours_va = (df_va["Start_Time"] - df_va["Created_At"]).dt.total_seconds() / 3600.0
        kelly_va[lead_hours_va.values <= 0] = -999

        oof_kelly[val_start:val_end] = kelly_va
        logger.info(
            "Fold %d: n_train=%d, n_val=%d, AUC=%.4f", fold_idx, len(x_tr), len(x_va), auc_va
        )

    logger.info("Mean CV AUC: %.4f", float(np.mean(fold_aucs)))
    return oof_kelly, fold_aucs


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    logger.info("Train: %d, Test: %d", len(train_df), len(test_df))

    # === OOF Kelly через CV ===
    logger.info("Запуск 5-fold time-series CV для OOF Kelly...")
    oof_kelly, fold_aucs = run_cv_oof(train_df)

    # Анализ OOF Kelly распределения для LOW Soccer 1x2
    lead_hours_train = (
        train_df["Start_Time"] - train_df["Created_At"]
    ).dt.total_seconds() / 3600.0
    train_low_mask = (
        (train_df["Market"].values == "1x2")
        & (train_df["Odds"].values < 1.8)
        & (lead_hours_train.values > 0)
        & ~np.isnan(oof_kelly)
    )
    kelly_oof_low = oof_kelly[train_low_mask]
    logger.info(
        "OOF LOW Soccer 1x2 pre-match: n=%d (из %d), kelly mean=%.4f std=%.4f",
        len(kelly_oof_low),
        int(train_low_mask.sum()),
        float(kelly_oof_low.mean()),
        float(kelly_oof_low.std()),
    )

    # Percentiles OOF Kelly
    percentiles = [60, 65, 70, 75, 80, 85]
    pct_oof = {p: float(np.percentile(kelly_oof_low, p)) for p in percentiles}
    for p, v in pct_oof.items():
        logger.info("  OOF p%d = %.4f", p, v)

    # === CV ROI для threshold selection ===
    # Используем тот же train_df и oof_kelly для sweep
    # Считаем "CV ROI" через oof_kelly:
    # для каждого порога: mask = train_low_mask & kelly >= t → ROI на train fold data
    cv_roi_results = []
    for p in percentiles:
        t = pct_oof[p]
        fold_mask = train_low_mask & (oof_kelly >= t)
        roi_cv, n_cv = calc_roi(train_df, fold_mask)
        cv_roi_results.append({"pct": p, "t": t, "roi": roi_cv, "n": n_cv})
        logger.info("  OOF p%d t=%.4f: ROI=%.4f%% n=%d (train OOF)", p, t, roi_cv, n_cv)

    # Выбрать threshold: максимум CV ROI при n >= 30
    valid = [r for r in cv_roi_results if r["n"] >= 30]
    best_cv = max(valid, key=lambda r: r["roi"]) if valid else cv_roi_results[0]
    chosen_pct = best_cv["pct"]
    chosen_threshold = best_cv["t"]
    logger.info(
        "Chosen by CV: p%d = %.4f (CV ROI=%.4f%% n=%d)",
        chosen_pct,
        chosen_threshold,
        best_cv["roi"],
        best_cv["n"],
    )

    # === Test evaluation ===
    # Переобучить финальную модель на полном train
    x_train = build_features(train_df)[FEATURE_NAMES]
    x_test = build_features(test_df)[FEATURE_NAMES]
    y_train = (train_df["Status"] == "won").astype(int)
    y_test = (test_df["Status"] == "won").astype(int)

    final_model = CatBoostClassifier(
        depth=7,
        learning_rate=0.1,
        iterations=500,
        cat_features=CAT_FEATURES,
        random_seed=42,
        verbose=0,
    )
    final_model.fit(x_train, y_train, cat_features=CAT_FEATURES, verbose=0)

    auc_test = roc_auc_score(y_test, final_model.predict_proba(x_test)[:, 1])
    logger.info("Final model test AUC: %.4f", auc_test)

    proba_test = final_model.predict_proba(x_test)[:, 1]
    kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
    lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly_test[lead_hours_test.values <= 0] = -999

    mkt_mask = test_df["Market"].values == "1x2"
    new_thresholds = {
        "low": chosen_threshold,
        "mid": SEGMENT_THRESHOLDS_BASELINE["mid"],
        "high": SEGMENT_THRESHOLDS_BASELINE["high"],
    }
    seg_mask_new = apply_seg_thresholds(test_df, kelly_test, new_thresholds)
    roi, n_bets = calc_roi(test_df, mkt_mask & seg_mask_new)

    # Baseline для сравнения
    seg_mask_base = apply_seg_thresholds(test_df, kelly_test, SEGMENT_THRESHOLDS_BASELINE)
    roi_base, n_base = calc_roi(test_df, mkt_mask & seg_mask_base)
    logger.info("Baseline (retrained): ROI=%.4f%% n=%d", roi_base, n_base)

    baseline_roi = 28.5833
    delta = roi - baseline_roi

    with mlflow.start_run(run_name="phase4/step_4_7_cv_kelly_threshold") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.7")
        mlflow.log_params(
            {
                "validation_scheme": "time_series_cv",
                "n_folds": N_FOLDS,
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
                "n_oof_low": len(kelly_oof_low),
                "chosen_percentile": chosen_pct,
                "chosen_threshold": round(chosen_threshold, 6),
                "threshold_selection": "max_cv_roi_n30",
                "depth": 7,
                "learning_rate": 0.1,
                "iterations": 500,
                "seg_low_baseline": SEGMENT_THRESHOLDS_BASELINE["low"],
                "seg_mid": SEGMENT_THRESHOLDS_BASELINE["mid"],
                "seg_high": SEGMENT_THRESHOLDS_BASELINE["high"],
            }
        )

        try:
            mlflow.log_metric("auc", auc_test)
            mlflow.log_metric("mean_cv_auc", float(np.mean(fold_aucs)))
            for i, auc_f in enumerate(fold_aucs):
                mlflow.log_metric(f"auc_fold_{i}", auc_f)

            for p, v in pct_oof.items():
                mlflow.log_metric(f"oof_p{p}", v)
            for r in cv_roi_results:
                mlflow.log_metric(f"cv_roi_p{r['pct']}", r["roi"])
                mlflow.log_metric(f"cv_n_p{r['pct']}", r["n"])

            mlflow.log_metrics({"roi": roi, "n_selected": n_bets, "roi_delta": delta})
            mlflow.log_metric("roi_baseline_retrained", roi_base)
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            if roi > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            logger.info(
                "RESULT: p%d threshold=%.4f → ROI=%.4f%% n=%d delta=%.4f (AUC=%.4f)",
                chosen_pct,
                chosen_threshold,
                roi,
                n_bets,
                delta,
                auc_test,
            )
            print(f"STEP_4_7_ROI={roi:.6f}")
            print(f"STEP_4_7_N={n_bets}")
            print(f"STEP_4_7_DELTA={delta:.4f}")
            print(f"STEP_4_7_THRESHOLD={chosen_threshold:.6f}")
            print(f"STEP_4_7_PERCENTILE={chosen_pct}")
            print(f"STEP_4_7_AUC={auc_test:.4f}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")
            print("\nCV sweep:")
            for r in cv_roi_results:
                print(f"  p{r['pct']} t={r['t']:.4f}: CV_ROI={r['roi']:.2f}% n={r['n']}")
            print(f"\nOOF percentiles: {pct_oof}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.7")
            raise


if __name__ == "__main__":
    main()
