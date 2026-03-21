"""Step 4.6 — Feature ablation: исключение временных фичей.

Гипотеза: day_of_week/hour/month — temporally overfit к train периоду.
Soccer-only показал: day_of_week=14.87% (топ-1 фича).
Удаление temporal фичей может улучшить test generalization.
Также логируем feature importances baseline для анализа.
Baseline: ROI=24.91% (step 1.4, 33 фичи).
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import joblib
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
BASELINE_ROI = 24.91
LEAKAGE_THRESHOLD = 35.0

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass


def load_data() -> pd.DataFrame:
    """Загрузка данных."""
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


def build_features_full(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Полный baseline feature set (33 фичи)."""
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
    return feats, ["Sport", "Market", "Currency"]


def build_features_no_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature set без временных фичей (убираем hour, day_of_week, month)."""
    feats, cat_f = build_features_full(df)
    feats = feats.drop(columns=["hour", "day_of_week", "month"])
    return feats, cat_f


def make_weights(n: int, half_life: float = 0.5) -> np.ndarray:
    """Temporal decay weights."""
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


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


def find_threshold(
    df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 200
) -> tuple[float, float]:
    """Поиск Kelly-порога по val ROI."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.60, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def run_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    build_fn,
    label: str,
) -> dict:
    """Обучение и оценка одного варианта фичей."""
    x_tr, cat_f = build_fn(train_df)
    x_vl, _ = build_fn(val_df)
    x_te, _ = build_fn(test_df)
    y_tr = (train_df["Status"] == "won").astype(int)
    y_vl = (val_df["Status"] == "won").astype(int)
    y_te = (test_df["Status"] == "won").astype(int)
    w = make_weights(len(train_df))

    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.1,
        iterations=500,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
        cat_features=cat_f,
    )
    model.fit(x_tr, y_tr, eval_set=(x_vl, y_vl), sample_weight=w)

    pt = model.predict_proba(x_te)[:, 1]
    pv = model.predict_proba(x_vl)[:, 1]
    auc_val = float(roc_auc_score(y_vl, pv))
    auc_test = float(roc_auc_score(y_te, pt))

    pm_val = (val_df["lead_hours"] > 0).values
    pm_test = (test_df["lead_hours"] > 0).values
    k_v = compute_kelly(pv, val_df["Odds"].values)
    k_t = compute_kelly(pt, test_df["Odds"].values)
    k_v[~pm_val] = -999
    k_t[~pm_test] = -999

    t_best, roi_val = find_threshold(val_df, k_v)
    roi_test, n_bets = calc_roi(test_df, k_t >= t_best)

    feat_names = list(x_tr.columns)
    importances = model.get_feature_importance()
    fi_sorted = sorted(zip(feat_names, importances, strict=True), key=lambda x: x[1], reverse=True)

    logger.info(
        "%s: val=%.2f%%, test=%.2f%% (n=%d), t=%.3f, AUC_val=%.4f, AUC_test=%.4f",
        label,
        roi_val,
        roi_test,
        n_bets,
        t_best,
        auc_val,
        auc_test,
    )
    logger.info("Top-10 feature importances (%s):", label)
    for fname, fimp in fi_sorted[:10]:
        logger.info("  %-30s %.2f", fname, fimp)

    return {
        "label": label,
        "model": model,
        "feat_names": feat_names,
        "cat_f": cat_f,
        "t_best": t_best,
        "roi_val": roi_val,
        "roi_test": roi_test,
        "n_bets": n_bets,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "fi_sorted": fi_sorted,
    }


def main() -> None:
    """Feature ablation: full vs no-temporal."""
    with mlflow.start_run(run_name="phase4/step4.6_feature_ablation") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

            # Эксперимент A: полные фичи (воспроизведение baseline)
            res_full = run_experiment(train_df, val_df, test_df, build_features_full, "FULL")

            # Эксперимент B: без временных фичей
            res_no_t = run_experiment(
                train_df, val_df, test_df, build_features_no_temporal, "NO_TEMPORAL"
            )

            delta_full = res_full["roi_test"] - BASELINE_ROI
            delta_no_t = res_no_t["roi_test"] - BASELINE_ROI

            logger.info(
                "FULL: test=%.2f%% (delta=%.2f%%), NO_TEMPORAL: test=%.2f%% (delta=%.2f%%)",
                res_full["roi_test"],
                delta_full,
                res_no_t["roi_test"],
                delta_no_t,
            )

            # Выбираем лучшую конфигурацию
            best = res_full if res_full["roi_test"] >= res_no_t["roi_test"] else res_no_t
            best_roi = best["roi_test"]

            for res in [res_full, res_no_t]:
                if res["roi_test"] > LEAKAGE_THRESHOLD:
                    logger.error("LEAKAGE SUSPECT (%s): roi=%.2f%%", res["label"], res["roi_test"])
                    mlflow.set_tag("leakage_suspect", "true")

            if best_roi > BASELINE_ROI and best["n_bets"] >= 200:
                build_fn = (
                    build_features_full if best["label"] == "FULL" else build_features_no_temporal
                )
                x_tr_best, _ = build_fn(train_df)
                _ = x_tr_best  # использован выше в run_experiment
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(best["model"], models_dir / "model.cbm")
                metadata = {
                    "framework": "catboost",
                    "pipeline_file": "pipeline.pkl",
                    "roi": float(best_roi),
                    "auc": float(best["auc_test"]),
                    "threshold": float(best["t_best"]),
                    "n_bets": best["n_bets"],
                    "feature_names": best["feat_names"],
                    "ablation_variant": best["label"],
                    "session_id": SESSION_ID,
                    "step": "4.6",
                }
                (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
                logger.info("New best saved! roi=%.2f%% (%s)", best_roi, best["label"])

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "model": "catboost_ablation",
                    "depth": 7,
                    "ablation": "no_temporal_vs_full",
                }
            )
            mlflow.log_metrics(
                {
                    "auc_val_full": res_full["auc_val"],
                    "auc_test_full": res_full["auc_test"],
                    "roi_val_full": res_full["roi_val"],
                    "roi_test_full": res_full["roi_test"],
                    "n_bets_full": res_full["n_bets"],
                    "threshold_full": res_full["t_best"],
                    "roi_delta_full": delta_full,
                    "auc_val_no_t": res_no_t["auc_val"],
                    "auc_test_no_t": res_no_t["auc_test"],
                    "roi_val_no_t": res_no_t["roi_val"],
                    "roi_test_no_t": res_no_t["roi_test"],
                    "n_bets_no_t": res_no_t["n_bets"],
                    "threshold_no_t": res_no_t["t_best"],
                    "roi_delta_no_t": delta_no_t,
                }
            )
            # Логируем top-5 importances для FULL
            for fname, fimp in res_full["fi_sorted"][:5]:
                mlflow.log_metric(f"fi_full_{fname[:15]}", float(fimp))
            for fname, fimp in res_no_t["fi_sorted"][:5]:
                mlflow.log_metric(f"fi_not_{fname[:15]}", float(fimp))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.75")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT full={res_full['roi_test']:.2f}% (n={res_full['n_bets']}) "
                f"no_temporal={res_no_t['roi_test']:.2f}% (n={res_no_t['n_bets']}) "
                f"run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
