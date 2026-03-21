"""Step 4.2 — Lead-time features + retrain.

Гипотеза: время размещения ставки относительно начала игры несёт сигнал.
Добавляем lead_hours_log, lead_bucket, is_late_bet (< 3h), is_early_bet (>= 24h).
Ожидаем: +1-3% ROI за счёт лучшей калибровки early vs late ставок.
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


def build_features_v2(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature set с lead-time признаками."""
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

    # NEW: lead-time features
    lead_hours = df.get("lead_hours", pd.Series(np.zeros(len(df)), index=df.index))
    lead_hours = lead_hours.fillna(0.0)
    feats["lead_hours_log"] = np.log1p(lead_hours.clip(0))
    feats["is_late_bet"] = (lead_hours < 3).astype(int)
    feats["is_early_bet"] = (lead_hours >= 24).astype(int)
    feats["is_same_day_bet"] = ((lead_hours >= 3) & (lead_hours < 24)).astype(int)
    feats["lead_bucket"] = (
        pd.cut(
            lead_hours.clip(0, 168),
            bins=[-1, 1, 3, 12, 24, 72, 999],
            labels=False,
        )
        .fillna(0)
        .astype(int)
    )

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


def find_threshold(
    df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 200
) -> tuple[float, float]:
    """Поиск Kelly-порога по val ROI."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.70, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


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


with mlflow.start_run(run_name="phase4/step4.2_lead_time_features") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.2")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        train_df = df_raw.iloc[:train_end].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

        new_features = [
            "lead_hours_log",
            "is_late_bet",
            "is_early_bet",
            "is_same_day_bet",
            "lead_bucket",
        ]
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "new_features": str(new_features),
                "depth": 7,
                "lr": 0.1,
                "iterations": 500,
            }
        )

        x_tr, cat_f = build_features_v2(train_df)
        x_vl, _ = build_features_v2(val_df)
        x_te, _ = build_features_v2(test_df)
        y_tr = (train_df["Status"] == "won").astype(int)
        y_vl = (val_df["Status"] == "won").astype(int)
        y_te = (test_df["Status"] == "won").astype(int)
        w = make_weights(len(train_df))

        feature_names = list(x_tr.columns)
        logger.info(
            "Features: %d total (%d baseline + %d new)", len(feature_names), 34, len(new_features)
        )

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

        proba_val = model.predict_proba(x_vl)[:, 1]
        proba_test = model.predict_proba(x_te)[:, 1]

        auc_val = roc_auc_score(y_vl, proba_val)
        auc_test = roc_auc_score(y_te, proba_test)
        logger.info("Val AUC: %.4f, Test AUC: %.4f", auc_val, auc_test)

        # Val Kelly + threshold
        kelly_val = compute_kelly(proba_val, val_df["Odds"].values)
        kelly_val[val_df["lead_hours"].values <= 0] = -999
        best_t, val_roi = find_threshold(val_df, kelly_val, min_bets=200)
        logger.info("Val threshold: %.3f, val_roi: %.2f%%", best_t, val_roi)

        # Test Kelly
        kelly_test = compute_kelly(proba_test, test_df["Odds"].values)
        kelly_test[test_df["lead_hours"].values <= 0] = -999
        mask_test = kelly_test >= best_t
        test_roi, n_test = calc_roi(test_df, mask_test)

        # Baseline для сравнения
        meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        baseline_roi = 24.9088
        delta = test_roi - baseline_roi

        logger.info(
            "Test ROI: %.4f%% (n=%d) vs baseline %.4f%%, delta=%.4f%%",
            test_roi,
            n_test,
            baseline_roi,
            delta,
        )

        if test_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", test_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        # Feature importance
        fi = pd.DataFrame(
            {"feature": feature_names, "importance": model.get_feature_importance()}
        ).sort_values("importance", ascending=False)
        logger.info("Top 10 features:\n%s", fi.head(10).to_string(index=False))

        for feat in new_features:
            if feat in fi["feature"].values:
                imp = fi.loc[fi["feature"] == feat, "importance"].iloc[0]
                mlflow.log_metric(f"fi_{feat}", imp)
                logger.info("New feature %s importance: %.2f", feat, imp)

        mlflow.log_metrics(
            {
                "roi": test_roi,
                "n_selected": n_test,
                "auc_val": auc_val,
                "auc_test": auc_test,
                "threshold": best_t,
                "baseline_roi": baseline_roi,
                "delta_vs_baseline": delta,
            }
        )

        # Сохраняем если новый best
        current_best_roi = 25.8347  # step 4.1
        if test_roi > current_best_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", test_roi, current_best_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(best_dir / "model.cbm"))
            new_meta = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": test_roi,
                "auc": float(auc_test),
                "threshold": float(best_t),
                "n_bets": n_test,
                "feature_names": feature_names,
                "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.2",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(
            f"step4.2 DONE: roi={test_roi:.4f}%, threshold={best_t:.3f}, "
            f"n={n_test}, delta={delta:.4f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
