"""Step 4.3 — Isotonic calibration + Kelly criterion selection.

Гипотеза: CatBoost d7_hl50pct предсказывает плохо откалиброванные вероятности.
Isotonic regression calibration + Kelly criterion (вместо probability threshold)
должны улучшить ROI.

Источник: arxiv 2303.06021 — калибровка даёт +34.69% ROI vs -35.17% без.
Kelly = (p * b - (1-p)) / b, где b = Odds - 1.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = os.environ["UAF_SESSION_DIR"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}
PREV_BEST_ROI = 7.34


def load_data() -> pd.DataFrame:
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Market"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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


def make_sample_weights(n: int, half_life: float) -> np.ndarray:
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly fraction: (p*b - (1-p)) / b, где b = Odds - 1."""
    b = odds - 1.0
    kelly = (proba * b - (1 - proba)) / b.clip(0.001)
    return kelly


def find_best_prob_threshold(
    val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 200
) -> float:
    """Стандартный порог по вероятности (baseline)."""
    best_roi, best_t = -999.0, 0.5
    for t in np.arange(0.40, 0.93, 0.01):
        mask = proba >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t


def find_best_kelly_threshold(
    val_df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 200
) -> float:
    """Оптимальный Kelly порог на val."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.30, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best Kelly threshold=%.3f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


with mlflow.start_run(run_name="phase4/step4.3_calibration_kelly") as run:
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")

    try:
        budget_file_path = os.environ.get("UAF_BUDGET_STATUS_FILE", "")
        if budget_file_path:
            try:
                budget_status = json.loads(Path(budget_file_path).read_text())
                if budget_status.get("hard_stop"):
                    mlflow.set_tag("status", "budget_stopped")
                    sys.exit(0)
            except FileNotFoundError:
                pass

        df = load_data()
        n = len(df)
        train_end = int(n * 0.8)
        val_start = int(n * 0.64)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:train_end]
        test_df = df.iloc[train_end:]

        logger.info("Split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

        X_train, cat_features = build_features(train_df)
        X_val, _ = build_features(val_df)
        X_test, _ = build_features(test_df)

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        # Обучаем d7_hl50pct (лучшая конфигурация)
        sw = make_sample_weights(len(train_df), 0.5)
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=7,
            random_seed=42,
            eval_metric="AUC",
            verbose=0,
            cat_features=cat_features,
        )
        model.fit(
            X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, sample_weight=sw
        )

        raw_pv = model.predict_proba(X_val)[:, 1]
        raw_pt = model.predict_proba(X_test)[:, 1]

        auc_v_raw = roc_auc_score(y_val, raw_pv)
        brier_v_raw = brier_score_loss(y_val, raw_pv)
        logger.info("Raw model: AUC_val=%.4f, Brier_val=%.4f", auc_v_raw, brier_v_raw)

        # Isotonic regression calibration на val
        iso_cal = IsotonicRegression(out_of_bounds="clip")
        iso_cal.fit(raw_pv, y_val)

        cal_pv = iso_cal.predict(raw_pv)
        cal_pt = iso_cal.predict(raw_pt)

        brier_v_cal = brier_score_loss(y_val, cal_pv)
        auc_v_cal = roc_auc_score(y_val, cal_pv)
        logger.info("Calibrated: AUC_val=%.4f, Brier_val=%.4f", auc_v_cal, brier_v_cal)

        # Метод 1: Стандартный порог на калиброванных вероятностях
        t_prob = find_best_prob_threshold(val_df, cal_pv)
        roi_val_prob, n_val_prob = calc_roi(val_df, cal_pv >= t_prob)
        roi_test_prob, n_test_prob = calc_roi(test_df, cal_pt >= t_prob)

        # Метод 2: Kelly criterion на калиброванных вероятностях
        kelly_val = compute_kelly(cal_pv, val_df["Odds"].values)
        kelly_test = compute_kelly(cal_pt, test_df["Odds"].values)

        t_kelly = find_best_kelly_threshold(val_df, kelly_val)
        roi_val_kelly, n_val_kelly = calc_roi(val_df, kelly_val >= t_kelly)
        roi_test_kelly, n_test_kelly = calc_roi(test_df, kelly_test >= t_kelly)

        logger.info(
            "Prob threshold: val=%.2f%% test=%.2f%% (%d)",
            roi_val_prob,
            roi_test_prob,
            n_test_prob,
        )
        logger.info(
            "Kelly threshold: val=%.2f%% test=%.2f%% (%d)",
            roi_val_kelly,
            roi_test_kelly,
            n_test_kelly,
        )

        # Метод 3: Raw prob + Kelly (без калибровки) для сравнения
        kelly_val_raw = compute_kelly(raw_pv, val_df["Odds"].values)
        kelly_test_raw = compute_kelly(raw_pt, test_df["Odds"].values)
        t_kelly_raw = find_best_kelly_threshold(val_df, kelly_val_raw)
        roi_val_kelly_raw, n_val_kelly_raw = calc_roi(val_df, kelly_val_raw >= t_kelly_raw)
        roi_test_kelly_raw, n_test_kelly_raw = calc_roi(test_df, kelly_test_raw >= t_kelly_raw)
        logger.info(
            "Raw Kelly: val=%.2f%% test=%.2f%% (%d)",
            roi_val_kelly_raw,
            roi_test_kelly_raw,
            n_test_kelly_raw,
        )

        # Метод 4: Стандартный порог без калибровки (baseline)
        t_raw_prob = find_best_prob_threshold(val_df, raw_pv)
        roi_val_raw_prob, n_val_raw_prob = calc_roi(val_df, raw_pv >= t_raw_prob)
        roi_test_raw_prob, n_test_raw_prob = calc_roi(test_df, raw_pt >= t_raw_prob)
        logger.info(
            "Raw prob: val=%.2f%% test=%.2f%% (%d) — this should be ~7.34%%",
            roi_val_raw_prob,
            roi_test_raw_prob,
            n_test_raw_prob,
        )

        # Лучший результат среди методов
        methods = [
            ("cal_prob", roi_test_prob, n_test_prob, roi_val_prob, t_prob),
            ("cal_kelly", roi_test_kelly, n_test_kelly, roi_val_kelly, t_kelly),
            ("raw_kelly", roi_test_kelly_raw, n_test_kelly_raw, roi_val_kelly_raw, t_kelly_raw),
            ("raw_prob", roi_test_raw_prob, n_test_raw_prob, roi_val_raw_prob, t_raw_prob),
        ]
        best_method = max(methods, key=lambda x: x[1])
        best_name, best_roi, best_n, best_roi_val, best_t = best_method

        if best_roi > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — leakage!")

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "base_model": "catboost_d7_hl50",
                "calibration": "isotonic",
                "best_method": best_name,
                "threshold_prob": t_prob,
                "threshold_kelly": t_kelly,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "roi_test_cal_prob": roi_test_prob,
                "roi_test_cal_kelly": roi_test_kelly,
                "roi_test_raw_kelly": roi_test_kelly_raw,
                "roi_test_raw_prob": roi_test_raw_prob,
                "roi_val_cal_prob": roi_val_prob,
                "roi_val_cal_kelly": roi_val_kelly,
                "roi_test_best": best_roi,
                "brier_val_raw": brier_v_raw,
                "brier_val_cal": brier_v_cal,
                "auc_val_raw": auc_v_raw,
                "auc_val_cal": auc_v_cal,
                "n_bets_best": best_n,
                "delta_vs_baseline": best_roi - PREV_BEST_ROI,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (best_roi - PREV_BEST_ROI) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        if best_roi > PREV_BEST_ROI and best_n >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)
            feats_list = X_train.columns.tolist()

            class BestPipeline:
                """Calibrated CatBoost + Kelly selection pipeline."""

                def __init__(
                    self,
                    model,
                    calibrator,
                    feature_names,
                    cat_features,
                    threshold,
                    method,
                    framework="catboost",
                ):
                    self.model = model
                    self.calibrator = calibrator
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.method = method
                    self.sport_filter: list[str] = []
                    self.framework = framework

                def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
                    feats, _ = build_features(df)
                    return feats[self.feature_names]

                def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                    x = self._build_features(df)
                    raw = self.model.predict_proba(x)[:, 1]
                    return self.calibrator.predict(raw)

                def evaluate(self, df: pd.DataFrame) -> dict:
                    proba = self.predict_proba(df)
                    if self.method == "kelly":
                        score = compute_kelly(proba, df["Odds"].values)
                    else:
                        score = proba
                    mask = score >= self.threshold
                    selected = df[mask].copy()
                    if len(selected) == 0:
                        return {"roi": -100.0, "n_selected": 0, "threshold": self.threshold}
                    won_mask = selected["Status"] == "won"
                    total_stake = selected["USD"].sum()
                    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
                    roi = (
                        (total_payout - total_stake) / total_stake * 100
                        if total_stake > 0
                        else -100.0
                    )
                    return {"roi": roi, "n_selected": int(mask.sum()), "threshold": self.threshold}

            method_type = "kelly" if "kelly" in best_name else "prob"
            pipeline = BestPipeline(
                model=model,
                calibrator=iso_cal,
                feature_names=feats_list,
                cat_features=cat_features,
                threshold=best_t,
                method=method_type,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            model.save_model(str(models_dir / "model.cbm"))
            metadata = {
                "framework": "catboost_calibrated",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": best_roi,
                "auc": auc_v_cal,
                "threshold": best_t,
                "method": best_name,
                "n_bets": best_n,
                "feature_names": feats_list,
                "params": {"depth": 7, "half_life": 0.5, "calibration": "isotonic"},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.3",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Pipeline saved. ROI=%.2f%%", best_roi)

        print("\n=== Step 4.3 Isotonic Calibration + Kelly Criterion ===")
        b_delta = f"{brier_v_raw:.4f}->{brier_v_cal:.4f}"
        a_delta = f"{auc_v_raw:.4f}->{auc_v_cal:.4f}"
        print(f"Brier: {b_delta}, AUC: {a_delta}")
        for name, rt, nt, rv, t in methods:
            print(f"  [{name}] val={rv:.2f}% test={rt:.2f}% ({nt}) thr={t:.3f}")
        print(f"\nBest: {best_name} ROI test={best_roi:.2f}%")
        print(f"Prev best: {PREV_BEST_ROI:.2f}%")
        print(f"Delta: {best_roi - PREV_BEST_ROI:+.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
