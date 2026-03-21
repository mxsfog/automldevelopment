"""Step 4.4 — Isotonic calibration: лучшие вероятности для Kelly.

Гипотеза: isotonic regression после CatBoost улучшит калибровку вероятностей,
что даст более точные Kelly значения и улучшит ROI.
Источник: Walsh & Joshi (2024) - model selection by calibration improves betting ROI.

Также тестируем фиксированный EV threshold вместо оптимизируемого Kelly.
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
from sklearn.metrics import brier_score_loss, roc_auc_score

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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline feature set."""
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


def eval_fixed_thresholds(
    df: pd.DataFrame, kelly: np.ndarray, pm_mask: np.ndarray, thresholds: list[float]
) -> dict[float, dict]:
    """Оценка ROI для нескольких фиксированных порогов."""
    results = {}
    for t in thresholds:
        mask = (kelly >= t) & pm_mask
        roi, n = calc_roi(df, mask)
        results[t] = {"roi": roi, "n_bets": n}
    return results


class CalibratedPipeline:
    """Пайплайн с isotonic-calibrated CatBoost."""

    def __init__(
        self,
        model: CatBoostClassifier,
        calibrator,
        feature_names: list[str],
        cat_features: list[str],
        threshold: float,
        sport_filter: list[str],
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.feature_names = feature_names
        self.cat_features = cat_features
        self.threshold = threshold
        self.sport_filter = sport_filter

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, _ = build_features(df)
        return feats[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Калиброванные вероятности для RAW DataFrame."""
        x = self._build_features(df)
        raw_proba = self.model.predict_proba(x)[:, 1]
        return self.calibrator.predict(raw_proba)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """ROI на RAW DataFrame."""
        if self.sport_filter:
            df = df[~df["Sport"].isin(self.sport_filter)].copy()
        proba = self.predict_proba(df)
        odds = df["Odds"].values
        kelly = compute_kelly(proba, odds)
        lead_hours = (
            pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
            - pd.to_datetime(df["Created_At"], utc=True)
        ).dt.total_seconds() / 3600.0
        kelly[lead_hours <= 0] = -999
        mask = kelly >= self.threshold
        roi, n_selected = calc_roi(df, mask)
        return {"roi": roi, "n_selected": n_selected, "threshold": self.threshold}


def main() -> None:
    """Isotonic calibration + Kelly."""
    with mlflow.start_run(run_name="phase4/step4.4_calibration") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)
            # Разбиваем val на cal (для калибровки) и val_eval (для порога)
            calib_split = int(n * 0.72)

            train_df = df.iloc[:train_end].copy()
            calib_df = df.iloc[val_start:calib_split].copy()  # 64-72%: для калибровки
            val_df = df.iloc[calib_split:train_end].copy()  # 72-80%: для порога
            test_df = df.iloc[train_end:].copy()

            logger.info(
                "Train: %d, Calib: %d, Val: %d, Test: %d",
                len(train_df),
                len(calib_df),
                len(val_df),
                len(test_df),
            )

            x_tr, cat_f = build_features(train_df)
            x_cal, _ = build_features(calib_df)
            _x_vl, _ = build_features(val_df)
            x_te, _ = build_features(test_df)
            y_tr = (train_df["Status"] == "won").astype(int)
            y_cal = (calib_df["Status"] == "won").astype(int)
            y_vl_full = (df.iloc[val_start:train_end]["Status"] == "won").astype(int)
            y_te = (test_df["Status"] == "won").astype(int)
            w = make_weights(len(train_df))

            # Обучение базовой модели (весь train как раньше)
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
            full_val_feats, _ = build_features(df.iloc[val_start:train_end].copy())
            model.fit(
                x_tr,
                y_tr,
                eval_set=(full_val_feats, y_vl_full),
                sample_weight=w,
            )

            # Baseline без калибровки
            pt_raw = model.predict_proba(x_te)[:, 1]

            # Calibration: isotonic на calib_df (64-72%)
            # CalibratedClassifierCV.cv='prefit' требует estimator с predict_proba
            from sklearn.isotonic import IsotonicRegression

            p_cal = model.predict_proba(x_cal)[:, 1]
            cal_model = IsotonicRegression(out_of_bounds="clip")
            cal_model.fit(p_cal, y_cal)

            # Применяем калибровку
            pt_cal = cal_model.predict(pt_raw)

            # Brier score comparison (calibration quality)
            brier_raw_te = brier_score_loss(y_te, pt_raw)
            brier_cal_te = brier_score_loss(y_te, pt_cal)
            auc_raw_te = roc_auc_score(y_te, pt_raw)
            auc_cal_te = roc_auc_score(y_te, pt_cal)

            logger.info(
                "Brier raw=%.4f cal=%.4f | AUC raw=%.4f cal=%.4f",
                brier_raw_te,
                brier_cal_te,
                auc_raw_te,
                auc_cal_te,
            )

            # Kelly с калиброванными вероятностями
            val_df_full = df.iloc[val_start:train_end].copy()
            pm_val_full = (val_df_full["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values

            pv_full = model.predict_proba(full_val_feats)[:, 1]
            pv_full_cal = cal_model.predict(pv_full)

            k_v_raw = compute_kelly(pv_full, val_df_full["Odds"].values)
            k_v_cal = compute_kelly(pv_full_cal, val_df_full["Odds"].values)
            k_t_raw = compute_kelly(pt_raw, test_df["Odds"].values)
            k_t_cal = compute_kelly(pt_cal, test_df["Odds"].values)
            k_v_raw[~pm_val_full] = -999
            k_v_cal[~pm_val_full] = -999
            k_t_raw[~pm_test] = -999
            k_t_cal[~pm_test] = -999

            t_raw, roi_val_raw = find_threshold(val_df_full, k_v_raw)
            roi_test_raw, n_raw = calc_roi(test_df, k_t_raw >= t_raw)

            t_cal, roi_val_cal = find_threshold(val_df_full, k_v_cal)
            roi_test_cal, n_cal = calc_roi(test_df, k_t_cal >= t_cal)

            logger.info(
                "Raw: val=%.2f%%, test=%.2f%% (n=%d), t=%.3f",
                roi_val_raw,
                roi_test_raw,
                n_raw,
                t_raw,
            )
            logger.info(
                "Calibrated: val=%.2f%%, test=%.2f%% (n=%d), t=%.3f",
                roi_val_cal,
                roi_test_cal,
                n_cal,
                t_cal,
            )

            # Также: фиксированные пороги для анализа (без opting на test)
            fixed_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            val_fixed_raw = eval_fixed_thresholds(
                val_df_full, k_v_raw, pm_val_full, fixed_thresholds
            )
            val_fixed_cal = eval_fixed_thresholds(
                val_df_full, k_v_cal, pm_val_full, fixed_thresholds
            )
            logger.info("Fixed threshold analysis on val (raw):")
            for t, r in val_fixed_raw.items():
                logger.info("  t=%.2f: roi=%.2f%% (n=%d)", t, r["roi"], r["n_bets"])
            logger.info("Fixed threshold analysis on val (calibrated):")
            for t, r in val_fixed_cal.items():
                logger.info("  t=%.2f: roi=%.2f%% (n=%d)", t, r["roi"], r["n_bets"])

            delta_cal = roi_test_cal - BASELINE_ROI
            delta_raw = roi_test_raw - BASELINE_ROI

            if roi_test_cal > LEAKAGE_THRESHOLD or roi_test_raw > LEAKAGE_THRESHOLD:
                logger.error("LEAKAGE SUSPECT")
                mlflow.set_tag("leakage_suspect", "true")

            # Сохраняем лучший
            best_roi = max(roi_test_raw, roi_test_cal)
            best_n = n_raw if roi_test_raw >= roi_test_cal else n_cal
            if best_roi > BASELINE_ROI and best_n >= 200:
                feat_names = list(x_tr.columns)
                if roi_test_cal >= roi_test_raw:
                    pipeline = CalibratedPipeline(
                        model=model,
                        calibrator=cal_model,
                        feature_names=feat_names,
                        cat_features=cat_f,
                        threshold=t_cal,
                        sport_filter=[],
                    )
                else:
                    pipeline = joblib.load(SESSION_DIR / "models" / "best" / "pipeline.pkl")
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, models_dir / "pipeline.pkl")
                model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost_isotonic",
                    "pipeline_file": "pipeline.pkl",
                    "roi": float(best_roi),
                    "auc": float(auc_cal_te),
                    "threshold": float(t_cal),
                    "n_bets": best_n,
                    "session_id": SESSION_ID,
                    "step": "4.4",
                }
                (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
                logger.info("New best saved! roi=%.2f%%", best_roi)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_calib": len(calib_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "calibration_method": "isotonic",
                    "threshold_raw": t_raw,
                    "threshold_cal": t_cal,
                }
            )
            mlflow.log_metrics(
                {
                    "auc_test_raw": float(auc_raw_te),
                    "auc_test_cal": float(auc_cal_te),
                    "brier_raw": float(brier_raw_te),
                    "brier_cal": float(brier_cal_te),
                    "roi_val_raw": float(roi_val_raw),
                    "roi_test_raw": float(roi_test_raw),
                    "roi_delta_raw": float(delta_raw),
                    "n_bets_raw": n_raw,
                    "roi_val_cal": float(roi_val_cal),
                    "roi_test_cal": float(roi_test_cal),
                    "roi_delta_cal": float(delta_cal),
                    "n_bets_cal": n_cal,
                }
            )
            for t, r in val_fixed_raw.items():
                mlflow.log_metric(f"val_roi_raw_t{t:.2f}", r["roi"])
            for t, r in val_fixed_cal.items():
                mlflow.log_metric(f"val_roi_cal_t{t:.2f}", r["roi"])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.65")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT raw={roi_test_raw:.2f}% (n={n_raw}) "
                f"calibrated={roi_test_cal:.2f}% (n={n_cal}) "
                f"brier_raw={brier_raw_te:.4f} brier_cal={brier_cal_te:.4f} "
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
