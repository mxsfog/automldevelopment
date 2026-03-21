"""Step 4.1 — Stacking CatBoost + LightGBM + isotonic calibration + PM Kelly.

Гипотеза:
- CatBoost (depth=7, hl=0.5) и LightGBM обучаются независимо как уровень 0
- Isotonic calibration на val-set для улучшения вероятностей
- Средневзвешенная комбинация + Kelly на pre-match ставках
- Ожидаемый прирост: +3-5% над 24.91%

Логика:
1. L0: CB + LGBM обучаются на train (0-80%)
2. Калибровка: isotonic fitted на val (64-80%)
3. Финальные prob = (w_cb * prob_cb_cal + w_lgbm * prob_lgbm_cal) / (w_cb + w_lgbm)
4. Kelly + PM filter → threshold из val → ROI на test
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
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}
PREV_BEST_ROI = 24.91


def load_data() -> pd.DataFrame:
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Market", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

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
    """Feature engineering — идентична step_4_5."""
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


def compute_kelly(proba: np.ndarray, odds: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    b = odds - 1.0
    return fraction * (proba * b - (1 - proba)) / b.clip(0.001)


def find_best_pm_threshold(
    val_df: pd.DataFrame, kelly_pm: np.ndarray, min_bets: int = 200
) -> tuple[float, float]:
    """Лучший порог Kelly на pre-match val ставках."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.60, 0.005):
        mask = kelly_pm >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def calibrate_proba(
    proba_train: np.ndarray, y_train: np.ndarray, proba_apply: np.ndarray
) -> np.ndarray:
    """Isotonic regression calibration."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba_train, y_train)
    return ir.transform(proba_apply)


with mlflow.start_run(run_name="phase4/step4.1_stacking_cb_lgbm") as run:
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

        logger.info("Загружаем данные...")
        df = load_data()
        n = len(df)
        train_end = int(n * 0.8)
        val_start = int(n * 0.64)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:train_end]
        test_df = df.iloc[train_end:]

        X_train, cat_features = build_features(train_df)
        X_val, _ = build_features(val_df)
        X_test, _ = build_features(test_df)

        y_train = (train_df["Status"] == "won").astype(int).values
        y_val = (val_df["Status"] == "won").astype(int).values
        y_test = (test_df["Status"] == "won").astype(int).values

        sw = make_sample_weights(len(train_df), 0.5)

        # === Уровень 0: CatBoost ===
        logger.info("Обучаем CatBoost (depth=7, hl=0.5)...")
        cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=7,
            random_seed=42,
            eval_metric="AUC",
            verbose=0,
            cat_features=cat_features,
        )
        cb.fit(
            X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, sample_weight=sw
        )

        prob_cb_val = cb.predict_proba(X_val)[:, 1]
        prob_cb_test = cb.predict_proba(X_test)[:, 1]
        auc_cb = roc_auc_score(y_test, prob_cb_test)
        logger.info("CatBoost AUC test: %.4f", auc_cb)

        # === Уровень 0: LightGBM ===
        logger.info("Обучаем LightGBM...")
        # LightGBM не поддерживает cat_features напрямую — кодируем ordinal
        X_train_lgbm = X_train.copy()
        X_val_lgbm = X_val.copy()
        X_test_lgbm = X_test.copy()
        for col in cat_features:
            cats = X_train_lgbm[col].astype("category")
            X_train_lgbm[col] = cats.cat.codes
            X_val_lgbm[col] = (
                X_val_lgbm[col]
                .astype("category")
                .cat.set_categories(cats.cat.categories)
                .cat.codes
            )
            X_test_lgbm[col] = (
                X_test_lgbm[col]
                .astype("category")
                .cat.set_categories(cats.cat.categories)
                .cat.codes
            )

        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        sw_lgbm = make_sample_weights(len(train_df), 0.5)
        lgbm.fit(
            X_train_lgbm,
            y_train,
            eval_set=[(X_val_lgbm, y_val)],
            callbacks=[],
            sample_weight=sw_lgbm,
        )

        prob_lgbm_val = lgbm.predict_proba(X_val_lgbm)[:, 1]
        prob_lgbm_test = lgbm.predict_proba(X_test_lgbm)[:, 1]
        auc_lgbm = roc_auc_score(y_test, prob_lgbm_test)
        logger.info("LightGBM AUC test: %.4f", auc_lgbm)

        # === Calibration: isotonic на val ===
        logger.info("Isotonic calibration...")
        prob_cb_val_cal = calibrate_proba(prob_cb_val, y_val, prob_cb_test)
        prob_lgbm_val_cal = calibrate_proba(prob_lgbm_val, y_val, prob_lgbm_test)

        # Calibration на val → predict на val тоже (для threshold search)
        # CB: fit на части val, apply на другой части — используем full val для simple isotonic
        prob_cb_val_cal_v = calibrate_proba(prob_cb_val, y_val, prob_cb_val)
        prob_lgbm_val_cal_v = calibrate_proba(prob_lgbm_val, y_val, prob_lgbm_val)

        # === Веса: оптимизация на val ===
        # Пробуем несколько соотношений CB:LGBM
        best_val_roi_blend = -999.0
        best_w_cb = 1.0
        best_blend_v = None
        best_blend_t = None

        for w_cb in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            w_lgbm = 1.0 - w_cb
            blend_v = w_cb * prob_cb_val_cal_v + w_lgbm * prob_lgbm_val_cal_v
            blend_t = w_cb * prob_cb_val_cal + w_lgbm * prob_lgbm_val_cal

            kelly_pm_v = compute_kelly(blend_v, val_df["Odds"].values, 1.0)
            pm_mask_val = (val_df["lead_hours"] > 0).values
            kelly_pm_v[~pm_mask_val] = -999

            t, roi_v = find_best_pm_threshold(val_df, kelly_pm_v, min_bets=200)
            if roi_v > best_val_roi_blend:
                best_val_roi_blend = roi_v
                best_w_cb = w_cb
                best_blend_v = blend_v.copy()
                best_blend_t = blend_t.copy()
                best_t = t

        logger.info(
            "Best blend: w_cb=%.1f val_roi=%.2f%% t=%.3f", best_w_cb, best_val_roi_blend, best_t
        )

        # === Финальная оценка на test ===
        kelly_pm_test = compute_kelly(best_blend_t, test_df["Odds"].values, 1.0)
        pm_mask_test = (test_df["lead_hours"] > 0).values
        kelly_pm_test[~pm_mask_test] = -999
        mask_test = kelly_pm_test >= best_t
        roi_test, n_test = calc_roi(test_df, mask_test)
        auc_blend = roc_auc_score(y_test, best_blend_t)

        logger.info("Blend ROI test: %.2f%% (%d bets)", roi_test, n_test)
        logger.info("Blend AUC test: %.4f", auc_blend)

        # Leakage guard
        if roi_test > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — leakage check!")

        # === Baseline: чистый CB без калибровки (для сравнения) ===
        kelly_cb_raw_v = compute_kelly(prob_cb_val, val_df["Odds"].values, 1.0)
        pm_mask_val = (val_df["lead_hours"] > 0).values
        kelly_cb_raw_v[~pm_mask_val] = -999
        t_cb_base, roi_cb_base_v = find_best_pm_threshold(val_df, kelly_cb_raw_v, min_bets=200)
        kelly_cb_raw_t = compute_kelly(prob_cb_test, test_df["Odds"].values, 1.0)
        kelly_cb_raw_t[~pm_mask_test] = -999
        roi_cb_base, n_cb_base = calc_roi(test_df, kelly_cb_raw_t >= t_cb_base)
        logger.info(
            "CB baseline (no cal): t=%.3f roi=%.2f%% (%d)", t_cb_base, roi_cb_base, n_cb_base
        )

        delta = roi_test - PREV_BEST_ROI
        logger.info("Delta vs prev best: %+.2f%%", delta)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "cb_depth": 7,
                "cb_lr": 0.1,
                "cb_iterations": 500,
                "lgbm_lr": 0.05,
                "lgbm_leaves": 63,
                "best_w_cb": best_w_cb,
                "best_t_pm": best_t,
                "min_bets_threshold": 200,
                "calibration": "isotonic",
            }
        )
        mlflow.log_metrics(
            {
                "auc_cb_test": auc_cb,
                "auc_lgbm_test": auc_lgbm,
                "auc_blend_test": auc_blend,
                "roi_test": roi_test,
                "roi_cb_baseline": roi_cb_base,
                "n_bets_test": n_test,
                "delta_vs_prev_best": delta,
                "val_roi_best_blend": best_val_roi_blend,
            }
        )
        mlflow.log_artifact(__file__)

        convergence = min(1.0, max(0.0, delta / 10.0 + 0.5))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))
        mlflow.set_tag("status", "success")

        # Сохраняем пайплайн если новый best
        if roi_test > PREV_BEST_ROI and n_test >= 100:
            models_dir = SESSION_DIR / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            feature_names = X_train.columns.tolist()
            w_lgbm_final = 1.0 - best_w_cb

            # Строим калибраторы на val
            ir_cb = IsotonicRegression(out_of_bounds="clip")
            ir_cb.fit(prob_cb_val, y_val)
            ir_lgbm = IsotonicRegression(out_of_bounds="clip")
            ir_lgbm.fit(prob_lgbm_val, y_val)

            # Кодировщик категорий для LGBM
            lgbm_cat_encoders = {}
            for col in cat_features:
                lgbm_cat_encoders[col] = X_train[col].astype("category").cat.categories

            class BestPipeline:
                """Stacking CB+LGBM + isotonic calibration + PM Kelly."""

                def __init__(
                    self,
                    cb_model,
                    lgbm_model,
                    feature_names,
                    cat_features,
                    lgbm_cat_encoders,
                    ir_cb,
                    ir_lgbm,
                    w_cb,
                    threshold,
                    kelly_fraction=1.0,
                ):
                    self.cb_model = cb_model
                    self.lgbm_model = lgbm_model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.lgbm_cat_encoders = lgbm_cat_encoders
                    self.ir_cb = ir_cb
                    self.ir_lgbm = ir_lgbm
                    self.w_cb = w_cb
                    self.threshold = threshold
                    self.kelly_fraction = kelly_fraction
                    self.sport_filter: list[str] = []
                    self.framework = "stacking_cb_lgbm"

                def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
                    feats, _ = build_features(df)
                    feats = feats[self.feature_names]
                    feats_lgbm = feats.copy()
                    for col in self.cat_features:
                        feats_lgbm[col] = (
                            feats_lgbm[col]
                            .astype("category")
                            .cat.set_categories(self.lgbm_cat_encoders[col])
                            .cat.codes
                        )
                    return feats, feats_lgbm

                def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                    feats_cb, feats_lgbm = self._build_features(df)
                    p_cb = self.cb_model.predict_proba(feats_cb)[:, 1]
                    p_lgbm = self.lgbm_model.predict_proba(feats_lgbm)[:, 1]
                    p_cb_cal = self.ir_cb.transform(p_cb)
                    p_lgbm_cal = self.ir_lgbm.transform(p_lgbm)
                    return self.w_cb * p_cb_cal + (1 - self.w_cb) * p_lgbm_cal

                def evaluate(self, df: pd.DataFrame) -> dict:
                    proba = self.predict_proba(df)
                    kelly = compute_kelly(proba, df["Odds"].values, self.kelly_fraction)
                    pm_mask = (df["lead_hours"] > 0).values
                    kelly[~pm_mask] = -999
                    mask = kelly >= self.threshold
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

            pipeline = BestPipeline(
                cb_model=cb,
                lgbm_model=lgbm,
                feature_names=feature_names,
                cat_features=cat_features,
                lgbm_cat_encoders=lgbm_cat_encoders,
                ir_cb=ir_cb,
                ir_lgbm=ir_lgbm,
                w_cb=best_w_cb,
                threshold=best_t,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            cb.save_model(str(models_dir / "model.cbm"))

            metadata = {
                "framework": "stacking_cb_lgbm_isotonic",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_test,
                "auc": auc_blend,
                "threshold": best_t,
                "n_bets": n_test,
                "feature_names": feature_names,
                "params": {
                    "cb_depth": 7,
                    "lgbm_leaves": 63,
                    "w_cb": best_w_cb,
                    "calibration": "isotonic",
                },
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.1",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("New best pipeline saved. ROI=%.2f%%", roi_test)

        print("\n=== Step 4.1 Stacking CB+LGBM + Isotonic Calibration ===")
        print(f"CB AUC test:       {auc_cb:.4f}")
        print(f"LGBM AUC test:     {auc_lgbm:.4f}")
        print(f"Blend AUC test:    {auc_blend:.4f}")
        print(f"CB baseline ROI:   {roi_cb_base:.2f}% ({n_cb_base} bets, t={t_cb_base:.3f})")
        print(
            f"Blend ROI test:    {roi_test:.2f}% ({n_test} bets,"
            f" t={best_t:.3f}, w_cb={best_w_cb:.1f})"
        )
        print(f"Prev best:         {PREV_BEST_ROI:.2f}%")
        print(f"Delta:             {delta:+.2f}%")
        print(f"MLflow run_id:     {run.info.run_id}")

    except Exception:
        import traceback

        tb = traceback.format_exc()
        mlflow.set_tag("status", "failed")
        mlflow.log_text(tb, "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        logger.error("Exception:\n%s", tb)
        raise
