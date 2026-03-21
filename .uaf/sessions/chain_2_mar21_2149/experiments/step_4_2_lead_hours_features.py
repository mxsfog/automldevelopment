"""Step 4.2 — Shadow Feature Trick: lead_hours + odds_tier как новые признаки.

Гипотеза:
- lead_hours (время до начала матча) — прямой сигнал предсказуемости ставки
  Чем раньше поставлена ставка, тем стабильнее сигнал. Признак не был в feature set.
- odds_tier (категориальный диапазон коэффициента) — разные диапазоны имеют разную
  эффективность рынка
- Ожидаемый прирост: +2-5% над 24.91%

Метод: Shadow Feature Trick
1. Baseline: исходный feature set (34 признака из step_4_5)
2. Candidate: + log_lead_hours + lead_tier + odds_tier (3 признака)
3. Обе модели обучаются с одинаковыми гиперпараметрами
4. delta = roi_candidate - roi_baseline
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


def build_features_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Базовый feature set (как в step_4_5)."""
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


def build_features_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Candidate: базовый + lead_hours + odds_tier."""
    feats, cat_features = build_features_baseline(df)

    # Shadow features: lead time
    lead_raw = df["lead_hours"].fillna(-1)
    feats["log_lead_hours"] = np.log1p(lead_raw.clip(0))
    # Bucket: [-inf, 0) = live, [0,1) = <1h, [1,12) = 1-12h, [12,48) = 12-48h, [48,+inf) = >48h
    bins = [-9999, 0, 1, 12, 48, 9999]
    labels = ["live", "lead_lt1h", "lead_1_12h", "lead_12_48h", "lead_gt48h"]
    feats["lead_tier"] = pd.cut(lead_raw, bins=bins, labels=labels).astype(str)

    # Shadow features: odds bucket
    odds_bins = [1.0, 1.5, 2.0, 3.0, 5.0, 100.0]
    odds_labels = ["odds_lt15", "odds_15_2", "odds_2_3", "odds_3_5", "odds_gt5"]
    feats["odds_tier"] = pd.cut(
        df["Odds"].clip(1.001, 100), bins=odds_bins, labels=odds_labels
    ).astype(str)

    new_cat = [*cat_features, "lead_tier", "odds_tier"]
    return feats, new_cat


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


def train_and_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    build_fn,
    tag: str,
) -> tuple[float, float, int, float]:
    """Обучает модель и возвращает (roi_val, roi_test, n_test, auc_test)."""
    x_train, cat_features = build_fn(train_df)
    x_val, _ = build_fn(val_df)
    x_test, _ = build_fn(test_df)

    y_train = (train_df["Status"] == "won").astype(int).values
    y_val = (val_df["Status"] == "won").astype(int).values
    y_test = (test_df["Status"] == "won").astype(int).values

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
        x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, sample_weight=sw
    )

    pv = model.predict_proba(x_val)[:, 1]
    pt = model.predict_proba(x_test)[:, 1]
    auc_t = roc_auc_score(y_test, pt)

    kelly_v = compute_kelly(pv, val_df["Odds"].values, 1.0)
    pm_mask_v = (val_df["lead_hours"] > 0).values
    kelly_v[~pm_mask_v] = -999
    t_best, roi_v = find_best_pm_threshold(val_df, kelly_v, min_bets=200)

    kelly_t = compute_kelly(pt, test_df["Odds"].values, 1.0)
    pm_mask_t = (test_df["lead_hours"] > 0).values
    kelly_t[~pm_mask_t] = -999
    roi_t, n_t = calc_roi(test_df, kelly_t >= t_best)

    logger.info(
        "[%s] t=%.3f val=%.2f%% test=%.2f%% (%d bets) AUC=%.4f",
        tag,
        t_best,
        roi_v,
        roi_t,
        n_t,
        auc_t,
    )
    return roi_v, roi_t, n_t, auc_t


with mlflow.start_run(run_name="phase4/step4.2_lead_hours_features") as run:
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

        logger.info("Обучаем baseline модель...")
        roi_v_base, roi_t_base, n_base, auc_base = train_and_eval(
            train_df, val_df, test_df, build_features_baseline, "baseline"
        )

        logger.info("Обучаем candidate модель (+ lead_hours + odds_tier)...")
        roi_v_cand, roi_t_cand, n_cand, auc_cand = train_and_eval(
            train_df, val_df, test_df, build_features_candidate, "candidate"
        )

        delta = roi_t_cand - roi_t_base
        delta_vs_prev = roi_t_cand - PREV_BEST_ROI
        logger.info(
            "Delta baseline→candidate: %+.2f%%, vs prev best: %+.2f%%",
            delta,
            delta_vs_prev,
        )

        # Leakage guard
        best_roi = max(roi_t_base, roi_t_cand)
        if best_roi > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — leakage check!")

        # Shadow feature decision
        if delta > 0.002:
            shadow_decision = "accept"
        elif delta <= 0:
            shadow_decision = "reject"
        else:
            shadow_decision = "marginal"
        logger.info("Shadow feature decision: %s (delta=%.4f)", shadow_decision, delta)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "new_features": "log_lead_hours,lead_tier,odds_tier",
                "shadow_method": "baseline_vs_candidate",
            }
        )
        mlflow.log_metrics(
            {
                "roi_test_baseline": roi_t_base,
                "roi_test_candidate": roi_t_cand,
                "roi_val_baseline": roi_v_base,
                "roi_val_candidate": roi_v_cand,
                "auc_test_baseline": auc_base,
                "auc_test_candidate": auc_cand,
                "delta_candidate_vs_baseline": delta,
                "delta_vs_prev_best": delta_vs_prev,
                "n_bets_baseline": n_base,
                "n_bets_candidate": n_cand,
            }
        )
        mlflow.set_tag("shadow_decision", shadow_decision)
        mlflow.log_artifact(__file__)

        convergence = min(1.0, max(0.0, delta_vs_prev / 10.0 + 0.5))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))
        mlflow.set_tag("status", "success")

        # Сохраняем пайплайн если новый best
        best_roi_final = max(roi_t_base, roi_t_cand)
        if best_roi_final > PREV_BEST_ROI:
            use_candidate = roi_t_cand > roi_t_base
            best_build = build_features_candidate if use_candidate else build_features_baseline

            # Переобучаем лучшую модель для сохранения
            X_tr, cat_f = best_build(train_df)
            X_v, _ = best_build(val_df)
            X_te, _ = best_build(test_df)
            y_tr = (train_df["Status"] == "won").astype(int).values
            y_v = (val_df["Status"] == "won").astype(int).values
            sw = make_sample_weights(len(train_df), 0.5)
            best_model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=7,
                random_seed=42,
                eval_metric="AUC",
                verbose=0,
                cat_features=cat_f,
            )
            best_model.fit(
                X_tr, y_tr, eval_set=(X_v, y_v), early_stopping_rounds=50, sample_weight=sw
            )
            pt = best_model.predict_proba(X_te)[:, 1]
            kelly_t = compute_kelly(pt, test_df["Odds"].values, 1.0)
            pm_t = (test_df["lead_hours"] > 0).values
            kelly_t[~pm_t] = -999
            pv = best_model.predict_proba(X_v)[:, 1]
            kelly_v = compute_kelly(pv, val_df["Odds"].values, 1.0)
            pm_v = (val_df["lead_hours"] > 0).values
            kelly_v[~pm_v] = -999
            t_save, _ = find_best_pm_threshold(val_df, kelly_v, min_bets=200)
            roi_save, n_save = calc_roi(test_df, kelly_t >= t_save)

            models_dir = SESSION_DIR / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            feature_names = X_tr.columns.tolist()

            class BestPipeline:
                """CatBoost + lead_hours/odds_tier + PM Kelly."""

                def __init__(self, model, feature_names, cat_features, threshold, use_candidate):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.use_candidate = use_candidate
                    self.sport_filter: list[str] = []
                    self.framework = "catboost_lead_features"

                def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
                    fn = (
                        build_features_candidate if self.use_candidate else build_features_baseline
                    )
                    feats, _ = fn(df)
                    return feats[self.feature_names]

                def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                    x = self._build_features(df)
                    return self.model.predict_proba(x)[:, 1]

                def evaluate(self, df: pd.DataFrame) -> dict:
                    proba = self.predict_proba(df)
                    kelly = compute_kelly(proba, df["Odds"].values, 1.0)
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
                model=best_model,
                feature_names=feature_names,
                cat_features=cat_f,
                threshold=t_save,
                use_candidate=use_candidate,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            best_model.save_model(str(models_dir / "model.cbm"))

            metadata = {
                "framework": "catboost_lead_features",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_save,
                "auc": roc_auc_score((test_df["Status"] == "won").astype(int), pt),
                "threshold": t_save,
                "n_bets": n_save,
                "feature_names": feature_names,
                "params": {"depth": 7, "use_candidate": use_candidate},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.2",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("New best pipeline saved. ROI=%.2f%%", roi_save)

        print("\n=== Step 4.2 Shadow Feature Trick: lead_hours + odds_tier ===")
        print(
            f"Baseline (no shadow): val={roi_v_base:.2f}% test={roi_t_base:.2f}%"
            f" ({n_base} bets) AUC={auc_base:.4f}"
        )
        print(
            f"Candidate (+ shadow): val={roi_v_cand:.2f}% test={roi_t_cand:.2f}%"
            f" ({n_cand} bets) AUC={auc_cand:.4f}"
        )
        print(f"Delta candidate-baseline: {delta:+.2f}%")
        print(f"Shadow decision: {shadow_decision}")
        print(f"Prev best: {PREV_BEST_ROI:.2f}%")
        print(f"Delta vs prev best: {delta_vs_prev:+.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        tb = traceback.format_exc()
        mlflow.set_tag("status", "failed")
        mlflow.log_text(tb, "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        logger.error("Exception:\n%s", tb)
        raise
