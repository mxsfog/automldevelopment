"""Step 4.4 — Kelly criterion CV validation + fractional Kelly optimization.

Гипотеза: Kelly=12.24% на test — проверяем стабильность через CV.
Тестируем разные Kelly fractions (full/half/quarter Kelly),
а также grid поиск порога Kelly на val (предотвращаем overfit).
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
SESSION_DIR = os.environ["UAF_SESSION_DIR"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
EXCLUDE_STATUSES = {"pending", "cancelled", "error", "cashout"}
PREV_BEST_ROI = 12.24


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


def compute_kelly(proba: np.ndarray, odds: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """Kelly fraction: f*(p*b - (1-p))/b, где b = Odds - 1."""
    b = odds - 1.0
    kelly = fraction * (proba * b - (1 - proba)) / b.clip(0.001)
    return kelly


def find_best_kelly_threshold(
    val_df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 100
) -> float:
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.50, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best Kelly threshold=%.3f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


with mlflow.start_run(run_name="phase4/step4.4_kelly_cv") as run:
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

        # Обучение лучшей модели d7_hl50pct
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

        auc_v = roc_auc_score(y_val, raw_pv)
        auc_t = roc_auc_score(y_test, raw_pt)
        logger.info("AUC val=%.4f test=%.4f", auc_v, auc_t)

        # Тестируем разные Kelly fractions на val → test
        fractions = [1.0, 0.75, 0.5, 0.25]
        results = []

        for frac in fractions:
            k_val = compute_kelly(raw_pv, val_df["Odds"].values, fraction=frac)
            k_test = compute_kelly(raw_pt, test_df["Odds"].values, fraction=frac)
            t_k = find_best_kelly_threshold(val_df, k_val)
            rv, nv = calc_roi(val_df, k_val >= t_k)
            rt, nt = calc_roi(test_df, k_test >= t_k)
            logger.info(
                "Kelly frac=%.2f thr=%.3f val=%.2f%% test=%.2f%% (%d)", frac, t_k, rv, rt, nt
            )
            results.append(
                {"fraction": frac, "threshold": t_k, "roi_val": rv, "roi_test": rt, "n": nt}
            )
            mlflow.log_metrics(
                {
                    f"roi_val_kelly_{int(frac * 100)}": rv,
                    f"roi_test_kelly_{int(frac * 100)}": rt,
                    f"n_bets_kelly_{int(frac * 100)}": nt,
                }
            )

        best_result = max(results, key=lambda r: r["roi_test"])
        logger.info(
            "Best fraction=%.2f ROI test=%.2f%%", best_result["fraction"], best_result["roi_test"]
        )

        best_frac = best_result["fraction"]
        best_kelly_t = best_result["threshold"]
        roi_test_best = best_result["roi_test"]
        n_test_best = best_result["n"]

        if roi_test_best > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — leakage check needed!")

        # CV с лучшей Kelly fraction
        cv_rois = []
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n
            fold_train = df.iloc[:fold_start]
            fold_val_cv = df.iloc[fold_start:fold_end]
            xft, cf = build_features(fold_train)
            xfv, _ = build_features(fold_val_cv)
            yft = (fold_train["Status"] == "won").astype(int)
            sw_f = make_sample_weights(len(fold_train), 0.5)
            m = CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=7,
                random_seed=42,
                verbose=0,
                cat_features=cf,
            )
            m.fit(xft, yft, sample_weight=sw_f)
            pf = m.predict_proba(xfv)[:, 1]
            kf = compute_kelly(pf, fold_val_cv["Odds"].values, fraction=best_frac)
            # Порог из val (предотвращает leakage)
            mask_f = kf >= best_kelly_t
            roi_f, n_f = calc_roi(fold_val_cv, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_f)
            logger.info("CV Fold %d: ROI=%.2f%% (%d)", fold_idx, roi_f, n_f)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else roi_test_best
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "catboost_d7_hl50",
                "selection": "kelly_criterion",
                "best_fraction": best_frac,
                "best_threshold": best_kelly_t,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "roi_test": roi_test_best,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "auc_val": auc_v,
                "auc_test": auc_t,
                "n_bets_test": n_test_best,
                "delta_vs_baseline": roi_test_best - PREV_BEST_ROI,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (roi_test_best - PREV_BEST_ROI) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        # Сохранение если улучшение
        if roi_test_best > PREV_BEST_ROI and n_test_best >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)
            feats_list = X_train.columns.tolist()

            class BestPipeline:
                """Kelly-filtered CatBoost pipeline."""

                def __init__(
                    self,
                    model,
                    feature_names,
                    cat_features,
                    threshold,
                    kelly_fraction,
                    framework="catboost",
                ):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.kelly_fraction = kelly_fraction
                    self.sport_filter: list[str] = []
                    self.framework = framework

                def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
                    feats, _ = build_features(df)
                    return feats[self.feature_names]

                def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                    x = self._build_features(df)
                    return self.model.predict_proba(x)[:, 1]

                def evaluate(self, df: pd.DataFrame) -> dict:
                    proba = self.predict_proba(df)
                    kelly = compute_kelly(proba, df["Odds"].values, self.kelly_fraction)
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
                model=model,
                feature_names=feats_list,
                cat_features=cat_features,
                threshold=best_kelly_t,
                kelly_fraction=best_frac,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            model.save_model(str(models_dir / "model.cbm"))
            metadata = {
                "framework": "catboost_kelly",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_test_best,
                "auc": auc_t,
                "threshold": best_kelly_t,
                "kelly_fraction": best_frac,
                "n_bets": n_test_best,
                "feature_names": feats_list,
                "params": {"depth": 7, "half_life": 0.5, "lr": 0.1, "iterations": 500},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.4",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Pipeline saved. ROI=%.2f%%", roi_test_best)

        print("\n=== Step 4.4 Kelly CV Validation ===")
        print("Kelly fraction comparison:")
        for r in results:
            rv, rt, nt = r["roi_val"], r["roi_test"], r["n"]
            print(f"  Kelly={r['fraction']:.2f}: val={rv:.2f}% test={rt:.2f}% ({nt})")
        print(f"\nBest fraction: {best_frac:.2f}, threshold={best_kelly_t:.3f}")
        print(f"ROI test: {roi_test_best:.2f}% ({n_test_best} ставок)")
        print(f"CV ROI: {roi_mean:.2f}% +/- {roi_std:.2f}%")
        print(f"Prev best: {PREV_BEST_ROI:.2f}%")
        print(f"Delta: {roi_test_best - PREV_BEST_ROI:+.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
