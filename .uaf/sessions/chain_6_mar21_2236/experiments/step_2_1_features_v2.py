"""Step 2.1 — Feature Engineering v2: week_of_year + lead_hours + odds buckets.

Гипотеза: добавление week_of_year (вместо day_of_week) + lead_hours + odds_bucket
+ дополнительные ratio-фичи улучшит ROI выше 24.91%.
Базируется на chain_3 шаге 2.2 (winner конфигурация).
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
    """Загрузка данных с ELO."""
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
            # Новые ELO фичи: delta ELO (изменение после предыдущего матча)
            elo_new_max=("New_ELO", "max"),
            elo_new_min=("New_ELO", "min"),
        )
        .reset_index()
    )
    elo_agg["elo_diff"] = elo_agg["elo_max"] - elo_agg["elo_min"]
    elo_agg["elo_ratio"] = elo_agg["elo_max"] / elo_agg["elo_min"].clip(1.0)
    # Изменение ELO (momentum): New - Old (положительное = рост формы)
    elo_agg["elo_delta_max"] = elo_agg["elo_new_max"] - elo_agg["elo_max"]
    elo_agg["elo_delta_min"] = elo_agg["elo_new_min"] - elo_agg["elo_min"]

    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature set v2: week_of_year + lead_hours + odds_bucket + ELO delta."""
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

    # Временные фичи: week_of_year вместо day_of_week (chain_3 winner)
    feats["hour"] = df["Created_At"].dt.hour
    feats["week_of_year"] = df["Created_At"].dt.isocalendar().week.astype(int)
    feats["month"] = df["Created_At"].dt.month
    feats["day_of_week"] = df["Created_At"].dt.dayofweek

    # Lead time фичи
    lead_hours = df["lead_hours"].fillna(-1)
    feats["lead_hours"] = lead_hours.clip(-1, 168)  # до 1 недели
    feats["is_prematch"] = (lead_hours > 0).astype(int)
    feats["lead_hours_log"] = np.log1p(lead_hours.clip(0))

    # Odds bucketing (нелинейные диапазоны)
    feats["odds_bucket"] = (
        pd.cut(
            df["Odds"].clip(1.01, 20),
            bins=[1.0, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 20.0],
            labels=False,
        )
        .fillna(0)
        .astype(int)
    )
    feats["is_favorite"] = (df["Odds"] < 2.0).astype(int)
    feats["is_longshot"] = (df["Odds"] > 5.0).astype(int)

    feats["odds_times_stake"] = feats["Odds"] * feats["USD"]
    feats["ml_edge_pos"] = feats["ml_edge"].clip(0)
    feats["ml_ev_pos"] = feats["ml_ev"].clip(0)

    # ELO фичи
    feats["elo_max"] = df["elo_max"].fillna(-1)
    feats["elo_min"] = df["elo_min"].fillna(-1)
    feats["elo_diff"] = df["elo_diff"].fillna(0.0)
    feats["elo_ratio"] = df["elo_ratio"].fillna(1.0)
    feats["elo_mean"] = df["elo_mean"].fillna(-1)
    feats["elo_std"] = df["elo_std"].fillna(0.0)
    feats["k_factor_mean"] = df["k_factor_mean"].fillna(-1)
    feats["has_elo"] = df["elo_count"].notna().astype(int)
    feats["elo_count"] = df["elo_count"].fillna(0)

    # Новые: ELO momentum (изменение после последнего матча)
    feats["elo_delta_max"] = df["elo_delta_max"].fillna(0.0).clip(-100, 100)
    feats["elo_delta_min"] = df["elo_delta_min"].fillna(0.0).clip(-100, 100)
    feats["elo_momentum"] = feats["elo_delta_max"] - feats["elo_delta_min"]

    # Взаимодействия
    feats["ml_edge_x_elo_diff"] = feats["ml_edge"] * feats["elo_diff"].clip(0, 500) / 500
    feats["elo_implied_agree"] = (
        feats["implied_prob"] - 1.0 / feats["elo_ratio"].clip(0.5, 2.0)
    ).abs()
    # Новое: edge * is_prematch (ценность только для pre-match ставок)
    feats["ml_edge_prematch"] = feats["ml_edge"] * feats["is_prematch"]
    feats["elo_diff_x_prematch"] = feats["elo_diff"] * feats["is_prematch"]

    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


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


class BestPipeline:
    """Пайплайн v2: расширенный feature set + CatBoost + Kelly."""

    def __init__(
        self,
        model: CatBoostClassifier,
        feature_names: list[str],
        cat_features: list[str],
        threshold: float,
        sport_filter: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.cat_features = cat_features
        self.threshold = threshold
        self.sport_filter = sport_filter

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, _ = build_features(df)
        return feats[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Вероятности для RAW DataFrame."""
        x = self._build_features(df)
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, df: pd.DataFrame) -> dict:
        """ROI и метрики на RAW DataFrame."""
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
    """Feature Engineering v2 + CatBoost."""
    with mlflow.start_run(run_name="phase2/step2.1_features_v2") as run:
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

            x_tr, cat_f = build_features(train_df)
            x_vl, _ = build_features(val_df)
            x_te, _ = build_features(test_df)
            y_tr = (train_df["Status"] == "won").astype(int)
            y_vl = (val_df["Status"] == "won").astype(int)
            y_te = (test_df["Status"] == "won").astype(int)
            w = make_weights(len(train_df))

            logger.info("Features: %d", len(x_tr.columns))

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

            pv = model.predict_proba(x_vl)[:, 1]
            pt = model.predict_proba(x_te)[:, 1]
            auc_val = roc_auc_score(y_vl, pv)
            auc_test = roc_auc_score(y_te, pt)

            pm_val = (val_df["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values
            k_v = compute_kelly(pv, val_df["Odds"].values)
            k_t = compute_kelly(pt, test_df["Odds"].values)
            k_v[~pm_val] = -999
            k_t[~pm_test] = -999

            t_best, roi_val = find_threshold(val_df, k_v)
            roi_test, n_bets = calc_roi(test_df, k_t >= t_best)
            delta = roi_test - BASELINE_ROI

            logger.info(
                "Features v2: val=%.2f%%, test=%.2f%% (%d bets), t=%.3f, AUC=%.4f, delta=%.2f%%",
                roi_val,
                roi_test,
                n_bets,
                t_best,
                auc_test,
                delta,
            )

            if roi_test > LEAKAGE_THRESHOLD:
                logger.error("LEAKAGE SUSPECT: roi=%.2f%%", roi_test)
                mlflow.set_tag("leakage_suspect", "true")

            # Feature importances top-10
            importances = model.get_feature_importance()
            feat_names = list(x_tr.columns)
            top_idx = np.argsort(importances)[::-1][:10]
            top_feats = {feat_names[i]: float(importances[i]) for i in top_idx}
            logger.info("Top features: %s", top_feats)

            # Сохраняем если новый best
            if roi_test > BASELINE_ROI and n_bets >= 200:
                feature_names = feat_names
                pipeline = BestPipeline(
                    model=model,
                    feature_names=feature_names,
                    cat_features=cat_f,
                    threshold=t_best,
                    sport_filter=[],
                )
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, models_dir / "pipeline.pkl")
                model.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "pipeline_file": "pipeline.pkl",
                    "roi": float(roi_test),
                    "auc": float(auc_test),
                    "threshold": float(t_best),
                    "n_bets": n_bets,
                    "feature_names": feat_names,
                    "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "step": "2.1",
                }
                (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
                logger.info("New best pipeline saved! roi=%.2f%%", roi_test)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "model": "catboost",
                    "n_features": len(x_tr.columns),
                    "new_features": "week_of_year,lead_hours,odds_bucket,elo_delta,elo_momentum",
                    "threshold": t_best,
                }
            )
            mlflow.log_metrics(
                {
                    "auc_val": float(auc_val),
                    "auc_test": float(auc_test),
                    "roi_val": float(roi_val),
                    "roi_test": float(roi_test),
                    "roi_delta": float(delta),
                    "n_bets": n_bets,
                    "kelly_threshold": float(t_best),
                }
            )
            for feat, imp in top_feats.items():
                mlflow.log_metric(f"imp_{feat}", imp)
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT roi_val={roi_val:.2f}% roi_test={roi_test:.2f}% "
                f"n_bets={n_bets} auc={auc_test:.4f} delta={delta:+.2f}% "
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
