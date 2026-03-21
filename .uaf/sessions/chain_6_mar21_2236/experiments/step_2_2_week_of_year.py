"""Step 2.2 — Shadow Feature: week_of_year вместо day_of_week.

Гипотеза: неделя года лучше захватывает сезонность, чем день недели.
Baseline: ROI=24.91% (step 1.4). Тест только одну замену.
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


def build_features_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline feature set (day_of_week)."""
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


def build_features_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Candidate: week_of_year вместо day_of_week."""
    feats, cat_f = build_features_baseline(df)
    # Заменяем day_of_week на week_of_year
    del feats["day_of_week"]
    feats["week_of_year"] = df["Created_At"].dt.isocalendar().week.astype(int)
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
    feature_builder: callable,
    run_suffix: str,
) -> tuple[float, float, int, float, float]:
    """Обучение модели и оценка ROI."""
    x_tr, cat_f = feature_builder(train_df)
    x_vl, _ = feature_builder(val_df)
    x_te, _ = feature_builder(test_df)
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

    pv = model.predict_proba(x_vl)[:, 1]
    pt = model.predict_proba(x_te)[:, 1]
    auc_test = roc_auc_score(y_te, pt)

    pm_val = (val_df["lead_hours"] > 0).values
    pm_test = (test_df["lead_hours"] > 0).values
    k_v = compute_kelly(pv, val_df["Odds"].values)
    k_t = compute_kelly(pt, test_df["Odds"].values)
    k_v[~pm_val] = -999
    k_t[~pm_test] = -999

    t_best, roi_val = find_threshold(val_df, k_v)
    roi_test, n_bets = calc_roi(test_df, k_t >= t_best)

    logger.info(
        "%s: val=%.2f%%, test=%.2f%% (%d bets), t=%.3f, AUC=%.4f",
        run_suffix,
        roi_val,
        roi_test,
        n_bets,
        t_best,
        auc_test,
    )
    return roi_val, roi_test, n_bets, auc_test, t_best, model, x_tr, cat_f


class BestPipeline:
    """Пайплайн с week_of_year."""

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
        feats, _ = build_features_candidate(df)
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
    """Shadow feature test: week_of_year vs day_of_week."""
    with mlflow.start_run(run_name="phase2/step2.2_week_of_year") as run:
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

            # Baseline run
            _roi_val_b, roi_test_b, n_bets_b, auc_b, t_b, *_ = run_experiment(
                train_df, val_df, test_df, build_features_baseline, "baseline"
            )
            # Candidate run
            result = run_experiment(
                train_df, val_df, test_df, build_features_candidate, "week_of_year"
            )
            roi_val_c, roi_test_c, n_bets_c, auc_c, t_c, model_c, x_tr_c, cat_f_c = result

            delta = roi_test_c - roi_test_b
            logger.info(
                "Delta: %.2f%% (candidate=%.2f%%, baseline=%.2f%%)",
                delta,
                roi_test_c,
                roi_test_b,
            )

            if roi_test_c > LEAKAGE_THRESHOLD:
                logger.error("LEAKAGE SUSPECT: roi=%.2f%%", roi_test_c)
                mlflow.set_tag("leakage_suspect", "true")

            # Сохраняем если улучшение
            if roi_test_c > BASELINE_ROI and n_bets_c >= 200:
                feat_names = list(x_tr_c.columns)
                pipeline = BestPipeline(
                    model=model_c,
                    feature_names=feat_names,
                    cat_features=cat_f_c,
                    threshold=t_c,
                    sport_filter=[],
                )
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, models_dir / "pipeline.pkl")
                model_c.save_model(str(models_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "pipeline_file": "pipeline.pkl",
                    "roi": float(roi_test_c),
                    "auc": float(auc_c),
                    "threshold": float(t_c),
                    "n_bets": n_bets_c,
                    "feature_names": feat_names,
                    "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "step": "2.2",
                }
                (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
                logger.info("New best pipeline saved! roi=%.2f%%", roi_test_c)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "shadow_feature": "week_of_year_vs_day_of_week",
                    "threshold_baseline": t_b,
                    "threshold_candidate": t_c,
                }
            )
            mlflow.log_metrics(
                {
                    "roi_test_baseline": float(roi_test_b),
                    "roi_test_candidate": float(roi_test_c),
                    "roi_val_candidate": float(roi_val_c),
                    "roi_delta": float(delta),
                    "auc_baseline": float(auc_b),
                    "auc_candidate": float(auc_c),
                    "n_bets_baseline": n_bets_b,
                    "n_bets_candidate": n_bets_c,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag(
                "shadow_result",
                "accepted" if delta > 0.002 else ("marginal" if delta > 0 else "rejected"),
            )
            mlflow.set_tag("convergence_signal", "0.65")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT baseline={roi_test_b:.2f}% candidate={roi_test_c:.2f}% "
                f"delta={delta:+.2f}% n_bets={n_bets_c} run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
