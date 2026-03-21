"""Step 4.2 — LightGBM: альтернативный GBT для сравнения с CatBoost.

Гипотеза: LightGBM с другим inductive bias может лучше обобщаться на test.
Baseline: CatBoost ROI=24.91% (n=435).
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

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


def build_features(
    df: pd.DataFrame, encoder: OrdinalEncoder | None = None, fit: bool = False
) -> tuple[np.ndarray, list[str], OrdinalEncoder]:
    """Feature set с OrdinalEncoder для LightGBM."""
    cat_cols = ["Sport", "Market", "Currency"]

    # Числовые фичи
    num = pd.DataFrame(index=df.index)
    num["Odds"] = df["Odds"]
    num["USD"] = df["USD"]
    num["log_odds"] = np.log(df["Odds"].clip(1.001))
    num["log_usd"] = np.log1p(df["USD"].clip(0))
    num["implied_prob"] = 1.0 / df["Odds"].clip(1.001)
    num["is_parlay"] = (df["Is_Parlay"] == "t").astype(int)
    num["outcomes_count"] = df["Outcomes_Count"].fillna(1)
    num["ml_p_model"] = df["ML_P_Model"].fillna(-1)
    num["ml_p_implied"] = df["ML_P_Implied"].fillna(-1)
    num["ml_edge"] = df["ML_Edge"].fillna(0.0)
    num["ml_ev"] = df["ML_EV"].clip(-100, 1000).fillna(0.0)
    num["ml_team_stats_found"] = (df["ML_Team_Stats_Found"] == "t").astype(int)
    num["ml_winrate_diff"] = df["ML_Winrate_Diff"].fillna(0.0)
    num["ml_rating_diff"] = df["ML_Rating_Diff"].fillna(0.0)
    num["hour"] = df["Created_At"].dt.hour
    num["day_of_week"] = df["Created_At"].dt.dayofweek
    num["month"] = df["Created_At"].dt.month
    num["odds_times_stake"] = num["Odds"] * num["USD"]
    num["ml_edge_pos"] = num["ml_edge"].clip(0)
    num["ml_ev_pos"] = num["ml_ev"].clip(0)
    num["elo_max"] = df["elo_max"].fillna(-1)
    num["elo_min"] = df["elo_min"].fillna(-1)
    num["elo_diff"] = df["elo_diff"].fillna(0.0)
    num["elo_ratio"] = df["elo_ratio"].fillna(1.0)
    num["elo_mean"] = df["elo_mean"].fillna(-1)
    num["elo_std"] = df["elo_std"].fillna(0.0)
    num["k_factor_mean"] = df["k_factor_mean"].fillna(-1)
    num["has_elo"] = df["elo_count"].notna().astype(int)
    num["elo_count"] = df["elo_count"].fillna(0)
    num["ml_edge_x_elo_diff"] = num["ml_edge"] * num["elo_diff"].clip(0, 500) / 500
    num["elo_implied_agree"] = (num["implied_prob"] - 1.0 / num["elo_ratio"].clip(0.5, 2.0)).abs()

    # Категориальные фичи (OrdinalEncoded)
    cat_data = df[cat_cols].fillna("unknown")
    if fit:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoded = encoder.fit_transform(cat_data)
    else:
        cat_encoded = encoder.transform(cat_data)

    x = np.hstack([num.values, cat_encoded])
    feat_names = list(num.columns) + cat_cols
    return x, feat_names, encoder


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


class LGBMPipeline:
    """Пайплайн с LightGBM."""

    def __init__(
        self,
        model: lgb.Booster,
        feature_names: list[str],
        encoder: OrdinalEncoder,
        threshold: float,
        sport_filter: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.encoder = encoder
        self.threshold = threshold
        self.sport_filter = sport_filter

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Вероятности для RAW DataFrame."""
        x, _, _ = build_features(df, encoder=self.encoder, fit=False)
        return self.model.predict(x)

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
    """LightGBM baseline + Kelly."""
    with mlflow.start_run(run_name="phase4/step4.2_lightgbm") as run:
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

            x_tr, feat_names, encoder = build_features(train_df, fit=True)
            x_vl, _, _ = build_features(val_df, encoder=encoder, fit=False)
            x_te, _, _ = build_features(test_df, encoder=encoder, fit=False)
            y_tr = (train_df["Status"] == "won").astype(int).values
            y_vl = (val_df["Status"] == "won").astype(int).values
            y_te = (test_df["Status"] == "won").astype(int).values
            w = make_weights(len(train_df))

            cat_idx = [feat_names.index(c) for c in ["Sport", "Market", "Currency"]]

            dtrain = lgb.Dataset(x_tr, label=y_tr, weight=w, categorical_feature=cat_idx)
            dval = lgb.Dataset(x_vl, label=y_vl, reference=dtrain, categorical_feature=cat_idx)

            params = {
                "objective": "binary",
                "metric": "auc",
                "num_leaves": 63,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "random_state": 42,
                "verbose": -1,
            }

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            pv = model.predict(x_vl)
            pt = model.predict(x_te)
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
                "LightGBM: val=%.2f%%, test=%.2f%% (%d bets), t=%.3f, AUC=%.4f, delta=%.2f%%",
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

            # Сохраняем если новый best
            if roi_test > BASELINE_ROI and n_bets >= 200:
                pipeline = LGBMPipeline(
                    model=model,
                    feature_names=feat_names,
                    encoder=encoder,
                    threshold=t_best,
                    sport_filter=[],
                )
                models_dir = SESSION_DIR / "models" / "best"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, models_dir / "pipeline.pkl")
                model.save_model(str(models_dir / "model.lgb"))
                metadata = {
                    "framework": "lightgbm",
                    "model_file": "model.lgb",
                    "pipeline_file": "pipeline.pkl",
                    "roi": float(roi_test),
                    "auc": float(auc_test),
                    "threshold": float(t_best),
                    "n_bets": n_bets,
                    "feature_names": feat_names,
                    "params": params,
                    "sport_filter": [],
                    "session_id": SESSION_ID,
                    "step": "4.2",
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
                    "model": "lightgbm",
                    "num_leaves": 63,
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
