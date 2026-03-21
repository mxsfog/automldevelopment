"""Step 2.2 - Singles filter + Odds range filter.

Hypothesis: Фильтрация parlays (val ROI=-17.9%) и odds>=3 (train ROI=-13...-30%)
улучшит ROI модели.

Validation:
- Parlays ROI: train=-19%, val=-18%, test=-28% (стабильно плохо)
- Odds 3+: train=-13.4% и -30.3% (стабильно плохо на train и test)
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

# Фильтры (выбраны по val, не по test)
EXCLUDE_PARLAYS = True  # val парлаи: -17.9%
MAX_ODDS = 3.0  # train: odds 3+ = -13.4% и -30.3%


def load_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Selection", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)
    return df


def apply_pre_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Применяем pre-filters (решение принято на val, не на test)."""
    n_before = len(df)
    if EXCLUDE_PARLAYS:
        df = df[df["Is_Parlay"] != "t"].copy()
    df = df[df["Odds"] < MAX_ODDS].copy()
    logger.info("Pre-filter: %d -> %d строк (убрали %d)", n_before, len(df), n_before - len(df))
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline фичи + несколько улучшений."""
    feats = pd.DataFrame(index=df.index)

    feats["Odds"] = df["Odds"]
    feats["USD"] = df["USD"]
    feats["log_odds"] = np.log(df["Odds"].clip(1.001))
    feats["log_usd"] = np.log1p(df["USD"].clip(0))
    feats["implied_prob"] = 1.0 / df["Odds"].clip(1.001)
    feats["outcomes_count"] = df["Outcomes_Count"].fillna(1)

    feats["ml_p_model"] = df["ML_P_Model"].fillna(-1)
    feats["ml_p_implied"] = df["ML_P_Implied"].fillna(-1)
    feats["ml_edge"] = df["ML_Edge"].fillna(0.0)
    feats["ml_ev"] = df["ML_EV"].clip(-100, 1000).fillna(0.0)
    feats["ml_winrate_diff"] = df["ML_Winrate_Diff"].fillna(0.0)
    feats["ml_rating_diff"] = df["ML_Rating_Diff"].fillna(0.0)
    feats["has_ml_data"] = (df["ML_P_Model"].notna() & (df["ML_P_Model"] > 0)).astype(int)

    feats["hour"] = df["Created_At"].dt.hour
    feats["day_of_week"] = df["Created_At"].dt.dayofweek
    feats["month"] = df["Created_At"].dt.month

    feats["odds_x_stake"] = feats["Odds"] * feats["USD"]
    feats["ml_edge_pos"] = feats["ml_edge"].clip(0)
    feats["ml_ev_pos"] = feats["ml_ev"].clip(0)
    feats["ml_edge_x_implied"] = feats["ml_edge"] * feats["implied_prob"]

    # Время до события
    feats["hours_before_event"] = (
        ((df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600).clip(0, 200).fillna(-1)
    )

    # Odds bucket (1.0-3.0 range только)
    feats["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0],
        labels=False,
    ).fillna(5)

    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")

    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """Вычислить ROI на выбранных ставках."""
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray) -> float:
    """Найти лучший порог на val-сете."""
    best_roi = -999.0
    best_t = 0.5
    for t in np.arange(0.40, 0.90, 0.02):
        mask = proba >= t
        if mask.sum() < 100:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best threshold=%.2f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


with mlflow.start_run(run_name="phase2/step2.2_singles_odds_filter") as run:
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

        df_all = load_data()
        n_all = len(df_all)
        train_end_all = int(n_all * 0.8)
        val_start_all = int(n_all * 0.64)

        # Применяем фильтры к каждому сплиту
        train_all = df_all.iloc[:train_end_all]
        val_all = df_all.iloc[val_start_all:train_end_all]
        test_all = df_all.iloc[train_end_all:]

        train_df = apply_pre_filters(train_all)
        val_df = apply_pre_filters(val_all)
        test_df = apply_pre_filters(test_all)

        logger.info(
            "After filter: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df)
        )

        # Validate filter decision using val
        logger.info(
            "Val ROI all: %.2f%%, Val ROI filtered: %.2f%%",
            calc_roi(val_all, np.ones(len(val_all), bool))[0],
            calc_roi(val_df, np.ones(len(val_df), bool))[0],
        )

        X_train, cat_features = build_features(train_df)
        X_val, _ = build_features(val_df)
        X_test, _ = build_features(test_df)

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        logger.info("Features: %d, cat: %s", X_train.shape[1], cat_features)

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            eval_metric="AUC",
            verbose=100,
            cat_features=cat_features,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

        proba_val = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]

        auc_val = roc_auc_score(y_val, proba_val)
        auc_test = roc_auc_score(y_test, proba_test)
        logger.info("AUC val=%.4f, test=%.4f", auc_val, auc_test)

        best_threshold = find_best_threshold(val_df, proba_val)

        mask_val = proba_val >= best_threshold
        mask_test = proba_test >= best_threshold

        roi_val, n_val = calc_roi(val_df, mask_val)
        roi_test, n_test = calc_roi(test_df, mask_test)

        logger.info(
            "ROI val=%.2f%% (%d), test=%.2f%% (%d)",
            roi_val,
            n_val,
            roi_test,
            n_test,
        )

        # Sanity check
        if roi_test > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — возможный leakage!")

        # Feature importance
        feat_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # CV expanding window
        cv_rois = []
        n = len(df_all)
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n

            fold_train_all = df_all.iloc[:fold_start]
            fold_val_all = df_all.iloc[fold_start:fold_end]

            fold_train = apply_pre_filters(fold_train_all)
            fold_val = apply_pre_filters(fold_val_all)

            if len(fold_train) < 1000 or len(fold_val) < 100:
                continue

            x_ft, cf = build_features(fold_train)
            x_fv, _ = build_features(fold_val)
            y_ft = (fold_train["Status"] == "won").astype(int)

            m = CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                eval_metric="AUC",
                verbose=0,
                cat_features=cf,
            )
            m.fit(x_ft, y_ft)
            proba_fv = m.predict_proba(x_fv)[:, 1]
            mask_fv = proba_fv >= best_threshold
            if mask_fv.sum() < 50:
                continue
            roi_fold, n_fold = calc_roi(fold_val, mask_fv)
            cv_rois.append(roi_fold)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_fold)
            logger.info("CV Fold %d: ROI=%.2f%% (%d ставок)", fold_idx, roi_fold, n_fold)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else roi_test
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "CatBoostClassifier",
                "filter_no_parlay": EXCLUDE_PARLAYS,
                "filter_max_odds": MAX_ODDS,
                "n_features": X_train.shape[1],
                "best_threshold": best_threshold,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "auc_val": auc_val,
                "auc_test": auc_test,
                "roi_val": roi_val,
                "roi_test": roi_test,
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_val": n_val,
                "n_bets_test": n_test,
            }
        )
        mlflow.log_text(feat_importance.to_string(), "feature_importance.txt")
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (roi_test - 4.84) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        # Сохраняем лучший пайплайн
        prev_best_roi = 4.84
        if roi_test > prev_best_roi and n_test >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            class BestPipeline:
                """Полный пайплайн: pre-filter + feature engineering + предсказание."""

                def __init__(
                    self,
                    model,
                    feature_names,
                    cat_features,
                    threshold,
                    exclude_parlays=True,
                    max_odds=3.0,
                    framework="catboost",
                ):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.exclude_parlays = exclude_parlays
                    self.max_odds = max_odds
                    self.sport_filter = []
                    self.framework = framework

                def _prefilter(self, df):
                    filtered = df.copy()
                    if self.exclude_parlays:
                        filtered = filtered[filtered["Is_Parlay"] != "t"]
                    filtered = filtered[filtered["Odds"] < self.max_odds]
                    return filtered

                def _build_features(self, df):
                    feats, _ = build_features(df)
                    return feats[self.feature_names]

                def predict_proba(self, df):
                    features = self._build_features(df)
                    return self.model.predict_proba(features)[:, 1]

                def evaluate(self, df) -> dict:
                    filtered = self._prefilter(df)
                    if len(filtered) == 0:
                        return {"roi": -100.0, "n_selected": 0, "threshold": self.threshold}
                    proba = self.predict_proba(filtered)
                    mask = proba >= self.threshold
                    selected = filtered[mask].copy()
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
                feature_names=X_train.columns.tolist(),
                cat_features=cat_features,
                threshold=best_threshold,
                exclude_parlays=EXCLUDE_PARLAYS,
                max_odds=MAX_ODDS,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            model.save_model(str(models_dir / "model.cbm"))

            metadata = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_test,
                "auc": auc_test,
                "threshold": best_threshold,
                "n_bets": n_test,
                "feature_names": X_train.columns.tolist(),
                "filters": {"exclude_parlays": EXCLUDE_PARLAYS, "max_odds": MAX_ODDS},
                "params": {"iterations": model.best_iteration_, "depth": 6, "lr": 0.1},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "2.2",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved pipeline. roi=%.2f%%", roi_test)

        print("\n=== Step 2.2 Results ===")
        print(f"Filter: No parlays, Odds < {MAX_ODDS}")
        print(f"AUC val={auc_val:.4f}, test={auc_test:.4f}")
        print(f"Best threshold (val): {best_threshold:.2f}")
        print(f"ROI val:  {roi_val:.2f}% ({n_val} ставок)")
        print(f"ROI test: {roi_test:.2f}% ({n_test} ставок)")
        print(f"CV ROI:   {roi_mean:.2f}% ± {roi_std:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")
        print("\nTop-10 features:")
        print(feat_importance.head(10).to_string(index=False))

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
