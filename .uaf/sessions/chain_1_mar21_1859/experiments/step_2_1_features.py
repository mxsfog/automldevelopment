"""Step 2.1 - Feature Engineering: rich market, odds, ELO, sport features.

Shadow feature trick: baseline vs enriched features.
Hypothesis: богатые market/odds/sport/ELO фичи улучшат ROI.
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


def load_data() -> pd.DataFrame:
    """Загрузка и объединение всех данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()

    # outcomes: первый leg каждого бета
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Selection", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    # Для парлаев: посчитать количество уникальных спортов (ставки на разные виды)
    outcomes_sport_count = (
        outcomes.groupby("Bet_ID")["Sport"].nunique().reset_index(name="n_sports_in_parlay")
    )
    outcomes_market_count = (
        outcomes.groupby("Bet_ID")["Market"].nunique().reset_index(name="n_markets_in_parlay")
    )

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df = df.merge(outcomes_sport_count, left_on="ID", right_on="Bet_ID", how="left")
    df = df.merge(outcomes_market_count, left_on="ID", right_on="Bet_ID", how="left")

    # ELO: sparse join по Bet_ID
    elo_home = (
        elo.groupby("Bet_ID")
        .agg(
            elo_team1=("Old_ELO", "first"),
            elo_team2=("Old_ELO", "last"),
            elo_change_team1=("ELO_Change", "first"),
        )
        .reset_index()
    )
    elo_home["elo_diff"] = (elo_home["elo_team1"] - elo_home["elo_team2"]).abs()
    elo_home["elo_sum"] = elo_home["elo_team1"] + elo_home["elo_team2"]
    elo_home["elo_has_data"] = 1
    elo_home = elo_home.rename(columns={"Bet_ID": "ELO_Bet_ID"})

    df = df.merge(elo_home, left_on="ID", right_on="ELO_Bet_ID", how="left")

    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    logger.info("Датасет: %d строк, %d колонок", len(df), df.shape[1])
    return df


def build_baseline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline фичи из step 1.4."""
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
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


def build_enriched_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Расширенные фичи Phase 2."""
    feats, cat_features = build_baseline_features(df)

    # --- Odds engineering ---
    # Odds buckets (ручные)
    feats["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 5.0, 1000.0],
        labels=False,
    ).fillna(7)

    # Bookmaker overround proxy (для single bets)
    feats["overround"] = feats["implied_prob"] - 0.5  # отклонение от fair 50/50
    feats["is_high_odds"] = (df["Odds"] >= 3.0).astype(int)
    feats["is_low_odds"] = (df["Odds"] < 1.5).astype(int)
    feats["is_medium_odds"] = ((df["Odds"] >= 1.5) & (df["Odds"] < 3.0)).astype(int)

    # --- ML edge engineering ---
    feats["ml_edge_sign"] = np.sign(feats["ml_edge"])
    feats["ml_edge_bucket"] = pd.cut(
        feats["ml_edge"], bins=[-100, -10, 0, 5, 10, 20, 100], labels=False
    ).fillna(0)
    feats["ml_ev_bucket"] = pd.cut(
        feats["ml_ev"], bins=[-100, -10, 0, 5, 10, 20, 100], labels=False
    ).fillna(0)
    feats["has_ml_data"] = (df["ML_P_Model"].notna() & (df["ML_P_Model"] > 0)).astype(int)

    # --- Parlay engineering ---
    feats["n_sports_in_parlay"] = df["n_sports_in_parlay"].fillna(1)
    feats["n_markets_in_parlay"] = df["n_markets_in_parlay"].fillna(1)
    feats["parlay_leg_count"] = df["Outcomes_Count"].fillna(1)
    feats["is_2leg_parlay"] = ((df["Is_Parlay"] == "t") & (df["Outcomes_Count"] == 2)).astype(int)
    feats["is_3plus_parlay"] = ((df["Is_Parlay"] == "t") & (df["Outcomes_Count"] >= 3)).astype(int)

    # --- ELO sparse features ---
    feats["elo_has_data"] = df["elo_has_data"].fillna(0).astype(int)
    feats["elo_diff"] = df["elo_diff"].fillna(0.0)
    feats["elo_sum"] = df["elo_sum"].fillna(3000.0)  # default 1500+1500
    feats["elo_change_team1"] = df["elo_change_team1"].fillna(0.0)
    # Нормализованный ELO diff (0 если данных нет)
    feats["elo_diff_norm"] = feats["elo_diff"] * feats["elo_has_data"]

    # --- Temporal features ---
    feats["days_since_start"] = (
        df["Created_At"] - df["Created_At"].min()
    ).dt.total_seconds() / 86400
    feats["week_of_data"] = (feats["days_since_start"] / 7).astype(int)

    # Время до начала события (если доступно)
    feats["hours_before_event"] = (
        ((df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600).clip(0, 200).fillna(-1)
    )
    feats["bet_before_event"] = (feats["hours_before_event"] >= 0).astype(int)

    # --- Market type encoding ---
    # Укрупнённая категория рынка
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Tournament"] = df["Tournament"].fillna("unknown")

    # Взаимодействия
    feats["ml_edge_x_implied"] = feats["ml_edge"] * feats["implied_prob"]
    feats["log_odds_x_edge"] = feats["log_odds"] * feats["ml_edge"].clip(-10, 10)
    feats["usd_log_x_odds"] = feats["log_usd"] * feats["log_odds"]

    # Stake size relative proxy (нормализованный rank)
    feats["usd_rank"] = pd.qcut(df["USD"], q=10, labels=False, duplicates="drop").fillna(5)

    cat_features_enriched = [*cat_features, "Tournament"]
    return feats, cat_features_enriched


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
        if mask.sum() < 200:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t


def run_cv(df, build_fn, best_threshold, n_folds=4) -> list[float]:
    """CV expanding window."""
    n = len(df)
    fold_size = n // 5
    cv_rois = []
    for fold_idx in range(1, n_folds + 1):
        fold_start = fold_idx * fold_size
        fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n

        fold_train = df.iloc[:fold_start]
        fold_val_cv = df.iloc[fold_start:fold_end]

        x_ft, cf = build_fn(fold_train)
        x_fv, _ = build_fn(fold_val_cv)
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
        roi_fold, _ = calc_roi(fold_val_cv, mask_fv)
        cv_rois.append(roi_fold)
        logger.info("CV Fold %d: ROI=%.2f%%", fold_idx, roi_fold)
    return cv_rois


with mlflow.start_run(run_name="phase2/step2.1_features") as run:
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

        logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        # === BASELINE (шаг 1.4) ===
        logger.info("--- Baseline ---")
        X_train_b, cat_b = build_baseline_features(train_df)
        X_val_b, _ = build_baseline_features(val_df)
        X_test_b, _ = build_baseline_features(test_df)
        y_train = (train_df["Status"] == "won").astype(int)

        model_b = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            eval_metric="AUC",
            verbose=100,
            cat_features=cat_b,
        )
        model_b.fit(X_train_b, y_train, eval_set=(X_val_b, y_val), early_stopping_rounds=50)
        proba_val_b = model_b.predict_proba(X_val_b)[:, 1]
        proba_test_b = model_b.predict_proba(X_test_b)[:, 1]
        auc_val_b = roc_auc_score(y_val, proba_val_b)
        threshold_b = find_best_threshold(val_df, proba_val_b)
        roi_val_b, _ = calc_roi(val_df, proba_val_b >= threshold_b)
        roi_test_b, n_test_b = calc_roi(test_df, proba_test_b >= threshold_b)
        logger.info(
            "BASELINE: AUC val=%.4f, ROI val=%.2f%%, test=%.2f%% (%d ставок)",
            auc_val_b,
            roi_val_b,
            roi_test_b,
            n_test_b,
        )

        # === ENRICHED ===
        logger.info("--- Enriched ---")
        X_train_e, cat_e = build_enriched_features(train_df)
        X_val_e, _ = build_enriched_features(val_df)
        X_test_e, _ = build_enriched_features(test_df)

        logger.info("Enriched features: %d (baseline: %d)", X_train_e.shape[1], X_train_b.shape[1])

        model_e = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            eval_metric="AUC",
            verbose=100,
            cat_features=cat_e,
        )
        model_e.fit(X_train_e, y_train, eval_set=(X_val_e, y_val), early_stopping_rounds=50)
        proba_val_e = model_e.predict_proba(X_val_e)[:, 1]
        proba_test_e = model_e.predict_proba(X_test_e)[:, 1]
        auc_val_e = roc_auc_score(y_val, proba_val_e)
        threshold_e = find_best_threshold(val_df, proba_val_e)
        roi_val_e, _ = calc_roi(val_df, proba_val_e >= threshold_e)
        roi_test_e, n_test_e = calc_roi(test_df, proba_test_e >= threshold_e)
        logger.info(
            "ENRICHED: AUC val=%.4f, ROI val=%.2f%%, test=%.2f%% (%d ставок)",
            auc_val_e,
            roi_val_e,
            roi_test_e,
            n_test_e,
        )

        # Shadow feature trick: delta
        delta_roi = roi_test_e - roi_test_b
        delta_auc = auc_val_e - auc_val_b
        logger.info("Delta ROI (test): %.2f%%, Delta AUC: %.4f", delta_roi, delta_auc)

        # Feature importance enriched
        feat_importance_e = pd.DataFrame(
            {"feature": X_train_e.columns, "importance": model_e.feature_importances_}
        ).sort_values("importance", ascending=False)
        logger.info("Top-15 enriched features:\n%s", feat_importance_e.head(15).to_string())

        # CV для enriched
        logger.info("CV для enriched модели...")
        cv_rois_e = run_cv(df, build_enriched_features, threshold_e)
        roi_mean_e = float(np.mean(cv_rois_e)) if cv_rois_e else roi_test_e
        roi_std_e = float(np.std(cv_rois_e)) if cv_rois_e else 0.0
        for i, r in enumerate(cv_rois_e):
            mlflow.log_metric(f"roi_fold_{i}", r)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "model": "CatBoostClassifier",
                "n_features_baseline": X_train_b.shape[1],
                "n_features_enriched": X_train_e.shape[1],
                "threshold_enriched": threshold_e,
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "auc_val_baseline": auc_val_b,
                "auc_val_enriched": auc_val_e,
                "roi_test_baseline": roi_test_b,
                "roi_test_enriched": roi_test_e,
                "roi_val_baseline": roi_val_b,
                "roi_val_enriched": roi_val_e,
                "roi_mean": roi_mean_e,
                "roi_std": roi_std_e,
                "delta_roi_test": delta_roi,
                "delta_auc_val": delta_auc,
                "n_bets_test": n_test_e,
            }
        )
        mlflow.log_text(feat_importance_e.to_string(), "feature_importance_enriched.txt")
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")

        # Решение по shadow feature trick
        if delta_roi > 0.002:
            mlflow.set_tag("feature_decision", "accepted")
            logger.info("Enriched features ПРИНЯТЫ (delta=%.2f%%)", delta_roi)
        elif delta_roi > 0:
            mlflow.set_tag("feature_decision", "marginal")
            logger.info("Enriched features MARGINALLY accepted (delta=%.2f%%)", delta_roi)
        else:
            mlflow.set_tag("feature_decision", "rejected")
            logger.info("Enriched features ОТКЛОНЕНЫ (delta=%.2f%%)", delta_roi)

        convergence = min(1.0, max(0.0, (roi_test_e - 4.84) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        # Сохраняем лучший пайплайн если улучшение
        best_roi = 4.84  # из step 1.4
        if roi_test_e > best_roi and n_test_e >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            class BestPipeline:
                """Полный пайплайн: feature engineering + предсказание + оценка."""

                def __init__(
                    self, model, feature_names, cat_features, threshold, framework="catboost"
                ):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.sport_filter = []
                    self.framework = framework

                def _build_features(self, df):
                    feats, _ = build_enriched_features(df)
                    return feats[self.feature_names]

                def predict_proba(self, df):
                    features = self._build_features(df)
                    return self.model.predict_proba(features)[:, 1]

                def evaluate(self, df) -> dict:
                    if self.sport_filter:
                        df = df[~df["Sport"].isin(self.sport_filter)].copy()
                    proba = self.predict_proba(df)
                    mask = proba >= self.threshold
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
                model=model_e,
                feature_names=X_train_e.columns.tolist(),
                cat_features=cat_e,
                threshold=threshold_e,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            model_e.save_model(str(models_dir / "model.cbm"))

            metadata = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_test_e,
                "auc": auc_val_e,
                "threshold": threshold_e,
                "n_bets": n_test_e,
                "feature_names": X_train_e.columns.tolist(),
                "params": {"iterations": model_e.best_iteration_, "depth": 6, "lr": 0.1},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "2.1",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved pipeline.pkl. roi=%.2f%%", roi_test_e)

        print("\n=== Step 2.1 Results ===")
        print(f"BASELINE: AUC={auc_val_b:.4f}, ROI test={roi_test_b:.2f}% ({n_test_b} ставок)")
        print(f"ENRICHED: AUC={auc_val_e:.4f}, ROI test={roi_test_e:.2f}% ({n_test_e} ставок)")
        print(f"Delta ROI: {delta_roi:+.2f}%, Delta AUC: {delta_auc:+.4f}")
        print(f"CV ROI: {roi_mean_e:.2f}% ± {roi_std_e:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")
        print("\nTop-15 enriched features:")
        print(feat_importance_e.head(15).to_string(index=False))

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
