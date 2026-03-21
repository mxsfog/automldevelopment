"""Step 4.1 — Optuna с CV-objective вместо val ROI.

Гипотеза: использование CV mean ROI как objective Optuna
вместо single val ROI устранит переобучение на val
и найдёт более генерализируемые гиперпараметры.
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
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

N_TRIALS = 25
N_CV_FOLDS = 3  # быстрее чем 5


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


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 150) -> float:
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


with mlflow.start_run(run_name="phase4/step4.1_optuna_cv") as run:
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

        # Train (80%) — для финальной модели
        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:train_end]
        test_df = df.iloc[train_end:]

        # Для CV objective — используем первые 80% без last 20%
        cv_data = df.iloc[:train_end]
        n_cv = len(cv_data)

        X_train, cat_features = build_features(train_df)
        X_val, _ = build_features(val_df)
        X_test, _ = build_features(test_df)

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        logger.info(
            "Data: train=%d val=%d test=%d, starting CV-Optuna...",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # Порог от шага 2.1 как фиксированный ориентир
        FIXED_THRESHOLD = 0.68

        def cv_objective(trial: optuna.Trial) -> float:
            depth = trial.suggest_int("depth", 5, 8)
            lr = trial.suggest_float("lr", 0.03, 0.15, log=True)
            iterations = trial.suggest_int("iterations", 300, 600)
            half_life = trial.suggest_float("half_life", 0.3, 0.7)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 8.0)

            fold_rois = []
            fold_size = n_cv // N_CV_FOLDS
            for fold_idx in range(1, N_CV_FOLDS + 1):
                fold_start = fold_idx * fold_size
                fold_end = (fold_idx + 1) * fold_size if fold_idx < N_CV_FOLDS else n_cv
                fold_train = cv_data.iloc[:fold_start]
                fold_val_cv = cv_data.iloc[fold_start:fold_end]
                if len(fold_val_cv) < 200:
                    continue
                xft, cf = build_features(fold_train)
                xfv, _ = build_features(fold_val_cv)
                yft = (fold_train["Status"] == "won").astype(int)
                sw = make_sample_weights(len(fold_train), half_life)
                m = CatBoostClassifier(
                    iterations=iterations,
                    learning_rate=lr,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    random_seed=42,
                    verbose=0,
                    cat_features=cf,
                )
                m.fit(xft, yft, sample_weight=sw)
                pf = m.predict_proba(xfv)[:, 1]
                # Порог на fold_val (предотвращаем leakage — fold_val не входит в test)
                t = find_best_threshold(fold_val_cv, pf, min_bets=100)
                roi_f, n_f = calc_roi(fold_val_cv, pf >= t)
                if n_f >= 50:
                    fold_rois.append(roi_f)

            if not fold_rois:
                return -100.0
            return float(np.mean(fold_rois))

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(cv_objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        best_cv_roi = study.best_value
        logger.info("Best CV ROI: %.2f%%  params: %s", best_cv_roi, best_params)

        # Финальная модель с лучшими CV-параметрами
        sw_final = make_sample_weights(len(train_df), best_params["half_life"])
        final_model = CatBoostClassifier(
            iterations=best_params["iterations"],
            learning_rate=best_params["lr"],
            depth=best_params["depth"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            random_seed=42,
            eval_metric="AUC",
            verbose=0,
            cat_features=cat_features,
        )
        final_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            sample_weight=sw_final,
        )

        pv = final_model.predict_proba(X_val)[:, 1]
        pt = final_model.predict_proba(X_test)[:, 1]

        auc_v = roc_auc_score(y_val, pv)
        auc_t = roc_auc_score(y_test, pt)

        threshold = find_best_threshold(val_df, pv)
        roi_val, n_val = calc_roi(val_df, pv >= threshold)
        roi_test, n_test = calc_roi(test_df, pt >= threshold)

        logger.info("Final: ROI val=%.2f%% test=%.2f%%", roi_val, roi_test)

        if roi_test > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — возможный leakage!")

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_trials": N_TRIALS,
                "n_cv_folds": N_CV_FOLDS,
                "objective": "cv_mean_roi",
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                **{f"best_{k}": v for k, v in best_params.items()},
                "threshold": threshold,
            }
        )
        mlflow.log_metrics(
            {
                "roi_test": roi_test,
                "roi_val": roi_val,
                "roi_cv_best": best_cv_roi,
                "auc_val": auc_v,
                "auc_test": auc_t,
                "n_bets_test": n_test,
                "delta_vs_baseline": roi_test - PREV_BEST_ROI,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (roi_test - PREV_BEST_ROI) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        # Сохранение если улучшение
        if roi_test > PREV_BEST_ROI and n_test >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)
            feats_list = X_train.columns.tolist()

            class BestPipeline:
                """CV-Optuna CatBoost pipeline."""

                def __init__(
                    self, model, feature_names, cat_features, threshold, framework="catboost"
                ):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
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
                model=final_model,
                feature_names=feats_list,
                cat_features=cat_features,
                threshold=threshold,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            final_model.save_model(str(models_dir / "model.cbm"))
            metadata = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_test,
                "auc": auc_t,
                "threshold": threshold,
                "n_bets": n_test,
                "feature_names": feats_list,
                "params": {**best_params},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "4.1",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Pipeline saved. ROI=%.2f%%", roi_test)

        print(f"\n=== Step 4.1 Optuna CV-objective ({N_TRIALS} trials) ===")
        print(f"Best CV ROI: {best_cv_roi:.2f}%")
        print(f"Best params: {best_params}")
        print(f"Final: ROI val={roi_val:.2f}%, test={roi_test:.2f}% ({n_test})")
        print(f"AUC val/test: {auc_v:.4f}/{auc_t:.4f}")
        print(f"Prev best: {PREV_BEST_ROI:.2f}%")
        print(f"Delta: {roi_test - PREV_BEST_ROI:+.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
