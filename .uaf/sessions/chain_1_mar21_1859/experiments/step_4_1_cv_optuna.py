"""Step 4.1 - Optuna с CV mean ROI как objective.

Hypothesis: Оптимизация по CV mean ROI (3 фолда) даст более стабильную модель
по сравнению с оптимизацией по val ROI одного периода.
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
N_TRIALS = 25


def classify_market(m: str) -> str:
    """Группировка рынков."""
    if pd.isna(m):
        return "unknown"
    m_l = m.lower()
    if "asian" in m_l and "total" in m_l:
        return "asian_total"
    if "asian" in m_l and "handicap" in m_l:
        return "asian_handicap"
    if "map" in m_l and "handicap" in m_l:
        return "map_handicap"
    if "map" in m_l and "winner" in m_l:
        return "map_winner"
    if "set" in m_l and "winner" in m_l:
        return "set_winner"
    if "set" in m_l:
        return "set_other"
    if "winner" in m_l and "twoway" in m_l.replace("-", "").replace(" ", ""):
        return "winner_2way"
    if "winner" in m_l and "threeway" in m_l.replace("-", "").replace(" ", ""):
        return "winner_3way"
    if "1x2" in m_l:
        return "match_result_3way"
    if "double chance" in m_l:
        return "double_chance"
    if "draw no bet" in m_l:
        return "draw_no_bet"
    if "both teams" in m_l:
        return "btts"
    if "handicap" in m_l:
        return "handicap"
    if "total" in m_l:
        return "total"
    if "innings" in m_l:
        return "innings"
    if "winner" in m_l:
        return "winner"
    if "half" in m_l:
        return "half_market"
    return "other"


def load_data() -> pd.DataFrame:
    """Загрузка данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()
    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    df = bets.merge(
        outcomes_first[["Bet_ID", "Sport", "Market", "Start_Time"]],
        left_on="ID",
        right_on="Bet_ID",
        how="left",
    )
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)
    df["market_type"] = df["Market"].apply(classify_market)
    return df


def build_features(df: pd.DataFrame, top_200_markets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Feature set v3b."""
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
    feats["hours_before_event"] = (
        ((df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600).clip(0, 200).fillna(-1)
    )
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["market_type"] = df["market_type"]
    feats["Currency"] = df["Currency"].fillna("unknown")
    top_200_set = set(top_200_markets)
    feats["Market_top200"] = df["Market"].apply(
        lambda x: x if (not pd.isna(x) and x in top_200_set) else "other"
    )
    cat_features = ["Sport", "market_type", "Currency", "Market_top200"]
    return feats, cat_features


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """ROI."""
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 200) -> float:
    """Лучший порог."""
    best_roi, best_t = -999.0, 0.5
    for t in np.arange(0.40, 0.90, 0.02):
        mask = proba >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t


with mlflow.start_run(run_name="phase4/step4.1_cv_optuna") as run:
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
        # CV используем 3 последних фолда из train (expanding window)
        # Фолды: [0-20%], [0-40%], [0-60%], [0-80%] train + соответствующий val
        # Для Optuna берём фолды 2, 3, 4 (обучение на достаточно данных)

        # Для финального обучения
        train_df = df.iloc[:train_end]
        val_start = int(n * 0.64)
        val_df = df.iloc[val_start:train_end]
        test_df = df.iloc[train_end:]

        # Топ-200 рынков (из всего train)
        top_200_markets = train_df["Market"].value_counts().head(200).index.tolist()

        logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

        # Подготовим CV fold splits внутри train
        fold_size = n // 5
        cv_folds = []
        for fi in range(2, 5):  # фолды 2, 3, 4 (достаточно данных для обучения)
            fi_start = fi * fold_size
            fi_end = (fi + 1) * fold_size if fi < 4 else train_end
            fold_train_data = df.iloc[:fi_start]
            fold_val_data = df.iloc[fi_start:fi_end]
            top_m_fi = fold_train_data["Market"].value_counts().head(200).index.tolist()
            x_ft, cf = build_features(fold_train_data, top_m_fi)
            x_fv, _ = build_features(fold_val_data, top_m_fi)
            y_ft = (fold_train_data["Status"] == "won").astype(int)
            cv_folds.append((x_ft, y_ft, x_fv, fold_val_data, cf, top_m_fi))
        logger.info("CV folds prepared: %d", len(cv_folds))

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective: средний CV ROI по 3 фолдам."""
            params = {
                "iterations": trial.suggest_int("iterations", 200, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                "depth": trial.suggest_int("depth", 4, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 20.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 2.0),
                "random_strength": trial.suggest_float("random_strength", 0.5, 3.0),
            }

            fold_rois = []
            for x_ft, y_ft, x_fv, fv_df, cf, _ in cv_folds:
                m = CatBoostClassifier(
                    **params,
                    random_seed=42,
                    eval_metric="AUC",
                    verbose=0,
                    cat_features=cf,
                )
                m.fit(x_ft, y_ft)
                pf = m.predict_proba(x_fv)[:, 1]
                # Порог выбираем на самом fold val (допустимо т.к. это внутри CV)
                t_best = find_best_threshold(fv_df, pf, min_bets=100)
                mask_f = pf >= t_best
                roi_f, n_f = calc_roi(fv_df, mask_f)
                if n_f < 100:
                    fold_rois.append(-100.0)
                else:
                    fold_rois.append(roi_f)

            cv_mean = float(np.mean(fold_rois))
            trial.set_user_attr("cv_rois", fold_rois)
            return cv_mean

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_trial = study.best_trial
        logger.info("Best trial: CV mean ROI=%.2f%%", best_trial.value)
        logger.info("CV fold ROIs: %s", best_trial.user_attrs.get("cv_rois", []))
        logger.info("Best params: %s", best_trial.params)

        # Финальное обучение с лучшими params
        best_params = best_trial.params
        x_train, cat_features = build_features(train_df, top_200_markets)
        x_val, _ = build_features(val_df, top_200_markets)
        x_test, _ = build_features(test_df, top_200_markets)

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        final_model = CatBoostClassifier(
            **best_params,
            random_seed=42,
            eval_metric="AUC",
            verbose=50,
            cat_features=cat_features,
            od_type="Iter",
            od_wait=30,
        )
        final_model.fit(x_train, y_train, eval_set=(x_val, y_val))

        proba_val = final_model.predict_proba(x_val)[:, 1]
        proba_test = final_model.predict_proba(x_test)[:, 1]

        auc_val = roc_auc_score(y_val, proba_val)
        auc_test = roc_auc_score(y_test, proba_test)

        best_threshold = find_best_threshold(val_df, proba_val, min_bets=200)

        mask_val = proba_val >= best_threshold
        mask_test = proba_test >= best_threshold

        roi_val, n_val = calc_roi(val_df, mask_val)
        roi_test, n_test = calc_roi(test_df, mask_test)

        logger.info(
            "Final: AUC val=%.4f test=%.4f, ROI val=%.2f%% (%d), test=%.2f%% (%d)",
            auc_val,
            auc_test,
            roi_val,
            n_val,
            roi_test,
            n_test,
        )

        if roi_test > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — leakage!")

        # CV на финальной модели
        cv_rois = []
        for fi in range(1, 5):
            fi_start = fi * fold_size
            fi_end = (fi + 1) * fold_size if fi < 4 else n

            fold_train = df.iloc[:fi_start]
            fold_val_cv = df.iloc[fi_start:fi_end]
            top_m_cv = fold_train["Market"].value_counts().head(200).index.tolist()

            x_ft, cf = build_features(fold_train, top_m_cv)
            x_fv, _ = build_features(fold_val_cv, top_m_cv)
            y_ft = (fold_train["Status"] == "won").astype(int)

            cm = CatBoostClassifier(**best_params, random_seed=42, verbose=0, cat_features=cf)
            cm.fit(x_ft, y_ft)
            pf = cm.predict_proba(x_fv)[:, 1]
            mask_f = pf >= best_threshold
            if mask_f.sum() < 50:
                continue
            roi_f, n_f = calc_roi(fold_val_cv, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fi}", roi_f)
            logger.info("CV Fold %d: ROI=%.2f%% (%d)", fi, roi_f, n_f)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else roi_test
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        feat_importance = pd.DataFrame(
            {"feature": x_train.columns, "importance": final_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        mlflow.log_params(
            {
                "validation_scheme": "time_series_cv",
                "seed": 42,
                "n_trials": N_TRIALS,
                "n_cv_folds": len(cv_folds),
                "best_threshold": best_threshold,
                **{f"best_{k}": v for k, v in best_params.items()},
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
                "optuna_best_cv_roi": best_trial.value,
            }
        )
        mlflow.log_text(feat_importance.to_string(), "feature_importance.txt")
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (roi_test - 5.09) / 8.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        prev_best = 5.09
        if roi_test > prev_best and n_test >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            t200 = top_200_markets
            fm = final_model
            bt = best_threshold

            class BestPipeline:
                """Полный пайплайн."""

                def __init__(
                    self,
                    model,
                    feature_names,
                    cat_features,
                    threshold,
                    top_200_markets,
                    framework="catboost",
                ):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.top_200_markets = top_200_markets
                    self.sport_filter = []
                    self.framework = framework

                def predict_proba(self, df):
                    feats, _ = build_features(df, self.top_200_markets)
                    return self.model.predict_proba(feats[self.feature_names])[:, 1]

                def evaluate(self, df) -> dict:
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
                model=fm,
                feature_names=x_train.columns.tolist(),
                cat_features=cat_features,
                threshold=bt,
                top_200_markets=t200,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            fm.save_model(str(models_dir / "model.cbm"))
            metadata = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi_test,
                "auc": auc_val,
                "threshold": bt,
                "n_bets": n_test,
                "feature_names": x_train.columns.tolist(),
                "best_params": best_params,
                "session_id": SESSION_ID,
                "step": "4.1",
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved pipeline. roi=%.2f%%", roi_test)

        print("\n=== Step 4.1 CV Optuna Results ===")
        print(f"Best CV mean ROI: {best_trial.value:.2f}%")
        print(f"CV fold ROIs: {best_trial.user_attrs.get('cv_rois', [])}")
        print(f"Best params: {best_params}")
        print(f"Threshold: {best_threshold:.2f}")
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
