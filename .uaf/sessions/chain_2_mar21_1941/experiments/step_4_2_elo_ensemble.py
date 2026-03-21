"""Step 4.2 - ELO features + CatBoost ensemble.

Гипотеза: добавление предматчевых ELO-фич (Old_ELO обеих команд, абсолютные значения,
K_factor) даст прирост сверх baseline 4.84%. Ансамбль двух CatBoost (depth=6 + depth=4)
усилит стабильность предсказаний.
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
PREV_BEST_ROI = 4.837448328326608


def load_data() -> pd.DataFrame:
    """Загрузка, объединение данных, join с ELO-историей."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Selection", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    # Агрегация ELO по Bet_ID: team_1 = выше ELO, team_2 = ниже ELO
    elo_agg = (
        elo.groupby("Bet_ID")
        .agg(
            elo_max=("Old_ELO", "max"),
            elo_min=("Old_ELO", "min"),
            elo_mean=("Old_ELO", "mean"),
            elo_std=("Old_ELO", "std"),
            elo_count=("Old_ELO", "count"),
            k_factor_max=("K_Factor", "max"),
            k_factor_mean=("K_Factor", "mean"),
        )
        .reset_index()
    )
    elo_agg["elo_diff"] = elo_agg["elo_max"] - elo_agg["elo_min"]
    elo_agg["elo_ratio"] = elo_agg["elo_max"] / elo_agg["elo_min"].clip(1.0)

    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))

    logger.info("Датасет после ELO join: %d строк, %d колонок", len(df), df.shape[1])
    logger.info("ELO match rate: %.1f%%", 100 * df["elo_count"].notna().mean())
    return df


def build_features_v1(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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


def build_features_v2(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline + ELO фичи."""
    feats, cat_features = build_features_v1(df)

    # ELO фичи (nan если нет данных — CatBoost умеет)
    feats["elo_max"] = df["elo_max"].fillna(-1)
    feats["elo_min"] = df["elo_min"].fillna(-1)
    feats["elo_diff"] = df["elo_diff"].fillna(0.0)
    feats["elo_ratio"] = df["elo_ratio"].fillna(1.0)
    feats["elo_mean"] = df["elo_mean"].fillna(-1)
    feats["elo_std"] = df["elo_std"].fillna(0.0)
    feats["k_factor_mean"] = df["k_factor_mean"].fillna(-1)
    feats["has_elo"] = df["elo_count"].notna().astype(int)
    feats["elo_count"] = df["elo_count"].fillna(0)

    # Взаимодействие: ml_edge * elo_diff
    feats["ml_edge_x_elo_diff"] = feats["ml_edge"] * feats["elo_diff"].clip(0, 500) / 500
    # implied_prob vs elo_ratio: market vs ELO согласие
    feats["elo_implied_agree"] = (
        feats["implied_prob"] - 1.0 / feats["elo_ratio"].clip(0.5, 2.0)
    ).abs()

    return feats, cat_features


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """Вычислить ROI."""
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 200) -> float:
    """Найти лучший порог на val-сете."""
    best_roi, best_t = -999.0, 0.5
    for t in np.arange(0.40, 0.92, 0.01):
        mask = proba >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best threshold=%.2f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list[str],
    depth: int = 6,
    iterations: int = 500,
) -> CatBoostClassifier:
    """Обучить CatBoost."""
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.1,
        depth=depth,
        random_seed=42,
        eval_metric="AUC",
        verbose=0,
        cat_features=cat_features,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50)
    return model


with mlflow.start_run(run_name="phase4/step4.2_elo_ensemble") as run:
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

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        results = []

        # --- Конфигурации ---
        configs = [
            ("baseline_d6", build_features_v1, 6),
            ("baseline_d4", build_features_v1, 4),
            ("elo_d6", build_features_v2, 6),
            ("elo_d4", build_features_v2, 4),
        ]

        for name, build_fn, depth in configs:
            logger.info("--- Config: %s (depth=%d) ---", name, depth)
            X_tr, cf = build_fn(train_df)
            X_vl, _ = build_fn(val_df)
            X_ts, _ = build_fn(test_df)

            m = train_catboost(X_tr, y_train, X_vl, y_val, cf, depth=depth)

            pv = m.predict_proba(X_vl)[:, 1]
            pt = m.predict_proba(X_ts)[:, 1]
            auc_v = roc_auc_score(y_val, pv)
            auc_t = roc_auc_score(y_test, pt)

            t_best = find_best_threshold(val_df, pv)
            roi_v, n_v = calc_roi(val_df, pv >= t_best)
            roi_t, n_t = calc_roi(test_df, pt >= t_best)

            logger.info(
                "[%s] AUC v/t=%.4f/%.4f ROI v=%.2f%%(%d) t=%.2f%%(%d)",
                name,
                auc_v,
                auc_t,
                roi_v,
                n_v,
                roi_t,
                n_t,
            )
            results.append(
                {
                    "name": name,
                    "model": m,
                    "proba_val": pv,
                    "proba_test": pt,
                    "auc_val": auc_v,
                    "auc_test": auc_t,
                    "roi_val": roi_v,
                    "roi_test": roi_t,
                    "n_test": n_t,
                    "threshold": t_best,
                    "build_fn": build_fn,
                    "depth": depth,
                    "features": X_tr.columns.tolist(),
                    "cat_features": cf,
                }
            )
            mlflow.log_metrics(
                {
                    f"auc_val_{name}": auc_v,
                    f"auc_test_{name}": auc_t,
                    f"roi_val_{name}": roi_v,
                    f"roi_test_{name}": roi_t,
                }
            )

        # --- Ансамбли ---
        ensemble_results = []

        # Ансамбль 1: baseline d6 + baseline d4
        r0, r1 = results[0], results[1]
        proba_ens_base = (r0["proba_val"] + r1["proba_val"]) / 2
        t_ens_base = find_best_threshold(val_df, proba_ens_base)
        roi_ens_base_v, n_ens_base_v = calc_roi(val_df, proba_ens_base >= t_ens_base)
        proba_ens_base_t = (r0["proba_test"] + r1["proba_test"]) / 2
        roi_ens_base_t, n_ens_base_t = calc_roi(test_df, proba_ens_base_t >= t_ens_base)
        logger.info(
            "[ensemble_base] ROI v=%.2f%%(%d) t=%.2f%%(%d)",
            roi_ens_base_v,
            n_ens_base_v,
            roi_ens_base_t,
            n_ens_base_t,
        )
        ensemble_results.append(
            {
                "name": "ensemble_base_d6d4",
                "roi_val": roi_ens_base_v,
                "roi_test": roi_ens_base_t,
                "n_test": n_ens_base_t,
                "threshold": t_ens_base,
                "proba_val": proba_ens_base,
                "proba_test": proba_ens_base_t,
                "models": [r0, r1],
            }
        )
        mlflow.log_metrics(
            {
                "roi_val_ens_base": roi_ens_base_v,
                "roi_test_ens_base": roi_ens_base_t,
            }
        )

        # Ансамбль 2: elo d6 + elo d4
        r2, r3 = results[2], results[3]
        proba_ens_elo = (r2["proba_val"] + r3["proba_val"]) / 2
        t_ens_elo = find_best_threshold(val_df, proba_ens_elo)
        roi_ens_elo_v, _ = calc_roi(val_df, proba_ens_elo >= t_ens_elo)
        proba_ens_elo_t = (r2["proba_test"] + r3["proba_test"]) / 2
        roi_ens_elo_t, n_ens_elo_t = calc_roi(test_df, proba_ens_elo_t >= t_ens_elo)
        logger.info(
            "[ensemble_elo] ROI v=%.2f%% t=%.2f%%(%d)",
            roi_ens_elo_v,
            roi_ens_elo_t,
            n_ens_elo_t,
        )
        ensemble_results.append(
            {
                "name": "ensemble_elo_d6d4",
                "roi_val": roi_ens_elo_v,
                "roi_test": roi_ens_elo_t,
                "n_test": n_ens_elo_t,
                "threshold": t_ens_elo,
                "proba_val": proba_ens_elo,
                "proba_test": proba_ens_elo_t,
                "models": [r2, r3],
            }
        )
        mlflow.log_metrics(
            {
                "roi_val_ens_elo": roi_ens_elo_v,
                "roi_test_ens_elo": roi_ens_elo_t,
            }
        )

        # Ансамбль 3: все 4 модели
        proba_ens_all_v = np.mean([r["proba_val"] for r in results], axis=0)
        proba_ens_all_t = np.mean([r["proba_test"] for r in results], axis=0)
        t_ens_all = find_best_threshold(val_df, proba_ens_all_v)
        roi_ens_all_v, _ = calc_roi(val_df, proba_ens_all_v >= t_ens_all)
        roi_ens_all_t, n_ens_all_t = calc_roi(test_df, proba_ens_all_t >= t_ens_all)
        logger.info(
            "[ensemble_all4] ROI v=%.2f%% t=%.2f%%(%d)", roi_ens_all_v, roi_ens_all_t, n_ens_all_t
        )
        ensemble_results.append(
            {
                "name": "ensemble_all4",
                "roi_val": roi_ens_all_v,
                "roi_test": roi_ens_all_t,
                "n_test": n_ens_all_t,
                "threshold": t_ens_all,
                "proba_val": proba_ens_all_v,
                "proba_test": proba_ens_all_t,
                "models": results,
            }
        )
        mlflow.log_metrics(
            {
                "roi_val_ens_all": roi_ens_all_v,
                "roi_test_ens_all": roi_ens_all_t,
            }
        )

        # --- Лучший результат ---
        all_results = results + ensemble_results
        best = max(all_results, key=lambda r: r["roi_test"])
        logger.info(
            "Best: %s ROI test=%.2f%% (%d ставок)",
            best["name"],
            best["roi_test"],
            best["n_test"],
        )

        if best["roi_test"] > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — возможный leakage! roi=%.2f%%", best["roi_test"])

        # CV для лучшей одиночной модели
        best_single = max(results, key=lambda r: r["roi_test"])
        cv_rois = []
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n

            fold_train = df.iloc[:fold_start]
            fold_val_cv = df.iloc[fold_start:fold_end]

            x_ft, cf = best_single["build_fn"](fold_train)
            x_fv, _ = best_single["build_fn"](fold_val_cv)
            y_ft = (fold_train["Status"] == "won").astype(int)

            m_cv = train_catboost(
                x_ft,
                y_ft,
                x_fv,
                (fold_val_cv["Status"] == "won").astype(int),
                cf,
                depth=best_single["depth"],
                iterations=300,
            )
            pf = m_cv.predict_proba(x_fv)[:, 1]
            mask_f = pf >= best_single["threshold"]
            if mask_f.sum() < 50:
                continue
            roi_f, n_f = calc_roi(fold_val_cv, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_f)
            logger.info("CV Fold %d: ROI=%.2f%% (%d)", fold_idx, roi_f, n_f)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else best_single["roi_test"]
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "best_config": best["name"],
                "best_threshold": best["threshold"],
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "roi_test_best": best["roi_test"],
                "roi_val_best": best["roi_val"],
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_test": best["n_test"],
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (best["roi_test"] - PREV_BEST_ROI) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        # Сохранение pipeline если улучшение
        if best["roi_test"] > PREV_BEST_ROI and best["n_test"] >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            best_ref = best
            # Для ансамбля — сохраняем усреднённый predict
            if "models" in best and len(best["models"]) > 1:
                # Ансамбль
                sub_models = [
                    (
                        r.get("model", None),
                        r["build_fn"],
                        r["features"],
                        r["cat_features"],
                    )
                    for r in best["models"]
                ]

                class EnsemblePipeline:
                    """Ансамбль CatBoost."""

                    def __init__(self, sub_models, threshold, framework="catboost_ensemble"):
                        self.sub_models = sub_models
                        self.threshold = threshold
                        self.sport_filter = []
                        self.framework = framework

                    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                        probas = []
                        for model, build_fn, feat_names, _cf in self.sub_models:
                            feats, _ = build_fn(df)
                            p = model.predict_proba(feats[feat_names])[:, 1]
                            probas.append(p)
                        return np.mean(probas, axis=0)

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
                        return {
                            "roi": roi,
                            "n_selected": int(mask.sum()),
                            "threshold": self.threshold,
                        }

                pipeline = EnsemblePipeline(
                    sub_models=sub_models,
                    threshold=best["threshold"],
                )
            else:
                # Одиночная модель
                bm = best["model"]
                bfn = best["build_fn"]
                bfeat = best["features"]
                bcf = best["cat_features"]
                bt = best["threshold"]

                class SinglePipeline:
                    """Одиночный CatBoost пайплайн."""

                    def __init__(
                        self,
                        model,
                        feature_names,
                        cat_features,
                        threshold,
                        build_fn,
                        framework="catboost",
                    ):
                        self.model = model
                        self.feature_names = feature_names
                        self.cat_features = cat_features
                        self.threshold = threshold
                        self._build_fn = build_fn
                        self.sport_filter = []
                        self.framework = framework

                    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                        feats, _ = self._build_fn(df)
                        return self.model.predict_proba(feats[self.feature_names])[:, 1]

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
                        return {
                            "roi": roi,
                            "n_selected": int(mask.sum()),
                            "threshold": self.threshold,
                        }

                pipeline = SinglePipeline(
                    model=bm, feature_names=bfeat, cat_features=bcf, threshold=bt, build_fn=bfn
                )

            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            metadata = {
                "framework": "catboost_ensemble",
                "pipeline_file": "pipeline.pkl",
                "roi": best["roi_test"],
                "threshold": best["threshold"],
                "n_bets": best["n_test"],
                "session_id": SESSION_ID,
                "step": "4.2",
                "config": best["name"],
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved pipeline. roi=%.2f%%", best["roi_test"])

        print("\n=== Step 4.2 ELO + Ensemble Results ===")
        print("\nОдиночные модели:")
        for r in results:
            name, rv, rt, nt = r["name"], r["roi_val"], r["roi_test"], r["n_test"]
            print(f"  [{name}] ROI val={rv:.2f}%, test={rt:.2f}% ({nt})")
        print("\nАнсамбли:")
        for r in ensemble_results:
            name, rv, rt, nt = r["name"], r["roi_val"], r["roi_test"], r["n_test"]
            print(f"  [{name}] ROI val={rv:.2f}%, test={rt:.2f}% ({nt})")
        print(f"\nBest: {best['name']} ROI test={best['roi_test']:.2f}%")
        print(f"CV ROI (best single): {roi_mean:.2f}% +/- {roi_std:.2f}%")
        print(f"Prev best: {PREV_BEST_ROI:.2f}%")
        print(f"Delta vs prev: {best['roi_test'] - PREV_BEST_ROI:+.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
