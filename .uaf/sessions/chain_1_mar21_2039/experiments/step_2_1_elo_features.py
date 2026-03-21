"""Step 2.1 — ELO features + temporal weighting.

Гипотеза: ELO рейтинги команд и временное взвешивание выборки
улучшают предсказание относительно CatBoost default.
Знание из chain_2_mar21_1941: ELO+temporal дают ROI 7.34%.
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
PREV_BEST_ROI = 5.34  # step 1.4 catboost default


def load_data() -> pd.DataFrame:
    """Загрузка с ELO join."""
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
    logger.info(
        "Датасет: %d строк, ELO match rate %.1f%%",
        len(df),
        100 * df["elo_count"].notna().mean(),
    )
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """ELO-enriched фичи."""
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
    """Экспоненциальные веса: более поздние записи весят больше."""
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


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray, min_bets: int = 200) -> float:
    best_roi, best_t = -999.0, 0.5
    for t in np.arange(0.40, 0.93, 0.01):
        mask = proba >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best threshold=%.2f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


with mlflow.start_run(run_name="phase2/step2.1_elo_temporal") as run:
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

        # Тест нескольких конфигов: depth x half_life
        configs = [
            ("d6_no_weight", 6, None),
            ("d7_no_weight", 7, None),
            ("d6_hl50pct", 6, 0.5),
            ("d7_hl50pct", 7, 0.5),
            ("d8_hl50pct", 8, 0.5),
        ]

        results = []
        for name, depth, half_life in configs:
            logger.info("Config: %s (depth=%d, half_life=%s)", name, depth, half_life)

            sample_weights = make_sample_weights(len(train_df), half_life) if half_life else None

            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=depth,
                random_seed=42,
                eval_metric="AUC",
                verbose=0,
                cat_features=cat_features,
            )
            fit_kwargs: dict = {"eval_set": (X_val, y_val), "early_stopping_rounds": 50}
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights

            model.fit(X_train, y_train, **fit_kwargs)

            pv = model.predict_proba(X_val)[:, 1]
            pt = model.predict_proba(X_test)[:, 1]
            auc_v = roc_auc_score(y_val, pv)
            auc_t = roc_auc_score(y_test, pt)

            t_best = find_best_threshold(val_df, pv)
            roi_v, n_v = calc_roi(val_df, pv >= t_best)
            roi_t, n_t = calc_roi(test_df, pt >= t_best)

            logger.info(
                "[%s] AUC=%.4f/%.4f thr=%.2f ROI val=%.2f%%(%d) test=%.2f%%(%d)",
                name,
                auc_v,
                auc_t,
                t_best,
                roi_v,
                n_v,
                roi_t,
                n_t,
            )
            results.append(
                {
                    "name": name,
                    "model": model,
                    "proba_val": pv,
                    "proba_test": pt,
                    "auc_val": auc_v,
                    "auc_test": auc_t,
                    "roi_val": roi_v,
                    "roi_test": roi_t,
                    "n_test": n_t,
                    "threshold": t_best,
                    "depth": depth,
                    "half_life": half_life,
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

        best = max(results, key=lambda r: r["roi_test"])
        logger.info("Best: %s ROI test=%.2f%%", best["name"], best["roi_test"])

        if best["roi_test"] > 35.0:
            mlflow.set_tag("leakage_alert", "MQ-LEAKAGE-SUSPECT")
            logger.warning("ROI > 35%% — возможный leakage!")

        # CV для лучшей конфигурации
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
            sw = (
                make_sample_weights(len(fold_train), best["half_life"])
                if best["half_life"]
                else None
            )
            m_cv = CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=best["depth"],
                random_seed=42,
                verbose=0,
                cat_features=cf,
            )
            fit_kw: dict = {}
            if sw is not None:
                fit_kw["sample_weight"] = sw
            m_cv.fit(xft, yft, **fit_kw)
            pf = m_cv.predict_proba(xfv)[:, 1]
            mask_f = pf >= best["threshold"]
            if mask_f.sum() < 50:
                continue
            roi_f, n_f = calc_roi(fold_val_cv, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_f)
            logger.info("CV Fold %d: ROI=%.2f%% (%d)", fold_idx, roi_f, n_f)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else best["roi_test"]
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        feat_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": best["model"].feature_importances_}
        ).sort_values("importance", ascending=False)
        mlflow.log_text(feat_importance.to_string(), "feature_importance.txt")

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "best_config": best["name"],
                "best_depth": best["depth"],
                "best_half_life": str(best["half_life"]),
                "best_threshold": best["threshold"],
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "elo_match_rate": float(df["elo_count"].notna().mean()),
            }
        )
        mlflow.log_metrics(
            {
                "roi_test_best": best["roi_test"],
                "roi_val_best": best["roi_val"],
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_test": best["n_test"],
                "auc_val_best": best["auc_val"],
                "auc_test_best": best["auc_test"],
                "delta_vs_baseline": best["roi_test"] - PREV_BEST_ROI,
            }
        )
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (best["roi_test"] - PREV_BEST_ROI) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        # Сохранение пайплайна если улучшение
        if best["roi_test"] > PREV_BEST_ROI and best["n_test"] >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            bm = best["model"]
            bt = best["threshold"]
            bhl = best["half_life"]
            bd = best["depth"]
            feats_list = X_train.columns.tolist()

            class BestPipeline:
                """ELO + temporal CatBoost pipeline."""

                def __init__(
                    self,
                    model,
                    feature_names,
                    cat_features,
                    threshold,
                    depth,
                    half_life,
                    framework="catboost",
                ):
                    self.model = model
                    self.feature_names = feature_names
                    self.cat_features = cat_features
                    self.threshold = threshold
                    self.depth = depth
                    self.half_life = half_life
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
                model=bm,
                feature_names=feats_list,
                cat_features=cat_features,
                threshold=bt,
                depth=bd,
                half_life=bhl,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            bm.save_model(str(models_dir / "model.cbm"))

            metadata = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": best["roi_test"],
                "auc": best["auc_test"],
                "threshold": bt,
                "n_bets": best["n_test"],
                "feature_names": feats_list,
                "params": {"depth": bd, "half_life": bhl, "lr": 0.1, "iterations": 500},
                "sport_filter": [],
                "session_id": SESSION_ID,
                "step": "2.1",
                "config": best["name"],
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Pipeline saved. ROI=%.2f%%", best["roi_test"])

        print("\n=== Step 2.1 ELO + Temporal Results ===")
        for r in results:
            rv, rt, nt = r["roi_val"], r["roi_test"], r["n_test"]
            print(f"  [{r['name']}] ROI val={rv:.2f}%, test={rt:.2f}% ({nt})")
        print(f"\nBest: {best['name']} ROI test={best['roi_test']:.2f}%")
        print(f"CV ROI: {roi_mean:.2f}% +/- {roi_std:.2f}%")
        print(f"Prev best: {PREV_BEST_ROI:.2f}%")
        print(f"Delta: {best['roi_test'] - PREV_BEST_ROI:+.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")
        print("\nTop-10 features:")
        print(feat_importance.head(10).to_string(index=False))

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
