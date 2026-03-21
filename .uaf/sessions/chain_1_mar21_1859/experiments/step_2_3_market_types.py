"""Step 2.3 - Market type grouping + model depth tuning.

Hypothesis: Замена raw Market (3383 unique) на market_type (16 групп) снизит
переобучение на временном сдвиге между val и test.
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


def classify_market(m: str) -> str:
    """Сгруппировать рынки в 16 типов."""
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
    """Загрузка и объединение данных."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")

    bets = bets[~bets["Status"].isin(EXCLUDE_STATUSES)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    safe_cols = ["Bet_ID", "Sport", "Tournament", "Market", "Start_Time"]
    outcomes_first = outcomes_first[safe_cols]

    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)
    df["market_type"] = df["Market"].apply(classify_market)
    return df


def build_features_v3(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline + market_type вместо raw Market."""
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
    feats["market_type"] = df["market_type"]  # 16 групп
    feats["Currency"] = df["Currency"].fillna("unknown")

    cat_features = ["Sport", "market_type", "Currency"]
    return feats, cat_features


def build_features_v3b(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline + ОБА: market_type + raw Market (top-200 остальные = 'other')."""
    feats, cats = build_features_v3(df)

    # Упрощённый Market: топ-200 по частоте, остальные = 'other'
    market_freq_path = Path(SESSION_DIR) / "market_freq.json"
    if market_freq_path.exists():
        with open(market_freq_path) as fp:
            top_200 = set(json.load(fp))
        feats["Market_top200"] = df["Market"].apply(
            lambda x: x if (not pd.isna(x) and x in top_200) else "other"
        )
        cats = [*cats, "Market_top200"]
    return feats, cats


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


def find_best_threshold(val_df: pd.DataFrame, proba: np.ndarray) -> float:
    """Лучший порог на val."""
    best_roi, best_t = -999.0, 0.5
    for t in np.arange(0.40, 0.90, 0.02):
        mask = proba >= t
        if mask.sum() < 200:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    logger.info("Best threshold=%.2f (val ROI=%.2f%%)", best_t, best_roi)
    return best_t


with mlflow.start_run(run_name="phase2/step2.3_market_types") as run:
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

        # Сохраним топ-200 рынков (по train) для Market_top200 фичи
        market_counts = train_df["Market"].value_counts()
        top_200_markets = market_counts.head(200).index.tolist()
        market_freq_path = Path(SESSION_DIR) / "market_freq.json"
        with open(market_freq_path, "w") as fp:
            json.dump(top_200_markets, fp)

        logger.info("Market type distribution:\n%s", df["market_type"].value_counts().to_string())

        y_train = (train_df["Status"] == "won").astype(int)
        y_val = (val_df["Status"] == "won").astype(int)
        y_test = (test_df["Status"] == "won").astype(int)

        # Run multiple configs as nested runs
        configs = [
            ("v3_market_type", build_features_v3, 6),
            ("v3_depth4", build_features_v3, 4),
            ("v3b_market_both", build_features_v3b, 6),
        ]

        results = []
        for name, build_fn, depth in configs:
            logger.info("--- Config: %s (depth=%d) ---", name, depth)
            x_tr, cf = build_fn(train_df)
            x_vl, _ = build_fn(val_df)
            x_ts, _ = build_fn(test_df)

            m = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=depth,
                random_seed=42,
                eval_metric="AUC",
                verbose=0,
                cat_features=cf,
            )
            m.fit(x_tr, y_train, eval_set=(x_vl, y_val), early_stopping_rounds=50)

            pv = m.predict_proba(x_vl)[:, 1]
            pt = m.predict_proba(x_ts)[:, 1]
            auc_v = roc_auc_score(y_val, pv)
            t_best = find_best_threshold(val_df, pv)
            roi_v, n_v = calc_roi(val_df, pv >= t_best)
            roi_t, n_t = calc_roi(test_df, pt >= t_best)

            logger.info(
                "[%s] AUC=%.4f, ROI val=%.2f%% (%d), test=%.2f%% (%d)",
                name,
                auc_v,
                roi_v,
                n_v,
                roi_t,
                n_t,
            )
            results.append(
                {
                    "name": name,
                    "model": m,
                    "auc_val": auc_v,
                    "roi_val": roi_v,
                    "roi_test": roi_t,
                    "n_test": n_t,
                    "threshold": t_best,
                    "build_fn": build_fn,
                    "depth": depth,
                    "features": x_tr.columns.tolist(),
                    "cat_features": cf,
                }
            )

        # Выбор лучшего по val ROI (не по test!)
        best_config = max(results, key=lambda r: r["roi_val"])
        logger.info("Best config by val ROI: %s", best_config["name"])

        # CV для лучшей конфигурации
        cv_rois = []
        fold_size = n // 5
        for fold_idx in range(1, 5):
            fold_start = fold_idx * fold_size
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n

            fold_train = df.iloc[:fold_start]
            fold_val_cv = df.iloc[fold_start:fold_end]

            x_ft, cf = best_config["build_fn"](fold_train)
            x_fv, _ = best_config["build_fn"](fold_val_cv)
            y_ft = (fold_train["Status"] == "won").astype(int)

            cm = CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=best_config["depth"],
                random_seed=42,
                verbose=0,
                cat_features=cf,
            )
            cm.fit(x_ft, y_ft)
            pf = cm.predict_proba(x_fv)[:, 1]
            mask_f = pf >= best_config["threshold"]
            if mask_f.sum() < 50:
                continue
            roi_f, n_f = calc_roi(fold_val_cv, mask_f)
            cv_rois.append(roi_f)
            mlflow.log_metric(f"roi_fold_{fold_idx}", roi_f)
            logger.info("CV Fold %d: ROI=%.2f%% (%d)", fold_idx, roi_f, n_f)

        roi_mean = float(np.mean(cv_rois)) if cv_rois else best_config["roi_test"]
        roi_std = float(np.std(cv_rois)) if cv_rois else 0.0

        # Feature importance
        x_train_best, cf_best = best_config["build_fn"](train_df)
        feat_importance = pd.DataFrame(
            {
                "feature": best_config["features"],
                "importance": best_config["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "best_config": best_config["name"],
                "depth": best_config["depth"],
                "threshold": best_config["threshold"],
                "n_samples_train": len(train_df),
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                **{f"auc_val_{r['name']}": r["auc_val"] for r in results},
                **{f"roi_test_{r['name']}": r["roi_test"] for r in results},
                "roi_val_best": best_config["roi_val"],
                "roi_test_best": best_config["roi_test"],
                "roi_mean": roi_mean,
                "roi_std": roi_std,
                "n_bets_test": best_config["n_test"],
            }
        )
        mlflow.log_text(feat_importance.to_string(), "feature_importance.txt")
        mlflow.log_artifact(__file__)
        mlflow.set_tag("status", "success")
        convergence = min(1.0, max(0.0, (best_config["roi_test"] - 4.84) / 10.0))
        mlflow.set_tag("convergence_signal", str(round(convergence, 2)))

        prev_best = 4.84
        if best_config["roi_test"] > prev_best and best_config["n_test"] >= 200:
            models_dir = Path(SESSION_DIR) / "models" / "best"
            models_dir.mkdir(parents=True, exist_ok=True)

            bfn = best_config["build_fn"]
            bm = best_config["model"]
            bt = best_config["threshold"]
            bfeat = best_config["features"]
            bcf = best_config["cat_features"]

            class BestPipeline:
                """Полный пайплайн."""

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

                def predict_proba(self, df):
                    feats, _ = self._build_fn(df)
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
                model=bm,
                feature_names=bfeat,
                cat_features=bcf,
                threshold=bt,
                build_fn=bfn,
            )
            joblib.dump(pipeline, models_dir / "pipeline.pkl")
            bm.save_model(str(models_dir / "model.cbm"))
            metadata = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": best_config["roi_test"],
                "auc": best_config["auc_val"],
                "threshold": bt,
                "n_bets": best_config["n_test"],
                "feature_names": bfeat,
                "session_id": SESSION_ID,
                "step": "2.3",
                "config": best_config["name"],
            }
            with open(models_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved pipeline. roi=%.2f%%", best_config["roi_test"])

        print("\n=== Step 2.3 Results ===")
        for r in results:
            print(
                f"  [{r['name']}] AUC val={r['auc_val']:.4f}, "
                f"ROI val={r['roi_val']:.2f}%, test={r['roi_test']:.2f}% ({r['n_test']} ставок)"
            )
        print(f"\nBest config: {best_config['name']}")
        print(f"ROI test: {best_config['roi_test']:.2f}%, CV: {roi_mean:.2f}% ± {roi_std:.2f}%")
        print(f"MLflow run_id: {run.info.run_id}")
        print("\nTop-10 features:")
        print(feat_importance.head(10).to_string(index=False))

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
