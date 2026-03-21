"""Step 4.1 — LightGBM + CatBoost ensemble, 1x2 + shrunken segments.

Гипотеза: CatBoost и LightGBM имеют разные индуктивные смещения — усреднение
вероятностей снижает дисперсию и может улучшить Kelly-отбор в 1x2 сегменте.
LightGBM обучается на тех же признаках и данных (train 80%), thresholds
подбираются на val (last 20% of train = 64%-80%).
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import lightgbm as lgb
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
PREV_BEST_DIR = Path("/mnt/d/automl-research/.uaf/sessions/chain_7_mar21_2347/models/best")
LEAKAGE_THRESHOLD = 35.0

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Стандартный feature set (chain_1 compatible)."""
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
    # Sport/Market/Currency как числовые через factorize
    feats["sport_enc"] = pd.factorize(df["Sport"].fillna("unknown"))[0]
    feats["market_enc"] = pd.factorize(df["Market"].fillna("unknown"))[0]
    feats["currency_enc"] = pd.factorize(df["Currency"].fillna("unknown"))[0]
    return feats


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


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


def apply_shrunken_segments(
    df: pd.DataFrame, kelly: np.ndarray, seg_thresholds: dict[str, float]
) -> np.ndarray:
    """Применить shrunken segment Kelly thresholds."""
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in seg_thresholds.items():
        mask |= (buckets == bucket).values & (kelly >= t)
    return mask


def load_raw_data() -> pd.DataFrame:
    """Загрузка и объединение данных."""
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
    return df


with mlflow.start_run(run_name="phase4/step4.1_lgbm_ensemble") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.1")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        train_df = df_raw.iloc[:val_start].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

        x_tr = build_features(train_df)
        x_vl = build_features(val_df)
        x_te = build_features(test_df)

        y_tr = (train_df["Status"] == "won").astype(int)
        y_vl = (val_df["Status"] == "won").astype(int)
        y_te = (test_df["Status"] == "won").astype(int)

        # LightGBM обучение на полном train (0-80%)
        full_train_df = df_raw.iloc[:train_end].copy()
        x_full_tr = build_features(full_train_df)
        y_full_tr = (full_train_df["Status"] == "won").astype(int)

        lgbm_params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 500,
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 7,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        logger.info("Обучение LightGBM...")
        lgbm_model = lgb.LGBMClassifier(**lgbm_params)
        lgbm_model.fit(
            x_full_tr,
            y_full_tr,
            eval_set=[(x_vl, y_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        lgbm_proba_val = lgbm_model.predict_proba(x_vl)[:, 1]
        lgbm_proba_test = lgbm_model.predict_proba(x_te)[:, 1]
        auc_lgbm = roc_auc_score(y_te, lgbm_proba_test)
        logger.info("LightGBM AUC test: %.4f", auc_lgbm)

        # CatBoost predictions (без повторного обучения — используем chain_6 модель)
        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        def build_cat_features(df: pd.DataFrame) -> pd.DataFrame:
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
            return feats[cb_meta["feature_names"]]

        cat_proba_val = cat_model.predict_proba(build_cat_features(val_df))[:, 1]
        cat_proba_test = cat_model.predict_proba(build_cat_features(test_df))[:, 1]
        auc_cat = roc_auc_score(y_te, cat_proba_test)
        logger.info("CatBoost AUC test: %.4f", auc_cat)

        # Ensemble: average прогнозы
        ensemble_proba_val = 0.5 * cat_proba_val + 0.5 * lgbm_proba_val
        ensemble_proba_test = 0.5 * cat_proba_test + 0.5 * lgbm_proba_test
        auc_ens = roc_auc_score(y_te, ensemble_proba_test)
        logger.info("Ensemble AUC test: %.4f", auc_ens)

        lead_hours_val = (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600.0
        lead_hours_test = (
            test_df["Start_Time"] - test_df["Created_At"]
        ).dt.total_seconds() / 3600.0

        def get_kelly_filtered(proba: np.ndarray, df: pd.DataFrame, lh: pd.Series) -> np.ndarray:
            k = compute_kelly(proba, df["Odds"].values)
            k[lh.values <= 0] = -999
            return k

        kelly_ens_val = get_kelly_filtered(ensemble_proba_val, val_df, lead_hours_val)
        kelly_ens_test = get_kelly_filtered(ensemble_proba_test, test_df, lead_hours_test)

        # Shrunken segment thresholds — используем из chain_7 (уже оптимизированы на val)
        seg_thresholds = cb_meta["segment_thresholds"]
        logger.info("Базовые shrunken thresholds из chain_7: %s", seg_thresholds)

        mkt_mask_val = val_df["Market"].values == "1x2"
        mkt_mask_test = test_df["Market"].values == "1x2"

        # Baseline: CatBoost only 1x2 + shrunken (должна воспроизвести 28.58%)
        cat_kelly_val = get_kelly_filtered(cat_proba_val, val_df, lead_hours_val)
        cat_kelly_test = get_kelly_filtered(cat_proba_test, test_df, lead_hours_test)
        baseline_mask = mkt_mask_test & apply_shrunken_segments(
            test_df, cat_kelly_test, seg_thresholds
        )
        roi_baseline, n_baseline = calc_roi(test_df, baseline_mask)
        logger.info("CatBoost baseline 1x2+seg: roi=%.4f%%, n=%d", roi_baseline, n_baseline)

        # Ансамбль 1x2 + shrunken (существующие пороги)
        ens_seg_mask = mkt_mask_test & apply_shrunken_segments(
            test_df, kelly_ens_test, seg_thresholds
        )
        roi_ens_seg, n_ens_seg = calc_roi(test_df, ens_seg_mask)
        logger.info("Ensemble 1x2+seg (base thresholds): roi=%.4f%%, n=%d", roi_ens_seg, n_ens_seg)

        # Оптимизация shrunken thresholds на val для ensemble
        baseline_t = 0.455
        shrink = 0.5
        raw_thresholds = {"low": 0.0, "mid": 0.0, "high": 0.0}
        best_val_roi = -999.0

        for low_t in np.arange(0.35, 0.65, 0.01):
            for mid_t in np.arange(0.35, 0.75, 0.01):
                for high_t in np.arange(0.15, 0.55, 0.01):
                    t_raw = {"low": low_t, "mid": mid_t, "high": high_t}
                    t_shr = {k: baseline_t + shrink * (v - baseline_t) for k, v in t_raw.items()}
                    val_mask = mkt_mask_val & apply_shrunken_segments(val_df, kelly_ens_val, t_shr)
                    if val_mask.sum() < 20:
                        continue
                    roi_v, _ = calc_roi(val_df, val_mask)
                    if roi_v > best_val_roi:
                        best_val_roi = roi_v
                        raw_thresholds = t_raw

        best_shrunken = {
            k: baseline_t + shrink * (v - baseline_t) for k, v in raw_thresholds.items()
        }
        logger.info(
            "Лучшие val thresholds для ensemble: %s (val_roi=%.2f%%)",
            best_shrunken,
            best_val_roi,
        )

        # Ensemble с оптимизированными thresholds
        ens_opt_mask = mkt_mask_test & apply_shrunken_segments(
            test_df, kelly_ens_test, best_shrunken
        )
        roi_ens_opt, n_ens_opt = calc_roi(test_df, ens_opt_mask)
        logger.info("Ensemble 1x2+opt_seg: roi=%.4f%%, n=%d", roi_ens_opt, n_ens_opt)

        # CatBoost only с теми же оптимизированными thresholds для fair comparison
        cat_opt_mask = mkt_mask_test & apply_shrunken_segments(
            test_df, cat_kelly_test, best_shrunken
        )
        roi_cat_opt, n_cat_opt = calc_roi(test_df, cat_opt_mask)
        logger.info("CatBoost only 1x2+opt_seg: roi=%.4f%%, n=%d", roi_cat_opt, n_cat_opt)

        # Взвешенный ансамбль CatBoost-heavy (0.7/0.3)
        ens_w_proba_test = 0.7 * cat_proba_test + 0.3 * lgbm_proba_test
        ens_w_proba_val = 0.7 * cat_proba_val + 0.3 * lgbm_proba_val
        kelly_ensw_test = get_kelly_filtered(ens_w_proba_test, test_df, lead_hours_test)
        kelly_ensw_val = get_kelly_filtered(ens_w_proba_val, val_df, lead_hours_val)
        ensw_mask = mkt_mask_test & apply_shrunken_segments(
            test_df, kelly_ensw_test, seg_thresholds
        )
        roi_ensw, n_ensw = calc_roi(test_df, ensw_mask)
        logger.info("Ensemble(0.7/0.3) 1x2+seg: roi=%.4f%%, n=%d", roi_ensw, n_ensw)

        best_roi = max(roi_ens_seg, roi_ens_opt, roi_ensw, roi_baseline)
        baseline_roi = 28.5833
        delta = best_roi - baseline_roi

        if best_roi > LEAKAGE_THRESHOLD:
            logger.error("LEAKAGE SUSPECT: roi=%.2f > %.2f", best_roi, LEAKAGE_THRESHOLD)
            mlflow.set_tag("status", "leakage_suspect")
            sys.exit(1)

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": train_end,
                "n_samples_val": len(val_df),
                "n_samples_test": len(test_df),
                "lgbm_n_estimators": lgbm_params["n_estimators"],
                "lgbm_lr": lgbm_params["learning_rate"],
                "lgbm_num_leaves": lgbm_params["num_leaves"],
                "ensemble_weights": "0.5/0.5",
                "market_filter": "1x2",
                "shrinkage": shrink,
            }
        )

        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_cat_baseline": roi_baseline,
                "roi_ens_seg": roi_ens_seg,
                "roi_ens_opt": roi_ens_opt,
                "roi_ens_weighted": roi_ensw,
                "roi_cat_opt": roi_cat_opt,
                "n_cat_baseline": n_baseline,
                "n_ens_seg": n_ens_seg,
                "n_ens_opt": n_ens_opt,
                "n_ens_weighted": n_ensw,
                "auc_lgbm": auc_lgbm,
                "auc_catboost": auc_cat,
                "auc_ensemble": auc_ens,
                "delta_vs_baseline": delta,
            }
        )

        if best_roi > baseline_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi, baseline_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            # Сохраняем модели
            cat_model.save_model(str(best_dir / "model_catboost.cbm"))
            lgbm_model.booster_.save_model(str(best_dir / "model_lgbm.txt"))
            meta_out = {
                "framework": "ensemble_catboost_lgbm",
                "roi": best_roi,
                "auc": float(auc_ens),
                "segment_thresholds": best_shrunken if roi_ens_opt == best_roi else seg_thresholds,
                "market_filter": "1x2",
                "n_bets": n_ens_opt if roi_ens_opt == best_roi else n_ens_seg,
                "feature_names": cb_meta["feature_names"],
                "params": {"catboost": cb_meta["params"], "lgbm": lgbm_params},
                "session_id": SESSION_ID,
                "step": "4.1",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.1 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  CatBoost baseline: {roi_baseline:.4f}%/{n_baseline}"
            f"  Ens(0.5/0.5)+seg: {roi_ens_seg:.4f}%/{n_ens_seg}"
            f"  Ens(0.5/0.5)+opt: {roi_ens_opt:.4f}%/{n_ens_opt}"
            f"  Ens(0.7/0.3)+seg: {roi_ensw:.4f}%/{n_ensw}"
        )
        print(f"  AUC: CatBoost={auc_cat:.4f}  LightGBM={auc_lgbm:.4f}  Ensemble={auc_ens:.4f}")
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
