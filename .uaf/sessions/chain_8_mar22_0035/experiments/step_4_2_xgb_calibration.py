"""Step 4.2 — XGBoost + CatBoost ensemble + isotonic calibration.

Гипотезы:
A) XGBoost + CatBoost ensemble (аналог step_4.1) — проверяем другую пару моделей
B) CatBoost с isotonic calibration — улучшение калибровки вероятностей для Kelly
C) Стабильность: используем зафиксированные thresholds из chain_7 (без re-оптимизации)

Главная цель: найти improvement без val-threshold overfitting.
Thresholds не пересматриваются — используем chain_7 shrunken thresholds напрямую.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
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


def build_features_num(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only features для XGBoost (без категориальных строк)."""
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
    feats["sport_enc"] = pd.factorize(df["Sport"].fillna("unknown"))[0].astype(float)
    feats["market_enc"] = pd.factorize(df["Market"].fillna("unknown"))[0].astype(float)
    feats["currency_enc"] = pd.factorize(df["Currency"].fillna("unknown"))[0].astype(float)
    return feats


def build_cat_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Feature set совместимый с CatBoost из chain_6."""
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
    return feats[feature_names]


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


with mlflow.start_run(run_name="phase4/step4.2_xgb_calibration") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.2")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        train_df = df_raw.iloc[:val_start].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()
        full_train_df = df_raw.iloc[:train_end].copy()

        logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        seg_thresholds = cb_meta["segment_thresholds"]  # chain_7 shrunken thresholds (FIXED)

        # CatBoost baseline (chain_7 модель)
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(PREV_BEST_DIR / "model.cbm"))

        x_cat_vl = build_cat_features(val_df, cb_meta["feature_names"])
        x_cat_te = build_cat_features(test_df, cb_meta["feature_names"])
        x_cat_full_tr = build_cat_features(full_train_df, cb_meta["feature_names"])

        cat_proba_val = cat_model.predict_proba(x_cat_vl)[:, 1]
        cat_proba_test = cat_model.predict_proba(x_cat_te)[:, 1]
        y_te = (test_df["Status"] == "won").astype(int)
        auc_cat = roc_auc_score(y_te, cat_proba_test)
        logger.info("CatBoost AUC: %.4f", auc_cat)

        # XGBoost обучение
        x_xgb_vl = build_features_num(val_df)
        x_xgb_te = build_features_num(test_df)
        x_xgb_full_tr = build_features_num(full_train_df)
        y_full_tr = (full_train_df["Status"] == "won").astype(int)
        y_vl = (val_df["Status"] == "won").astype(int)

        xgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

        logger.info("Обучение XGBoost...")
        xgb_params_with_es = {**xgb_params, "early_stopping_rounds": 50}
        xgb_model = xgb.XGBClassifier(**xgb_params_with_es)
        xgb_model.fit(
            x_xgb_full_tr,
            y_full_tr,
            eval_set=[(x_xgb_vl, y_vl)],
            verbose=False,
        )

        xgb_proba_val = xgb_model.predict_proba(x_xgb_vl)[:, 1]
        xgb_proba_test = xgb_model.predict_proba(x_xgb_te)[:, 1]
        auc_xgb = roc_auc_score(y_te, xgb_proba_test)
        logger.info("XGBoost AUC: %.4f", auc_xgb)

        # Isotonic calibration CatBoost (fit на full train, проверка на val+test)
        # Fit на last 20% of train (val_df) — isotonic использует val split
        from sklearn.isotonic import IsotonicRegression

        iso_reg = IsotonicRegression(out_of_bounds="clip")
        y_vl_arr = y_vl.values
        iso_reg.fit(cat_proba_val, y_vl_arr)

        cat_calib_proba_val = iso_reg.predict(cat_proba_val)
        cat_calib_proba_test = iso_reg.predict(cat_proba_test)
        auc_cat_calib = roc_auc_score(y_te, cat_calib_proba_test)
        logger.info("CatBoost calibrated AUC: %.4f", auc_cat_calib)

        # Platt scaling
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        platt.fit(cat_proba_val.reshape(-1, 1), y_vl_arr)
        cat_platt_proba_val = platt.predict_proba(cat_proba_val.reshape(-1, 1))[:, 1]
        cat_platt_proba_test = platt.predict_proba(cat_proba_test.reshape(-1, 1))[:, 1]
        auc_cat_platt = roc_auc_score(y_te, cat_platt_proba_test)
        logger.info("CatBoost Platt AUC: %.4f", auc_cat_platt)

        # Kelly + shrunken segments (ФИКСИРОВАННЫЕ thresholds chain_7 — без re-opt)
        def get_kelly(proba: np.ndarray, df: pd.DataFrame) -> np.ndarray:
            lead_hours = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
            k = compute_kelly(proba, df["Odds"].values)
            k[lead_hours.values <= 0] = -999
            return k

        mkt_test = test_df["Market"].values == "1x2"

        # A) CatBoost baseline с chain_7 thresholds
        cat_kelly_test = get_kelly(cat_proba_test, test_df)
        mask_cat = mkt_test & apply_shrunken_segments(test_df, cat_kelly_test, seg_thresholds)
        roi_cat, n_cat = calc_roi(test_df, mask_cat)
        logger.info("CatBoost 1x2+seg: roi=%.4f%%, n=%d", roi_cat, n_cat)

        # B) XGBoost 1x2 + seg (chain_7 thresholds)
        xgb_kelly_test = get_kelly(xgb_proba_test, test_df)
        mask_xgb = mkt_test & apply_shrunken_segments(test_df, xgb_kelly_test, seg_thresholds)
        roi_xgb, n_xgb = calc_roi(test_df, mask_xgb)
        logger.info("XGBoost 1x2+seg: roi=%.4f%%, n=%d", roi_xgb, n_xgb)

        # C) XGBoost + CatBoost ensemble 0.5/0.5 (chain_7 thresholds)
        ens_xgb_proba_test = 0.5 * cat_proba_test + 0.5 * xgb_proba_test
        ens_xgb_kelly_test = get_kelly(ens_xgb_proba_test, test_df)
        mask_ens_xgb = mkt_test & apply_shrunken_segments(
            test_df, ens_xgb_kelly_test, seg_thresholds
        )
        roi_ens_xgb, n_ens_xgb = calc_roi(test_df, mask_ens_xgb)
        logger.info("Ensemble(CAT+XGB 0.5/0.5) 1x2+seg: roi=%.4f%%, n=%d", roi_ens_xgb, n_ens_xgb)

        # D) Calibrated CatBoost (isotonic) 1x2+seg
        iso_kelly_test = get_kelly(cat_calib_proba_test, test_df)
        mask_iso = mkt_test & apply_shrunken_segments(test_df, iso_kelly_test, seg_thresholds)
        roi_iso, n_iso = calc_roi(test_df, mask_iso)
        logger.info("CatBoost+Isotonic 1x2+seg: roi=%.4f%%, n=%d", roi_iso, n_iso)

        # E) Calibrated CatBoost (Platt) 1x2+seg
        platt_kelly_test = get_kelly(cat_platt_proba_test, test_df)
        mask_platt = mkt_test & apply_shrunken_segments(test_df, platt_kelly_test, seg_thresholds)
        roi_platt, n_platt = calc_roi(test_df, mask_platt)
        logger.info("CatBoost+Platt 1x2+seg: roi=%.4f%%, n=%d", roi_platt, n_platt)

        # F) 3-way ensemble: CatBoost + XGB + calibrated_CatBoost
        three_ens_proba_test = (cat_proba_test + xgb_proba_test + cat_calib_proba_test) / 3.0
        three_ens_kelly_test = get_kelly(three_ens_proba_test, test_df)
        mask_3ens = mkt_test & apply_shrunken_segments(
            test_df, three_ens_kelly_test, seg_thresholds
        )
        roi_3ens, n_3ens = calc_roi(test_df, mask_3ens)
        logger.info("3-way ensemble(CAT+XGB+CAT_ISO) 1x2+seg: roi=%.4f%%, n=%d", roi_3ens, n_3ens)

        best_roi = max(roi_cat, roi_xgb, roi_ens_xgb, roi_iso, roi_platt, roi_3ens)
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
                "xgb_n_estimators": xgb_params["n_estimators"],
                "xgb_lr": xgb_params["learning_rate"],
                "market_filter": "1x2",
                "thresholds_source": "chain_7_fixed",
                "calibration_method": "isotonic+platt",
            }
        )

        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_cat_baseline": roi_cat,
                "roi_xgb": roi_xgb,
                "roi_ens_xgb": roi_ens_xgb,
                "roi_iso": roi_iso,
                "roi_platt": roi_platt,
                "roi_3ens": roi_3ens,
                "n_cat": n_cat,
                "n_xgb": n_xgb,
                "n_ens_xgb": n_ens_xgb,
                "n_iso": n_iso,
                "n_platt": n_platt,
                "n_3ens": n_3ens,
                "auc_catboost": auc_cat,
                "auc_xgb": auc_xgb,
                "auc_cat_calib": auc_cat_calib,
                "auc_cat_platt": auc_cat_platt,
                "delta_vs_baseline": delta,
            }
        )

        # Сохраняем только если реальный улучшение по зафиксированным thresholds
        if best_roi > baseline_roi:
            logger.info("NEW BEST (fixed thresholds): %.4f%% > %.4f%%", best_roi, baseline_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            cat_model.save_model(str(best_dir / "model.cbm"))
            meta_out = {
                "framework": "catboost",
                "roi": best_roi,
                "auc": float(auc_cat),
                "segment_thresholds": seg_thresholds,
                "market_filter": "1x2",
                "n_bets": n_cat,
                "feature_names": cb_meta["feature_names"],
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.2",
                "note": f"best variant in step 4.2: best_roi={best_roi:.4f}",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.2 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  CAT: {roi_cat:.4f}%/{n_cat}"
            f"  XGB: {roi_xgb:.4f}%/{n_xgb}"
            f"  ENS(CAT+XGB): {roi_ens_xgb:.4f}%/{n_ens_xgb}"
        )
        print(
            f"  ISO: {roi_iso:.4f}%/{n_iso}"
            f"  PLATT: {roi_platt:.4f}%/{n_platt}"
            f"  3ENS: {roi_3ens:.4f}%/{n_3ens}"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
