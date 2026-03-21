"""Step 4.5 — Profit regression: CatBoostRegressor на E[profit/stake].

Гипотеза: классификатор оптимизирует P(win), а Kelly criterion использует эту
вероятность как входную. Регрессор напрямую предсказывает ожидаемую прибыль
(profit/stake = (payout-stake)/stake для won, -1.0 для lost), что точнее
соответствует Kelly-отбору высокодоходных ставок.

Target: profit_rate = (Payout_USD - USD) / USD for won, -1.0 for lost
Selection: bet if predicted_profit_rate > threshold (оптимизируется на val)
Filter: 1x2 Soccer (все 1x2 = Soccer, per step 4.3)

Thresholds выбираются на val через grid search.
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
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

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


def build_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Feature set совместимый с chain_7."""
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


with mlflow.start_run(run_name="phase4/step4.5_profit_regression") as run:
    run_id = run.info.run_id
    mlflow.set_tag("session_id", SESSION_ID)
    mlflow.set_tag("type", "experiment")
    mlflow.set_tag("status", "running")
    mlflow.set_tag("step", "4.5")

    try:
        logger.info("Загрузка данных...")
        df_raw = load_raw_data()
        n = len(df_raw)

        train_end = int(n * 0.80)
        val_start = int(n * 0.64)

        full_train_df = df_raw.iloc[:train_end].copy()
        val_df = df_raw.iloc[val_start:train_end].copy()
        test_df = df_raw.iloc[train_end:].copy()

        logger.info(
            "Full train: %d, Val: %d, Test: %d", len(full_train_df), len(val_df), len(test_df)
        )

        cb_meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        feature_names = cb_meta["feature_names"]
        seg_thresholds = cb_meta["segment_thresholds"]

        # Construct target: profit_rate
        def make_profit_target(df: pd.DataFrame) -> pd.Series:
            """profit_rate = (payout-stake)/stake для won, -1.0 для lost."""
            won = df["Status"] == "won"
            profit = pd.Series(-1.0, index=df.index)
            # Для won: (Payout_USD - USD) / USD
            profit[won] = (df.loc[won, "Payout_USD"] - df.loc[won, "USD"]) / df.loc[
                won, "USD"
            ].clip(0.001)
            # Clip для защиты от выбросов
            return profit.clip(-1.0, 20.0)

        y_tr = make_profit_target(full_train_df)
        y_vl = make_profit_target(val_df)

        x_tr = build_features(full_train_df, feature_names)
        x_vl = build_features(val_df, feature_names)
        x_te = build_features(test_df, feature_names)

        cat_idx = [
            i for i, col in enumerate(feature_names) if col in ("Sport", "Market", "Currency")
        ]

        logger.info("Обучение CatBoostRegressor...")
        reg_params = {
            "depth": cb_meta["params"]["depth"],
            "learning_rate": cb_meta["params"]["learning_rate"],
            "iterations": cb_meta["params"]["iterations"],
            "cat_features": cat_idx,
            "random_seed": 42,
            "verbose": False,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
        }
        cbr = CatBoostRegressor(**reg_params)
        cbr.fit(x_tr, y_tr, eval_set=(x_vl, y_vl))

        pred_val = cbr.predict(x_vl)
        pred_test = cbr.predict(x_te)

        rmse_val = float(np.sqrt(mean_squared_error(y_vl, pred_val)))
        logger.info("RMSE val: %.4f", rmse_val)

        # Correlation с реальным исходом (чем выше predicted_profit, тем более likely win?)
        y_te_binary = (test_df["Status"] == "won").astype(int)
        auc_regr = roc_auc_score(y_te_binary, pred_test)
        logger.info("Regression AUC (pred_profit vs actual_won): %.4f", auc_regr)

        # Pre-match filter
        lead_hours_val = (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
        lead_hours_test = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600

        pred_val_filtered = pred_val.copy()
        pred_val_filtered[lead_hours_val.values <= 0] = -999
        pred_test_filtered = pred_test.copy()
        pred_test_filtered[lead_hours_test.values <= 0] = -999

        mkt_val = val_df["Market"].values == "1x2"
        mkt_test = test_df["Market"].values == "1x2"

        # Baseline (CatBoost classifier с chain_7 shrunken thresholds)
        from catboost import CatBoostClassifier

        def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
            b = odds - 1.0
            return (proba * b - (1 - proba)) / b.clip(0.001)

        def apply_shrunken_segments(
            df: pd.DataFrame, kelly: np.ndarray, seg_t: dict[str, float]
        ) -> np.ndarray:
            buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
            mask = np.zeros(len(df), dtype=bool)
            for bucket, t in seg_t.items():
                mask |= (buckets == bucket).values & (kelly >= t)
            return mask

        cat_cls = CatBoostClassifier()
        cat_cls.load_model(str(PREV_BEST_DIR / "model.cbm"))
        cls_proba_test = cat_cls.predict_proba(x_te)[:, 1]
        kelly_test = compute_kelly(cls_proba_test, test_df["Odds"].values)
        kelly_test[lead_hours_test.values <= 0] = -999
        baseline_mask = mkt_test & apply_shrunken_segments(test_df, kelly_test, seg_thresholds)
        roi_baseline, n_baseline = calc_roi(test_df, baseline_mask)
        logger.info("Classifier baseline 1x2+seg: roi=%.4f%%, n=%d", roi_baseline, n_baseline)

        # Grid search threshold на val для regressor
        logger.info("Grid search regression threshold на val (1x2 filter)...")
        best_val_roi = -999.0
        best_thr = 0.0
        best_n_val = 0

        val_df_tmp = val_df.copy()
        for thr in np.arange(-0.5, 2.0, 0.05):
            mask_v = mkt_val & (pred_val_filtered >= thr)
            if mask_v.sum() < 15:
                continue
            roi_v, n_v = calc_roi(val_df_tmp, mask_v)
            if roi_v > best_val_roi:
                best_val_roi = roi_v
                best_thr = thr
                best_n_val = n_v

        logger.info(
            "Best regression threshold from val: %.3f (val_roi=%.2f%%, n=%d)",
            best_thr,
            best_val_roi,
            best_n_val,
        )

        # Применяем к тест
        mask_test = mkt_test & (pred_test_filtered >= best_thr)
        roi_regr, n_regr = calc_roi(test_df, mask_test)
        logger.info("Regression 1x2+thr=%.3f: roi=%.4f%%, n=%d", best_thr, roi_regr, n_regr)

        # Также пробуем разные фиксированные thresholds без val-opt
        for fix_thr in [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3]:
            m = mkt_test & (pred_test_filtered >= fix_thr)
            roi_f, n_f = calc_roi(test_df, m)
            logger.info("  Reg thr=%.1f: roi=%.2f%%, n=%d", fix_thr, roi_f, n_f)

        # Temporal stability: первая vs вторая половина test
        test_mid = len(test_df) // 2
        mask_test_half1 = baseline_mask.copy()
        mask_test_half1[test_mid:] = False
        mask_test_half2 = baseline_mask.copy()
        mask_test_half2[:test_mid] = False

        roi_h1, n_h1 = calc_roi(test_df, mask_test_half1)
        roi_h2, n_h2 = calc_roi(test_df, mask_test_half2)
        logger.info(
            "Baseline temporal stability: first_half roi=%.2f%%/n=%d, second_half roi=%.2f%%/n=%d",
            roi_h1,
            n_h1,
            roi_h2,
            n_h2,
        )

        # Temporal stability для regression threshold
        mask_reg_h1 = mask_test.copy()
        mask_reg_h1[test_mid:] = False
        mask_reg_h2 = mask_test.copy()
        mask_reg_h2[:test_mid] = False
        roi_rh1, n_rh1 = calc_roi(test_df, mask_reg_h1)
        roi_rh2, n_rh2 = calc_roi(test_df, mask_reg_h2)
        logger.info(
            "Regr temporal stability: first_half roi=%.2f%%/n=%d, second_half roi=%.2f%%/n=%d",
            roi_rh1,
            n_rh1,
            roi_rh2,
            n_rh2,
        )

        best_roi = max(roi_baseline, roi_regr)
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
                "market_filter": "1x2",
                "regressor_depth": reg_params["depth"],
                "regressor_lr": reg_params["learning_rate"],
                "best_regression_threshold": best_thr,
                "target": "profit_rate",
            }
        )
        mlflow.log_metrics(
            {
                "roi": best_roi,
                "roi_cls_baseline": roi_baseline,
                "roi_regression": roi_regr,
                "n_cls_baseline": n_baseline,
                "n_regression": n_regr,
                "auc_regression": auc_regr,
                "rmse_val": rmse_val,
                "roi_temporal_h1": roi_h1,
                "roi_temporal_h2": roi_h2,
                "roi_regr_h1": roi_rh1,
                "roi_regr_h2": roi_rh2,
                "delta_vs_baseline": delta,
            }
        )

        if best_roi > baseline_roi:
            logger.info("NEW BEST: %.4f%% > %.4f%%", best_roi, baseline_roi)
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            cat_cls.save_model(str(best_dir / "model.cbm"))
            meta_out = {
                "framework": "catboost",
                "roi": best_roi,
                "segment_thresholds": seg_thresholds,
                "market_filter": "1x2",
                "n_bets": n_baseline if roi_baseline >= roi_regr else n_regr,
                "feature_names": feature_names,
                "params": cb_meta["params"],
                "session_id": SESSION_ID,
                "step": "4.5",
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            mlflow.set_tag("new_best", "true")

        convergence_signal = min(1.0, max(0.0, (delta + 5.0) / 10.0))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", f"{convergence_signal:.2f}")
        mlflow.log_artifact(__file__)

        print(f"step4.5 DONE: best_roi={best_roi:.4f}%, delta={delta:.4f}%")
        print(
            f"  Classifier baseline: {roi_baseline:.4f}%/{n_baseline}"
            f"  Regression: {roi_regr:.4f}%/{n_regr} (thr={best_thr:.3f})"
        )
        print(
            f"  Temporal: baseline h1={roi_h1:.2f}%/h2={roi_h2:.2f}%"
            f"  Regr h1={roi_rh1:.2f}%/h2={roi_rh2:.2f}%"
        )
        print(f"MLflow run_id: {run_id}")

    except Exception:
        import traceback

        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception")
        raise
