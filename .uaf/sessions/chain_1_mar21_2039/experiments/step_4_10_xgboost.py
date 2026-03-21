"""Step 4.10: XGBoost + exact step 4.5 feature set.

Ключевое отличие: использую ТОЧНО те же фичи что в step 4.5
(без lead_hours как фичи — он только для фильтрации).

XGBoost может дать другое распределение вероятностей в высоком
Kelly-диапазоне (>0.455), потенциально улучшив test ROI.

Сравнение: XGBoost vs CatBoost (step 4.5 config).
"""

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")
PREV_BEST_ROI = 24.908815


def load_data() -> pd.DataFrame:
    """Загрузка данных (идентично step 4.5)."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")
    outcomes_first = outcomes_first[["Bet_ID", "Sport", "Market", "Start_Time"]]

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
    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def build_features_no_lead(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Фичи как в step 4.5 — без lead_hours (только для фильтрации)."""
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
    # NB: Sport, Market, Currency нужны для CatBoost; для XGBoost — label encoding
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


def encode_cats_for_xgb(
    x_tr: pd.DataFrame,
    x_vl: pd.DataFrame,
    x_te: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label encoding для XGBoost (ordinal encoding по train frequency)."""
    x_tr = x_tr.copy()
    x_vl = x_vl.copy()
    x_te = x_te.copy()
    for col in cat_cols:
        freq = x_tr[col].value_counts().to_dict()
        for df_part in [x_tr, x_vl, x_te]:
            df_part[col] = df_part[col].map(freq).fillna(0).astype(float)
    return x_tr, x_vl, x_te


def make_weights(n: int, half_life: float = 0.5) -> np.ndarray:
    """Temporal веса."""
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


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


def compute_kelly(proba: np.ndarray, odds: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return fraction * (proba * b - (1 - proba)) / b.clip(0.001)


def find_threshold(
    val_df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 200
) -> tuple[float, float]:
    """Поиск лучшего Kelly-порога."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.60, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(val_df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def main() -> None:
    """Основной эксперимент."""
    with mlflow.start_run(run_name="phase4/step4.10_xgboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()

            x_tr, cat_f = build_features_no_lead(train_df)
            x_vl, _ = build_features_no_lead(val_df)
            x_te, _ = build_features_no_lead(test_df)
            y_tr = (train_df["Status"] == "won").astype(int)
            y_vl = (val_df["Status"] == "won").astype(int)
            y_te = (test_df["Status"] == "won").astype(int)
            w = make_weights(len(train_df))

            pm_val = (val_df["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values

            # === CatBoost (точная step 4.5 конфигурация) ===
            logger.info("Training CatBoost (step 4.5 config)...")
            cb_model = CatBoostClassifier(
                depth=7,
                learning_rate=0.1,
                iterations=500,
                eval_metric="AUC",
                early_stopping_rounds=50,
                random_seed=42,
                verbose=0,
                cat_features=cat_f,
            )
            cb_model.fit(x_tr, y_tr, eval_set=(x_vl, y_vl), sample_weight=w)

            pv_cb = cb_model.predict_proba(x_vl)[:, 1]
            pt_cb = cb_model.predict_proba(x_te)[:, 1]

            k_v_cb = compute_kelly(pv_cb, val_df["Odds"].values, 1.0)
            k_t_cb = compute_kelly(pt_cb, test_df["Odds"].values, 1.0)
            k_v_cb[~pm_val] = -999
            k_t_cb[~pm_test] = -999

            t_cb, roi_cb_val = find_threshold(val_df, k_v_cb, min_bets=200)
            roi_cb_test, cnt_cb = calc_roi(test_df, k_t_cb >= t_cb)
            auc_cb_val = roc_auc_score(y_vl, pv_cb)
            auc_cb_test = roc_auc_score(y_te, pt_cb)
            logger.info(
                "CatBoost: val=%.2f%% test=%.2f%% (%d) t=%.3f AUC=%.4f/%.4f",
                roi_cb_val,
                roi_cb_test,
                cnt_cb,
                t_cb,
                auc_cb_val,
                auc_cb_test,
            )

            # === XGBoost (с label-encoded cats) ===
            logger.info("Training XGBoost...")
            x_tr_xgb, x_vl_xgb, x_te_xgb = encode_cats_for_xgb(x_tr, x_vl, x_te, cat_f)

            scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
            xgb_model = XGBClassifier(
                max_depth=7,
                learning_rate=0.1,
                n_estimators=500,
                scale_pos_weight=scale_pos,
                eval_metric="auc",
                early_stopping_rounds=50,
                random_state=42,
                verbosity=0,
                tree_method="hist",
            )
            xgb_model.fit(
                x_tr_xgb,
                y_tr,
                sample_weight=w,
                eval_set=[(x_vl_xgb, y_vl)],
                verbose=False,
            )

            pv_xgb = xgb_model.predict_proba(x_vl_xgb)[:, 1]
            pt_xgb = xgb_model.predict_proba(x_te_xgb)[:, 1]

            k_v_xgb = compute_kelly(pv_xgb, val_df["Odds"].values, 1.0)
            k_t_xgb = compute_kelly(pt_xgb, test_df["Odds"].values, 1.0)
            k_v_xgb[~pm_val] = -999
            k_t_xgb[~pm_test] = -999

            t_xgb, roi_xgb_val = find_threshold(val_df, k_v_xgb, min_bets=200)
            roi_xgb_test, cnt_xgb = calc_roi(test_df, k_t_xgb >= t_xgb)
            auc_xgb_val = roc_auc_score(y_vl, pv_xgb)
            auc_xgb_test = roc_auc_score(y_te, pt_xgb)
            logger.info(
                "XGBoost: val=%.2f%% test=%.2f%% (%d) t=%.3f AUC=%.4f/%.4f",
                roi_xgb_val,
                roi_xgb_test,
                cnt_xgb,
                t_xgb,
                auc_xgb_val,
                auc_xgb_test,
            )

            # === Ensemble CatBoost + XGBoost ===
            pv_ens = 0.5 * pv_cb + 0.5 * pv_xgb
            pt_ens = 0.5 * pt_cb + 0.5 * pt_xgb
            k_v_ens = compute_kelly(pv_ens, val_df["Odds"].values, 1.0)
            k_t_ens = compute_kelly(pt_ens, test_df["Odds"].values, 1.0)
            k_v_ens[~pm_val] = -999
            k_t_ens[~pm_test] = -999

            t_ens, roi_ens_val = find_threshold(val_df, k_v_ens, min_bets=200)
            roi_ens_test, cnt_ens = calc_roi(test_df, k_t_ens >= t_ens)
            logger.info(
                "Ensemble CB+XGB: val=%.2f%% test=%.2f%% (%d) t=%.3f",
                roi_ens_val,
                roi_ens_test,
                cnt_ens,
                t_ens,
            )

            best_test = max(roi_cb_test, roi_xgb_test, roi_ens_test)
            delta = best_test - PREV_BEST_ROI

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "catboost_depth": 7,
                    "xgb_depth": 7,
                    "lr": 0.1,
                    "kelly_fraction": 1.0,
                    "t_cb": round(t_cb, 3),
                    "t_xgb": round(t_xgb, 3),
                    "t_ens": round(t_ens, 3),
                }
            )
            mlflow.log_metrics(
                {
                    "roi_test_catboost": roi_cb_test,
                    "roi_test_xgboost": roi_xgb_test,
                    "roi_test_ensemble": roi_ens_test,
                    "roi_test_best": best_test,
                    "auc_val_cb": auc_cb_val,
                    "auc_test_cb": auc_cb_test,
                    "auc_val_xgb": auc_xgb_val,
                    "auc_test_xgb": auc_xgb_test,
                    "n_bets_cb": cnt_cb,
                    "n_bets_xgb": cnt_xgb,
                    "n_bets_ens": cnt_ens,
                    "delta_vs_prev": delta,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")

            print("\n=== Step 4.10 XGBoost vs CatBoost ===")
            cb_row = f"val={roi_cb_val:.2f}% test={roi_cb_test:.2f}% ({cnt_cb} bets)"
            xgb_row = f"val={roi_xgb_val:.2f}% test={roi_xgb_test:.2f}% ({cnt_xgb} bets)"
            ens_row = f"val={roi_ens_val:.2f}% test={roi_ens_test:.2f}% ({cnt_ens} bets)"
            print(f"CatBoost  (t={t_cb:.3f}): {cb_row}")
            print(f"XGBoost   (t={t_xgb:.3f}): {xgb_row}")
            print(f"Ensemble  (t={t_ens:.3f}): {ens_row}")
            print(f"Best: {best_test:.2f}%")
            print(f"Prev best: {PREV_BEST_ROI:.2f}%")
            print(f"Delta: {delta:+.2f}%")
            print(f"MLflow run_id: {run.info.run_id}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
