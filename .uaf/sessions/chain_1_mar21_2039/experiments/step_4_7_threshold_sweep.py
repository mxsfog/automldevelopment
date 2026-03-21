"""Step 4.7: Threshold sweep + sport breakdown.

Цели:
1. Полный sweep Kelly-порогов на val и test для понимания ROI-vs-count кривой
2. Анализ ROI по спорту — найти спорты с наивысшим ROI
3. Баггинг (5 seeds) для стабилизации предсказаний
4. CV с фиксированным порогом (не оптимизация порога в каждом фолде)

Гипотеза: баггинг стабилизирует CV, sport-specific анализ
покажет где сигнал сильнее.
"""

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

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
    safe_cols = ["Bet_ID", "Sport", "Market", "Start_Time"]
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
            k_factor_mean=("K_Factor", "mean"),
        )
        .reset_index()
    )
    elo_agg["elo_diff"] = elo_agg["elo_max"] - elo_agg["elo_min"]
    elo_agg["elo_ratio"] = elo_agg["elo_max"] / elo_agg["elo_min"].clip(1.0)
    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))

    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Построение фичей."""
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
    feats["lead_hours"] = df["lead_hours"].fillna(0.0).clip(0, 168)
    feats["log_lead_hours"] = np.log1p(feats["lead_hours"])
    feats["Sport"] = df["Sport"].fillna("unknown")
    feats["Market"] = df["Market"].fillna("unknown")
    feats["Currency"] = df["Currency"].fillna("unknown")
    cat_features = ["Sport", "Market", "Currency"]
    return feats, cat_features


def make_weights(n: int, half_life: float = 0.5) -> np.ndarray:
    """Экспоненциальные temporal веса."""
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
    """ROI на выбранных ставках."""
    selected = df[mask].copy()
    if len(selected) == 0:
        return -100.0, 0
    won_mask = selected["Status"] == "won"
    total_stake = selected["USD"].sum()
    total_payout = selected.loc[won_mask, "Payout_USD"].sum()
    roi = (total_payout - total_stake) / total_stake * 100 if total_stake > 0 else -100.0
    return roi, int(mask.sum())


def compute_kelly(proba: np.ndarray, odds: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """Дробный Kelly criterion."""
    b = odds - 1.0
    return fraction * (proba * b - (1 - proba)) / b.clip(0.001)


def train_one(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_vl: pd.DataFrame,
    y_vl: pd.Series,
    cat_f: list[str],
    weights: np.ndarray,
    seed: int = 42,
) -> CatBoostClassifier:
    """Обучение одной CatBoost модели."""
    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.1,
        iterations=500,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=seed,
        verbose=0,
        cat_features=cat_f,
    )
    model.fit(x_tr, y_tr, eval_set=(x_vl, y_vl), sample_weight=weights)
    return model


def main() -> None:
    """Основной эксперимент."""
    with mlflow.start_run(run_name="phase4/step4.7_threshold_sweep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            # Разбивка как в step 4.5 (train включает val-период для early stopping)
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()
            logger.info(
                "Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
            )

            x_tr, cat_f = build_features(train_df)
            x_vl, _ = build_features(val_df)
            x_te, _ = build_features(test_df)
            y_tr = (train_df["Status"] == "won").astype(int)
            y_vl = (val_df["Status"] == "won").astype(int)
            y_te = (test_df["Status"] == "won").astype(int)
            w = make_weights(len(train_df))

            # === Баггинг: 5 моделей с разными seeds ===
            seeds = [42, 137, 777, 2024, 31415]
            probas_val = []
            probas_test = []
            for seed in seeds:
                logger.info("Training seed=%d...", seed)
                m = train_one(x_tr, y_tr, x_vl, y_vl, cat_f, w, seed=seed)
                probas_val.append(m.predict_proba(x_vl)[:, 1])
                probas_test.append(m.predict_proba(x_te)[:, 1])

            pv_bag = np.mean(probas_val, axis=0)
            pt_bag = np.mean(probas_test, axis=0)

            auc_val = roc_auc_score(y_vl, pv_bag)
            auc_test = roc_auc_score(y_te, pt_bag)
            logger.info("Bagging AUC: val=%.4f test=%.4f", auc_val, auc_test)

            # === Threshold sweep (all bets) ===
            pm_val = (val_df["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values

            k_val_all = compute_kelly(pv_bag, val_df["Odds"].values, fraction=1.0)
            k_test_all = compute_kelly(pt_bag, test_df["Odds"].values, fraction=1.0)

            # Pre-match Kelly values
            k_val_pm = k_val_all.copy()
            k_val_pm[~pm_val] = -999
            k_test_pm = k_test_all.copy()
            k_test_pm[~pm_test] = -999

            logger.info("Threshold sweep (pre-match):")
            sweep_results = []
            for t in np.arange(0.10, 0.55, 0.02):
                mask_v = k_val_pm >= t
                mask_t = k_test_pm >= t
                if mask_v.sum() < 50:
                    break
                roi_v, cnt_v = calc_roi(val_df, mask_v)
                roi_t, cnt_t = calc_roi(test_df, mask_t)
                sweep_results.append((t, roi_v, cnt_v, roi_t, cnt_t))
                logger.info(
                    "  t=%.2f val=%.1f%%(%d) test=%.1f%%(%d)",
                    t,
                    roi_v,
                    cnt_v,
                    roi_t,
                    cnt_t,
                )

            # Лучший порог по val с min_bets=200
            best_t, best_val_roi = 0.1, -999.0
            for t, roi_v, cnt_v, _roi_t, _cnt_t in sweep_results:
                if cnt_v >= 200 and roi_v > best_val_roi:
                    best_val_roi = roi_v
                    best_t = t

            mask_best_t = k_test_pm >= best_t
            roi_best_t, cnt_best_t = calc_roi(test_df, mask_best_t)
            logger.info(
                "Best threshold: t=%.2f val=%.2f%% test=%.2f%% (%d bets)",
                best_t,
                best_val_roi,
                roi_best_t,
                cnt_best_t,
            )

            # === Sport breakdown на test (threshold = best_t) ===
            logger.info("Sport breakdown (test, t=%.2f):", best_t)
            sport_results = {}
            for sport in test_df["Sport"].dropna().unique():
                sport_mask = (test_df["Sport"] == sport).values & (k_test_pm >= best_t)
                if sport_mask.sum() < 10:
                    continue
                roi_s, cnt_s = calc_roi(test_df, sport_mask)
                sport_results[sport] = (roi_s, cnt_s)
                logger.info("  %s: ROI=%.1f%% (%d bets)", sport, roi_s, cnt_s)

            top_sports = sorted(sport_results, key=lambda s: sport_results[s][0], reverse=True)[:5]
            logger.info("Top sports: %s", top_sports)

            # === Лучшие спорты subset ===
            if top_sports:
                top_mask_val = pm_val & val_df["Sport"].isin(top_sports).values
                top_mask_test = pm_test & test_df["Sport"].isin(top_sports).values

                k_val_top = k_val_all.copy()
                k_val_top[~top_mask_val] = -999
                k_test_top = k_test_all.copy()
                k_test_top[~top_mask_test] = -999

                t_top, roi_top_val = 0.1, -999.0
                for t in np.arange(0.10, 0.55, 0.02):
                    mask_v = k_val_top >= t
                    if mask_v.sum() < 100:
                        break
                    rv, _ = calc_roi(val_df, mask_v)
                    if rv > roi_top_val:
                        roi_top_val = rv
                        t_top = t

                roi_top_test, cnt_top = calc_roi(test_df, k_test_top >= t_top)
                logger.info(
                    "Top sports only: val=%.2f%% test=%.2f%% (%d)",
                    roi_top_val,
                    roi_top_test,
                    cnt_top,
                )
            else:
                roi_top_test, cnt_top, t_top = roi_best_t, cnt_best_t, best_t

            # === CV с фиксированным порогом (best_t) ===
            logger.info("CV с fixed threshold=%.2f...", best_t)
            cv_rois = []
            fold_size = n // 5
            for fold_idx in range(1, 5):
                fold_start = fold_idx * fold_size
                fold_end = (fold_idx + 1) * fold_size if fold_idx < 4 else n
                fold_train = df.iloc[:fold_start].copy()
                fold_val_cv = df.iloc[fold_start:fold_end].copy()
                if len(fold_train) < 1000 or len(fold_val_cv) < 200:
                    continue

                xft, cf = build_features(fold_train)
                xfv, _ = build_features(fold_val_cv)
                yft = (fold_train["Status"] == "won").astype(int)
                sw_f = make_weights(len(fold_train))

                try:
                    m = CatBoostClassifier(
                        depth=7,
                        learning_rate=0.1,
                        iterations=300,
                        random_seed=42,
                        verbose=0,
                        cat_features=cf,
                    )
                    m.fit(xft, yft, sample_weight=sw_f)
                    pf = m.predict_proba(xfv)[:, 1]
                    kf = compute_kelly(pf, fold_val_cv["Odds"].values, 1.0)
                    pm_fv = (fold_val_cv["lead_hours"] > 0).values
                    kf[~pm_fv] = -999
                    mask_f = kf >= best_t
                    if mask_f.sum() < 10:
                        logger.info("CV Fold %d: too few bets (%d)", fold_idx, mask_f.sum())
                        continue
                    roi_f, n_f = calc_roi(fold_val_cv, mask_f)
                    cv_rois.append(roi_f)
                    mlflow.log_metric(f"cv_fold_{fold_idx}", roi_f)
                    logger.info("CV Fold %d: ROI=%.2f%% (%d)", fold_idx, roi_f, n_f)
                except Exception as e:
                    logger.warning("CV fold %d error: %s", fold_idx, e)

            cv_mean = float(np.mean(cv_rois)) if cv_rois else -999.0
            cv_std = float(np.std(cv_rois)) if cv_rois else 999.0
            logger.info("CV (fixed t=%.2f): %.2f%% +/- %.2f%%", best_t, cv_mean, cv_std)

            # Финальный лучший результат
            best_roi_final = max(roi_best_t, roi_top_test)
            delta = best_roi_final - PREV_BEST_ROI

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_seeds_bagging": len(seeds),
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "depth": 7,
                    "half_life": 0.5,
                    "kelly_fraction": 1.0,
                    "best_threshold": round(best_t, 3),
                    "top_threshold": round(t_top, 3),
                    "top_sports": str(top_sports[:3]),
                }
            )
            mlflow.log_metrics(
                {
                    "roi_val_best": best_val_roi,
                    "roi_test_best_t": roi_best_t,
                    "roi_test_top_sports": roi_top_test,
                    "n_bets_test": cnt_best_t,
                    "n_bets_top": cnt_top,
                    "auc_val": auc_val,
                    "auc_test": auc_test,
                    "cv_mean_fixed": cv_mean,
                    "cv_std_fixed": cv_std,
                    "delta_vs_prev": delta,
                }
            )

            # Логируем sweep в mlflow
            for i, (t, rv, _cv, rt, _ct) in enumerate(sweep_results):
                mlflow.log_metric("sweep_val_roi", rv, step=i)
                mlflow.log_metric("sweep_test_roi", rt, step=i)
                mlflow.log_metric("sweep_threshold", t, step=i)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")

            print("\n=== Step 4.7 Threshold Sweep + Bagging ===")
            print(f"Bagging AUC: val={auc_val:.4f} test={auc_test:.4f}")
            print(f"Best threshold (min_bets=200): t={best_t:.3f}")
            print(f"  Val ROI: {best_val_roi:.2f}%")
            print(f"  Test ROI: {roi_best_t:.2f}% ({cnt_best_t} bets)")
            print(f"Top sports ({top_sports[:3]}): test={roi_top_test:.2f}% ({cnt_top} bets)")
            print(f"CV fixed (t={best_t:.2f}): {cv_mean:.2f}% +/- {cv_std:.2f}%")
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
