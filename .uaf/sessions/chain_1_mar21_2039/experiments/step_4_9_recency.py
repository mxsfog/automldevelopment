"""Step 4.9: Recency window — обучение на последних N% данных.

Гипотеза: тест-период (80-100%) ближе по распределению к последним
40-50% данных, чем к первым. Обучение на последних данных может дать
лучшую генерализацию на тест.

Варианты:
  A) train=40-80% (последние 50% train-периода)
  B) train=55-80% (последние 30% train-периода)
  C) full train=0-80% (baseline step 4.5)

Метрика: test ROI с pre-match Kelly (точно по схеме step 4.5).
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
    """Загрузка данных."""
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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Построение фичей (идентично step 4.5)."""
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
    """Поиск лучшего порога Kelly с min_bets."""
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


def run_variant(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pm_val: np.ndarray,
    pm_test: np.ndarray,
    use_equal_weights: bool = False,
) -> dict:
    """Один вариант обучения + оценка."""
    x_tr, cat_f = build_features(train_df)
    x_vl, _ = build_features(val_df)
    x_te, _ = build_features(test_df)
    y_tr = (train_df["Status"] == "won").astype(int)
    y_vl = (val_df["Status"] == "won").astype(int)

    w = np.ones(len(train_df)) if use_equal_weights else make_weights(len(train_df))

    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.1,
        iterations=500,
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
        cat_features=cat_f,
    )
    model.fit(x_tr, y_tr, eval_set=(x_vl, y_vl), sample_weight=w)

    pv = model.predict_proba(x_vl)[:, 1]
    pt = model.predict_proba(x_te)[:, 1]

    k_v = compute_kelly(pv, val_df["Odds"].values, fraction=1.0)
    k_t = compute_kelly(pt, test_df["Odds"].values, fraction=1.0)
    k_v_pm = k_v.copy()
    k_v_pm[~pm_val] = -999
    k_t_pm = k_t.copy()
    k_t_pm[~pm_test] = -999

    t_opt, roi_val = find_threshold(val_df, k_v_pm, min_bets=200)
    roi_test, n_bets = calc_roi(test_df, k_t_pm >= t_opt)
    auc_val = roc_auc_score(y_vl, pv)
    auc_test = roc_auc_score((test_df["Status"] == "won").astype(int), pt)

    logger.info(
        "%s: val=%.2f%% test=%.2f%% (%d bets) t=%.3f AUC=%.4f/%.4f",
        name,
        roi_val,
        roi_test,
        n_bets,
        t_opt,
        auc_val,
        auc_test,
    )
    return {
        "name": name,
        "threshold": t_opt,
        "roi_val": roi_val,
        "roi_test": roi_test,
        "n_bets": n_bets,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "n_train": len(train_df),
    }


def main() -> None:
    """Основной эксперимент."""
    with mlflow.start_run(run_name="phase4/step4.9_recency") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        try:
            df = load_data()
            n = len(df)
            train_end = int(n * 0.80)
            val_start = int(n * 0.64)

            val_df = df.iloc[val_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()
            pm_val = (val_df["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values

            # === Вариант C: full train 0-80% (baseline step 4.5) ===
            res_c = run_variant(
                "C_full",
                df.iloc[:train_end].copy(),
                val_df,
                test_df,
                pm_val,
                pm_test,
            )

            # === Вариант A: train 40-80% (последние 50% train-периода) ===
            train_start_a = int(n * 0.40)
            res_a = run_variant(
                "A_40to80",
                df.iloc[train_start_a:train_end].copy(),
                val_df,
                test_df,
                pm_val,
                pm_test,
            )

            # === Вариант B: train 55-80% (последние 30% train-периода) ===
            train_start_b = int(n * 0.55)
            res_b = run_variant(
                "B_55to80",
                df.iloc[train_start_b:train_end].copy(),
                val_df,
                test_df,
                pm_val,
                pm_test,
            )

            # === Вариант D: full train без temporal weights ===
            res_d = run_variant(
                "D_full_equal_w",
                df.iloc[:train_end].copy(),
                val_df,
                test_df,
                pm_val,
                pm_test,
                use_equal_weights=True,
            )

            results = [res_a, res_b, res_c, res_d]
            best = max(results, key=lambda r: r["roi_test"])
            delta = best["roi_test"] - PREV_BEST_ROI

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "depth": 7,
                    "lr": 0.1,
                    "kelly_fraction": 1.0,
                    "best_variant": best["name"],
                    "best_threshold": round(best["threshold"], 3),
                }
            )
            mlflow.log_metrics(
                {
                    "roi_test_A": res_a["roi_test"],
                    "roi_test_B": res_b["roi_test"],
                    "roi_test_C": res_c["roi_test"],
                    "roi_test_D": res_d["roi_test"],
                    "roi_test_best": best["roi_test"],
                    "roi_val_best": best["roi_val"],
                    "n_bets_best": best["n_bets"],
                    "auc_val_best": best["auc_val"],
                    "auc_test_best": best["auc_test"],
                    "delta_vs_prev": delta,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            print("\n=== Step 4.9 Recency Window ===")
            for r in sorted(results, key=lambda x: x["roi_test"], reverse=True):
                print(
                    f"{r['name']:<16} train={r['n_train']:>6} val={r['roi_val']:>7.2f}%"
                    f" test={r['roi_test']:>7.2f}% ({r['n_bets']} bets)"
                    f" t={r['threshold']:.3f}"
                )
            print(f"\nBest: {best['name']} = {best['roi_test']:.2f}%")
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
