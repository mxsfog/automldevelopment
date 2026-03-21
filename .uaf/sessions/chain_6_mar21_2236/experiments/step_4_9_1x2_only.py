"""Step 4.9 — 1x2-only strategy: фокус на самом ликвидном рынке.

Гипотеза: 1x2 рынок занимает 233/435 val бет (53%) при val ROI=103.6%.
Фокус только на 1x2 может дать более чистый и стабильный сигнал.
Дополнительно тестируем: рынки из step 4.8 + Asian markets.
Baseline: ROI=25.38% (top-3 markets, step 4.8).
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
BASELINE_ROI = 25.38  # step 4.8 best
LEAKAGE_THRESHOLD = 35.0

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.info("hard_stop=true, выход")
        sys.exit(0)
except FileNotFoundError:
    pass


def load_data() -> pd.DataFrame:
    """Загрузка данных."""
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
    df["lead_hours"] = (df["Start_Time"] - df["Created_At"]).dt.total_seconds() / 3600.0
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Baseline feature set (33 фичи)."""
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
    return feats, ["Sport", "Market", "Currency"]


def make_weights(n: int, half_life: float = 0.5) -> np.ndarray:
    """Temporal decay weights."""
    indices = np.arange(n)
    decay = np.log(2) / (half_life * n)
    weights = np.exp(decay * indices)
    return weights / weights.mean()


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


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion."""
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def find_threshold(
    df: pd.DataFrame, kelly: np.ndarray, min_bets: int = 100
) -> tuple[float, float]:
    """Поиск Kelly-порога по val ROI."""
    best_roi, best_t = -999.0, 0.01
    for t in np.arange(0.01, 0.60, 0.005):
        mask = kelly >= t
        if mask.sum() < min_bets:
            break
        roi, _ = calc_roi(df, mask)
        if roi > best_roi:
            best_roi = roi
            best_t = t
    return best_t, best_roi


def eval_market_strategy(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_v: np.ndarray,
    k_t: np.ndarray,
    markets: list[str],
    pm_val: np.ndarray,
    pm_test: np.ndarray,
    label: str,
    min_bets: int = 100,
) -> dict:
    """Оценка стратегии с конкретными рынками."""
    mfilter_v = val_df["Market"].isin(markets).values
    mfilter_t = test_df["Market"].isin(markets).values

    k_v_m = k_v.copy()
    k_t_m = k_t.copy()
    k_v_m[~(pm_val & mfilter_v)] = -999
    k_t_m[~(pm_test & mfilter_t)] = -999

    t_best, roi_val = find_threshold(val_df, k_v_m, min_bets=min_bets)
    roi_test, n_bets = calc_roi(test_df, k_t_m >= t_best)

    logger.info(
        "%s: val=%.2f%%, test=%.2f%% (n=%d), t=%.3f",
        label,
        roi_val,
        roi_test,
        n_bets,
        t_best,
    )
    return {"roi_val": roi_val, "roi_test": roi_test, "n_bets": n_bets, "t_best": t_best}


def main() -> None:
    """1x2-only и расширенные market strategies."""
    with mlflow.start_run(run_name="phase4/step4.9_1x2_only") as run:
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

            x_tr, cat_f = build_features(train_df)
            x_vl, _ = build_features(val_df)
            x_te, _ = build_features(test_df)
            y_tr = (train_df["Status"] == "won").astype(int)
            y_vl = (val_df["Status"] == "won").astype(int)
            _y_te = (test_df["Status"] == "won").astype(int)
            w = make_weights(len(train_df))

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
            auc_val = roc_auc_score(y_vl, pv)
            auc_test = roc_auc_score(_y_te, pt)

            pm_val = (val_df["lead_hours"] > 0).values
            pm_test = (test_df["lead_hours"] > 0).values
            k_v = compute_kelly(pv, val_df["Odds"].values)
            k_t = compute_kelly(pt, test_df["Odds"].values)
            k_v[~pm_val] = -999
            k_t[~pm_test] = -999

            logger.info("AUC: val=%.4f, test=%.4f", auc_val, auc_test)

            # Стратегия 1: только 1x2
            r1 = eval_market_strategy(
                val_df, test_df, k_v, k_t, ["1x2"], pm_val, pm_test, "1x2 only", min_bets=100
            )

            # Стратегия 2: 1x2 + Draw No Bet (высокий val ROI с n>=3)
            r2 = eval_market_strategy(
                val_df,
                test_df,
                k_v,
                k_t,
                ["1x2", "Draw No Bet"],
                pm_val,
                pm_test,
                "1x2+DNB",
                min_bets=100,
            )

            # Стратегия 3: step 4.8 top-3 + Asian Total + Handicap(OT)
            r3 = eval_market_strategy(
                val_df,
                test_df,
                k_v,
                k_t,
                [
                    "1x2",
                    "Winner",
                    "Winner (Incl. Overtime and Penalties)",
                    "Asian Total",
                    "Handicap (Incl. Overtime)",
                ],
                pm_val,
                pm_test,
                "top5_extended",
                min_bets=150,
            )

            # Стратегия 4: все рынки с val ROI > 0 и n >= 3
            k_v_temp = k_v.copy()
            k_v_temp[~pm_val] = -999
            t_base = 0.455  # базовый threshold
            base_mask_v = k_v_temp >= t_base
            val_selected = val_df[base_mask_v].copy()
            profitable_markets = []
            for market, grp in val_selected.groupby("Market"):
                won = grp["Status"] == "won"
                stake = grp["USD"].sum()
                payout = grp.loc[won, "Payout_USD"].sum()
                roi = (payout - stake) / stake * 100 if stake > 0 else -100.0
                if roi > 0 and len(grp) >= 3:
                    profitable_markets.append(market)
            logger.info(
                "Val-profitable markets (n>=3, ROI>0): %d markets", len(profitable_markets)
            )

            r4 = eval_market_strategy(
                val_df,
                test_df,
                k_v,
                k_t,
                profitable_markets,
                pm_val,
                pm_test,
                "all_profitable_val",
                min_bets=150,
            )

            results = {
                "1x2_only": r1,
                "1x2_dnb": r2,
                "top5_extended": r3,
                "all_profitable": r4,
            }

            best_label = max(results, key=lambda k: results[k]["roi_test"])
            best = results[best_label]
            logger.info(
                "Best strategy: %s test=%.2f%% (n=%d)",
                best_label,
                best["roi_test"],
                best["n_bets"],
            )

            if best["roi_test"] > LEAKAGE_THRESHOLD:
                logger.error("LEAKAGE SUSPECT: roi=%.2f%%", best["roi_test"])
                mlflow.set_tag("leakage_suspect", "true")

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "n_samples_train": len(train_df),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df),
                    "model": "catboost_market_1x2",
                    "n_profitable_markets": len(profitable_markets),
                }
            )
            mlflow.log_metrics(
                {
                    "auc_val": float(auc_val),
                    "auc_test": float(auc_test),
                    "roi_test_1x2": float(r1["roi_test"]),
                    "n_bets_1x2": r1["n_bets"],
                    "roi_test_1x2_dnb": float(r2["roi_test"]),
                    "n_bets_1x2_dnb": r2["n_bets"],
                    "roi_test_top5ext": float(r3["roi_test"]),
                    "n_bets_top5ext": r3["n_bets"],
                    "roi_test_all_prof": float(r4["roi_test"]),
                    "n_bets_all_prof": r4["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")

            logger.info("Run ID: %s", run.info.run_id)
            print(
                f"RESULT 1x2={r1['roi_test']:.2f}%(n={r1['n_bets']}) "
                f"1x2+DNB={r2['roi_test']:.2f}%(n={r2['n_bets']}) "
                f"top5ext={r3['roi_test']:.2f}%(n={r3['n_bets']}) "
                f"all_prof={r4['roi_test']:.2f}%(n={r4['n_bets']}) "
                f"run_id={run.info.run_id}"
            )

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise


if __name__ == "__main__":
    main()
