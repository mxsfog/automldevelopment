"""Step 4.5 — Multi-seed CatBoost ensemble для снижения дисперсии.

Гипотеза: усреднение 5 CatBoost моделей с разными seeds + subsampling
даёт более стабильные вероятности → более точные Kelly значения →
лучший отбор ставок при тех же thresholds.

Ключевое отличие от step 4.1 (LightGBM+CatBoost):
- Одна архитектура (CatBoost), разные seed + subsample/rsm
- Нет проблем с разными диапазонами вероятностей
- Те же Kelly thresholds (калибровка не меняется drastically)

Baseline: ROI=28.5833% (n=233), AUC=0.786
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

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

SEGMENT_THRESHOLDS = {"low": 0.475, "mid": 0.545, "high": 0.325}
SEEDS = [42, 123, 456, 789, 1024]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def check_budget() -> bool:
    try:
        status = json.loads(BUDGET_FILE.read_text())
        return bool(status.get("hard_stop"))
    except FileNotFoundError:
        return False


def load_raw_data() -> pd.DataFrame:
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return feats


def compute_kelly(proba: np.ndarray, odds: np.ndarray) -> np.ndarray:
    b = odds - 1.0
    return (proba * b - (1 - proba)) / b.clip(0.001)


def calc_roi(df: pd.DataFrame, mask: np.ndarray) -> tuple[float, int]:
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
    buckets = pd.cut(df["Odds"], bins=[0, 1.8, 3.0, np.inf], labels=["low", "mid", "high"])
    mask = np.zeros(len(df), dtype=bool)
    for bucket, t in seg_thresholds.items():
        mask |= (buckets == bucket).values & (kelly >= t)
    return mask


def train_single_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    seed: int,
    subsample: float,
    rsm: float,
) -> CatBoostClassifier:
    """Обучить одну CatBoost модель."""
    model = CatBoostClassifier(
        depth=7,
        learning_rate=0.1,
        iterations=500,
        cat_features=cat_features,
        random_seed=seed,
        subsample=subsample,
        rsm=rsm,
        verbose=0,
    )
    model.fit(x_train, y_train, cat_features=cat_features, verbose=0)
    return model


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)

    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    cat_features = ["Sport", "Market", "Currency"]
    x_train = build_features(train_df)
    x_test = build_features(test_df)

    y_train = (train_df["Status"] == "won").astype(int)
    y_test = (test_df["Status"] == "won").astype(int)

    feature_names = list(x_train.columns)
    logger.info(
        "Train: %d, Test: %d | Features: %d", len(train_df), len(test_df), len(feature_names)
    )

    with mlflow.start_run(run_name="phase4/step_4_5_multiseed_ensemble") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.5")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed_list": str(SEEDS),
                "n_models": len(SEEDS),
                "n_samples_train": len(x_train),
                "n_samples_test": len(x_test),
                "n_features": len(feature_names),
                "depth": 7,
                "learning_rate": 0.1,
                "iterations": 500,
                "subsample": 0.8,
                "rsm": 0.8,
                "seg_low": SEGMENT_THRESHOLDS["low"],
                "seg_mid": SEGMENT_THRESHOLDS["mid"],
                "seg_high": SEGMENT_THRESHOLDS["high"],
            }
        )

        try:
            # Обучить 5 моделей
            models = []
            for seed in SEEDS:
                logger.info("Обучаю модель seed=%d...", seed)
                m = train_single_model(x_train, y_train, cat_features, seed, 0.8, 0.8)
                models.append(m)
                auc_i = roc_auc_score(y_test, m.predict_proba(x_test)[:, 1])
                logger.info("  seed=%d AUC=%.4f", seed, auc_i)
                mlflow.log_metric(f"auc_seed_{seed}", auc_i)

            # Averaged probabilities
            proba_arrays = [m.predict_proba(x_test)[:, 1] for m in models]
            proba_ensemble = np.mean(proba_arrays, axis=0)

            auc_ensemble = roc_auc_score(y_test, proba_ensemble)
            logger.info("Ensemble AUC: %.4f (baseline=0.786)", auc_ensemble)
            mlflow.log_metric("auc", auc_ensemble)

            # Корреляции между моделями
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                corr = np.corrcoef(proba_arrays[i], proba_arrays[j])[0, 1]
                logger.info("  corr(seed_%d, seed_%d) = %.4f", SEEDS[i], SEEDS[j], corr)
                mlflow.log_metric(f"corr_{SEEDS[i]}_{SEEDS[j]}", corr)

            # ROI с Kelly thresholds на ensemble
            kelly_ensemble = compute_kelly(proba_ensemble, test_df["Odds"].values)
            lead_hours = (
                test_df["Start_Time"] - test_df["Created_At"]
            ).dt.total_seconds() / 3600.0
            kelly_ensemble[lead_hours.values <= 0] = -999

            mkt_mask = test_df["Market"].values == "1x2"
            seg_mask = apply_shrunken_segments(test_df, kelly_ensemble, SEGMENT_THRESHOLDS)
            final_mask = mkt_mask & seg_mask

            roi, n_bets = calc_roi(test_df, final_mask)
            logger.info("Ensemble ROI: %.4f%% (n=%d)", roi, n_bets)
            mlflow.log_metrics({"roi": roi, "n_selected": n_bets})

            # Сравнить с baseline single model
            baseline_roi = 28.5833
            delta = roi - baseline_roi
            mlflow.log_metric("roi_delta", delta)
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            if roi > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                sys.exit(1)

            # Для сравнения: single model baseline (seed=42, no subsample)
            model_single = CatBoostClassifier(
                depth=7,
                learning_rate=0.1,
                iterations=500,
                cat_features=cat_features,
                random_seed=42,
                verbose=0,
            )
            model_single.fit(x_train, y_train, cat_features=cat_features, verbose=0)
            proba_single = model_single.predict_proba(x_test)[:, 1]
            kelly_single = compute_kelly(proba_single, test_df["Odds"].values)
            kelly_single[lead_hours.values <= 0] = -999
            seg_mask_single = apply_shrunken_segments(test_df, kelly_single, SEGMENT_THRESHOLDS)
            roi_single, n_single = calc_roi(test_df, mkt_mask & seg_mask_single)
            logger.info(
                "Single model (seed=42, no subsample): ROI=%.4f%% n=%d", roi_single, n_single
            )
            mlflow.log_metrics({"roi_single": roi_single, "n_single": n_single})

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            logger.info(
                "RESULT: ensemble=%.4f%% n=%d | single=%.4f%% n=%d | delta=%.4f",
                roi,
                n_bets,
                roi_single,
                n_single,
                delta,
            )
            print(f"STEP_4_5_ROI={roi:.6f}")
            print(f"STEP_4_5_N={n_bets}")
            print(f"STEP_4_5_AUC={auc_ensemble:.4f}")
            print(f"STEP_4_5_DELTA={delta:.4f}")
            print(f"STEP_4_5_SINGLE_ROI={roi_single:.4f}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.5")
            raise


if __name__ == "__main__":
    main()
