"""Step 4.1 — Selection/Tournament target encoding.

Гипотеза: добавить признаки из outcomes.csv:
- is_draw: ставка на ничью
- selection_winrate: smoothed target encoding Selection (команда)
- tournament_winrate: smoothed target encoding Tournament
- is_big5: Premier League / La Liga / Bundesliga / Serie A / Ligue 1
- is_favorite: Selection == low_odds team (odds < 2.0)

Baseline: ROI=28.5833% (n=233)
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

# Baseline thresholds из chain_6/7
SEGMENT_THRESHOLDS = {"low": 0.475, "mid": 0.545, "high": 0.325}
BIG5_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def check_budget() -> bool:
    try:
        status = json.loads(BUDGET_FILE.read_text())
        return bool(status.get("hard_stop"))
    except FileNotFoundError:
        return False


def load_raw_data() -> pd.DataFrame:
    """Загрузка данных с Selection и Tournament из outcomes.csv."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time", "Selection", "Tournament"]
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


def smoothed_target_encode(
    train_series: pd.Series,
    train_labels: pd.Series,
    test_series: pd.Series,
    global_mean: float,
    min_samples_leaf: int = 10,
    smoothing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed mean target encoding с Bayesian smoothing.

    smoother = 1 / (1 + exp(-(count - min_samples_leaf) / smoothing))
    encoded = smoother * category_mean + (1 - smoother) * global_mean
    """
    stats = train_series.to_frame("cat").assign(label=train_labels.values)
    agg = stats.groupby("cat")["label"].agg(["mean", "count"]).reset_index()
    agg.columns = ["cat", "cat_mean", "cat_count"]
    smoother = 1.0 / (1.0 + np.exp(-(agg["cat_count"] - min_samples_leaf) / smoothing))
    agg["encoded"] = smoother * agg["cat_mean"] + (1 - smoother) * global_mean
    encode_map = dict(zip(agg["cat"], agg["encoded"], strict=True))

    train_enc = train_series.map(encode_map).fillna(global_mean).values
    test_enc = test_series.map(encode_map).fillna(global_mean).values
    return train_enc, test_enc


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set с новыми Selection/Tournament признаками."""
    feats = pd.DataFrame(index=df.index)
    # Базовые (идентичны chain_8)
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
    # Новые фичи (заполняются снаружи через target encoding)
    feats["is_draw"] = (df["Selection"].fillna("") == "Draw").astype(int)
    feats["is_big5"] = df["Tournament"].fillna("").isin(BIG5_LEAGUES).astype(int)
    feats["is_favorite"] = (df["Odds"] < 2.0).astype(int)
    # selection_winrate и tournament_winrate добавляются снаружи
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


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    df_raw = load_raw_data()
    n = len(df_raw)
    train_end = int(n * 0.80)
    train_df = df_raw.iloc[:train_end].copy()
    test_df = df_raw.iloc[train_end:].copy()

    # Метка для target encoding
    y_train_raw = (train_df["Status"] == "won").astype(int)
    global_mean = y_train_raw.mean()
    logger.info(
        "Train: %d, Test: %d | global_win_rate: %.3f", len(train_df), len(test_df), global_mean
    )

    # Target encoding Selection
    sel_train, sel_test = smoothed_target_encode(
        train_df["Selection"].fillna("__missing__"),
        y_train_raw,
        test_df["Selection"].fillna("__missing__"),
        global_mean=global_mean,
        min_samples_leaf=15,
        smoothing=1.5,
    )
    train_df["selection_winrate"] = sel_train
    test_df["selection_winrate"] = sel_test

    # Target encoding Tournament
    tourn_train, tourn_test = smoothed_target_encode(
        train_df["Tournament"].fillna("__missing__"),
        y_train_raw,
        test_df["Tournament"].fillna("__missing__"),
        global_mean=global_mean,
        min_samples_leaf=20,
        smoothing=1.5,
    )
    train_df["tournament_winrate"] = tourn_train
    test_df["tournament_winrate"] = tourn_test

    # Feature matrices
    x_train = build_features(train_df)
    x_train["selection_winrate"] = sel_train
    x_train["tournament_winrate"] = tourn_train

    x_test = build_features(test_df)
    x_test["selection_winrate"] = sel_test
    x_test["tournament_winrate"] = tourn_test

    cat_features = ["Sport", "Market", "Currency"]
    y_train = (train_df["Status"] == "won").astype(int)
    y_test = (test_df["Status"] == "won").astype(int)

    feature_names = list(x_train.columns)
    logger.info("Features: %d (baseline was 34, new: %d)", len(feature_names), len(feature_names))

    with mlflow.start_run(run_name="phase4/step_4_1_selection_features") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.1")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "n_samples_train": len(x_train),
                "n_samples_test": len(x_test),
                "n_features": len(feature_names),
                "new_features": "is_draw,is_big5,is_favorite,selection_winrate,tournament_winrate",
                "smoothing_selection": 1.5,
                "smoothing_tournament": 1.5,
                "min_samples_leaf": 15,
                "depth": 7,
                "learning_rate": 0.1,
                "iterations": 500,
                "seg_low": SEGMENT_THRESHOLDS["low"],
                "seg_mid": SEGMENT_THRESHOLDS["mid"],
                "seg_high": SEGMENT_THRESHOLDS["high"],
            }
        )

        try:
            model = CatBoostClassifier(
                depth=7,
                learning_rate=0.1,
                iterations=500,
                cat_features=cat_features,
                random_seed=42,
                verbose=100,
                eval_metric="AUC",
            )

            # Split train into sub-train/val (80/20) for early stopping
            val_start = int(len(x_train) * 0.8)
            x_sub_train = x_train.iloc[:val_start]
            x_val = x_train.iloc[val_start:]
            y_sub_train = y_train.iloc[:val_start]
            y_val = y_train.iloc[val_start:]

            model.fit(
                x_sub_train,
                y_sub_train,
                cat_features=cat_features,
                eval_set=(x_val, y_val),
                early_stopping_rounds=50,
                verbose=100,
            )

            auc_test = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
            logger.info("Test AUC: %.4f", auc_test)
            mlflow.log_metric("auc", auc_test)

            # Оценка ROI с baseline Kelly thresholds
            test_df_eval = test_df.copy()
            proba_test = model.predict_proba(x_test)[:, 1]
            odds_test = test_df_eval["Odds"].values
            kelly_test = compute_kelly(proba_test, odds_test)

            lead_hours = (
                test_df_eval["Start_Time"] - test_df_eval["Created_At"]
            ).dt.total_seconds() / 3600.0
            kelly_test[lead_hours.values <= 0] = -999

            mkt_mask = test_df_eval["Market"].values == "1x2"
            seg_mask = apply_shrunken_segments(test_df_eval, kelly_test, SEGMENT_THRESHOLDS)
            final_mask = mkt_mask & seg_mask

            roi, n_bets = calc_roi(test_df_eval, final_mask)
            logger.info("Test ROI: %.4f%% (n=%d)", roi, n_bets)
            mlflow.log_metrics({"roi": roi, "n_selected": n_bets})

            baseline_roi = 28.5833
            delta = roi - baseline_roi
            mlflow.log_metric("roi_delta", delta)
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            # Проверка на leakage
            if roi > 35.0:
                logger.warning("ROI > 35%% — подозрение на leakage! Прерываем.")
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("failure_reason", f"ROI={roi:.2f} > 35% — leakage suspected")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            logger.info("RESULT: ROI=%.4f%% n=%d delta=%.4f vs baseline", roi, n_bets, delta)
            print(f"STEP_4_1_ROI={roi:.6f}")
            print(f"STEP_4_1_N={n_bets}")
            print(f"STEP_4_1_AUC={auc_test:.4f}")
            print(f"STEP_4_1_DELTA={delta:.4f}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")

            # Сохранить новый pipeline если улучшение > 0
            if delta > 0.0:
                logger.info("Улучшение: delta=%.4f. Сохраняем pipeline.", delta)
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)

                # Retrain на полном train для pipeline
                model_full = CatBoostClassifier(
                    depth=7,
                    learning_rate=0.1,
                    iterations=model.best_iteration_ or 500,
                    cat_features=cat_features,
                    random_seed=42,
                    verbose=100,
                )
                model_full.fit(x_train, y_train, cat_features=cat_features, verbose=100)

                # Сохранить encode maps для pipeline
                sel_encode_map = _build_encode_map(
                    train_df["Selection"].fillna("__missing__"), y_train_raw, global_mean, 15, 1.5
                )
                tourn_encode_map = _build_encode_map(
                    train_df["Tournament"].fillna("__missing__"),
                    y_train_raw,
                    global_mean,
                    20,
                    1.5,
                )

                pipeline_data = {
                    "model": model_full,
                    "feature_names": feature_names,
                    "segment_thresholds": SEGMENT_THRESHOLDS,
                    "sel_encode_map": sel_encode_map,
                    "tourn_encode_map": tourn_encode_map,
                    "global_mean": global_mean,
                    "cat_features": cat_features,
                }
                joblib.dump(pipeline_data, best_dir / "pipeline_step4_1.pkl")
                model_full.save_model(str(best_dir / "model.cbm"))

                metadata = {
                    "framework": "catboost",
                    "roi": roi,
                    "auc": auc_test,
                    "segment_thresholds": SEGMENT_THRESHOLDS,
                    "market_filter": "1x2",
                    "n_bets": n_bets,
                    "feature_names": feature_names,
                    "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
                    "session_id": SESSION_ID,
                    "step": "4.1",
                    "new_features": [
                        "is_draw",
                        "is_big5",
                        "is_favorite",
                        "selection_winrate",
                        "tournament_winrate",
                    ],
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Pipeline сохранён: %s", best_dir)

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.1")
            raise


def _build_encode_map(
    series: pd.Series,
    labels: pd.Series,
    global_mean: float,
    min_samples_leaf: int,
    smoothing: float,
) -> dict[str, float]:
    stats = series.to_frame("cat").assign(label=labels.values)
    agg = stats.groupby("cat")["label"].agg(["mean", "count"]).reset_index()
    agg.columns = ["cat", "cat_mean", "cat_count"]
    smoother = 1.0 / (1.0 + np.exp(-(agg["cat_count"] - min_samples_leaf) / smoothing))
    agg["encoded"] = smoother * agg["cat_mean"] + (1 - smoother) * global_mean
    return dict(zip(agg["cat"], agg["encoded"], strict=True))


if __name__ == "__main__":
    main()
