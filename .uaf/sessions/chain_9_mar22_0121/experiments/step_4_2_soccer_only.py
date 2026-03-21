"""Step 4.2 — Soccer-only CatBoost + H/D/A direction features.

Гипотеза: модель, обученная только на Soccer бетах, даст лучшую калибровку
для Soccer 1x2 бетов. Дополнительно: признак bet_direction (Home/Draw/Away),
выведенный из Match без использования target (не leakage).

Признаки:
- bet_direction: H/D/A из Match + Selection сравнения (no leakage)
- is_home / is_draw / is_away: binary flags

Baseline: ROI=28.5833% (n=233) — chain_8 pipeline.pkl
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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def check_budget() -> bool:
    try:
        status = json.loads(BUDGET_FILE.read_text())
        return bool(status.get("hard_stop"))
    except FileNotFoundError:
        return False


def extract_direction(match: str | None, selection: str | None) -> str:
    """Определить H/D/A направление ставки без использования результата.

    Возвращает: 'H' (home), 'D' (draw), 'A' (away), 'U' (unknown).
    """
    if pd.isna(selection) or pd.isna(match):
        return "U"
    sel = str(selection).strip().lower()
    if sel == "draw":
        return "D"
    match_str = str(match)
    if " vs " in match_str:
        home_team = match_str.split(" vs ")[0].strip().lower()
        away_team = match_str.split(" vs ")[1].strip().lower()
        # Нечёткое сравнение: проверяем первые 8 символов (обрезание артефактов)
        if sel[:8] == home_team[:8] or sel in home_team or home_team in sel:
            return "H"
        if sel[:8] == away_team[:8] or sel in away_team or away_team in sel:
            return "A"
    return "U"


def load_raw_data(soccer_only: bool = False) -> pd.DataFrame:
    """Загрузка данных с Match/Selection из outcomes.csv."""
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    exclude = {"pending", "cancelled", "error", "cashout"}
    bets = bets[~bets["Status"].isin(exclude)].copy()

    outcomes_first = outcomes.drop_duplicates(subset="Bet_ID", keep="first")[
        ["Bet_ID", "Sport", "Market", "Start_Time", "Selection", "Match"]
    ]
    df = bets.merge(outcomes_first, left_on="ID", right_on="Bet_ID", how="left")
    df["Created_At"] = pd.to_datetime(df["Created_At"], utc=True)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    df = df.sort_values("Created_At").reset_index(drop=True)

    if soccer_only:
        df = df[df["Sport"] == "Soccer"].reset_index(drop=True)
        logger.info("Soccer-only filter: %d bets", len(df))

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
    """Feature set с добавленными direction фичами."""
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
    # Новые direction фичи
    feats["bet_direction"] = df["bet_direction"].fillna("U")  # категориальная
    feats["is_home"] = (df["bet_direction"] == "H").astype(int)
    feats["is_draw"] = (df["bet_direction"] == "D").astype(int)
    feats["is_away"] = (df["bet_direction"] == "A").astype(int)
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


def evaluate_on_test(model: CatBoostClassifier, test_df: pd.DataFrame) -> tuple[float, int]:
    """Вычислить ROI на test с baseline Kelly thresholds."""
    x_test = build_features(test_df)
    proba = model.predict_proba(x_test)[:, 1]
    odds = test_df["Odds"].values
    kelly = compute_kelly(proba, odds)

    lead_hours = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600.0
    kelly[lead_hours.values <= 0] = -999

    mkt_mask = test_df["Market"].values == "1x2"
    seg_mask = apply_shrunken_segments(test_df, kelly, SEGMENT_THRESHOLDS)
    final_mask = mkt_mask & seg_mask
    return calc_roi(test_df, final_mask)


def main() -> None:
    if check_budget():
        logger.info("hard_stop=true, выход")
        sys.exit(0)

    # Загружаем данные — test set всегда из полного набора (временной split)
    df_all = load_raw_data(soccer_only=False)
    n_all = len(df_all)
    train_all_end = int(n_all * 0.80)

    # Для test — только Soccer (правило применения фильтра)
    # Но нам нужен test по времени, поэтому берём последние 20% всего набора
    test_df_all = df_all.iloc[train_all_end:].copy()

    # Добавить direction фичу для test
    test_df_all["bet_direction"] = [
        extract_direction(m, s)
        for m, s in zip(test_df_all["Match"], test_df_all["Selection"], strict=False)
    ]

    # Soccer-only train split по времени
    df_soccer = load_raw_data(soccer_only=True)
    n_soccer = len(df_soccer)
    train_soccer_end = int(n_soccer * 0.80)
    train_df = df_soccer.iloc[:train_soccer_end].copy()
    # val_df — для анализа (не для threshold tuning из-за inflation)
    val_df = df_soccer.iloc[train_soccer_end:].copy()

    logger.info(
        "Soccer train: %d, Soccer val: %d, All test: %d",
        len(train_df),
        len(val_df),
        len(test_df_all),
    )

    # Добавить direction фичи
    train_df["bet_direction"] = [
        extract_direction(m, s)
        for m, s in zip(train_df["Match"], train_df["Selection"], strict=False)
    ]
    val_df["bet_direction"] = [
        extract_direction(m, s) for m, s in zip(val_df["Match"], val_df["Selection"], strict=False)
    ]

    # direction distribution в Soccer 1x2
    soccer_1x2_train = train_df[train_df["Market"] == "1x2"]
    dir_counts = soccer_1x2_train["bet_direction"].value_counts()
    logger.info("Direction distribution (Soccer 1x2 train): %s", dict(dir_counts))

    x_train = build_features(train_df)
    x_val_eval = build_features(val_df)

    cat_features = ["Sport", "Market", "Currency", "bet_direction"]
    y_train = (train_df["Status"] == "won").astype(int)
    y_val = (val_df["Status"] == "won").astype(int)

    feature_names = list(x_train.columns)
    logger.info("Features: %d", len(feature_names))

    with mlflow.start_run(run_name="phase4/step_4_2_soccer_only") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.2")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": 42,
                "train_scope": "soccer_only",
                "n_samples_train": len(x_train),
                "n_samples_val": len(x_val_eval),
                "n_samples_test": len(test_df_all),
                "n_features": len(feature_names),
                "new_features": "bet_direction,is_home,is_draw,is_away",
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

            model.fit(
                x_train,
                y_train,
                cat_features=cat_features,
                eval_set=(x_val_eval, y_val),
                early_stopping_rounds=50,
                verbose=100,
            )

            # AUC на Soccer val
            auc_val = roc_auc_score(y_val, model.predict_proba(x_val_eval)[:, 1])
            logger.info("Val AUC (Soccer): %.4f", auc_val)
            mlflow.log_metric("auc_val", auc_val)

            # ROI на test (full temporal split)
            roi, n_bets = evaluate_on_test(model, test_df_all)
            logger.info("Test ROI: %.4f%% (n=%d)", roi, n_bets)
            mlflow.log_metrics({"roi": roi, "n_selected": n_bets})

            baseline_roi = 28.5833
            delta = roi - baseline_roi
            mlflow.log_metric("roi_delta", delta)
            mlflow.set_tag("convergence_signal", str(min(1.0, max(0.0, delta / 5.0 + 0.5))))

            if roi > 35.0:
                mlflow.set_tag("alert", "MQ-LEAKAGE-SUSPECT")
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("failure_reason", f"ROI={roi:.2f} > 35%")
                sys.exit(1)

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            logger.info("RESULT: ROI=%.4f%% n=%d AUC=%.4f delta=%.4f", roi, n_bets, auc_val, delta)
            print(f"STEP_4_2_ROI={roi:.6f}")
            print(f"STEP_4_2_N={n_bets}")
            print(f"STEP_4_2_AUC={auc_val:.4f}")
            print(f"STEP_4_2_DELTA={delta:.4f}")
            print(f"MLFLOW_RUN_ID={run.info.run_id}")

            if delta > 0.0:
                logger.info("Улучшение delta=%.4f. Сохраняем pipeline.", delta)
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(best_dir / "model.cbm"))
                metadata = {
                    "framework": "catboost",
                    "roi": roi,
                    "auc": auc_val,
                    "segment_thresholds": SEGMENT_THRESHOLDS,
                    "market_filter": "1x2",
                    "n_bets": n_bets,
                    "feature_names": feature_names,
                    "params": {
                        "depth": 7,
                        "learning_rate": 0.1,
                        "iterations": model.best_iteration_ or 500,
                    },
                    "session_id": SESSION_ID,
                    "step": "4.2",
                    "train_scope": "soccer_only",
                    "new_features": ["bet_direction", "is_home", "is_draw", "is_away"],
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Сохранено: %s", best_dir)

        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            logger.exception("Ошибка в step 4.2")
            raise


if __name__ == "__main__":
    main()
