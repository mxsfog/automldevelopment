"""Phase 2: Feature Engineering с Shadow Feature Trick.

Тестируем 5 групп фичей:
1. Implied probability и value gap
2. Временные фичи (hour, dow, is_weekend)
3. Нелинейные трансформации odds (log, squared)
4. Rolling win rate по Sport (historical, only train)
5. ELO фичи (Old_ELO разница для покрытых ставок)

Shadow Feature Trick:
- baseline = лучшая модель из Phase 1 (CatBoost, те же параметры)
- candidate = baseline + shadow features
- delta > 0.002: принять
- delta <= 0: отклонить
- 0 < delta <= 0.002: marginal
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")

budget_file = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
try:
    status = json.loads(budget_file.read_text())
    if status.get("hard_stop"):
        logger.warning("Budget hard stop detected, exiting")
        sys.exit(0)
except FileNotFoundError:
    pass

BASELINE_NUM = [
    "Odds",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "Outcomes_Count",
    "USD",
]
BASELINE_CAT = ["Sport", "Market", "is_parlay_str"]
BEST_THRESHOLD = 0.55


def compute_roi_at_threshold(df_val: pd.DataFrame, proba: np.ndarray, threshold: float) -> float:
    """ROI на ставках где P(won) > threshold."""
    selected = proba >= threshold
    if selected.sum() == 0:
        return 0.0
    sel = df_val[selected]
    total_staked = sel["USD"].sum()
    total_returned = sel.loc[sel["Status"] == "won", "Payout_USD"].sum()
    return (total_returned - total_staked) / total_staked * 100


def prepare_base_data() -> pd.DataFrame:
    """Загрузка и подготовка базового датасета."""
    bets = pd.read_csv(DATA_DIR / "bets.csv", low_memory=False)
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv", low_memory=False)

    exclude = ["pending", "cancelled", "error", "cashout"]
    df = bets[~bets["Status"].isin(exclude)].copy()
    df["Created_At"] = pd.to_datetime(df["Created_At"])
    df = df.sort_values("Created_At").reset_index(drop=True)

    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(Sport=("Sport", "first"), Market=("Market", "first"))
        .reset_index()
    )
    df = df.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")

    df["is_parlay_str"] = df["Is_Parlay"].astype(str)
    for col in ["ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV"]:
        df[col] = df[col].fillna(0.0)
    for col in BASELINE_CAT:
        df[col] = df[col].fillna("unknown").astype(str)

    return df


def add_implied_value_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Группа 1: implied probability и value gap."""
    df = df.copy()
    df["implied_prob"] = 1.0 / df["Odds"]
    df["value_gap"] = df["ML_P_Model"] / 100.0 - df["implied_prob"]
    df["ml_vs_implied_ratio"] = np.where(
        df["ML_P_Implied"] > 0,
        df["ML_P_Model"] / (df["ML_P_Implied"] + 1e-8),
        0.0,
    )
    new_features = ["implied_prob", "value_gap", "ml_vs_implied_ratio"]
    return df, new_features


def add_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Группа 2: временные фичи."""
    df = df.copy()
    df["hour"] = df["Created_At"].dt.hour
    df["dow"] = df["Created_At"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    new_features = ["hour", "dow", "is_weekend"]
    return df, new_features


def add_odds_transforms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Группа 3: нелинейные трансформации odds."""
    df = df.copy()
    df["log_odds"] = np.log1p(df["Odds"])
    df["odds_sq"] = df["Odds"] ** 2
    df["odds_bucket"] = pd.cut(
        df["Odds"],
        bins=[0, 1.3, 1.6, 2.0, 2.5, 3.5, 5.0, 100, 10000],
        labels=False,
    ).fillna(7)
    new_features = ["log_odds", "odds_sq", "odds_bucket"]
    return df, new_features


def add_sport_winrate(df: pd.DataFrame, train_end: int) -> tuple[pd.DataFrame, list[str]]:
    """Группа 4: rolling sport win rate (fit only on train)."""
    df = df.copy()
    train = df.iloc[:train_end]
    sport_wr = train.groupby("Sport")["Status"].apply(lambda x: (x == "won").mean()).to_dict()
    df["sport_hist_winrate"] = df["Sport"].map(sport_wr).fillna(0.5)

    market_wr = train.groupby("Market")["Status"].apply(lambda x: (x == "won").mean()).to_dict()
    df["market_hist_winrate"] = df["Market"].map(market_wr).fillna(0.5)
    new_features = ["sport_hist_winrate", "market_hist_winrate"]
    return df, new_features


def add_elo_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Группа 5: ELO фичи из elo_history."""
    elo_hist = pd.read_csv(DATA_DIR / "elo_history.csv", low_memory=False)
    elo_agg = (
        elo_hist.groupby("Bet_ID")
        .agg(
            elo_home=("Old_ELO", "first"),
            elo_away=("Old_ELO", "last"),
            elo_diff=("Old_ELO", lambda x: x.iloc[0] - x.iloc[-1] if len(x) > 1 else 0),
        )
        .reset_index()
    )
    df = df.merge(elo_agg, left_on="ID", right_on="Bet_ID", how="left", suffixes=("", "_elo"))
    for col in ["elo_home", "elo_away", "elo_diff"]:
        df[col] = df[col].fillna(0.0)
    new_features = ["elo_home", "elo_away", "elo_diff"]
    return df, new_features


def run_shadow_test(
    df: pd.DataFrame,
    baseline_features: list[str],
    candidate_features: list[str],
    cat_features_baseline: list[str],
    cat_features_candidate: list[str],
    group_name: str,
    parent_run_id: str,
) -> dict:
    """Shadow feature trick: сравнение baseline vs candidate."""
    y_binary = (df["Status"] == "won").astype(int)
    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 1)

    baseline_rois = []
    candidate_rois = []
    baseline_aucs = []
    candidate_aucs = []

    cat_idx_base = [baseline_features.index(c) for c in cat_features_baseline]
    cat_idx_cand = [candidate_features.index(c) for c in cat_features_candidate]

    for fold_idx in range(n_splits):
        train_end = fold_size * (fold_idx + 1)
        val_start = train_end
        val_end = train_end + fold_size
        if fold_idx == n_splits - 1:
            val_end = n

        # Для sport_winrate нужно пересчитать на каждом фолде
        if "sport_hist_winrate" in candidate_features:
            df, _ = add_sport_winrate(df, train_end)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        y_train = y_binary.iloc[:train_end].values
        y_val = y_binary.iloc[val_start:val_end].values

        # Baseline
        model_base = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            cat_features=cat_idx_base,
            verbose=0,
            auto_class_weights="Balanced",
        )
        model_base.fit(train_df[baseline_features], y_train)
        proba_base = model_base.predict_proba(val_df[baseline_features])[:, 1]
        roi_base = compute_roi_at_threshold(val_df, proba_base, BEST_THRESHOLD)
        auc_base = roc_auc_score(y_val, proba_base)

        # Candidate
        model_cand = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            cat_features=cat_idx_cand,
            verbose=0,
            auto_class_weights="Balanced",
        )
        model_cand.fit(train_df[candidate_features], y_train)
        proba_cand = model_cand.predict_proba(val_df[candidate_features])[:, 1]
        roi_cand = compute_roi_at_threshold(val_df, proba_cand, BEST_THRESHOLD)
        auc_cand = roc_auc_score(y_val, proba_cand)

        baseline_rois.append(roi_base)
        candidate_rois.append(roi_cand)
        baseline_aucs.append(auc_base)
        candidate_aucs.append(auc_cand)

    roi_base_mean = np.mean(baseline_rois)
    roi_cand_mean = np.mean(candidate_rois)
    delta = roi_cand_mean - roi_base_mean

    if delta > 0.2:
        decision = "accept"
    elif delta <= 0:
        decision = "reject"
    else:
        decision = "marginal"

    result = {
        "group": group_name,
        "roi_baseline": roi_base_mean,
        "roi_candidate": roi_cand_mean,
        "delta": delta,
        "auc_baseline": np.mean(baseline_aucs),
        "auc_candidate": np.mean(candidate_aucs),
        "decision": decision,
    }

    with mlflow.start_run(run_name=f"phase2/{group_name}", nested=True):
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "shadow_test")
        mlflow.set_tag("group", group_name)
        mlflow.set_tag("decision", decision)
        mlflow.log_metrics(
            {
                "roi_baseline": round(roi_base_mean, 4),
                "roi_candidate": round(roi_cand_mean, 4),
                "roi_delta": round(delta, 4),
                "auc_baseline": round(np.mean(baseline_aucs), 4),
                "auc_candidate": round(np.mean(candidate_aucs), 4),
            }
        )

    logger.info(
        "%s: baseline_roi=%.4f, candidate_roi=%.4f, delta=%.4f -> %s",
        group_name,
        roi_base_mean,
        roi_cand_mean,
        delta,
        decision,
    )
    return result


def main() -> None:
    logger.info("Загрузка данных")
    df = prepare_base_data()
    logger.info("Подготовленный датасет: %d строк", len(df))

    baseline_all = BASELINE_NUM + BASELINE_CAT

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="phase2/feature_engineering") as parent_run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "feature_engineering")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("phase", "2")

        try:
            results = []

            # Группа 1: implied_prob + value_gap
            df1, new_feats_1 = add_implied_value_features(df)
            r1 = run_shadow_test(
                df1,
                baseline_all,
                baseline_all + new_feats_1,
                BASELINE_CAT,
                BASELINE_CAT,
                "implied_value",
                parent_run.info.run_id,
            )
            results.append(r1)

            # Группа 2: time features
            df2, new_feats_2 = add_time_features(df)
            r2 = run_shadow_test(
                df2,
                baseline_all,
                baseline_all + new_feats_2,
                BASELINE_CAT,
                BASELINE_CAT,
                "time_features",
                parent_run.info.run_id,
            )
            results.append(r2)

            # Группа 3: odds transforms
            df3, new_feats_3 = add_odds_transforms(df)
            r3 = run_shadow_test(
                df3,
                baseline_all,
                baseline_all + new_feats_3,
                BASELINE_CAT,
                BASELINE_CAT,
                "odds_transforms",
                parent_run.info.run_id,
            )
            results.append(r3)

            # Группа 4: sport/market winrate
            df4, new_feats_4 = add_sport_winrate(df, len(df) // 2)
            r4 = run_shadow_test(
                df4,
                baseline_all,
                baseline_all + new_feats_4,
                BASELINE_CAT,
                BASELINE_CAT,
                "sport_market_winrate",
                parent_run.info.run_id,
            )
            results.append(r4)

            # Группа 5: ELO features
            df5, new_feats_5 = add_elo_features(df)
            r5 = run_shadow_test(
                df5,
                baseline_all,
                baseline_all + new_feats_5,
                BASELINE_CAT,
                BASELINE_CAT,
                "elo_features",
                parent_run.info.run_id,
            )
            results.append(r5)

            # Подведём итоги
            accepted = [r for r in results if r["decision"] == "accept"]
            marginal = [r for r in results if r["decision"] == "marginal"]
            rejected = [r for r in results if r["decision"] == "reject"]

            logger.info("Accepted: %s", [r["group"] for r in accepted])
            logger.info("Marginal: %s", [r["group"] for r in marginal])
            logger.info("Rejected: %s", [r["group"] for r in rejected])

            mlflow.log_params(
                {
                    "n_groups_tested": len(results),
                    "n_accepted": len(accepted),
                    "n_marginal": len(marginal),
                    "n_rejected": len(rejected),
                    "accepted_groups": ",".join(r["group"] for r in accepted),
                }
            )

            # Финальная модель со всеми принятыми фичами
            accepted_features = []
            final_df = df.copy()
            for r in accepted + marginal:
                if r["group"] == "implied_value":
                    final_df, feats = add_implied_value_features(final_df)
                    accepted_features.extend(feats)
                elif r["group"] == "time_features":
                    final_df, feats = add_time_features(final_df)
                    accepted_features.extend(feats)
                elif r["group"] == "odds_transforms":
                    final_df, feats = add_odds_transforms(final_df)
                    accepted_features.extend(feats)
                elif r["group"] == "sport_market_winrate":
                    final_df, feats = add_sport_winrate(final_df, len(final_df) // 2)
                    accepted_features.extend(feats)
                elif r["group"] == "elo_features":
                    final_df, feats = add_elo_features(final_df)
                    accepted_features.extend(feats)

            mlflow.log_param("accepted_features", ",".join(accepted_features))

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")

            logger.info("Run ID: %s", parent_run.info.run_id)
            print(f"ACCEPTED: {[r['group'] for r in accepted]}")
            print(f"MARGINAL: {[r['group'] for r in marginal]}")
            print(f"REJECTED: {[r['group'] for r in rejected]}")
            print(f"RUN_ID: {parent_run.info.run_id}")

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "feature engineering failed")
            logger.exception("Phase 2 failed")
            raise


if __name__ == "__main__":
    main()
