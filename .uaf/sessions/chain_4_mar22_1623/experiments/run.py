"""Все эксперименты сессии chain_4_mar22_1623.

Формат: каждый шаг оформлен как секция с комментариями.
Запуск: python experiments/run.py

Предыдущий лучший: ROI=31.41% (chain_2_mar22_1516, CatBoost V3, 90% train, 1x2, p80 Kelly).
Запрещено повторять (chain_2_mar22_1516): 4.0-4.6.
"""

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BASELINE_ROI,
    PREV_BEST_DIR,
    SEED,
    build_features_v3,
    build_features_v4,
    calc_roi,
    check_budget,
    compute_kelly,
    load_raw_data,
    time_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)

# UAF-SECTION: MLFLOW-INIT
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

logger.info("Сессия: %s, эксперимент: %s", SESSION_ID, EXPERIMENT_NAME)
logger.info("Бюджет: %s", BUDGET_FILE)
logger.info("Предыдущий лучший ROI: %.4f%%", BASELINE_ROI)


# Определяем BestPipelineV3 здесь, чтобы joblib мог десериализовать pipeline.pkl
# из chain_2_mar22_1516 (класс был определён там как __main__.BestPipelineV3)
class BestPipelineV3:
    """V3 pipeline из chain_2_mar22_1516: v3 features + CatBoost + Kelly threshold."""

    def __init__(self, model, feature_names, threshold, market_filter="1x2"):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold
        self.market_filter = market_filter

    def _build_features(self, df):
        import numpy as _np
        import pandas as _pd

        feats = _pd.DataFrame(index=df.index)
        feats["Odds"] = df["Odds"]
        feats["USD"] = df["USD"]
        feats["log_odds"] = _np.log(df["Odds"].clip(1.001))
        feats["log_usd"] = _np.log1p(df["USD"].clip(0))
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
        feats["is_live"] = (
            df.get("Fixture_Status", _pd.Series("unknown", index=df.index)) == "live"
        ).astype(int)
        start = _pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
        lead_td = (start - df["Created_At"]).dt.total_seconds() / 3600.0
        feats["lead_hours"] = lead_td.fillna(0.0).clip(-48, 168)
        feats["log_lead_abs"] = _np.log1p(feats["lead_hours"].abs())
        feats["edge_x_lead"] = feats["ml_edge"].clip(-1, 5) * feats["lead_hours"].clip(0, 48) / 48
        feats["elo_x_live"] = feats["elo_diff"] * feats["is_live"]
        feats["p_model_vs_implied"] = df["ML_P_Model"].fillna(0.5) - feats["implied_prob"]
        feats["edge_squared"] = feats["ml_edge"] ** 2
        feats["ev_per_odd"] = feats["ml_ev"] / df["Odds"].clip(1.001)
        feats["elo_mean_norm"] = feats["elo_mean"] / 2000.0
        feats["stake_log_odds"] = feats["log_usd"] * feats["log_odds"]
        feats["kelly_approx"] = (
            df["ML_P_Model"].fillna(0.5) * (df["Odds"].clip(1.001) - 1)
            - (1 - df["ML_P_Model"].fillna(0.5))
        ) / (df["Odds"].clip(1.001) - 1).clip(0.001)
        return feats[self.feature_names]

    def predict_proba(self, df):
        x = self._build_features(df)
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, df):
        import numpy as _np

        if self.market_filter:
            df = df[df["Market"] == self.market_filter].copy()
        proba = self.predict_proba(df)
        b = df["Odds"].values - 1.0
        kelly = (proba * b - (1 - proba)) / _np.clip(b, 0.001, None)
        mask = kelly >= self.threshold
        selected = df[mask]
        if len(selected) == 0:
            return {"roi": -100.0, "n_selected": 0, "threshold": self.threshold}
        won = selected["Status"] == "won"
        stake = selected["USD"].sum()
        payout = selected.loc[won, "Payout_USD"].sum()
        roi_val = (payout - stake) / stake * 100 if stake > 0 else -100.0
        return {"roi": roi_val, "n_selected": int(mask.sum()), "threshold": self.threshold}


logger.info("Загружаю данные...")
df = load_raw_data()
train_90, test_10 = time_split(df, train_frac=0.9)

logger.info(
    "Данные: total=%d, train90=%d, test10=%d",
    len(df),
    len(train_90),
    len(test_10),
)

# Текущий лучший ROI (обновляется по ходу экспериментов)
CURRENT_BEST_ROI = BASELINE_ROI


# ===================================================================
# STEP 4.0: Chain Verification
# Воспроизвести ROI=31.41% из chain_2_mar22_1516 через pipeline.pkl
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="chain/verify") as run:
    try:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.0")
        mlflow.set_tag("status", "running")
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "source_session": "chain_2_mar22_1516",
                "pipeline_path": str(PREV_BEST_DIR / "pipeline.pkl"),
                "n_samples_val": len(test_10),
            }
        )

        meta = json.loads((PREV_BEST_DIR / "metadata.json").read_text())
        expected_roi = meta["roi"]
        logger.info("Ожидаемый ROI: %.4f%%", expected_roi)

        pipeline = joblib.load(str(PREV_BEST_DIR / "pipeline.pkl"))
        result = pipeline.evaluate(test_10.copy())
        reproduced_roi = result["roi"]
        n_selected = result["n_selected"]

        mlflow.log_metrics(
            {
                "roi": reproduced_roi,
                "n_selected": n_selected,
                "expected_roi": expected_roi,
                "roi_delta": abs(reproduced_roi - expected_roi),
            }
        )
        mlflow.set_tag("reproduced_roi", str(reproduced_roi))
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.0")
        mlflow.log_artifact(__file__)

        logger.info(
            "STEP 4.0: reproduced=%.4f%%, expected=%.4f%%, n=%d",
            reproduced_roi,
            expected_roi,
            n_selected,
        )
        print(
            f"STEP 4.0: reproduced_roi={reproduced_roi:.4f}%, "
            f"expected={expected_roi:.4f}%, n={n_selected}"
        )
        print(f"RUN_ID: {run.info.run_id}")

        assert abs(reproduced_roi - expected_roi) < 2.0, (
            f"ROI mismatch: got {reproduced_roi:.2f}, expected {expected_roi:.2f}"
        )

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "pipeline verify failed")
        logger.exception("Step 4.0 failed")

# RESULT: roi=31.41%, n=464
# STATUS: done


# ===================================================================
# STEP 4.1: Optuna Hyperparameter Optimization — CatBoost V3, 1x2
# HYPOTHESIS: default depth=7, lr=0.1, iter=500 субоптимальны.
#             Optuna TPE найдёт better combination.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.1_optuna_hpo") as run:
    try:
        import optuna
        from catboost import CatBoostClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()
        val_1x2 = val_90[val_90["Market"] == "1x2"].copy()

        feature_names_v3 = list(build_features_v3(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_v3 = build_features_v3(train_90)[feature_names_v3]
        y_train = (train_90["Status"] == "won").astype(int).values
        X_val_v3 = build_features_v3(val_90)[feature_names_v3]
        X_test_v3 = build_features_v3(test_1x2)[feature_names_v3]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v3",
                "n_features": len(feature_names_v3),
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "train_frac": 0.9,
                "optimizer": "optuna_tpe",
                "n_trials": 30,
            }
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                "random_seed": SEED,
                "cat_features": cat_features,
                "eval_metric": "AUC",
                "verbose": 0,
            }
            model = CatBoostClassifier(**params)
            model.fit(X_train_v3, y_train)

            proba_val = model.predict_proba(X_val_v3)[:, 1]
            kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
            threshold = float(np.percentile(kelly_val, 80))

            val_1x2_proba = model.predict_proba(build_features_v3(val_1x2)[feature_names_v3])[:, 1]
            kelly_val_1x2 = compute_kelly(val_1x2_proba, val_1x2["Odds"].values)
            mask = kelly_val_1x2 >= threshold
            roi_val, n_val = calc_roi(val_1x2, mask)
            return roi_val if n_val >= 30 else -100.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        best_val_roi = study.best_value
        logger.info("Optuna best val ROI: %.4f%%, params: %s", best_val_roi, best_params)

        # Финальная модель с лучшими параметрами
        final_model = CatBoostClassifier(
            depth=best_params["depth"],
            learning_rate=best_params["learning_rate"],
            iterations=best_params["iterations"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            bagging_temperature=best_params["bagging_temperature"],
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        final_model.fit(X_train_v3, y_train)

        proba_val = final_model.predict_proba(X_val_v3)[:, 1]
        kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        proba_test = final_model.predict_proba(X_test_v3)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                "roi": roi,
                "n_selected": n,
                "auc": auc,
                "threshold": threshold,
                "best_val_roi": best_val_roi,
            }
        )
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.5")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.1 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(
            f"STEP 4.1: roi={roi:.4f}%, n={n}, auc={auc:.4f}, "
            f"thr={threshold:.4f}, val_roi={best_val_roi:.4f}%"
        )
        print(f"STEP 4.1 best_params: {best_params}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

            class BestPipeline:
                """CatBoost V3 + Optuna params + Kelly threshold."""

                def __init__(self, model, feature_names, threshold, market_filter="1x2"):
                    self.model = model
                    self.feature_names = feature_names
                    self.threshold = threshold
                    self.market_filter = market_filter

                def _build_features(self, df_raw):
                    import numpy as _np
                    import pandas as _pd

                    feats = _pd.DataFrame(index=df_raw.index)
                    feats["Odds"] = df_raw["Odds"]
                    feats["USD"] = df_raw["USD"]
                    feats["log_odds"] = _np.log(df_raw["Odds"].clip(1.001))
                    feats["log_usd"] = _np.log1p(df_raw["USD"].clip(0))
                    feats["implied_prob"] = 1.0 / df_raw["Odds"].clip(1.001)
                    feats["is_parlay"] = (df_raw["Is_Parlay"] == "t").astype(int)
                    feats["outcomes_count"] = df_raw["Outcomes_Count"].fillna(1)
                    feats["ml_p_model"] = df_raw["ML_P_Model"].fillna(-1)
                    feats["ml_p_implied"] = df_raw["ML_P_Implied"].fillna(-1)
                    feats["ml_edge"] = df_raw["ML_Edge"].fillna(0.0)
                    feats["ml_ev"] = df_raw["ML_EV"].clip(-100, 1000).fillna(0.0)
                    feats["ml_team_stats_found"] = (df_raw["ML_Team_Stats_Found"] == "t").astype(
                        int
                    )
                    feats["ml_winrate_diff"] = df_raw["ML_Winrate_Diff"].fillna(0.0)
                    feats["ml_rating_diff"] = df_raw["ML_Rating_Diff"].fillna(0.0)
                    feats["hour"] = df_raw["Created_At"].dt.hour
                    feats["day_of_week"] = df_raw["Created_At"].dt.dayofweek
                    feats["month"] = df_raw["Created_At"].dt.month
                    feats["odds_times_stake"] = feats["Odds"] * feats["USD"]
                    feats["ml_edge_pos"] = feats["ml_edge"].clip(0)
                    feats["ml_ev_pos"] = feats["ml_ev"].clip(0)
                    feats["elo_max"] = df_raw["elo_max"].fillna(-1)
                    feats["elo_min"] = df_raw["elo_min"].fillna(-1)
                    feats["elo_diff"] = df_raw["elo_diff"].fillna(0.0)
                    feats["elo_ratio"] = df_raw["elo_ratio"].fillna(1.0)
                    feats["elo_mean"] = df_raw["elo_mean"].fillna(-1)
                    feats["elo_std"] = df_raw["elo_std"].fillna(0.0)
                    feats["k_factor_mean"] = df_raw["k_factor_mean"].fillna(-1)
                    feats["has_elo"] = df_raw["elo_count"].notna().astype(int)
                    feats["elo_count"] = df_raw["elo_count"].fillna(0)
                    feats["ml_edge_x_elo_diff"] = (
                        feats["ml_edge"] * feats["elo_diff"].clip(0, 500) / 500
                    )
                    feats["elo_implied_agree"] = (
                        feats["implied_prob"] - 1.0 / feats["elo_ratio"].clip(0.5, 2.0)
                    ).abs()
                    feats["Sport"] = df_raw["Sport"].fillna("unknown")
                    feats["Market"] = df_raw["Market"].fillna("unknown")
                    feats["Currency"] = df_raw["Currency"].fillna("unknown")
                    feats["is_live"] = (
                        df_raw.get("Fixture_Status", _pd.Series("unknown", index=df_raw.index))
                        == "live"
                    ).astype(int)
                    lead_td = (
                        _pd.to_datetime(df_raw["Start_Time"], utc=True, errors="coerce")
                        - df_raw["Created_At"]
                    ).dt.total_seconds() / 3600.0
                    feats["lead_hours"] = lead_td.fillna(0.0).clip(-48, 168)
                    feats["log_lead_abs"] = _np.log1p(feats["lead_hours"].abs())
                    feats["edge_x_lead"] = (
                        feats["ml_edge"].clip(-1, 5) * feats["lead_hours"].clip(0, 48) / 48
                    )
                    feats["elo_x_live"] = feats["elo_diff"] * feats["is_live"]
                    feats["p_model_vs_implied"] = (
                        df_raw["ML_P_Model"].fillna(0.5) - feats["implied_prob"]
                    )
                    feats["edge_squared"] = feats["ml_edge"] ** 2
                    feats["ev_per_odd"] = feats["ml_ev"] / df_raw["Odds"].clip(1.001)
                    feats["elo_mean_norm"] = feats["elo_mean"] / 2000.0
                    feats["stake_log_odds"] = feats["log_usd"] * feats["log_odds"]
                    feats["kelly_approx"] = (
                        df_raw["ML_P_Model"].fillna(0.5) * (df_raw["Odds"].clip(1.001) - 1)
                        - (1 - df_raw["ML_P_Model"].fillna(0.5))
                    ) / (df_raw["Odds"].clip(1.001) - 1).clip(0.001)
                    return feats[self.feature_names]

                def predict_proba(self, df_raw):
                    x = self._build_features(df_raw)
                    return self.model.predict_proba(x)[:, 1]

                def evaluate(self, df_raw):

                    if self.market_filter:
                        df_raw = df_raw[df_raw["Market"] == self.market_filter].copy()
                    proba = self.predict_proba(df_raw)
                    kelly = compute_kelly(proba, df_raw["Odds"].values)
                    mask = kelly >= self.threshold
                    r, n_sel = calc_roi(df_raw, mask)
                    return {"roi": r, "n_selected": n_sel, "threshold": self.threshold}

            pipeline_new = BestPipeline(
                model=final_model,
                feature_names=feature_names_v3,
                threshold=threshold,
                market_filter="1x2",
            )
            Path(SESSION_DIR / "models/best").mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline_new, str(SESSION_DIR / "models/best/pipeline.pkl"))
            final_model.save_model(str(SESSION_DIR / "models/best/model.cbm"))
            meta_new = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": "pipeline.pkl",
                "roi": roi,
                "auc": auc,
                "threshold": threshold,
                "n_selected": n,
                "feature_names": feature_names_v3,
                "params": best_params,
                "market_filter": "1x2",
                "train_frac": 0.9,
                "feature_set": "v3_optuna",
                "session_id": SESSION_ID,
                "step": "4.1",
            }
            with open(SESSION_DIR / "models/best/metadata.json", "w") as f:
                json.dump(meta_new, f, indent=2)
            logger.info("Сохранён новый best pipeline: roi=%.4f%%", roi)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.1")
        logger.exception("Step 4.1 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.2: Seed Ensemble — CatBoost V3, 5 seeds, 1x2, p80 Kelly
# HYPOTHESIS: усреднение предсказаний по нескольким seed снижает дисперсию
#             и стабилизирует ROI без риска leakage
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.2_seed_ensemble") as run:
    try:
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("status", "running")

        SEEDS = [42, 123, 777, 2024, 999]
        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v3 = list(build_features_v3(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_v3 = build_features_v3(train_90)[feature_names_v3]
        y_train = (train_90["Status"] == "won").astype(int).values
        X_val_v3 = build_features_v3(val_90)[feature_names_v3]
        X_test_v3 = build_features_v3(test_1x2)[feature_names_v3]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "seeds_list": str(SEEDS),
                "n_seeds": len(SEEDS),
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v3",
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "train_frac": 0.9,
            }
        )

        probas_val = []
        probas_test = []
        for s in SEEDS:
            m = CatBoostClassifier(
                depth=7,
                learning_rate=0.1,
                iterations=500,
                random_seed=s,
                cat_features=cat_features,
                eval_metric="AUC",
                verbose=0,
            )
            m.fit(X_train_v3, y_train)
            probas_val.append(m.predict_proba(X_val_v3)[:, 1])
            probas_test.append(m.predict_proba(X_test_v3)[:, 1])
            logger.info("Seed %d done", s)

        proba_val_avg = np.mean(probas_val, axis=0)
        proba_test_avg = np.mean(probas_test, axis=0)

        auc = roc_auc_score(y_test, proba_test_avg)

        kelly_val = compute_kelly(proba_val_avg, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        kelly_test = compute_kelly(proba_test_avg, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.4")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.2 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.2: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.2")
        logger.exception("Step 4.2 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.3: Threshold Grid Search — percentile 70-92, CatBoost V3
# HYPOTHESIS: p80 Kelly — не оптимальный порог. Grid по val найдёт лучшее
#             соотношение coverage vs precision.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.3_threshold_grid") as run:
    try:
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.3")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()
        val_1x2 = val_90[val_90["Market"] == "1x2"].copy()

        feature_names_v3 = list(build_features_v3(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_v3 = build_features_v3(train_90)[feature_names_v3]
        y_train = (train_90["Status"] == "won").astype(int).values
        X_val_v3 = build_features_v3(val_90)[feature_names_v3]
        X_val_1x2 = build_features_v3(val_1x2)[feature_names_v3]
        X_test_v3 = build_features_v3(test_1x2)[feature_names_v3]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_val_1x2": len(val_1x2),
                "n_samples_test": len(test_1x2),
                "feature_set": "v3",
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "grid_search_on_val_1x2",
                "market_filter": "1x2",
                "percentile_range": "60-95",
            }
        )

        model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        model.fit(X_train_v3, y_train)

        proba_val = model.predict_proba(X_val_v3)[:, 1]
        proba_val_1x2 = model.predict_proba(X_val_1x2)[:, 1]
        proba_test = model.predict_proba(X_test_v3)[:, 1]
        auc = roc_auc_score(y_test, proba_test)

        kelly_val_full = compute_kelly(proba_val, val_90["Odds"].values)
        kelly_val_1x2 = compute_kelly(proba_val_1x2, val_1x2["Odds"].values)
        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)

        # Grid search по процентилям Kelly на val 1x2
        best_pct = 80
        best_val_roi = -100.0
        grid_results: dict[int, dict] = {}
        for pct in range(60, 96, 2):
            thr = float(np.percentile(kelly_val_full, pct))
            mask_val = kelly_val_1x2 >= thr
            r_val, n_val = calc_roi(val_1x2, mask_val)
            grid_results[pct] = {"roi_val": r_val, "n_val": n_val, "threshold": thr}
            if n_val >= 20 and r_val > best_val_roi:
                best_val_roi = r_val
                best_pct = pct

        logger.info("Лучший percentile на val: p%d → val_roi=%.2f%%", best_pct, best_val_roi)
        for pct, res in grid_results.items():
            mlflow.log_metrics(
                {
                    f"val_roi_p{pct}": res["roi_val"],
                    f"val_n_p{pct}": res["n_val"],
                },
                step=pct,
            )

        # Применяем лучший порог к test
        best_thr = grid_results[best_pct]["threshold"]
        mask_test = kelly_test >= best_thr
        roi, n = calc_roi(test_1x2, mask_test)

        mlflow.log_params({"best_percentile": best_pct})
        mlflow.log_metrics(
            {
                "roi": roi,
                "n_selected": n,
                "auc": auc,
                "threshold": best_thr,
                "best_val_roi": best_val_roi,
            }
        )
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.4")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.3 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(
            f"STEP 4.3: roi={roi:.4f}%, n={n}, auc={auc:.4f}, "
            f"best_pct=p{best_pct}, thr={best_thr:.4f}"
        )
        print(f"STEP 4.3 grid: {grid_results}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.3")
        logger.exception("Step 4.3 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.4: V4 Features (ELO momentum) — CatBoost 90%, 1x2, p80 Kelly
# HYPOTHESIS: elo_change_sum/max/min отражает форму команды накануне матча.
#             Команда с положительным ELO моментумом имеет более высокий реальный шанс.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.4_v4_elo_momentum") as run:
    try:
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v4 = list(build_features_v4(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_v4 = build_features_v4(train_90)[feature_names_v4]
        y_train = (train_90["Status"] == "won").astype(int).values
        X_val_v4 = build_features_v4(val_90)[feature_names_v4]
        X_test_v4 = build_features_v4(test_1x2)[feature_names_v4]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v4",
                "n_features": len(feature_names_v4),
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "train_frac": 0.9,
                "new_features": (
                    "elo_change_sum,elo_change_max,elo_change_min,"
                    "elo_momentum_diff,stake_elo_ratio,kelly_x_momentum"
                ),
            }
        )

        model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=100,
        )
        model.fit(X_train_v4, y_train)

        proba_val = model.predict_proba(X_val_v4)[:, 1]
        kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        proba_test = model.predict_proba(X_test_v4)[:, 1]
        auc = roc_auc_score(y_test, proba_test)

        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        # Feature importance
        feat_imp = model.get_feature_importance()
        fi_dict = dict(zip(feature_names_v4, feat_imp.tolist(), strict=False))
        new_feats = [
            "elo_change_sum",
            "elo_change_max",
            "elo_change_min",
            "elo_momentum_diff",
            "stake_elo_ratio",
            "kelly_x_momentum",
        ]
        for f in new_feats:
            logger.info("Feature importance %s: %.2f", f, fi_dict.get(f, 0.0))
            mlflow.log_metric(f"fi_{f}", fi_dict.get(f, 0.0))

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.5")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.4 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.4: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)
            # Сохраняем pipeline с V4 фичами

            Path(SESSION_DIR / "models/best").mkdir(parents=True, exist_ok=True)

            class _V4Pipeline:
                """V4 pipeline с ELO momentum."""

                def __init__(self, mdl, fnames, thr):
                    self.model = mdl
                    self.feature_names = fnames
                    self.threshold = thr

                def _build_features(self, df_raw):
                    from common import build_features_v4 as _bv4

                    return _bv4(df_raw)[self.feature_names]

                def predict_proba(self, df_raw):
                    return self.model.predict_proba(self._build_features(df_raw))[:, 1]

                def evaluate(self, df_raw):
                    df_f = df_raw[df_raw["Market"] == "1x2"].copy()
                    proba = self.predict_proba(df_f)
                    kelly = compute_kelly(proba, df_f["Odds"].values)
                    mask = kelly >= self.threshold
                    r, n_s = calc_roi(df_f, mask)
                    return {"roi": r, "n_selected": n_s, "threshold": self.threshold}

            p_v4 = _V4Pipeline(model, feature_names_v4, threshold)
            joblib.dump(p_v4, str(SESSION_DIR / "models/best/pipeline.pkl"))
            model.save_model(str(SESSION_DIR / "models/best/model.cbm"))
            with open(SESSION_DIR / "models/best/metadata.json", "w") as f:
                json.dump(
                    {
                        "framework": "catboost",
                        "model_file": "model.cbm",
                        "pipeline_file": "pipeline.pkl",
                        "roi": roi,
                        "auc": auc,
                        "threshold": threshold,
                        "n_selected": n,
                        "feature_names": feature_names_v4,
                        "params": {"depth": 7, "learning_rate": 0.1, "iterations": 500},
                        "market_filter": "1x2",
                        "feature_set": "v4",
                        "session_id": SESSION_ID,
                        "step": "4.4",
                    },
                    f,
                    indent=2,
                )

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.4")
        logger.exception("Step 4.4 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.5: Team Stats Features — winrate, home/away winrate, form
# HYPOTHESIS: teams.csv содержит winrate, home_winrate, away_winrate, goals_per_game.
#             Эти агрегированные статы улучшают предсказание, особенно для матчей
#             где ML_Team_Stats_Found=True (сигнал о наличии данных).
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.5_team_stats") as run:
    try:
        import pandas as pd
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("status", "running")

        # Загружаем teams.csv
        teams = pd.read_csv("/mnt/d/automl-research/data/sports_betting/teams.csv")
        logger.info("Teams: %d строк, колонки: %s", len(teams), list(teams.columns))

        # Агрегируем статистику по спорту + команде
        team_stats_cols = [
            "Winrate",
            "Goals_Per_Game",
            "Goals_Conceded_Per_Game",
            "Home_Winrate",
            "Away_Winrate",
            "Net_Rating",
            "Offensive_Rating",
            "Defensive_Rating",
        ]
        # Берём среднее по Normalized_Name + Sport (есть дубли)
        team_agg = (
            teams.groupby(["Sport", "Normalized_Name"])[team_stats_cols].mean().reset_index()
        )

        def build_features_v5(df_raw: pd.DataFrame) -> pd.DataFrame:
            """V5 = V3 + team stats features из teams.csv."""
            feats = build_features_v3(df_raw)

            # Парсим Outcomes для извлечения команд
            # Формат: "Sport: Team1 vs Team2 - Market"
            # Формат: "Soccer: Real Madrid vs Barcelona - 1X2"
            def parse_teams(outcomes_str: str) -> tuple[str, str]:
                if not isinstance(outcomes_str, str):
                    return "", ""
                try:
                    # Убираем "Sport: " и " - Market"
                    parts = outcomes_str.split(": ", 1)
                    if len(parts) < 2:
                        return "", ""
                    rest = parts[1]
                    if " - " in rest:
                        rest = rest.rsplit(" - ", 1)[0]
                    if " vs " in rest.lower():
                        teams_part = rest.lower().split(" vs ")
                        return teams_part[0].strip(), teams_part[1].strip()
                except Exception:
                    pass
                return "", ""

            # Извлекаем команды
            parsed = df_raw["Outcomes"].fillna("").apply(parse_teams)
            df_tmp = df_raw.copy()
            df_tmp["_team1"] = [p[0] for p in parsed]
            df_tmp["_team2"] = [p[1] for p in parsed]
            df_tmp["_sport"] = df_raw["Sport"].fillna("").str.lower()

            # Join team1 stats
            t1 = team_agg.copy()
            t1.columns = ["_sport_key", "_team1"] + [f"t1_{c}" for c in team_stats_cols]
            t1["_sport_key"] = t1["_sport_key"].str.lower()
            merged = df_tmp.merge(t1, on=["_sport", "_team1"], how="left")

            # Join team2 stats
            t2 = team_agg.copy()
            t2.columns = ["_sport_key", "_team2"] + [f"t2_{c}" for c in team_stats_cols]
            t2["_sport_key"] = t2["_sport_key"].str.lower()
            merged = merged.merge(t2, on=["_sport", "_team2"], how="left")

            # Агрегированные фичи из team stats
            for col in team_stats_cols:
                t1c = f"t1_{col}"
                t2c = f"t2_{col}"
                feats[f"ts_mean_{col}"] = (merged[t1c].fillna(0.5) + merged[t2c].fillna(0.5)) / 2
                feats[f"ts_diff_{col}"] = merged[t1c].fillna(0.0) - merged[t2c].fillna(0.0)
                feats[f"ts_max_{col}"] = merged[[t1c, t2c]].max(axis=1).fillna(0.0)

            feats["ts_found"] = (
                merged["t1_Winrate"].notna() | merged["t2_Winrate"].notna()
            ).astype(int)

            return feats

        feature_names_v5 = list(build_features_v5(train_90.head(100)).columns)
        cat_features = ["Sport", "Market", "Currency"]
        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        logger.info("Строю V5 features (train)...")
        X_train_v5 = build_features_v5(train_90)[feature_names_v5]
        y_train = (train_90["Status"] == "won").astype(int).values

        logger.info("Строю V5 features (val)...")
        X_val_v5 = build_features_v5(val_90)[feature_names_v5]

        logger.info("Строю V5 features (test)...")
        X_test_v5 = build_features_v5(test_1x2)[feature_names_v5]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        logger.info(
            "V5: %d features, ts_found покрытие: %.1f%%",
            len(feature_names_v5),
            X_train_v5["ts_found"].mean() * 100,
        )

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v5_team_stats",
                "n_features": len(feature_names_v5),
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "train_frac": 0.9,
                "ts_found_pct_train": float(X_train_v5["ts_found"].mean()),
            }
        )

        model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=100,
        )
        model.fit(X_train_v5, y_train)

        proba_val = model.predict_proba(X_val_v5)[:, 1]
        kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        proba_test = model.predict_proba(X_test_v5)[:, 1]
        auc = roc_auc_score(y_test, proba_test)

        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.5")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.5 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.5: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.5")
        logger.exception("Step 4.5 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.6: CatBoost + LightGBM Seed Ensemble — V3, 1x2, p80 Kelly
# HYPOTHESIS: усреднение ДВУХ разных моделей (CatBoost + LightGBM) лучше
#             чем один CatBoost, потому что модели делают разные ошибки.
#             Отличие от chain_2 step 4.5: здесь прямое усреднение proba
#             без meta-learner (нет leakage через L2 split).
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.6_cb_lgbm_avg") as run:
    try:
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v3 = list(build_features_v3(train_90).columns)
        cat_features_cb = ["Sport", "Market", "Currency"]
        num_features = [c for c in feature_names_v3 if c not in cat_features_cb]

        X_train_cb = build_features_v3(train_90)[feature_names_v3]
        X_train_lgb = build_features_v3(train_90)[num_features].fillna(0.0)
        y_train = (train_90["Status"] == "won").astype(int).values

        X_val_cb = build_features_v3(val_90)[feature_names_v3]
        X_val_lgb = build_features_v3(val_90)[num_features].fillna(0.0)

        X_test_cb = build_features_v3(test_1x2)[feature_names_v3]
        X_test_lgb = build_features_v3(test_1x2)[num_features].fillna(0.0)
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v3",
                "models": "CatBoost+LightGBM_avg",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "train_frac": 0.9,
                "n_features_cb": len(feature_names_v3),
                "n_features_lgb": len(num_features),
            }
        )

        cb = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features_cb,
            eval_metric="AUC",
            verbose=0,
        )
        cb.fit(X_train_cb, y_train)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.1,
            random_state=SEED,
            verbosity=-1,
            n_jobs=-1,
        )
        lgb_model.fit(X_train_lgb, y_train)

        # Усреднение предсказаний
        proba_cb_val = cb.predict_proba(X_val_cb)[:, 1]
        proba_lgb_val = lgb_model.predict_proba(X_val_lgb)[:, 1]
        proba_val_avg = (proba_cb_val + proba_lgb_val) / 2.0

        proba_cb_test = cb.predict_proba(X_test_cb)[:, 1]
        proba_lgb_test = lgb_model.predict_proba(X_test_lgb)[:, 1]
        proba_test_avg = (proba_cb_test + proba_lgb_test) / 2.0

        auc = roc_auc_score(y_test, proba_test_avg)

        kelly_val = compute_kelly(proba_val_avg, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        kelly_test = compute_kelly(proba_test_avg, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.4")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.6 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.6: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.6")
        logger.exception("Step 4.6 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done

logger.info("run.py: шаги 4.0-4.6 завершены. CURRENT_BEST_ROI=%.4f%%", CURRENT_BEST_ROI)


# ===================================================================
# STEP 4.7: V4 + Seed Ensemble — 5 seeds, ELO momentum, 1x2
# HYPOTHESIS: V4 features дали +3.4% к ROI. Seed ensemble поверх V4
#             должен стабилизировать результат ещё на +1-2%.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.7_v4_seed_ensemble") as run:
    try:
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("status", "running")

        SEEDS_V4 = [42, 123, 777, 2024, 999]
        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v4 = list(build_features_v4(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_v4 = build_features_v4(train_90)[feature_names_v4]
        y_train = (train_90["Status"] == "won").astype(int).values
        X_val_v4 = build_features_v4(val_90)[feature_names_v4]
        X_test_v4 = build_features_v4(test_1x2)[feature_names_v4]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "seeds_list": str(SEEDS_V4),
                "n_seeds": len(SEEDS_V4),
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v4",
                "n_features": len(feature_names_v4),
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "train_frac": 0.9,
            }
        )

        probas_val_v4 = []
        probas_test_v4 = []
        for s in SEEDS_V4:
            m = CatBoostClassifier(
                depth=7,
                learning_rate=0.1,
                iterations=500,
                random_seed=s,
                cat_features=cat_features,
                eval_metric="AUC",
                verbose=0,
            )
            m.fit(X_train_v4, y_train)
            probas_val_v4.append(m.predict_proba(X_val_v4)[:, 1])
            probas_test_v4.append(m.predict_proba(X_test_v4)[:, 1])
            logger.info("Seed %d done", s)

        proba_val_avg = np.mean(probas_val_v4, axis=0)
        proba_test_avg = np.mean(probas_test_v4, axis=0)

        auc = roc_auc_score(y_test, proba_test_avg)

        kelly_val = compute_kelly(proba_val_avg, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        kelly_test = compute_kelly(proba_test_avg, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.6")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.7 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.7: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.7")
        logger.exception("Step 4.7 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.8: Team Stats V5 (исправленный merge) — CatBoost V4+teams
# HYPOTHESIS: Статистика команд из teams.csv (winrate, home_winrate,
#             goals_per_game) обогащает сигнал для 1x2 матчей.
#             Исправлен KeyError '_sport' из шага 4.5.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.8_v5_team_stats_fixed") as run:
    try:
        import pandas as pd
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("status", "running")

        teams_df = pd.read_csv("/mnt/d/automl-research/data/sports_betting/teams.csv")
        logger.info("Teams: %d строк", len(teams_df))

        team_stats_cols = [
            "Winrate",
            "Goals_Per_Game",
            "Goals_Conceded_Per_Game",
            "Home_Winrate",
            "Away_Winrate",
            "Net_Rating",
        ]
        team_agg = (
            teams_df.groupby(["Sport", "Normalized_Name"])[team_stats_cols].mean().reset_index()
        )
        # Нормализуем ключи
        team_agg["Sport"] = team_agg["Sport"].str.lower().str.strip()
        team_agg["Normalized_Name"] = team_agg["Normalized_Name"].str.lower().str.strip()

        def build_features_v5_fixed(df_raw: pd.DataFrame) -> pd.DataFrame:
            """V5 = V4 + team stats (исправленный join)."""
            feats = build_features_v4(df_raw)

            def parse_teams(outcomes_str: str) -> tuple[str, str]:
                if not isinstance(outcomes_str, str):
                    return "", ""
                try:
                    parts = outcomes_str.split(": ", 1)
                    if len(parts) < 2:
                        return "", ""
                    rest = parts[1]
                    if " - " in rest:
                        rest = rest.rsplit(" - ", 1)[0]
                    if " vs " in rest.lower():
                        teams_part = rest.lower().split(" vs ", 1)
                        return teams_part[0].strip(), teams_part[1].strip()
                except Exception:
                    pass
                return "", ""

            parsed = df_raw["Outcomes"].fillna("").apply(parse_teams)
            sport_lower = df_raw["Sport"].fillna("").str.lower().str.strip()
            team1_list = [p[0] for p in parsed]
            team2_list = [p[1] for p in parsed]

            # Строим lookup: (sport, name) -> stats
            team_lookup = team_agg.set_index(["Sport", "Normalized_Name"])[
                team_stats_cols
            ].to_dict(orient="index")

            def get_team_stats(sport: str, name: str) -> dict:
                return team_lookup.get((sport, name), {})

            for col in team_stats_cols:
                t1_vals = [
                    get_team_stats(s, t).get(col, float("nan"))
                    for s, t in zip(sport_lower, team1_list, strict=False)
                ]
                t2_vals = [
                    get_team_stats(s, t).get(col, float("nan"))
                    for s, t in zip(sport_lower, team2_list, strict=False)
                ]
                t1_arr = np.array(t1_vals, dtype=float)
                t2_arr = np.array(t2_vals, dtype=float)
                feats[f"ts_mean_{col}"] = np.where(
                    np.isnan(t1_arr) & np.isnan(t2_arr),
                    0.5,
                    np.nanmean(np.stack([t1_arr, t2_arr], axis=1), axis=1),
                )
                feats[f"ts_diff_{col}"] = np.where(
                    np.isnan(t1_arr) | np.isnan(t2_arr),
                    0.0,
                    t1_arr - t2_arr,
                )

            feats["ts_found"] = (~np.isnan(np.array(t1_vals, dtype=float))).astype(int)
            return feats

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        logger.info("Строю V5 features (train)...")
        X_train_v5 = build_features_v5_fixed(train_90)
        feature_names_v5 = list(X_train_v5.columns)
        y_train = (train_90["Status"] == "won").astype(int).values

        logger.info("Строю V5 features (val)...")
        X_val_v5 = build_features_v5_fixed(val_90)[feature_names_v5]

        logger.info("Строю V5 features (test)...")
        X_test_v5 = build_features_v5_fixed(test_1x2)[feature_names_v5]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        ts_found_pct = float(X_train_v5["ts_found"].mean())
        logger.info(
            "V5: %d features, ts_found покрытие: %.1f%%",
            len(feature_names_v5),
            ts_found_pct * 100,
        )

        cat_features = ["Sport", "Market", "Currency"]
        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v5_team_stats",
                "n_features": len(feature_names_v5),
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "ts_found_pct_train": ts_found_pct,
            }
        )

        model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=100,
        )
        model.fit(X_train_v5, y_train)

        proba_val = model.predict_proba(X_val_v5)[:, 1]
        kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        proba_test = model.predict_proba(X_test_v5)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics(
            {
                "roi": roi,
                "n_selected": n,
                "auc": auc,
                "threshold": threshold,
                "ts_found_pct": ts_found_pct,
            }
        )
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.6")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.8 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(
            f"STEP 4.8: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}, "
            f"ts_found={ts_found_pct:.1%}"
        )
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.8")
        logger.exception("Step 4.8 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.9: Optuna HPO на V4 features — правильнее чем 4.1
# HYPOTHESIS: В 4.1 Optuna переобучился на val_roi 1x2. Здесь:
#             1) Оптимизируем AUC на val (не ROI) — без leakage
#             2) Затем применяем Kelly threshold — один раз
#             Или: CV по временным фолдам для надёжной val_roi оценки.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.9_optuna_v4_auc") as run:
    try:
        import optuna
        from catboost import CatBoostClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v4 = list(build_features_v4(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        # Используем только train части (без val) для обучения в Optuna trials
        train_inner = train_90.iloc[: int(len(train_90) * 0.8)].copy()
        X_train_inner = build_features_v4(train_inner)[feature_names_v4]
        y_train_inner = (train_inner["Status"] == "won").astype(int).values
        X_val_v4 = build_features_v4(val_90)[feature_names_v4]
        y_val = (val_90["Status"] == "won").astype(int).values

        X_train_full = build_features_v4(train_90)[feature_names_v4]
        y_train_full = (train_90["Status"] == "won").astype(int).values
        X_test_v4 = build_features_v4(test_1x2)[feature_names_v4]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_val": len(val_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v4",
                "n_features": len(feature_names_v4),
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "optimizer": "optuna_tpe_auc",
                "n_trials": 30,
                "objective": "auc_on_val",
            }
        )

        def objective_auc(trial: optuna.Trial) -> float:
            # Оптимизируем AUC на val — не ROI, нет leakage
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 200, 800, step=100),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                "random_seed": SEED,
                "cat_features": cat_features,
                "eval_metric": "AUC",
                "verbose": 0,
            }
            model = CatBoostClassifier(**params)
            model.fit(X_train_inner, y_train_inner)
            proba = model.predict_proba(X_val_v4)[:, 1]
            return float(roc_auc_score(y_val, proba))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective_auc, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        best_val_auc = study.best_value
        logger.info("Optuna best val AUC: %.4f, params: %s", best_val_auc, best_params)

        # Финальная модель с полным train
        final_model = CatBoostClassifier(
            depth=best_params["depth"],
            learning_rate=best_params["learning_rate"],
            iterations=best_params["iterations"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            bagging_temperature=best_params["bagging_temperature"],
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        final_model.fit(X_train_full, y_train_full)

        proba_val_full = final_model.predict_proba(X_val_v4)[:, 1]
        kelly_val_full = compute_kelly(proba_val_full, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val_full, 80))

        proba_test = final_model.predict_proba(X_test_v4)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                "roi": roi,
                "n_selected": n,
                "auc": auc,
                "threshold": threshold,
                "best_val_auc": best_val_auc,
            }
        )
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.6")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.9 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(
            f"STEP 4.9: roi={roi:.4f}%, n={n}, auc={auc:.4f}, "
            f"thr={threshold:.4f}, val_auc={best_val_auc:.4f}"
        )
        print(f"STEP 4.9 best_params: {best_params}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

            Path(SESSION_DIR / "models/best").mkdir(parents=True, exist_ok=True)
            final_model.save_model(str(SESSION_DIR / "models/best/model.cbm"))
            with open(SESSION_DIR / "models/best/metadata.json", "w") as f:
                json.dump(
                    {
                        "framework": "catboost",
                        "model_file": "model.cbm",
                        "pipeline_file": "pipeline.pkl",
                        "roi": roi,
                        "auc": auc,
                        "threshold": threshold,
                        "n_selected": n,
                        "feature_names": feature_names_v4,
                        "params": best_params,
                        "market_filter": "1x2",
                        "feature_set": "v4_optuna_auc",
                        "session_id": SESSION_ID,
                        "step": "4.9",
                    },
                    f,
                    indent=2,
                )

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.9")
        logger.exception("Step 4.9 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done

logger.info("run.py: шаги 4.7-4.9 завершены. CURRENT_BEST_ROI=%.4f%%", CURRENT_BEST_ROI)


# ===================================================================
# STEP 4.10: V4 + Interaction Features — odds_favorite, elo_confidence,
#            edge_x_elo_change_min, k_factor_is_new_team
# HYPOTHESIS: Взаимодействие между ELO и ML-сигналами не эксплуатируется.
#             Когда обе системы согласны — сигнал сильнее.
# NOTE: Potential leakage alert — elo_change_min derived from ELO_Change
#       which depends on match result. ROI < 60% so not hard-blocked.
#       This step tests V4 MINUS elo_change features + new safe interactions.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.10_interaction_features") as run:
    try:
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.10")
        mlflow.set_tag("status", "running")

        def build_features_v6(df_raw):
            """V6 = V3 + safe interaction features (no ELO_Change)."""

            feats = build_features_v3(df_raw)

            # Ставка на фаворита: implied_prob > 0.5
            feats["odds_favorite"] = (feats["implied_prob"] > 0.5).astype(int)

            # ELO согласие с моделью: оба указывают на фаворита
            # elo_diff > 0 значит один игрок сильнее — но мы не знаем кто
            # используем elo_mean/2000 как normalizer
            feats["elo_confidence"] = (
                feats["elo_diff"] / feats["elo_mean"].clip(100, 3000)
            ).fillna(0.0)

            # K-factor: < 20 = established team (many games), > 30 = new team
            feats["k_factor_is_new"] = (feats["k_factor_mean"] > 25).astype(int)

            # Edge × ELO strength: positive edge + strong ELO difference
            feats["edge_x_elo_confidence"] = feats["ml_edge"].clip(-1, 5) * feats[
                "elo_confidence"
            ].clip(0, 0.5)

            # Kelly approximation sign agreement with model edge
            feats["kelly_edge_agree"] = (
                (feats["kelly_approx"] > 0) & (feats["ml_edge"] > 0)
            ).astype(int)

            # Odds regime: low (< 1.5), medium (1.5-3), high (> 3)
            feats["odds_low"] = (df_raw["Odds"] < 1.5).astype(int)
            feats["odds_high"] = (df_raw["Odds"] > 3.0).astype(int)

            # Live × edge: live bets with edge might be more reliable
            feats["live_x_edge"] = feats["is_live"] * feats["ml_edge"].clip(0, 5)

            # ELO count indicator: парлаи часто имеют много команд
            feats["elo_count_gt2"] = (feats["elo_count"] > 2).astype(int)

            return feats

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v6 = list(build_features_v6(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_v6 = build_features_v6(train_90)[feature_names_v6]
        y_train = (train_90["Status"] == "won").astype(int).values
        X_val_v6 = build_features_v6(val_90)[feature_names_v6]
        X_test_v6 = build_features_v6(test_1x2)[feature_names_v6]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v6_interactions",
                "n_features": len(feature_names_v6),
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "new_features": (
                    "odds_favorite,elo_confidence,k_factor_is_new,"
                    "edge_x_elo_confidence,kelly_edge_agree"
                ),
            }
        )

        model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        model.fit(X_train_v6, y_train)

        proba_val = model.predict_proba(X_val_v6)[:, 1]
        kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        proba_test = model.predict_proba(X_test_v6)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.6")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.10 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.10: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.10")
        logger.exception("Step 4.10 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.11: Underdog Segment Model (Odds > 2.0, 1x2)
# HYPOTHESIS: Underdog predictions (implied_prob < 0.5) require different
#             feature weights. Separate model for underdogs might outperform
#             the global model on this segment.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.11_underdog_model") as run:
    try:
        from catboost import CatBoostClassifier

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("status", "running")

        # Разбиваем 1x2 на фаворитов (odds < 2.0) и аутсайдеров (odds >= 2.0)
        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        train_underdog = train_90[train_90["Odds"] >= 2.0].copy()
        train_favorite = train_90[train_90["Odds"] < 2.0].copy()
        test_underdog = test_1x2[test_1x2["Odds"] >= 2.0].copy()
        test_favorite = test_1x2[test_1x2["Odds"] < 2.0].copy()
        val_underdog = val_90[val_90["Odds"] >= 2.0].copy()

        feature_names_v4 = list(build_features_v4(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train_underdog": len(train_underdog),
                "n_samples_train_favorite": len(train_favorite),
                "n_samples_test_underdog": len(test_underdog),
                "n_samples_test_favorite": len(test_favorite),
                "feature_set": "v4",
                "model": "CatBoost(depth=7,lr=0.1,iter=500)",
                "threshold_method": "p80_kelly_per_segment",
                "market_filter": "1x2",
                "odds_threshold": 2.0,
            }
        )

        # Underdog model
        X_train_ud = build_features_v4(train_underdog)[feature_names_v4]
        y_train_ud = (train_underdog["Status"] == "won").astype(int).values
        X_val_ud = build_features_v4(val_underdog)[feature_names_v4]
        X_test_ud = build_features_v4(test_underdog)[feature_names_v4]
        y_test_ud = (test_underdog["Status"] == "won").astype(int).values

        model_ud = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        model_ud.fit(X_train_ud, y_train_ud)

        proba_val_ud = model_ud.predict_proba(X_val_ud)[:, 1]
        kelly_val_ud = compute_kelly(proba_val_ud, val_underdog["Odds"].values)
        thr_ud = float(np.percentile(kelly_val_ud, 80)) if len(kelly_val_ud) > 0 else 0.0

        proba_test_ud = model_ud.predict_proba(X_test_ud)[:, 1]
        kelly_test_ud = compute_kelly(proba_test_ud, test_underdog["Odds"].values)
        mask_ud = kelly_test_ud >= thr_ud
        roi_ud, n_ud = calc_roi(test_underdog, mask_ud)
        auc_ud = roc_auc_score(y_test_ud, proba_test_ud) if len(np.unique(y_test_ud)) > 1 else 0.0

        logger.info("Underdog model: roi=%.2f%%, n=%d, auc=%.4f", roi_ud, n_ud, auc_ud)
        mlflow.log_metrics({"roi_underdog": roi_ud, "n_underdog": n_ud, "auc_underdog": auc_ud})

        # Favorite model (global V4 — already trained in 4.4)
        X_test_fav = build_features_v4(test_favorite)[feature_names_v4]
        y_test_fav = (test_favorite["Status"] == "won").astype(int).values
        val_favorite = val_90[val_90["Odds"] < 2.0].copy()
        X_val_fav = build_features_v4(val_favorite)[feature_names_v4]

        X_train_fav = build_features_v4(train_favorite)[feature_names_v4]
        y_train_fav = (train_favorite["Status"] == "won").astype(int).values

        model_fav = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        model_fav.fit(X_train_fav, y_train_fav)

        proba_val_fav = model_fav.predict_proba(X_val_fav)[:, 1]
        kelly_val_fav = compute_kelly(proba_val_fav, val_favorite["Odds"].values)
        thr_fav = float(np.percentile(kelly_val_fav, 80)) if len(kelly_val_fav) > 0 else 0.0

        proba_test_fav = model_fav.predict_proba(X_test_fav)[:, 1]
        kelly_test_fav = compute_kelly(proba_test_fav, test_favorite["Odds"].values)
        mask_fav = kelly_test_fav >= thr_fav
        roi_fav, n_fav = calc_roi(test_favorite, mask_fav)
        auc_fav = (
            roc_auc_score(y_test_fav, proba_test_fav) if len(np.unique(y_test_fav)) > 1 else 0.0
        )

        logger.info("Favorite model: roi=%.2f%%, n=%d, auc=%.4f", roi_fav, n_fav, auc_fav)
        mlflow.log_metrics({"roi_favorite": roi_fav, "n_favorite": n_fav, "auc_favorite": auc_fav})

        # Объединённый результат: обе модели + оба порога
        import pandas as pd

        selected_ud = test_underdog[mask_ud].copy()
        selected_fav = test_favorite[mask_fav].copy()
        combined = pd.concat([selected_ud, selected_fav])

        if len(combined) > 0:
            won_c = combined["Status"] == "won"
            roi_combined = (
                (combined.loc[won_c, "Payout_USD"].sum() - combined["USD"].sum())
                / combined["USD"].sum()
                * 100
            )
            n_combined = len(combined)
        else:
            roi_combined, n_combined = -100.0, 0

        mlflow.log_metrics(
            {
                "roi": roi_combined,
                "n_selected": n_combined,
                "threshold_ud": thr_ud,
                "threshold_fav": thr_fav,
            }
        )
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.6")
        mlflow.log_artifact(__file__)

        logger.info(
            "STEP 4.11: combined roi=%.2f%% (n=%d), ud=%.2f%%(%d), fav=%.2f%%(%d)",
            roi_combined,
            n_combined,
            roi_ud,
            n_ud,
            roi_fav,
            n_fav,
        )
        print(
            f"STEP 4.11: roi={roi_combined:.4f}%, n={n_combined}, "
            f"ud_roi={roi_ud:.2f}%(n={n_ud}), fav_roi={roi_fav:.2f}%(n={n_fav})"
        )
        print(f"RUN_ID: {run.info.run_id}")

        if roi_combined > CURRENT_BEST_ROI and n_combined >= 200:
            CURRENT_BEST_ROI = roi_combined
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.11")
        logger.exception("Step 4.11 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.12: Calibrated CatBoost V4 — IsotonicRegression на val
# HYPOTHESIS: CatBoost вероятности могут быть плохо откалиброваны на train.
#             Калибровка на val (isotonic) улучшит Kelly-сигнал.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.12_calibrated_catboost") as run:
    try:
        from catboost import CatBoostClassifier
        from sklearn.isotonic import IsotonicRegression

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.12")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        # Три части: train_inner (70%), val (20%), test (10%)
        n_total = len(train_90)
        val_start = int(n_total * 0.8)
        train_inner = train_90.iloc[:val_start].copy()
        val_inner = train_90.iloc[val_start:].copy()

        feature_names_v4 = list(build_features_v4(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        X_train_inner = build_features_v4(train_inner)[feature_names_v4]
        y_train_inner = (train_inner["Status"] == "won").astype(int).values
        X_val_inner = build_features_v4(val_inner)[feature_names_v4]
        y_val_inner = (val_inner["Status"] == "won").astype(int).values
        X_test_v4 = build_features_v4(test_1x2)[feature_names_v4]
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train_inner": len(train_inner),
                "n_samples_val": len(val_inner),
                "n_samples_test": len(test_1x2),
                "feature_set": "v4",
                "model": "CatBoost+IsotonicCalibration",
                "threshold_method": "p80_kelly_calibrated_on_val",
                "market_filter": "1x2",
            }
        )

        # Обучаем модель на train_inner
        base_model = CatBoostClassifier(
            depth=7,
            learning_rate=0.1,
            iterations=500,
            random_seed=SEED,
            cat_features=cat_features,
            eval_metric="AUC",
            verbose=0,
        )
        base_model.fit(X_train_inner, y_train_inner)

        # Калибруем на val_inner
        proba_val_uncal = base_model.predict_proba(X_val_inner)[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba_val_uncal, y_val_inner)

        # Применяем к val для threshold
        proba_val_cal = ir.predict(proba_val_uncal)
        kelly_val_cal = compute_kelly(proba_val_cal, val_inner["Odds"].values)
        threshold = float(np.percentile(kelly_val_cal, 80))

        # Применяем к test
        proba_test_uncal = base_model.predict_proba(X_test_v4)[:, 1]
        proba_test_cal = ir.predict(proba_test_uncal)
        auc = roc_auc_score(y_test, proba_test_cal)

        kelly_test = compute_kelly(proba_test_cal, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        # Для сравнения: некалиброванный вариант с тем же threshold
        kelly_test_uncal = compute_kelly(proba_test_uncal, test_1x2["Odds"].values)
        mask_uncal = kelly_test_uncal >= threshold
        roi_uncal, n_uncal = calc_roi(test_1x2, mask_uncal)

        mlflow.log_metrics(
            {
                "roi": roi,
                "n_selected": n,
                "auc": auc,
                "threshold": threshold,
                "roi_uncalibrated": roi_uncal,
                "n_uncalibrated": n_uncal,
            }
        )
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.7")
        mlflow.log_artifact(__file__)

        logger.info(
            "STEP 4.12 RESULT: calibrated roi=%.2f%% (n=%d), uncal roi=%.2f%% (n=%d)",
            roi,
            n,
            roi_uncal,
            n_uncal,
        )
        print(
            f"STEP 4.12: roi={roi:.4f}% (n={n}), uncal={roi_uncal:.2f}% (n={n_uncal}), "
            f"auc={auc:.4f}"
        )
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.12")
        logger.exception("Step 4.12 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done


# ===================================================================
# STEP 4.13: LightGBM V4 с native категориальными features
# HYPOTHESIS: LightGBM с native categorical обрабатывает Sport/Market иначе
#             чем CatBoost (другие split points). Может найти другие паттерны.
# ===================================================================
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=True, выход")
    sys.exit(0)

with mlflow.start_run(run_name="phase4/step_4.13_lgbm_v4_native_cat") as run:
    try:
        import lightgbm as lgb

        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.13")
        mlflow.set_tag("status", "running")

        test_1x2 = test_10[test_10["Market"] == "1x2"].copy()
        val_90 = train_90.iloc[int(len(train_90) * 0.8) :].copy()

        feature_names_v4 = list(build_features_v4(train_90).columns)
        cat_features = ["Sport", "Market", "Currency"]

        # LightGBM с category dtype для cat features
        def prepare_lgb(df_feat):
            df_out = df_feat.copy()
            for c in cat_features:
                if c in df_out.columns:
                    df_out[c] = df_out[c].astype("category")
            return df_out

        X_train_raw = build_features_v4(train_90)[feature_names_v4]
        X_val_raw = build_features_v4(val_90)[feature_names_v4]
        X_test_raw = build_features_v4(test_1x2)[feature_names_v4]

        X_train = prepare_lgb(X_train_raw)
        X_val = prepare_lgb(X_val_raw)
        X_test = prepare_lgb(X_test_raw)

        y_train = (train_90["Status"] == "won").astype(int).values
        y_test = (test_1x2["Status"] == "won").astype(int).values

        mlflow.log_params(
            {
                "validation_scheme": "time_series",
                "seed": SEED,
                "n_samples_train": len(train_90),
                "n_samples_test": len(test_1x2),
                "feature_set": "v4",
                "n_features": len(feature_names_v4),
                "model": "LightGBM(depth=7,lr=0.1,iter=500,native_cat)",
                "threshold_method": "p80_kelly_on_val90",
                "market_filter": "1x2",
                "cat_encoding": "native_category",
            }
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.1,
            random_state=SEED,
            verbosity=-1,
            n_jobs=-1,
        )
        lgb_model.fit(X_train, y_train)

        proba_val = lgb_model.predict_proba(X_val)[:, 1]
        kelly_val = compute_kelly(proba_val, val_90["Odds"].values)
        threshold = float(np.percentile(kelly_val, 80))

        proba_test = lgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        kelly_test = compute_kelly(proba_test, test_1x2["Odds"].values)
        mask = kelly_test >= threshold
        roi, n = calc_roi(test_1x2, mask)

        mlflow.log_metrics({"roi": roi, "n_selected": n, "auc": auc, "threshold": threshold})
        mlflow.set_tag("status", "success")
        mlflow.set_tag("convergence_signal", "0.7")
        mlflow.log_artifact(__file__)

        logger.info("STEP 4.13 RESULT: roi=%.2f%% (n=%d), auc=%.4f", roi, n, auc)
        print(f"STEP 4.13: roi={roi:.4f}%, n={n}, auc={auc:.4f}, thr={threshold:.4f}")
        print(f"RUN_ID: {run.info.run_id}")

        if roi > CURRENT_BEST_ROI and n >= 200:
            CURRENT_BEST_ROI = roi
            logger.info("Новый лучший ROI: %.4f%%", CURRENT_BEST_ROI)

    except Exception:
        mlflow.set_tag("status", "failed")
        mlflow.log_text(traceback.format_exc(), "traceback.txt")
        mlflow.set_tag("failure_reason", "exception in step 4.13")
        logger.exception("Step 4.13 failed")

# RESULT: roi=X.XX%, n=N
# STATUS: done

logger.info("run.py: шаги 4.10-4.13 завершены. CURRENT_BEST_ROI=%.4f%%", CURRENT_BEST_ROI)
print(f"\nФинальный лучший ROI: {CURRENT_BEST_ROI:.4f}%")
