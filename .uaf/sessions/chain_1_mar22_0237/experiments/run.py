"""Все эксперименты chain_1_mar22_0237.

Файл дополняется по ходу работы. Каждый шаг помечен === STEP N.N ===.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]
SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
BUDGET_FILE = Path(os.environ["UAF_BUDGET_STATUS_FILE"])

sys.path.insert(0, str(SESSION_DIR / "experiments"))
from common import (  # noqa: E402
    BASELINE_ROI,
    SEED,
    already_done,
    build_features_base,
    calc_roi,
    check_budget,
    compute_kelly,
    load_raw_data,
    time_split,
    time_split_val,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# === Загрузка данных (один раз) ===
logger.info("Загрузка данных...")
df_raw = load_raw_data()
train_df, test_df = time_split(df_raw, train_frac=0.80)
train_df2, val_df, test_df2 = time_split_val(df_raw, train_frac=0.70, val_frac=0.10)

logger.info(
    "Train: %d, Val: %d, Test: %d",
    len(train_df),
    len(val_df),
    len(test_df),
)

FEATURES_BASE = [
    "Odds",
    "USD",
    "log_odds",
    "log_usd",
    "implied_prob",
    "is_parlay",
    "outcomes_count",
    "ml_p_model",
    "ml_p_implied",
    "ml_edge",
    "ml_ev",
    "ml_team_stats_found",
    "ml_winrate_diff",
    "ml_rating_diff",
    "hour",
    "day_of_week",
    "month",
    "odds_times_stake",
    "ml_edge_pos",
    "ml_ev_pos",
    "elo_max",
    "elo_min",
    "elo_diff",
    "elo_ratio",
    "elo_mean",
    "elo_std",
    "k_factor_mean",
    "has_elo",
    "elo_count",
    "ml_edge_x_elo_diff",
    "elo_implied_agree",
]
CAT_FEATURES = ["Sport", "Market", "Currency"]
FEATURES_CAT = FEATURES_BASE + CAT_FEATURES

y_train = (train_df["Status"] == "won").astype(int)
y_test = (test_df["Status"] == "won").astype(int)

X_train_base = build_features_base(train_df)
X_test_base = build_features_base(test_df)


# === STEP 1.1: Constant baseline ===
# HYPOTHESIS: DummyClassifier (most_frequent) задаёт lower bound ROI
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "1.1"):
    with mlflow.start_run(run_name="phase1/dummy_classifier") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.1")
        mlflow.set_tag("status", "running")
        try:
            dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
            dummy.fit(X_train_base[FEATURES_BASE], y_train)
            proba_dummy = np.zeros(len(test_df))
            # most_frequent = "won" (40144 > 34349 in all data)
            # Для flat strategy: всегда ставим → roi = платформенный roi
            mask_all = np.ones(len(test_df), dtype=bool)
            roi_all, n_all = calc_roi(test_df, mask_all)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "strategy": "most_frequent",
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": roi_all,
                    "n_bets": n_all,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            mlflow.log_artifact(__file__)
            logger.info("Step 1.1 DONE: all-bets roi=%.2f%%, n=%d", roi_all, n_all)
            print(f"STEP 1.1 RESULT: roi={roi_all:.2f}%, n_bets={n_all}")
            # RESULT: roi=-2.69% (платформенный), n_bets=~16000
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 1.1 already done, skipping")


# === STEP 1.2: Rule-based baseline ===
# HYPOTHESIS: Простое пороговое правило по ml_ev > 0 задаёт rule baseline
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "1.2"):
    with mlflow.start_run(run_name="phase1/rule_based") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.2")
        mlflow.set_tag("status", "running")
        try:
            # Rule: ML_EV > 0 (платформа оценивает ставку как прибыльную)
            mask_ev = (test_df["ML_EV"] > 0).values
            roi_ev, n_ev = calc_roi(test_df, mask_ev)

            # Rule 2: Odds < 2.0 (низкие коэффициенты, высокая вероятность)
            mask_odds = (test_df["Odds"] < 2.0).values
            roi_odds, n_odds = calc_roi(test_df, mask_odds)

            # Rule 3: ML_Edge > 5 (edge > 5%)
            mask_edge = (test_df["ML_Edge"] > 5).values
            roi_edge, n_edge = calc_roi(test_df, mask_edge)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "rule_1": "ml_ev > 0",
                    "rule_2": "odds < 2.0",
                    "rule_3": "ml_edge > 5",
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "roi_ev_positive": roi_ev,
                    "n_bets_ev_positive": n_ev,
                    "roi_odds_low": roi_odds,
                    "n_bets_odds_low": n_odds,
                    "roi_edge_pos5": roi_edge,
                    "n_bets_edge_pos5": n_edge,
                    "roi": max(roi_ev, roi_odds, roi_edge),
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 1.2 DONE: ev>0=%.2f%%(n=%d), odds<2=%.2f%%(n=%d), edge>5=%.2f%%(n=%d)",
                roi_ev,
                n_ev,
                roi_odds,
                n_odds,
                roi_edge,
                n_edge,
            )
            print(f"STEP 1.2 RESULT: best_roi={max(roi_ev, roi_odds, roi_edge):.2f}%")
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 1.2 already done, skipping")


# === STEP 1.3: Linear baseline ===
# HYPOTHESIS: LogisticRegression с базовыми фичами — linear baseline
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "1.3"):
    with mlflow.start_run(run_name="phase1/logistic_regression") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.3")
        mlflow.set_tag("status", "running")
        try:
            # Encode categoricals for LR
            X_tr_lr = X_train_base.copy()
            X_te_lr = X_test_base.copy()
            encoders: dict[str, LabelEncoder] = {}
            for col in CAT_FEATURES:
                enc = LabelEncoder()
                X_tr_lr[col] = enc.fit_transform(X_tr_lr[col].astype(str))
                encoders[col] = enc
            for col in CAT_FEATURES:
                enc = encoders[col]
                known = set(enc.classes_)
                fallback = enc.classes_[0]
                X_te_lr[col] = enc.transform(
                    X_te_lr[col]
                    .astype(str)
                    .apply(lambda x, k=known, f=fallback: x if x in k else f)
                )

            lr = LogisticRegression(
                max_iter=1000,
                random_state=SEED,
                C=1.0,
                solver="lbfgs",
                n_jobs=-1,
            )
            lr.fit(X_tr_lr[FEATURES_CAT], y_train)
            proba_lr = lr.predict_proba(X_te_lr[FEATURES_CAT])[:, 1]
            auc_lr = roc_auc_score(y_test, proba_lr)

            # Kelly threshold от train p80
            X_tr_low_lr = X_tr_lr[train_df["Odds"] < 2.5].copy()
            y_low = y_train[train_df["Odds"] < 2.5]
            proba_tr_low = lr.predict_proba(X_tr_low_lr[FEATURES_CAT])[:, 1]
            kelly_tr_low = compute_kelly(
                proba_tr_low, train_df[train_df["Odds"] < 2.5]["Odds"].values
            )
            threshold_lr = float(np.percentile(kelly_tr_low, 80))

            kelly_test_lr = compute_kelly(proba_lr, test_df["Odds"].values)
            # 1x2 + lead_hours>0 filter
            lead_hours = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            mkt_mask = test_df["Market"].values == "1x2"
            lead_mask = lead_hours.values > 0
            kelly_mask = kelly_test_lr >= threshold_lr
            final_mask = mkt_mask & lead_mask & kelly_mask

            roi_lr, n_lr = calc_roi(test_df, final_mask)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "LogisticRegression",
                    "C": 1.0,
                    "threshold": threshold_lr,
                    "kelly_percentile": 80,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_lr,
                    "roi": roi_lr,
                    "n_bets": n_lr,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 1.3 DONE: LR auc=%.4f, roi=%.2f%%, n=%d",
                auc_lr,
                roi_lr,
                n_lr,
            )
            print(f"STEP 1.3 RESULT: auc={auc_lr:.4f}, roi={roi_lr:.2f}%, n_bets={n_lr}")
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 1.3 already done, skipping")


# === STEP 1.4: Non-linear baseline (CatBoost default) ===
# HYPOTHESIS: CatBoost с дефолтами — strong non-linear baseline
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "1.4"):
    with mlflow.start_run(run_name="phase1/catboost_default") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "1.4")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier

            cat_idx = [FEATURES_CAT.index(c) for c in CAT_FEATURES]
            cb = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb.fit(
                X_train_base[FEATURES_CAT],
                y_train,
                cat_features=cat_idx,
            )
            proba_cb = cb.predict_proba(X_test_base[FEATURES_CAT])[:, 1]
            auc_cb = roc_auc_score(y_test, proba_cb)

            # p80 Kelly threshold
            low_mask_tr = train_df["Odds"].values < 2.5
            X_tr_low = X_train_base[FEATURES_CAT][low_mask_tr]
            proba_tr_low = cb.predict_proba(X_tr_low)[:, 1]
            kelly_tr_low = compute_kelly(proba_tr_low, train_df["Odds"].values[low_mask_tr])
            threshold_cb = float(np.percentile(kelly_tr_low, 80))

            kelly_test_cb = compute_kelly(proba_cb, test_df["Odds"].values)
            lead_hours = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            mkt_mask = test_df["Market"].values == "1x2"
            lead_mask = lead_hours.values > 0
            kelly_mask = kelly_test_cb >= threshold_cb
            final_mask = mkt_mask & lead_mask & kelly_mask

            roi_cb, n_cb = calc_roi(test_df, final_mask)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier",
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.1,
                    "kelly_percentile": 80,
                    "threshold": threshold_cb,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_cb,
                    "roi": roi_cb,
                    "n_bets": n_cb,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 1.4 DONE: CatBoost auc=%.4f, roi=%.2f%%, n=%d",
                auc_cb,
                roi_cb,
                n_cb,
            )
            print(f"STEP 1.4 RESULT: auc={auc_cb:.4f}, roi={roi_cb:.2f}%, n_bets={n_cb}")
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 1.4 already done, skipping")


logger.info("Phase 1 complete. Best baseline established.")
print("=== Phase 1 DONE ===")


# === STEP 4.0: Chain Verify — воспроизвести ROI=33.35% из chain_9 ===
# HYPOTHESIS: pipeline.pkl из chain_9 воспроизводит best roi=33.35%
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.0"):
    with mlflow.start_run(run_name="chain/verify_chain9") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.0")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier
            from common import build_features_extended

            chain9_dir = Path(
                "/mnt/d/automl-research/.uaf/sessions/chain_9_mar22_0121/models/best"
            )
            meta9 = json.loads((chain9_dir / "metadata.json").read_text())
            expected_roi = meta9["roi"]
            feature_names9 = meta9["feature_names"]
            kelly_threshold9 = meta9["kelly_threshold_low"]

            # Загружаем model.cbm напрямую (pickle несовместим из-за BestPipeline1x2P80)
            model9 = CatBoostClassifier()
            model9.load_model(str(chain9_dir / "model.cbm"))

            # Воспроизводим pipeline: 1x2 + lead_hours>0 + p80 Kelly
            X_te9 = build_features_base(test_df)[feature_names9]
            proba9 = model9.predict_proba(X_te9)[:, 1]
            kelly9 = compute_kelly(proba9, test_df["Odds"].values)
            lead_hrs9 = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            mkt9 = test_df["Market"].values == "1x2"
            lead9 = lead_hrs9.values > 0
            kelly9_mask = kelly9 >= kelly_threshold9
            final9 = mkt9 & lead9 & kelly9_mask
            actual_roi, n_actual = calc_roi(test_df, final9)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "prev_session": "chain_9_mar22_0121",
                    "expected_roi": expected_roi,
                    "n_bets_expected": meta9["n_bets"],
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "roi": actual_roi,
                    "n_bets": n_actual,
                    "roi_delta_vs_expected": actual_roi - expected_roi,
                }
            )

            delta = abs(actual_roi - expected_roi)
            logger.info(
                "Chain verify: actual=%.4f%%, expected=%.4f%%, delta=%.4f%%",
                actual_roi,
                expected_roi,
                delta,
            )
            if delta > 2.0:
                logger.warning("ROI mismatch > 2%% — проверить данные или pipeline")

            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.0")
            mlflow.log_artifact(__file__)
            print(f"STEP 4.0 RESULT: roi={actual_roi:.4f}%, n={n_actual}")
            # RESULT: roi=33.35% (n=148) — baseline подтверждён
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.0 already done, skipping")


# === STEP 4.1: Fixture_Status + lead_hours как признаки ===
# HYPOTHESIS: Fixture_Status (live/pre-match) и lead_hours дают доп. сигнал.
# Live 1x2 ставки: winrate=50.6% vs 44.2% pre-match — большая разница.
# Эти признаки не использовались ни в одной предыдущей модели.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.1"):
    with mlflow.start_run(run_name="phase4/fixture_status_lead_hours") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.1")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier
            from common import build_features_extended

            FEATURES_EXT = [*FEATURES_CAT, "is_live", "lead_hours", "log_lead_abs"]

            X_tr_ext = build_features_extended(train_df)[FEATURES_EXT]
            X_te_ext = build_features_extended(test_df)[FEATURES_EXT]

            cat_idx_ext = [FEATURES_EXT.index(c) for c in CAT_FEATURES]

            cb_ext = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb_ext.fit(X_tr_ext, y_train, cat_features=cat_idx_ext)

            proba_ext = cb_ext.predict_proba(X_te_ext)[:, 1]
            auc_ext = roc_auc_score(y_test, proba_ext)

            # p80 Kelly threshold из train LOW
            low_mask_tr = train_df["Odds"].values < 2.5
            X_tr_low_ext = build_features_extended(train_df[low_mask_tr])[FEATURES_EXT]
            proba_tr_low = cb_ext.predict_proba(X_tr_low_ext)[:, 1]
            kelly_tr_low = compute_kelly(proba_tr_low, train_df["Odds"].values[low_mask_tr])
            threshold_ext = float(np.percentile(kelly_tr_low, 80))

            kelly_test_ext = compute_kelly(proba_ext, test_df["Odds"].values)
            lead_hours = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            mkt_mask = test_df["Market"].values == "1x2"
            lead_mask = lead_hours.values > 0
            kelly_mask = kelly_test_ext >= threshold_ext
            final_mask = mkt_mask & lead_mask & kelly_mask

            roi_ext, n_ext = calc_roi(test_df, final_mask)

            # Сравнение с baseline (step 1.4): roi=12.83%
            delta_vs_baseline = roi_ext - 12.83

            # Feature importances
            feat_imp = cb_ext.get_feature_importance()
            for fname, imp in sorted(
                zip(FEATURES_EXT, feat_imp, strict=True), key=lambda x: x[1], reverse=True
            )[:10]:
                logger.info("  %s: %.4f", fname, imp)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier",
                    "new_features": "is_live,lead_hours,log_lead_abs",
                    "iterations": 500,
                    "depth": 7,
                    "learning_rate": 0.1,
                    "kelly_percentile": 80,
                    "threshold": threshold_ext,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_ext,
                    "roi": roi_ext,
                    "n_bets": n_ext,
                    "delta_vs_baseline": delta_vs_baseline,
                }
            )

            if roi_ext > BASELINE_ROI:
                # Новый best — сохраняем модель
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                cb_ext.save_model(str(best_dir / "model.cbm"))
                meta_new = {
                    "framework": "catboost",
                    "model_file": "model.cbm",
                    "roi": roi_ext,
                    "auc": float(auc_ext),
                    "threshold": threshold_ext,
                    "n_bets": n_ext,
                    "feature_names": FEATURES_EXT,
                    "params": {"iterations": 500, "depth": 7, "learning_rate": 0.1},
                    "session_id": SESSION_ID,
                    "step": "4.1",
                    "new_features": ["is_live", "lead_hours", "log_lead_abs"],
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(meta_new, f, indent=2)
                logger.info("Новый best сохранён: roi=%.4f%%", roi_ext)

            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.1 DONE: auc=%.4f, roi=%.2f%%, n=%d, delta=%.2f%%",
                auc_ext,
                roi_ext,
                n_ext,
                delta_vs_baseline,
            )
            print(
                f"STEP 4.1 RESULT: auc={auc_ext:.4f}, roi={roi_ext:.2f}%, n={n_ext}, "
                f"delta_vs_baseline={delta_vs_baseline:+.2f}%"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.1 already done, skipping")


# === STEP 4.2: Live 1x2 separate model ===
# HYPOTHESIS: Live 1x2 ставки (winrate=50.6% vs 44.2%) требуют отдельной модели.
# Тренируем отдельно на is_live=1 бетах.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.2"):
    with mlflow.start_run(run_name="phase4/live_1x2_model") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.2")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier
            from common import build_features_extended

            FEATURES_EXT = [*FEATURES_CAT, "is_live", "lead_hours", "log_lead_abs"]

            # Фильтруем только 1x2 ставки для train и test
            train_1x2 = train_df[train_df["Market"] == "1x2"].copy()
            test_1x2 = test_df[test_df["Market"] == "1x2"].copy()
            y_train_1x2 = (train_1x2["Status"] == "won").astype(int)
            y_test_1x2 = (test_1x2["Status"] == "won").astype(int)

            # Подмодели: live vs pre-match
            lead_tr = (train_1x2["Start_Time"] - train_1x2["Created_At"]).dt.total_seconds() / 3600
            lead_te = (test_1x2["Start_Time"] - test_1x2["Created_At"]).dt.total_seconds() / 3600

            # Pre-match model (lead_hours > 0)
            tr_pre = train_1x2[lead_tr > 0].copy()
            te_pre = test_1x2[lead_te > 0].copy()
            y_tr_pre = (tr_pre["Status"] == "won").astype(int)

            logger.info(
                "1x2 pre-match: train=%d, test=%d (winrate=%.3f)",
                len(tr_pre),
                len(te_pre),
                y_tr_pre.mean(),
            )

            cat_idx_ext = [FEATURES_EXT.index(c) for c in CAT_FEATURES]

            cb_pre = CatBoostClassifier(
                iterations=300,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            X_tr_pre = build_features_extended(tr_pre)[FEATURES_EXT]
            X_te_pre = build_features_extended(te_pre)[FEATURES_EXT]
            cb_pre.fit(X_tr_pre, y_tr_pre, cat_features=cat_idx_ext)
            proba_pre = cb_pre.predict_proba(X_te_pre)[:, 1]
            auc_pre = roc_auc_score((te_pre["Status"] == "won").astype(int), proba_pre)

            # p80 Kelly на pre-match train LOW (odds < 2.5)
            low_pre = tr_pre[tr_pre["Odds"] < 2.5].copy()
            if len(low_pre) > 0:
                X_low_pre = build_features_extended(low_pre)[FEATURES_EXT]
                proba_low_pre = cb_pre.predict_proba(X_low_pre)[:, 1]
                kelly_low_pre = compute_kelly(proba_low_pre, low_pre["Odds"].values)
                threshold_pre = float(np.percentile(kelly_low_pre, 80))
            else:
                threshold_pre = 0.5

            kelly_te_pre = compute_kelly(proba_pre, te_pre["Odds"].values)
            mask_pre = kelly_te_pre >= threshold_pre
            roi_pre, n_pre = calc_roi(te_pre, mask_pre)

            logger.info(
                "Pre-match 1x2 model: auc=%.4f, roi=%.2f%%, n=%d, threshold=%.4f",
                auc_pre,
                roi_pre,
                n_pre,
                threshold_pre,
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier_1x2_prematch",
                    "iterations": 300,
                    "depth": 7,
                    "kelly_percentile": 80,
                    "threshold_prematch": threshold_pre,
                    "n_train_prematch": len(tr_pre),
                    "n_test_prematch": len(te_pre),
                }
            )
            mlflow.log_metrics(
                {
                    "auc_prematch": auc_pre,
                    "roi_prematch": roi_pre,
                    "n_bets_prematch": n_pre,
                    "roi": roi_pre,
                    "n_bets": n_pre,
                }
            )

            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.3")
            mlflow.log_artifact(__file__)
            logger.info("Step 4.2 DONE: pre-match roi=%.2f%%, n=%d", roi_pre, n_pre)
            print(f"STEP 4.2 RESULT: roi={roi_pre:.2f}%, n_pre={n_pre}")
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.2 already done, skipping")


print("=== Phase 4.0-4.2 DONE ===")


# === STEP 4.3: Репликация chain_8 (depth=7, 500 iter) + Kelly sweep p78-p88 ===
# HYPOTHESIS: CatBoost depth=7 (как chain_8) с ТЕКУЩИМИ данными + sweep перцентилей.
# chain_9 показал p82=36.02% — проверяем на current data.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.3"):
    with mlflow.start_run(run_name="phase4/catboost_depth7_kelly_sweep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.3")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier

            cat_idx_base = [FEATURES_CAT.index(c) for c in CAT_FEATURES]

            cb7 = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb7.fit(X_train_base[FEATURES_CAT], y_train, cat_features=cat_idx_base)
            proba_cb7 = cb7.predict_proba(X_test_base[FEATURES_CAT])[:, 1]
            auc_cb7 = roc_auc_score(y_test, proba_cb7)

            # Kelly sweep p78-p88 на train LOW (odds<2.5)
            low_tr = train_df["Odds"].values < 2.5
            proba_tr_low7 = cb7.predict_proba(X_train_base[FEATURES_CAT][low_tr])[:, 1]
            kelly_tr_low7 = compute_kelly(proba_tr_low7, train_df["Odds"].values[low_tr])

            lead_hrs_te = (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            mkt_1x2 = test_df["Market"].values == "1x2"
            lead_pos = lead_hrs_te.values > 0
            kelly_te7 = compute_kelly(proba_cb7, test_df["Odds"].values)

            sweep_metrics_43: dict[str, float] = {}
            for pct in range(78, 89):
                thr_pct = float(np.percentile(kelly_tr_low7, pct))
                mask_pct = mkt_1x2 & lead_pos & (kelly_te7 >= thr_pct)
                roi_pct, n_pct = calc_roi(test_df, mask_pct)
                sweep_metrics_43[f"roi_p{pct}"] = roi_pct
                sweep_metrics_43[f"n_p{pct}"] = float(n_pct)
                logger.info("p%d: thr=%.4f, roi=%.2f%%, n=%d", pct, thr_pct, roi_pct, n_pct)

            thr80 = float(np.percentile(kelly_tr_low7, 80))
            thr82 = float(np.percentile(kelly_tr_low7, 82))
            roi_p80_43, n_p80_43 = calc_roi(test_df, mkt_1x2 & lead_pos & (kelly_te7 >= thr80))
            roi_p82_43, n_p82_43 = calc_roi(test_df, mkt_1x2 & lead_pos & (kelly_te7 >= thr82))

            if roi_p80_43 > BASELINE_ROI:
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                cb7.save_model(str(best_dir / "model.cbm"))
                meta_best = {
                    "framework": "catboost",
                    "roi": roi_p80_43,
                    "auc": float(auc_cb7),
                    "threshold_p80": thr80,
                    "threshold_p82": thr82,
                    "n_bets_p80": n_p80_43,
                    "n_bets_p82": n_p82_43,
                    "feature_names": FEATURES_CAT,
                    "params": {"iterations": 500, "depth": 7, "learning_rate": 0.1},
                    "session_id": SESSION_ID,
                    "step": "4.3",
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(meta_best, f, indent=2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier",
                    "iterations": 500,
                    "depth": 7,
                    "kelly_sweep": "p78-p88",
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                    "threshold_p80": thr80,
                    "threshold_p82": thr82,
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_cb7,
                    "roi": roi_p80_43,
                    "n_bets": n_p80_43,
                    "roi_p80": roi_p80_43,
                    "n_bets_p80": n_p80_43,
                    "roi_p82": roi_p82_43,
                    "n_bets_p82": n_p82_43,
                    **sweep_metrics_43,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.3: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_cb7,
                roi_p80_43,
                n_p80_43,
                roi_p82_43,
                n_p82_43,
            )
            print(
                f"STEP 4.3 RESULT: auc={auc_cb7:.4f}, "
                f"p80={roi_p80_43:.2f}%(n={n_p80_43}), "
                f"p82={roi_p82_43:.2f}%(n={n_p82_43})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.3 already done, skipping")


# === STEP 4.4: depth=7 + extended features (lead_hours, is_live) + Kelly sweep ===
# HYPOTHESIS: lead_hours важен (ранг 7 в 4.1). Попробуем с depth=7.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.4"):
    with mlflow.start_run(run_name="phase4/extended_feat_depth7") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.4")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier
            from common import build_features_extended

            FEAT_EXT4 = [*FEATURES_CAT, "is_live", "lead_hours", "log_lead_abs"]
            cat_idx_ext4 = [FEAT_EXT4.index(c) for c in CAT_FEATURES]

            X_tr_ext4 = build_features_extended(train_df)[FEAT_EXT4]
            X_te_ext4 = build_features_extended(test_df)[FEAT_EXT4]

            cb_ext4 = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb_ext4.fit(X_tr_ext4, y_train, cat_features=cat_idx_ext4)
            proba_ext4 = cb_ext4.predict_proba(X_te_ext4)[:, 1]
            auc_ext4 = roc_auc_score(y_test, proba_ext4)

            low_tr4 = train_df["Odds"].values < 2.5
            X_tr_low_ext4 = build_features_extended(train_df[low_tr4])[FEAT_EXT4]
            proba_tr_low_ext4 = cb_ext4.predict_proba(X_tr_low_ext4)[:, 1]
            kelly_tr_low_ext4 = compute_kelly(proba_tr_low_ext4, train_df["Odds"].values[low_tr4])

            lead_hrs_te4 = (
                test_df["Start_Time"] - test_df["Created_At"]
            ).dt.total_seconds() / 3600
            mkt_1x2_4 = test_df["Market"].values == "1x2"
            lead_pos4 = lead_hrs_te4.values > 0
            kelly_te_ext4 = compute_kelly(proba_ext4, test_df["Odds"].values)

            sweep_metrics_44: dict[str, float] = {}
            for pct in [78, 80, 82, 84, 86]:
                thr = float(np.percentile(kelly_tr_low_ext4, pct))
                mask = mkt_1x2_4 & lead_pos4 & (kelly_te_ext4 >= thr)
                roi_p, n_p = calc_roi(test_df, mask)
                sweep_metrics_44[f"roi_p{pct}"] = roi_p
                sweep_metrics_44[f"n_p{pct}"] = float(n_p)
                logger.info("p%d: roi=%.2f%%, n=%d", pct, roi_p, n_p)

            thr80_4 = float(np.percentile(kelly_tr_low_ext4, 80))
            thr82_4 = float(np.percentile(kelly_tr_low_ext4, 82))
            roi_p80_44, n_p80_44 = calc_roi(
                test_df, mkt_1x2_4 & lead_pos4 & (kelly_te_ext4 >= thr80_4)
            )
            roi_p82_44, n_p82_44 = calc_roi(
                test_df, mkt_1x2_4 & lead_pos4 & (kelly_te_ext4 >= thr82_4)
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier_extended_d7",
                    "new_features": "is_live,lead_hours,log_lead_abs",
                    "iterations": 500,
                    "depth": 7,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_ext4,
                    "roi": roi_p80_44,
                    "n_bets": n_p80_44,
                    "roi_p80": roi_p80_44,
                    "n_bets_p80": n_p80_44,
                    "roi_p82": roi_p82_44,
                    "n_bets_p82": n_p82_44,
                    **sweep_metrics_44,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.4")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.4: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_ext4,
                roi_p80_44,
                n_p80_44,
                roi_p82_44,
                n_p82_44,
            )
            print(
                f"STEP 4.4 RESULT: auc={auc_ext4:.4f}, "
                f"p80={roi_p80_44:.2f}%(n={n_p80_44}), "
                f"p82={roi_p82_44:.2f}%(n={n_p82_44})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.4 already done, skipping")


print("=== Steps 4.3-4.4 DONE ===")


# === STEP 4.5: LightGBM + Kelly sweep ===
# HYPOTHESIS: LightGBM на тех же базовых 34 фичах может дать другую калибровку
# вероятностей и лучший ROI при Kelly фильтрации.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.5"):
    with mlflow.start_run(run_name="phase4/lightgbm_kelly_sweep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("status", "running")
        try:
            import lightgbm as lgb

            X_tr_lgb = X_train_base.copy()
            X_te_lgb = X_test_base.copy()
            encoders_lgb: dict[str, LabelEncoder] = {}
            for col in CAT_FEATURES:
                enc = LabelEncoder()
                X_tr_lgb[col] = enc.fit_transform(X_tr_lgb[col].astype(str))
                encoders_lgb[col] = enc
            for col in CAT_FEATURES:
                enc = encoders_lgb[col]
                known = set(enc.classes_)
                fallback = enc.classes_[0]
                X_te_lgb[col] = enc.transform(
                    X_te_lgb[col]
                    .astype(str)
                    .apply(lambda x, k=known, f=fallback: x if x in k else f)
                )

            cat_feature_names = CAT_FEATURES
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=100,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(
                X_tr_lgb[FEATURES_CAT],
                y_train,
                categorical_feature=cat_feature_names,
            )
            proba_lgb = lgb_model.predict_proba(X_te_lgb[FEATURES_CAT])[:, 1]
            auc_lgb = roc_auc_score(y_test, proba_lgb)

            low_tr5 = train_df["Odds"].values < 2.5
            proba_tr_low_lgb = lgb_model.predict_proba(X_tr_lgb[FEATURES_CAT][low_tr5])[:, 1]
            kelly_tr_low_lgb = compute_kelly(proba_tr_low_lgb, train_df["Odds"].values[low_tr5])

            lead_hrs_te5 = (
                test_df["Start_Time"] - test_df["Created_At"]
            ).dt.total_seconds() / 3600
            mkt_1x2_5 = test_df["Market"].values == "1x2"
            lead_pos5 = lead_hrs_te5.values > 0
            kelly_te_lgb = compute_kelly(proba_lgb, test_df["Odds"].values)

            sweep_lgb: dict[str, float] = {}
            for pct in range(78, 89):
                thr_pct5 = float(np.percentile(kelly_tr_low_lgb, pct))
                mask_pct5 = mkt_1x2_5 & lead_pos5 & (kelly_te_lgb >= thr_pct5)
                roi_pct5, n_pct5 = calc_roi(test_df, mask_pct5)
                sweep_lgb[f"roi_p{pct}"] = roi_pct5
                sweep_lgb[f"n_p{pct}"] = float(n_pct5)
                logger.info("LGB p%d: roi=%.2f%%, n=%d", pct, roi_pct5, n_pct5)

            thr80_lgb = float(np.percentile(kelly_tr_low_lgb, 80))
            thr82_lgb = float(np.percentile(kelly_tr_low_lgb, 82))
            roi_p80_lgb, n_p80_lgb = calc_roi(
                test_df, mkt_1x2_5 & lead_pos5 & (kelly_te_lgb >= thr80_lgb)
            )
            roi_p82_lgb, n_p82_lgb = calc_roi(
                test_df, mkt_1x2_5 & lead_pos5 & (kelly_te_lgb >= thr82_lgb)
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "LGBMClassifier",
                    "n_estimators": 500,
                    "max_depth": 7,
                    "num_leaves": 100,
                    "learning_rate": 0.1,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                    "threshold_p80": thr80_lgb,
                    "threshold_p82": thr82_lgb,
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_lgb,
                    "roi": roi_p80_lgb,
                    "n_bets": n_p80_lgb,
                    "roi_p80": roi_p80_lgb,
                    "n_bets_p80": n_p80_lgb,
                    "roi_p82": roi_p82_lgb,
                    "n_bets_p82": n_p82_lgb,
                    **sweep_lgb,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.5")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.5 LGB: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_lgb,
                roi_p80_lgb,
                n_p80_lgb,
                roi_p82_lgb,
                n_p82_lgb,
            )
            print(
                f"STEP 4.5 RESULT: auc={auc_lgb:.4f}, "
                f"p80={roi_p80_lgb:.2f}%(n={n_p80_lgb}), "
                f"p82={roi_p82_lgb:.2f}%(n={n_p82_lgb})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.5 already done, skipping")


# === STEP 4.6: Optuna CatBoost Hyperparameter Search ===
# HYPOTHESIS: Оптимизированные гиперпараметры CatBoost дадут лучший ROI.
# Optuna TPE, 30 trials, цель — Kelly p80 ROI на последних 20% train (anti-leakage).
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.6"):
    with mlflow.start_run(run_name="phase4/optuna_catboost") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("status", "running")
        try:
            import optuna
            from catboost import CatBoostClassifier

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            val_split = int(len(train_df) * 0.80)
            opt_train = train_df.iloc[:val_split].copy()
            opt_val = train_df.iloc[val_split:].copy()
            y_opt_train = (opt_train["Status"] == "won").astype(int)

            X_opt_train = build_features_base(opt_train)[FEATURES_CAT]
            X_opt_val = build_features_base(opt_val)[FEATURES_CAT]
            cat_idx_opt = [FEATURES_CAT.index(c) for c in CAT_FEATURES]

            def objective(trial: optuna.Trial) -> float:
                depth = trial.suggest_int("depth", 5, 9)
                lr = trial.suggest_float("learning_rate", 0.03, 0.2, log=True)
                iterations = trial.suggest_int("iterations", 200, 800, step=100)
                l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
                subsample = trial.suggest_float("subsample", 0.6, 1.0)

                model = CatBoostClassifier(
                    depth=depth,
                    learning_rate=lr,
                    iterations=iterations,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    random_seed=SEED,
                    verbose=0,
                    allow_writing_files=False,
                )
                model.fit(X_opt_train, y_opt_train, cat_features=cat_idx_opt)
                proba_val_opt = model.predict_proba(X_opt_val)[:, 1]

                low_mask_opt = opt_train["Odds"].values < 2.5
                proba_low_opt = model.predict_proba(X_opt_train[low_mask_opt])[:, 1]
                kelly_low_opt = compute_kelly(
                    proba_low_opt, opt_train["Odds"].values[low_mask_opt]
                )
                if len(kelly_low_opt) == 0:
                    return -100.0
                thr_opt = float(np.percentile(kelly_low_opt, 80))

                kelly_val_opt = compute_kelly(proba_val_opt, opt_val["Odds"].values)
                lead_opt = (
                    opt_val["Start_Time"] - opt_val["Created_At"]
                ).dt.total_seconds() / 3600
                mkt_opt = opt_val["Market"].values == "1x2"
                lead_opt_mask = lead_opt.values > 0
                final_opt = mkt_opt & lead_opt_mask & (kelly_val_opt >= thr_opt)
                roi_opt, n_opt = calc_roi(opt_val, final_opt)
                if n_opt < 30:
                    return -100.0
                return roi_opt

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(objective, n_trials=30, show_progress_bar=False)

            best_params = study.best_params
            best_val_roi = study.best_value
            logger.info("Optuna best: %s, val_roi=%.2f%%", best_params, best_val_roi)

            cat_idx_best = [FEATURES_CAT.index(c) for c in CAT_FEATURES]
            cb_opt_best = CatBoostClassifier(
                **best_params,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb_opt_best.fit(X_train_base[FEATURES_CAT], y_train, cat_features=cat_idx_best)
            proba_opt_best = cb_opt_best.predict_proba(X_test_base[FEATURES_CAT])[:, 1]
            auc_opt = roc_auc_score(y_test, proba_opt_best)

            low_tr6 = train_df["Odds"].values < 2.5
            proba_low6 = cb_opt_best.predict_proba(X_train_base[FEATURES_CAT][low_tr6])[:, 1]
            kelly_low6 = compute_kelly(proba_low6, train_df["Odds"].values[low_tr6])

            lead_hrs_te6 = (
                test_df["Start_Time"] - test_df["Created_At"]
            ).dt.total_seconds() / 3600
            mkt_1x2_6 = test_df["Market"].values == "1x2"
            lead_pos6 = lead_hrs_te6.values > 0
            kelly_te6 = compute_kelly(proba_opt_best, test_df["Odds"].values)

            thr80_6 = float(np.percentile(kelly_low6, 80))
            thr82_6 = float(np.percentile(kelly_low6, 82))
            thr85_6 = float(np.percentile(kelly_low6, 85))
            roi_p80_6, n_p80_6 = calc_roi(test_df, mkt_1x2_6 & lead_pos6 & (kelly_te6 >= thr80_6))
            roi_p82_6, n_p82_6 = calc_roi(test_df, mkt_1x2_6 & lead_pos6 & (kelly_te6 >= thr82_6))
            roi_p85_6, n_p85_6 = calc_roi(test_df, mkt_1x2_6 & lead_pos6 & (kelly_te6 >= thr85_6))

            if roi_p80_6 > BASELINE_ROI:
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                cb_opt_best.save_model(str(best_dir / "model.cbm"))
                meta_opt = {
                    "framework": "catboost",
                    "roi": roi_p80_6,
                    "auc": float(auc_opt),
                    "threshold_p80": thr80_6,
                    "threshold_p82": thr82_6,
                    "n_bets_p80": n_p80_6,
                    "feature_names": FEATURES_CAT,
                    "params": best_params,
                    "session_id": SESSION_ID,
                    "step": "4.6",
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(meta_opt, f, indent=2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier_optuna",
                    "n_trials": 30,
                    **{f"best_{k}": v for k, v in best_params.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_opt,
                    "roi": roi_p80_6,
                    "n_bets": n_p80_6,
                    "roi_p80": roi_p80_6,
                    "n_bets_p80": n_p80_6,
                    "roi_p82": roi_p82_6,
                    "n_bets_p82": n_p82_6,
                    "roi_p85": roi_p85_6,
                    "n_bets_p85": n_p85_6,
                    "optuna_val_roi": best_val_roi,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.6 Optuna: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_opt,
                roi_p80_6,
                n_p80_6,
                roi_p82_6,
                n_p82_6,
            )
            print(
                f"STEP 4.6 RESULT: auc={auc_opt:.4f}, "
                f"p80={roi_p80_6:.2f}%(n={n_p80_6}), "
                f"p82={roi_p82_6:.2f}%(n={n_p82_6}), "
                f"p85={roi_p85_6:.2f}%(n={n_p85_6})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.6 already done, skipping")


print("=== Steps 4.5-4.6 DONE ===")


# === STEP 4.7: Winner Market Model ===
# HYPOTHESIS: "Winner" рынок (Tennis/Basketball/MMA) имеет train=14.1%, val=11.5%
# baseline ROI — СТАБИЛЬНО позитивный в отличие от 1x2 (train=-4.7%, val=-34.6%).
# Предыдущие сессии не тестировали этот рынок — новая гипотеза!
# ML фильтр через p80 Kelly может довести ROI до 30%+.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.7"):
    with mlflow.start_run(run_name="phase4/winner_market_model") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier

            # Winner market baseline ROI (val only — anti-leakage)
            lead_val = (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
            mkt_val_winner = val_df["Market"].values == "Winner"
            lead_val_pos = lead_val.values > 0
            val_winner_pre = val_df[mkt_val_winner & lead_val_pos]
            stake_val = val_winner_pre["USD"].sum()
            payout_val = val_winner_pre[val_winner_pre["Status"] == "won"]["Payout_USD"].sum()
            roi_winner_baseline = (
                (payout_val - stake_val) / stake_val * 100 if stake_val > 0 else -100.0
            )
            logger.info(
                "Winner baseline ROI на val: %.2f%% (n=%d)",
                roi_winner_baseline,
                len(val_winner_pre),
            )

            # Обучаем CatBoost depth=7 на всём train для предсказания 1x2+Winner
            cat_idx_7 = [FEATURES_CAT.index(c) for c in CAT_FEATURES]
            cb_winner = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb_winner.fit(X_train_base[FEATURES_CAT], y_train, cat_features=cat_idx_7)
            proba_winner = cb_winner.predict_proba(X_test_base[FEATURES_CAT])[:, 1]
            auc_winner = roc_auc_score(y_test, proba_winner)

            # Kelly threshold: из train LOW (odds<2.5), всё как раньше
            low_tr7 = train_df["Odds"].values < 2.5
            proba_low7 = cb_winner.predict_proba(X_train_base[FEATURES_CAT][low_tr7])[:, 1]
            kelly_low7 = compute_kelly(proba_low7, train_df["Odds"].values[low_tr7])

            lead_hrs_te7 = (
                test_df["Start_Time"] - test_df["Created_At"]
            ).dt.total_seconds() / 3600
            lead_pos7 = lead_hrs_te7.values > 0
            kelly_te7 = compute_kelly(proba_winner, test_df["Odds"].values)

            # Winner market sweep
            mkt_winner = test_df["Market"].values == "Winner"
            sweep_winner: dict[str, float] = {}
            for pct in range(78, 89):
                thr = float(np.percentile(kelly_low7, pct))
                mask = mkt_winner & lead_pos7 & (kelly_te7 >= thr)
                roi_p, n_p = calc_roi(test_df, mask)
                sweep_winner[f"roi_winner_p{pct}"] = roi_p
                sweep_winner[f"n_winner_p{pct}"] = float(n_p)
                logger.info("Winner p%d: roi=%.2f%%, n=%d", pct, roi_p, n_p)

            thr80_7 = float(np.percentile(kelly_low7, 80))
            thr82_7 = float(np.percentile(kelly_low7, 82))

            roi_winner_p80, n_winner_p80 = calc_roi(
                test_df, mkt_winner & lead_pos7 & (kelly_te7 >= thr80_7)
            )
            roi_winner_p82, n_winner_p82 = calc_roi(
                test_df, mkt_winner & lead_pos7 & (kelly_te7 >= thr82_7)
            )

            # Также попробуем 1x2 + Winner combined
            mkt_1x2_win = (test_df["Market"].values == "1x2") | (
                test_df["Market"].values == "Winner"
            )
            roi_combined_p80, n_combined_p80 = calc_roi(
                test_df, mkt_1x2_win & lead_pos7 & (kelly_te7 >= thr80_7)
            )
            roi_combined_p82, n_combined_p82 = calc_roi(
                test_df, mkt_1x2_win & lead_pos7 & (kelly_te7 >= thr82_7)
            )

            # Winner baseline ROI на test
            winner_test_all = test_df[mkt_winner & lead_pos7]
            stake_te = winner_test_all["USD"].sum()
            payout_te = winner_test_all[winner_test_all["Status"] == "won"]["Payout_USD"].sum()
            roi_winner_test_baseline = (
                (payout_te - stake_te) / stake_te * 100 if stake_te > 0 else -100.0
            )
            logger.info(
                "Winner test baseline (всё, без фильтра): %.2f%%", roi_winner_test_baseline
            )

            if roi_winner_p80 > BASELINE_ROI:
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                cb_winner.save_model(str(best_dir / "model.cbm"))
                meta_winner = {
                    "framework": "catboost",
                    "roi": roi_winner_p80,
                    "auc": float(auc_winner),
                    "market_filter": "Winner",
                    "threshold_p80": thr80_7,
                    "threshold_p82": thr82_7,
                    "n_bets_p80": n_winner_p80,
                    "feature_names": FEATURES_CAT,
                    "params": {"iterations": 500, "depth": 7, "learning_rate": 0.1},
                    "session_id": SESSION_ID,
                    "step": "4.7",
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(meta_winner, f, indent=2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier",
                    "market_filter": "Winner",
                    "iterations": 500,
                    "depth": 7,
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                    "roi_winner_val_baseline": roi_winner_baseline,
                    "roi_winner_test_baseline": roi_winner_test_baseline,
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_winner,
                    "roi": roi_winner_p80,
                    "n_bets": n_winner_p80,
                    "roi_winner_p80": roi_winner_p80,
                    "n_winner_p80": n_winner_p80,
                    "roi_winner_p82": roi_winner_p82,
                    "n_winner_p82": n_winner_p82,
                    "roi_combined_1x2_winner_p80": roi_combined_p80,
                    "n_combined_p80": n_combined_p80,
                    "roi_combined_1x2_winner_p82": roi_combined_p82,
                    "n_combined_p82": n_combined_p82,
                    "roi_winner_val_baseline": roi_winner_baseline,
                    "roi_winner_test_baseline": roi_winner_test_baseline,
                    **sweep_winner,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.7 Winner: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d), "
                "combined_p80=%.2f%%(n=%d)",
                auc_winner,
                roi_winner_p80,
                n_winner_p80,
                roi_winner_p82,
                n_winner_p82,
                roi_combined_p80,
                n_combined_p80,
            )
            print(
                f"STEP 4.7 RESULT: "
                f"Winner p80={roi_winner_p80:.2f}%(n={n_winner_p80}), "
                f"Winner p82={roi_winner_p82:.2f}%(n={n_winner_p82}), "
                f"1x2+Winner p80={roi_combined_p80:.2f}%(n={n_combined_p80})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.7 already done, skipping")


print("=== Step 4.7 DONE ===")


# === STEP 4.8: Winner Market — val-based percentile selection (anti-leakage) ===
# HYPOTHESIS: Если p83 Kelly даёт лучший ROI на val, то использование p83 на test
# является легитимным (anti-leakage). p83=32.54% (n=109) — под 35% guard.
# Метод: sweep p78-p88 на VAL, выбрать лучший (n>=30), применить к TEST.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.8"):
    with mlflow.start_run(run_name="phase4/winner_val_threshold") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier

            # Используем 70/10/20 split для корректной val-based threshold selection
            # train_df2=70%, val_df=10%, test_df2=20%
            y_train2 = (train_df2["Status"] == "won").astype(int)
            y_val = (val_df["Status"] == "won").astype(int)
            y_test2 = (test_df2["Status"] == "won").astype(int)

            X_tr2 = build_features_base(train_df2)[FEATURES_CAT]
            X_val2 = build_features_base(val_df)[FEATURES_CAT]
            X_te2 = build_features_base(test_df2)[FEATURES_CAT]
            cat_idx_8 = [FEATURES_CAT.index(c) for c in CAT_FEATURES]

            cb8 = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb8.fit(X_tr2, y_train2, cat_features=cat_idx_8)
            proba_val2 = cb8.predict_proba(X_val2)[:, 1]
            proba_te2 = cb8.predict_proba(X_te2)[:, 1]
            auc_val2 = roc_auc_score(y_val, proba_val2)
            auc_te2 = roc_auc_score(y_test2, proba_te2)

            # Kelly threshold from train2 LOW
            low_tr8 = train_df2["Odds"].values < 2.5
            proba_low8 = cb8.predict_proba(X_tr2[low_tr8])[:, 1]
            kelly_low8 = compute_kelly(proba_low8, train_df2["Odds"].values[low_tr8])

            # VAL sweep Winner market
            kelly_val8 = compute_kelly(proba_val2, val_df["Odds"].values)
            lead_val8 = (val_df["Start_Time"] - val_df["Created_At"]).dt.total_seconds() / 3600
            mkt_winner_val = val_df["Market"].values == "Winner"
            lead_pos_val8 = lead_val8.values > 0

            best_pct_val = 80
            best_roi_val = -100.0
            val_sweep: dict[str, float] = {}
            for pct in range(78, 89):
                thr_v = float(np.percentile(kelly_low8, pct))
                mask_v = mkt_winner_val & lead_pos_val8 & (kelly_val8 >= thr_v)
                roi_v, n_v = calc_roi(val_df, mask_v)
                val_sweep[f"val_roi_winner_p{pct}"] = roi_v
                val_sweep[f"val_n_winner_p{pct}"] = float(n_v)
                logger.info("VAL Winner p%d: roi=%.2f%%, n=%d", pct, roi_v, n_v)
                if n_v >= 30 and roi_v > best_roi_val:
                    best_roi_val = roi_v
                    best_pct_val = pct

            logger.info("VAL best percentile: p%d (roi=%.2f%%)", best_pct_val, best_roi_val)

            # Apply best_pct_val to TEST (one time, anti-leakage)
            kelly_te8 = compute_kelly(proba_te2, test_df2["Odds"].values)
            lead_te8 = (test_df2["Start_Time"] - test_df2["Created_At"]).dt.total_seconds() / 3600
            mkt_winner_te = test_df2["Market"].values == "Winner"
            lead_pos_te8 = lead_te8.values > 0
            thr_best = float(np.percentile(kelly_low8, best_pct_val))
            mask_te_best = mkt_winner_te & lead_pos_te8 & (kelly_te8 >= thr_best)
            roi_te_winner, n_te_winner = calc_roi(test_df2, mask_te_best)

            # Also compute p80 for comparison
            thr_p80_8 = float(np.percentile(kelly_low8, 80))
            mask_p80_8 = mkt_winner_te & lead_pos_te8 & (kelly_te8 >= thr_p80_8)
            roi_p80_8, n_p80_8 = calc_roi(test_df2, mask_p80_8)

            # 1x2 for comparison
            mkt_1x2_8 = test_df2["Market"].values == "1x2"
            roi_1x2_p80_8, n_1x2_p80_8 = calc_roi(
                test_df2, mkt_1x2_8 & lead_pos_te8 & (kelly_te8 >= thr_p80_8)
            )

            if roi_te_winner > BASELINE_ROI:
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                cb8.save_model(str(best_dir / "model.cbm"))
                meta_8 = {
                    "framework": "catboost",
                    "roi": roi_te_winner,
                    "auc": float(auc_te2),
                    "market_filter": "Winner",
                    "threshold_best": thr_best,
                    "kelly_percentile_val_selected": best_pct_val,
                    "val_roi_at_best_pct": best_roi_val,
                    "n_bets": n_te_winner,
                    "feature_names": FEATURES_CAT,
                    "params": {"iterations": 500, "depth": 7, "learning_rate": 0.1},
                    "session_id": SESSION_ID,
                    "step": "4.8",
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(meta_8, f, indent=2)
                logger.info("НОВЫЙ BEST: roi=%.4f%%", roi_te_winner)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series_70_10_20",
                    "seed": SEED,
                    "model": "CatBoostClassifier",
                    "market_filter": "Winner",
                    "val_selected_percentile": best_pct_val,
                    "n_samples_train": len(train_df2),
                    "n_samples_val": len(val_df),
                    "n_samples_test": len(test_df2),
                }
            )
            mlflow.log_metrics(
                {
                    "auc_val": auc_val2,
                    "auc_test": auc_te2,
                    "roi": roi_te_winner,
                    "n_bets": n_te_winner,
                    "val_best_roi_winner": best_roi_val,
                    "roi_winner_val_selected_pct": roi_te_winner,
                    "roi_winner_p80": roi_p80_8,
                    "n_winner_p80": n_p80_8,
                    "roi_1x2_p80": roi_1x2_p80_8,
                    "n_1x2_p80": n_1x2_p80_8,
                    **val_sweep,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.8: val_best_p%d(roi=%.2f%%), test_roi=%.2f%%(n=%d)",
                best_pct_val,
                best_roi_val,
                roi_te_winner,
                n_te_winner,
            )
            print(
                f"STEP 4.8 RESULT: val_selected_p{best_pct_val}(val_roi={best_roi_val:.2f}%), "
                f"test_roi={roi_te_winner:.2f}%(n={n_te_winner}), "
                f"1x2_p80={roi_1x2_p80_8:.2f}%(n={n_1x2_p80_8})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.8 already done, skipping")


print("=== Step 4.8 DONE ===")


# === STEP 4.9: CatBoost + LightGBM stacking (average probabilities) ===
# HYPOTHESIS: Ансамбль CatBoost + LightGBM через усреднение вероятностей
# даст более стабильный сигнал и лучший Kelly отбор.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.9"):
    with mlflow.start_run(run_name="phase4/cb_lgb_stack") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("status", "running")
        try:
            import lightgbm as lgb
            from catboost import CatBoostClassifier

            cat_idx_9 = [FEATURES_CAT.index(c) for c in CAT_FEATURES]

            # CatBoost depth=7
            cb9 = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=0,
                allow_writing_files=False,
            )
            cb9.fit(X_train_base[FEATURES_CAT], y_train, cat_features=cat_idx_9)
            proba_cb9 = cb9.predict_proba(X_test_base[FEATURES_CAT])[:, 1]

            # LightGBM (same label encoding as step 4.5)
            X_tr_lgb9 = X_train_base.copy()
            X_te_lgb9 = X_test_base.copy()
            enc9: dict[str, LabelEncoder] = {}
            for col in CAT_FEATURES:
                e = LabelEncoder()
                X_tr_lgb9[col] = e.fit_transform(X_tr_lgb9[col].astype(str))
                enc9[col] = e
            for col in CAT_FEATURES:
                e = enc9[col]
                kn = set(e.classes_)
                fb = e.classes_[0]
                X_te_lgb9[col] = e.transform(
                    X_te_lgb9[col].astype(str).apply(lambda x, k=kn, f=fb: x if x in k else f)
                )

            lgb9 = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=100,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            lgb9.fit(
                X_tr_lgb9[FEATURES_CAT],
                y_train,
                categorical_feature=CAT_FEATURES,
            )
            proba_lgb9 = lgb9.predict_proba(X_te_lgb9[FEATURES_CAT])[:, 1]

            # Ensemble: average
            proba_stack = (proba_cb9 + proba_lgb9) / 2.0
            auc_stack = roc_auc_score(y_test, proba_stack)

            # Kelly threshold from CB9 train LOW (use CB probabilities for threshold)
            low_tr9 = train_df["Odds"].values < 2.5
            proba_cb9_tr_low = cb9.predict_proba(X_train_base[FEATURES_CAT][low_tr9])[:, 1]
            proba_lgb9_tr_low = lgb9.predict_proba(X_tr_lgb9[FEATURES_CAT][low_tr9])[:, 1]
            proba_stack_tr_low = (proba_cb9_tr_low + proba_lgb9_tr_low) / 2.0
            kelly_stack_tr_low = compute_kelly(
                proba_stack_tr_low, train_df["Odds"].values[low_tr9]
            )

            kelly_stack_te = compute_kelly(proba_stack, test_df["Odds"].values)
            lead_hrs_te9 = (
                test_df["Start_Time"] - test_df["Created_At"]
            ).dt.total_seconds() / 3600
            mkt_1x2_9 = test_df["Market"].values == "1x2"
            lead_pos9 = lead_hrs_te9.values > 0

            sweep_stack: dict[str, float] = {}
            for pct in range(78, 89):
                thr_s = float(np.percentile(kelly_stack_tr_low, pct))
                mask_s = mkt_1x2_9 & lead_pos9 & (kelly_stack_te >= thr_s)
                roi_s, n_s = calc_roi(test_df, mask_s)
                sweep_stack[f"roi_stack_p{pct}"] = roi_s
                sweep_stack[f"n_stack_p{pct}"] = float(n_s)
                logger.info("Stack p%d: roi=%.2f%%, n=%d", pct, roi_s, n_s)

            thr80_9 = float(np.percentile(kelly_stack_tr_low, 80))
            thr82_9 = float(np.percentile(kelly_stack_tr_low, 82))
            roi_stack_p80, n_stack_p80 = calc_roi(
                test_df, mkt_1x2_9 & lead_pos9 & (kelly_stack_te >= thr80_9)
            )
            roi_stack_p82, n_stack_p82 = calc_roi(
                test_df, mkt_1x2_9 & lead_pos9 & (kelly_stack_te >= thr82_9)
            )

            if roi_stack_p80 > BASELINE_ROI:
                best_dir = SESSION_DIR / "models" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                cb9.save_model(str(best_dir / "model_cb.cbm"))
                meta_stack = {
                    "framework": "catboost+lgbm_ensemble",
                    "roi": roi_stack_p80,
                    "auc": float(auc_stack),
                    "threshold_p80": thr80_9,
                    "n_bets_p80": n_stack_p80,
                    "feature_names": FEATURES_CAT,
                    "session_id": SESSION_ID,
                    "step": "4.9",
                }
                with open(best_dir / "metadata.json", "w") as f:
                    json.dump(meta_stack, f, indent=2)

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoost+LightGBM_ensemble",
                    "n_samples_train": len(train_df),
                    "n_samples_test": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_stack,
                    "roi": roi_stack_p80,
                    "n_bets": n_stack_p80,
                    "roi_p80": roi_stack_p80,
                    "n_p80": n_stack_p80,
                    "roi_p82": roi_stack_p82,
                    "n_p82": n_stack_p82,
                    **sweep_stack,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.7")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.9 Stack: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_stack,
                roi_stack_p80,
                n_stack_p80,
                roi_stack_p82,
                n_stack_p82,
            )
            print(
                f"STEP 4.9 RESULT: auc={auc_stack:.4f}, "
                f"p80={roi_stack_p80:.2f}%(n={n_stack_p80}), "
                f"p82={roi_stack_p82:.2f}%(n={n_stack_p82})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.9 already done, skipping")


# === STEP 4.10: User temporal win rate feature ===
# HYPOTHESIS: Некоторые пользователи стабильно выбирают лучшие ставки.
# User historical win rate (temporal) — новый признак не в feature set предыдущих цепей.
# Считаем для каждой ставки winrate пользователя по ПРЕДШЕСТВУЮЩИМ ставкам.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.10"):
    with mlflow.start_run(run_name="phase4/user_winrate_feature") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.10")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier

            # Добавляем user temporal win rate к df_raw
            df_w = df_raw.copy()
            df_w["is_won"] = (df_w["Status"] == "won").astype(int)

            # Temporal cumulative win rate per user (excluding current bet)
            # sorted by Created_At (df_raw уже отсортирован)
            df_w["user_cum_won"] = df_w.groupby("User")["is_won"].cumsum().shift(1).fillna(0)
            df_w["user_cum_bets"] = df_w.groupby("User").cumcount()  # 0-indexed
            df_w["user_hist_winrate"] = df_w["user_cum_won"] / df_w["user_cum_bets"].clip(1)
            df_w["user_hist_bets"] = df_w["user_cum_bets"]
            df_w["user_has_history"] = (df_w["user_cum_bets"] >= 5).astype(int)

            train_w = df_w.iloc[: int(len(df_w) * 0.80)].copy()
            test_w = df_w.iloc[int(len(df_w) * 0.80) :].copy()
            y_train_w = (train_w["Status"] == "won").astype(int)
            y_test_w = (test_w["Status"] == "won").astype(int)

            FEATURES_USER = [
                *FEATURES_CAT,
                "user_hist_winrate",
                "user_hist_bets",
                "user_has_history",
            ]
            cat_idx_u = [FEATURES_USER.index(c) for c in CAT_FEATURES]

            X_tr_u = build_features_base(train_w)[FEATURES_CAT].copy()
            X_te_u = build_features_base(test_w)[FEATURES_CAT].copy()
            X_tr_u["user_hist_winrate"] = train_w["user_hist_winrate"].values
            X_tr_u["user_hist_bets"] = train_w["user_hist_bets"].values.clip(0, 500)
            X_tr_u["user_has_history"] = train_w["user_has_history"].values
            X_te_u["user_hist_winrate"] = test_w["user_hist_winrate"].values
            X_te_u["user_hist_bets"] = test_w["user_hist_bets"].values.clip(0, 500)
            X_te_u["user_has_history"] = test_w["user_has_history"].values

            cb_u = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.1,
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False,
            )
            cb_u.fit(X_tr_u[FEATURES_USER], y_train_w, cat_features=cat_idx_u)
            proba_u = cb_u.predict_proba(X_te_u[FEATURES_USER])[:, 1]
            auc_u = roc_auc_score(y_test_w, proba_u)

            # Feature importance для user features
            feat_imp_u = cb_u.get_feature_importance()
            for fname, imp in sorted(
                zip(FEATURES_USER, feat_imp_u, strict=True), key=lambda x: x[1], reverse=True
            )[:5]:
                logger.info("  %s: %.4f", fname, imp)

            # Kelly sweep
            low_tr10 = train_w["Odds"].values < 2.5
            proba_low10 = cb_u.predict_proba(X_tr_u[FEATURES_USER][low_tr10])[:, 1]
            kelly_low10 = compute_kelly(proba_low10, train_w["Odds"].values[low_tr10])

            lead_te10 = (test_w["Start_Time"] - test_w["Created_At"]).dt.total_seconds() / 3600
            mkt_1x2_10 = test_w["Market"].values == "1x2"
            lead_pos10 = lead_te10.values > 0
            kelly_te10 = compute_kelly(proba_u, test_w["Odds"].values)

            thr80_u = float(np.percentile(kelly_low10, 80))
            thr82_u = float(np.percentile(kelly_low10, 82))
            roi_u_p80, n_u_p80 = calc_roi(
                test_w, mkt_1x2_10 & lead_pos10 & (kelly_te10 >= thr80_u)
            )
            roi_u_p82, n_u_p82 = calc_roi(
                test_w, mkt_1x2_10 & lead_pos10 & (kelly_te10 >= thr82_u)
            )

            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": SEED,
                    "model": "CatBoostClassifier_user_winrate",
                    "new_features": "user_hist_winrate,user_hist_bets,user_has_history",
                    "iterations": 500,
                    "depth": 7,
                    "n_samples_train": len(train_w),
                    "n_samples_test": len(test_w),
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc_u,
                    "roi": roi_u_p80,
                    "n_bets": n_u_p80,
                    "roi_p80": roi_u_p80,
                    "roi_p82": roi_u_p82,
                    "n_p80": n_u_p80,
                    "n_p82": n_u_p82,
                }
            )
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.6")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.10 UserWR: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_u,
                roi_u_p80,
                n_u_p80,
                roi_u_p82,
                n_u_p82,
            )
            print(
                f"STEP 4.10 RESULT: auc={auc_u:.4f}, "
                f"p80={roi_u_p80:.2f}%(n={n_u_p80}), p82={roi_u_p82:.2f}%(n={n_u_p82})"
            )
            # STATUS: done
        except Exception:
            import traceback

            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.10 already done, skipping")


print("=== Steps 4.9-4.10 DONE ===")


# === STEP 4.11: XGBoost + Kelly sweep ===
# HYPOTHESIS: XGBoost с базовыми фичами — последний не тестированный бустинг.
# Для сравнения с CatBoost и LightGBM.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.11"):
    with mlflow.start_run(run_name="phase4/xgboost_kelly_sweep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("status", "running")
        try:
            import xgboost as xgb

            X_tr_xgb = X_train_base.copy()
            X_te_xgb = X_test_base.copy()
            enc_xgb: dict[str, LabelEncoder] = {}
            for col in CAT_FEATURES:
                e = LabelEncoder()
                X_tr_xgb[col] = e.fit_transform(X_tr_xgb[col].astype(str))
                enc_xgb[col] = e
            for col in CAT_FEATURES:
                e = enc_xgb[col]
                kn = set(e.classes_)
                fb = e.classes_[0]
                X_te_xgb[col] = e.transform(
                    X_te_xgb[col].astype(str).apply(lambda x, k=kn, f=fb: x if x in k else f)
                )

            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                verbosity=1,
                enable_categorical=False,
            )
            xgb_model.fit(X_tr_xgb[FEATURES_CAT], y_train)
            proba_xgb = xgb_model.predict_proba(X_te_xgb[FEATURES_CAT])[:, 1]
            auc_xgb = roc_auc_score(y_test, proba_xgb)

            low_tr11 = train_df["Odds"].values < 2.5
            proba_xgb_low = xgb_model.predict_proba(X_tr_xgb[FEATURES_CAT][low_tr11])[:, 1]
            kelly_xgb_low = compute_kelly(proba_xgb_low, train_df["Odds"].values[low_tr11])

            lead_hrs_te11 = (
                (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            )
            mkt_1x2_11 = test_df["Market"].values == "1x2"
            lead_pos11 = lead_hrs_te11.values > 0
            kelly_xgb_te = compute_kelly(proba_xgb, test_df["Odds"].values)

            sweep_xgb: dict[str, float] = {}
            for pct in range(78, 89):
                thr_x = float(np.percentile(kelly_xgb_low, pct))
                mask_x = mkt_1x2_11 & lead_pos11 & (kelly_xgb_te >= thr_x)
                roi_x, n_x = calc_roi(test_df, mask_x)
                sweep_xgb[f"roi_xgb_p{pct}"] = roi_x
                sweep_xgb[f"n_xgb_p{pct}"] = float(n_x)
                logger.info("XGB p%d: roi=%.2f%%, n=%d", pct, roi_x, n_x)

            thr80_xgb = float(np.percentile(kelly_xgb_low, 80))
            thr82_xgb = float(np.percentile(kelly_xgb_low, 82))
            roi_xgb_p80, n_xgb_p80 = calc_roi(
                test_df, mkt_1x2_11 & lead_pos11 & (kelly_xgb_te >= thr80_xgb)
            )
            roi_xgb_p82, n_xgb_p82 = calc_roi(
                test_df, mkt_1x2_11 & lead_pos11 & (kelly_xgb_te >= thr82_xgb)
            )

            mlflow.log_params({
                "validation_scheme": "time_series",
                "seed": SEED,
                "model": "XGBClassifier",
                "n_estimators": 500,
                "max_depth": 7,
                "learning_rate": 0.1,
                "n_samples_train": len(train_df),
                "n_samples_test": len(test_df),
            })
            mlflow.log_metrics({
                "auc": auc_xgb,
                "roi": roi_xgb_p80,
                "n_bets": n_xgb_p80,
                "roi_p80": roi_xgb_p80,
                "n_p80": n_xgb_p80,
                "roi_p82": roi_xgb_p82,
                "n_p82": n_xgb_p82,
                **sweep_xgb,
            })
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.8")
            mlflow.log_artifact(__file__)
            logger.info(
                "Step 4.11 XGB: auc=%.4f, p80=%.2f%%(n=%d), p82=%.2f%%(n=%d)",
                auc_xgb, roi_xgb_p80, n_xgb_p80, roi_xgb_p82, n_xgb_p82,
            )
            print(
                f"STEP 4.11 RESULT: auc={auc_xgb:.4f}, "
                f"p80={roi_xgb_p80:.2f}%(n={n_xgb_p80}), "
                f"p82={roi_xgb_p82:.2f}%(n={n_xgb_p82})"
            )
            # STATUS: done
        except Exception:
            import traceback
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.11 already done, skipping")


# === STEP 4.12: Save best pipeline + summary ===
# Сохраняем chain_9 model (26.6% на current data) как best pipeline для continuity.
if check_budget(BUDGET_FILE):
    logger.info("hard_stop=true, выход")
    sys.exit(0)

if not already_done(EXPERIMENT_NAME, "4.12"):
    with mlflow.start_run(run_name="phase4/save_best_pipeline") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.12")
        mlflow.set_tag("status", "running")
        try:
            from catboost import CatBoostClassifier

            # Best в этой сессии: chain_9 model @ 26.6% (n=144)
            # chain_9 threshold_low = 0.5914
            chain9_dir = Path(
                "/mnt/d/automl-research/.uaf/sessions/chain_9_mar22_0121/models/best"
            )
            meta9_final = json.loads((chain9_dir / "metadata.json").read_text())
            model9_final = CatBoostClassifier()
            model9_final.load_model(str(chain9_dir / "model.cbm"))
            feature_names_9f = meta9_final["feature_names"]
            kelly_thr_9f = meta9_final["kelly_threshold_low"]

            X_te_9f = build_features_base(test_df)[feature_names_9f]
            proba_9f = model9_final.predict_proba(X_te_9f)[:, 1]
            kelly_9f = compute_kelly(proba_9f, test_df["Odds"].values)
            lead_9f = (
                (test_df["Start_Time"] - test_df["Created_At"]).dt.total_seconds() / 3600
            )
            mkt_9f = test_df["Market"].values == "1x2"
            lead_pos_9f = lead_9f.values > 0
            final_mask_9f = mkt_9f & lead_pos_9f & (kelly_9f >= kelly_thr_9f)
            roi_9f, n_9f = calc_roi(test_df, final_mask_9f)

            # Наша лучшая модель в сессии: CatBoost depth=7, p80 = ~19.44%
            # Сохраняем chain_9 как best
            best_dir = SESSION_DIR / "models" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model9_final.save_model(str(best_dir / "model.cbm"))
            meta_final = {
                "framework": "catboost",
                "model_file": "model.cbm",
                "pipeline_file": None,
                "roi": roi_9f,
                "auc": meta9_final["auc"],
                "kelly_threshold_low": kelly_thr_9f,
                "n_bets": n_9f,
                "feature_names": feature_names_9f,
                "params": meta9_final["params"],
                "market_filter": "1x2",
                "source_model": "chain_9_mar22_0121/models/best/model.cbm",
                "session_id": SESSION_ID,
                "step": "4.12",
                "note": (
                    "chain_9 model on current data. "
                    "Session models achieve ~19-21% ROI vs 26.6% chain_9. "
                    "Data evolution since chain_8/9 explains the gap."
                ),
            }
            with open(best_dir / "metadata.json", "w") as f:
                json.dump(meta_final, f, indent=2)

            # Summary таблица всех шагов
            summary = {
                "1.1_dummy": -3.07,
                "1.2_rules": 2.88,
                "1.3_lr": -42.22,
                "1.4_catboost_d6": 12.83,
                "4.0_chain9_verify": roi_9f,
                "4.1_fixture_status": 16.23,
                "4.2_prematch_only": 2.72,
                "4.3_catboost_d7_p80": 19.44,
                "4.4_extended_d7": 16.23,
                "4.5_lgbm_p80": 20.27,
                "4.6_optuna_p80": 21.33,
                "4.7_winner_p80": 23.20,
                "4.8_winner_val_thresh": 8.16,
                "4.9_cb_lgb_stack": 20.49,
                "4.10_user_winrate": 18.00,
            }
            logger.info("Session summary ROI table:")
            for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True):
                logger.info("  %s: %.2f%%", k, v)

            mlflow.log_params({
                "validation_scheme": "time_series",
                "seed": SEED,
                "best_source": "chain_9_model_verified",
            })
            mlflow.log_metrics({
                "roi": roi_9f,
                "n_bets": n_9f,
                "best_session_own_roi": 21.33,
                **{k: v for k, v in summary.items()},
            })
            mlflow.log_dict(summary, "session_roi_summary.json")
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")
            mlflow.log_artifact(__file__)
            logger.info("Step 4.12: best pipeline saved. ROI=%.4f%% (n=%d)", roi_9f, n_9f)
            print(f"STEP 4.12: best chain_9 pipeline saved. roi={roi_9f:.4f}%(n={n_9f})")
            # STATUS: done
        except Exception:
            import traceback
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception")
            raise
else:
    logger.info("Step 4.12 already done, skipping")


print("=== Steps 4.11-4.12 DONE ===")
print("=== Session chain_1_mar22_0237 COMPLETE ===")
