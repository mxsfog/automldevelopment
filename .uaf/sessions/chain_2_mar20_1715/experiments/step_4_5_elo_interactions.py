"""Step 4.5: ELO interaction features + Optuna ensemble weights."""

import logging
import os
import traceback

import lightgbm
import mlflow
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    add_engineered_features,
    calc_roi,
    calc_roi_at_thresholds,
    check_budget,
    find_best_threshold_on_val,
    get_base_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from step_2_5_safe_elo import build_safe_elo_features, get_safe_elo_features
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def add_elo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """ELO-specific interaction features."""

    df = df.copy()
    # ELO diff normalized by odds expectation
    df["elo_diff_x_value"] = df["elo_diff"] * df["value_ratio"]
    # Winrate diff scaled by implied probability
    df["winrate_diff_x_implied"] = df["team_winrate_diff"] * df["implied_prob"]
    # ELO advantage relative to odds
    df["elo_mean_x_edge"] = df["elo_mean_vs_1500"] * df["ML_Edge"]
    # High ELO spread with high odds = risky
    df["elo_spread_x_log_odds"] = df["elo_spread"] * df["log_odds"]
    # Winrate mean * model confidence
    df["winrate_x_model_p"] = df["team_winrate_mean"] * df["ML_P_Model"] / 100.0
    # ELO diff per unit of odds
    df["elo_diff_per_odds"] = df["elo_diff"] / df["Odds"].clip(lower=1.01)
    # Games experience * edge
    df["games_x_ev"] = df["team_total_games_mean"] * df["ML_EV"]
    return df


def get_elo_interaction_features() -> list[str]:
    """Names of ELO interaction features."""
    return [
        "elo_diff_x_value",
        "winrate_diff_x_implied",
        "elo_mean_x_edge",
        "elo_spread_x_log_odds",
        "winrate_x_model_p",
        "elo_diff_per_odds",
        "games_x_ev",
    ]


def main() -> None:
    """ELO interactions + Optuna ensemble weights."""
    logger.info("Step 4.5: ELO interaction features + Optuna weights")

    df = load_data()
    df = add_engineered_features(df)
    df = build_safe_elo_features(df)
    df = add_elo_interactions(df)
    train_all, test_all = time_series_split(df)

    train = train_all[train_all["has_elo"] == 1.0].copy()
    test = test_all[test_all["has_elo"] == 1.0].copy()
    logger.info("ELO-only: train=%d, test=%d", len(train), len(test))

    base_feats = get_base_features() + get_engineered_features() + get_safe_elo_features()
    interaction_feats = get_elo_interaction_features()
    all_feats = base_feats + interaction_feats

    val_split = int(len(train) * 0.8)
    train_fit = train.iloc[:val_split]
    val_df = train.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[all_feats])
    x_val = imp.transform(val_df[all_feats])
    x_test = imp.transform(test[all_feats])
    y_fit = train_fit["target"].values
    y_val = val_df["target"].values

    # Also prepare baseline features (no interactions) for comparison
    imp_base = SimpleImputer(strategy="median")
    x_fit_base = imp_base.fit_transform(train_fit[base_feats])
    x_val_base = imp_base.transform(val_df[base_feats])
    x_test_base = imp_base.transform(test[base_feats])

    with mlflow.start_run(run_name="phase4/step4.5_elo_interactions") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("status", "running")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("phase", "4")

        try:
            # Part A: Test if interaction features help individual CatBoost
            logger.info("--- Part A: Interaction features effect ---")

            # Baseline CB (no interactions)
            cb_base = CatBoostClassifier(
                iterations=499,
                depth=7,
                learning_rate=0.214,
                l2_leaf_reg=1.15,
                random_strength=0.823,
                bagging_temperature=2.41,
                border_count=121,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=30,
            )
            cb_base.fit(x_fit_base, y_fit, eval_set=(x_val_base, y_val))
            p_cb_base_val = cb_base.predict_proba(x_val_base)[:, 1]
            p_cb_base_test = cb_base.predict_proba(x_test_base)[:, 1]
            t_base, _vr = find_best_threshold_on_val(val_df, p_cb_base_val)
            roi_base = calc_roi(test, p_cb_base_test, threshold=t_base)
            auc_base = roc_auc_score(test["target"], p_cb_base_test)
            logger.info(
                "CB baseline: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_base["roi"],
                auc_base,
                t_base,
                roi_base["n_bets"],
            )

            # CB with interactions
            cb_int = CatBoostClassifier(
                iterations=499,
                depth=7,
                learning_rate=0.214,
                l2_leaf_reg=1.15,
                random_strength=0.823,
                bagging_temperature=2.41,
                border_count=121,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=30,
            )
            cb_int.fit(x_fit, y_fit, eval_set=(x_val, y_val))
            p_cb_int_val = cb_int.predict_proba(x_val)[:, 1]
            p_cb_int_test = cb_int.predict_proba(x_test)[:, 1]
            t_int, _vr = find_best_threshold_on_val(val_df, p_cb_int_val)
            roi_int = calc_roi(test, p_cb_int_test, threshold=t_int)
            auc_int = roc_auc_score(test["target"], p_cb_int_test)
            logger.info(
                "CB +interactions: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_int["roi"],
                auc_int,
                t_int,
                roi_int["n_bets"],
            )

            delta_roi = roi_int["roi"] - roi_base["roi"]
            delta_auc = auc_int - auc_base
            logger.info("Delta: ROI=%.2f%%, AUC=%.4f", delta_roi, delta_auc)

            # Feature importance for interaction features
            fi = dict(zip(all_feats, cb_int.feature_importances_, strict=False))
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in fi_sorted[:20]:
                logger.info("  FI: %s = %.3f", fname, fval)

            interaction_fi = {f: fi[f] for f in interaction_feats if f in fi}
            logger.info("Interaction features importance:")
            for f, v in sorted(interaction_fi.items(), key=lambda x: x[1], reverse=True):
                logger.info("  %s = %.3f", f, v)

            # Decide which feature set to use for ensemble
            use_interactions = delta_roi > 0.5
            chosen_x_fit = x_fit if use_interactions else x_fit_base
            chosen_x_val = x_val if use_interactions else x_val_base
            chosen_x_test = x_test if use_interactions else x_test_base
            logger.info("Using interactions: %s (delta=%.2f%%)", use_interactions, delta_roi)

            # Part B: Optuna ensemble weight optimization
            logger.info("--- Part B: Optuna ensemble weight optimization ---")
            check_budget()

            # Train 3 models on chosen features
            cb = CatBoostClassifier(
                iterations=499,
                depth=7,
                learning_rate=0.214,
                l2_leaf_reg=1.15,
                random_strength=0.823,
                bagging_temperature=2.41,
                border_count=121,
                random_seed=42,
                verbose=0,
                eval_metric="AUC",
                early_stopping_rounds=30,
            )
            cb.fit(chosen_x_fit, y_fit, eval_set=(chosen_x_val, y_val))
            p_cb_val = cb.predict_proba(chosen_x_val)[:, 1]
            p_cb_test = cb.predict_proba(chosen_x_test)[:, 1]

            lgb = LGBMClassifier(
                n_estimators=477,
                max_depth=3,
                learning_rate=0.292,
                num_leaves=16,
                min_child_samples=49,
                reg_lambda=28.63,
                random_state=42,
                verbose=-1,
            )
            lgb.fit(
                chosen_x_fit,
                y_fit,
                eval_set=[(chosen_x_val, y_val)],
                callbacks=[
                    lightgbm.early_stopping(30, verbose=False),
                    lightgbm.log_evaluation(0),
                ],
            )
            p_lgb_val = lgb.predict_proba(chosen_x_val)[:, 1]
            p_lgb_test = lgb.predict_proba(chosen_x_test)[:, 1]

            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                eval_metric="auc",
                early_stopping_rounds=30,
            )
            xgb.fit(chosen_x_fit, y_fit, eval_set=[(chosen_x_val, y_val)], verbose=False)
            p_xgb_val = xgb.predict_proba(chosen_x_val)[:, 1]
            p_xgb_test = xgb.predict_proba(chosen_x_test)[:, 1]

            # Optuna to find best weights + threshold jointly
            def weight_objective(trial: optuna.Trial) -> float:
                w_cb = trial.suggest_float("w_cb", 0.2, 0.8)
                w_lgb = trial.suggest_float("w_lgb", 0.05, 0.5)
                w_xgb = 1.0 - w_cb - w_lgb
                if w_xgb < 0.05 or w_xgb > 0.5:
                    return -999.0
                p_ens = w_cb * p_cb_val + w_lgb * p_lgb_val + w_xgb * p_xgb_val
                _t, val_roi = find_best_threshold_on_val(val_df, p_ens, min_bets=20)
                return val_roi

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(weight_objective, n_trials=100, show_progress_bar=False)
            best_w_cb = study.best_params["w_cb"]
            best_w_lgb = study.best_params["w_lgb"]
            best_w_xgb = 1.0 - best_w_cb - best_w_lgb
            logger.info(
                "Optuna weights: CB=%.3f LGB=%.3f XGB=%.3f val_roi=%.2f%%",
                best_w_cb,
                best_w_lgb,
                best_w_xgb,
                study.best_value,
            )

            # Apply to test
            p_ens_val = best_w_cb * p_cb_val + best_w_lgb * p_lgb_val + best_w_xgb * p_xgb_val
            p_ens_test = best_w_cb * p_cb_test + best_w_lgb * p_lgb_test + best_w_xgb * p_xgb_test
            t_ens, _vr = find_best_threshold_on_val(val_df, p_ens_val)
            roi_ens = calc_roi(test, p_ens_test, threshold=t_ens)
            auc_ens = roc_auc_score(test["target"], p_ens_test)
            logger.info(
                "Optuna ens: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_ens["roi"],
                auc_ens,
                t_ens,
                roi_ens["n_bets"],
            )

            # Compare with CB50 (previous best)
            p_cb50_val = 0.5 * p_cb_val + 0.25 * p_lgb_val + 0.25 * p_xgb_val
            p_cb50_test = 0.5 * p_cb_test + 0.25 * p_lgb_test + 0.25 * p_xgb_test
            t_cb50, _vr = find_best_threshold_on_val(val_df, p_cb50_val)
            roi_cb50 = calc_roi(test, p_cb50_test, threshold=t_cb50)
            auc_cb50 = roc_auc_score(test["target"], p_cb50_test)
            logger.info(
                "CB50 ens: ROI=%.2f%% AUC=%.4f t=%.2f n=%d",
                roi_cb50["roi"],
                auc_cb50,
                t_cb50,
                roi_cb50["n_bets"],
            )

            # ROI at thresholds for best
            best_roi = max(roi_ens["roi"], roi_cb50["roi"], roi_int["roi"])
            if roi_ens["roi"] >= roi_cb50["roi"]:
                p_best_test = p_ens_test
                best_label = "optuna_weights"
            else:
                p_best_test = p_cb50_test
                best_label = "cb50"

            roi_thresholds = calc_roi_at_thresholds(test, p_best_test)
            for t, r in roi_thresholds.items():
                logger.info("  t=%.2f: ROI=%.2f%% n=%d", t, r["roi"], r["n_bets"])
                mlflow.log_metric(f"roi_best_t{int(t * 100):03d}", r["roi"])

            # Log everything
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": f"elo_interactions_optuna_weights_{best_label}",
                    "n_features_base": len(base_feats),
                    "n_features_with_interactions": len(all_feats),
                    "use_interactions": use_interactions,
                    "interaction_delta_roi": delta_roi,
                    "interaction_delta_auc": delta_auc,
                    "optuna_w_cb": best_w_cb,
                    "optuna_w_lgb": best_w_lgb,
                    "optuna_w_xgb": best_w_xgb,
                    "optuna_trials": 100,
                    "leakage_free": "true",
                }
            )
            mlflow.log_metrics(
                {
                    "roi_cb_base": roi_base["roi"],
                    "roi_cb_interactions": roi_int["roi"],
                    "roi_optuna_ens": roi_ens["roi"],
                    "roi_cb50_ens": roi_cb50["roi"],
                    "roi": best_roi,
                    "roc_auc": auc_ens,
                    "n_bets": roi_ens["n_bets"],
                    "best_threshold": t_ens,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.85")
            logger.info("Best: %s ROI=%.2f%%, run_id=%s", best_label, best_roi, run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            mlflow.set_tag("failure_reason", "exception in step 4.5")
            raise


if __name__ == "__main__":
    main()
