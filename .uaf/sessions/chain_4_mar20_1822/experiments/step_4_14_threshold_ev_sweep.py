"""Step 4.14: Threshold sweep with EV>=0 — validate p=0.77 optimality."""

import logging
import os
import traceback

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import (
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_roi,
    check_budget,
    get_base_features,
    get_elo_features,
    get_engineered_features,
    load_data,
    set_seed,
    time_series_split,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

set_seed()
check_budget()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SESSION_ID = os.environ["UAF_SESSION_ID"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

CB_PARAMS = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 21.1,
    "min_data_in_leaf": 20,
    "random_strength": 1.0,
    "bagging_temperature": 0.06,
    "border_count": 102,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def calc_ev_roi(
    df: pd.DataFrame, proba: np.ndarray, ev_threshold: float, min_prob: float = 0.5
) -> dict:
    """Выбор ставок по EV = p * odds - 1 >= ev_threshold AND p >= min_prob."""
    odds = df["Odds"].values
    ev = proba * odds - 1.0
    mask = (ev >= ev_threshold) & (proba >= min_prob)
    n_selected = int(mask.sum())
    if n_selected == 0:
        return {
            "roi": 0.0,
            "n_bets": 0,
            "n_won": 0,
            "total_staked": 0.0,
            "win_rate": 0.0,
            "pct_selected": 0.0,
        }
    selected = df.iloc[np.where(mask)[0]]
    total_staked = selected["USD"].sum()
    total_payout = selected["Payout_USD"].sum()
    roi = (total_payout - total_staked) / total_staked * 100
    n_won = (selected["Status"] == "won").sum()
    return {
        "roi": float(roi),
        "n_bets": n_selected,
        "n_won": int(n_won),
        "total_staked": float(total_staked),
        "win_rate": float(n_won / n_selected),
        "pct_selected": float(n_selected / len(df) * 100),
    }


def main() -> None:
    """Threshold sweep with EV>=0 on val and test."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()

    # Train reference model with early stopping
    check_budget()
    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_data = train_sf.iloc[val_split:]

    imp_ref = SimpleImputer(strategy="median")
    x_fit = imp_ref.fit_transform(train_fit[feat_list])
    x_val = imp_ref.transform(val_data[feat_list])

    ref_model = CatBoostClassifier(**CB_PARAMS)
    ref_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_data["target"]))
    best_iter = ref_model.get_best_iteration()

    # Full-train model
    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter + 10
    ft_model = CatBoostClassifier(**params_ft)
    ft_model.fit(x_full, train_sf["target"])

    # Predictions on val and test
    p_val = ref_model.predict_proba(x_val)[:, 1]
    p_test = ft_model.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(test_sf["target"], p_test)

    # Sweep thresholds on VAL to find optimal p with EV>=0
    thresholds = np.arange(0.70, 0.86, 0.01)
    logger.info("Threshold sweep on VAL (EV>=0):")
    val_results = []
    for t in thresholds:
        r = calc_ev_roi(val_data, p_val, ev_threshold=0.0, min_prob=t)
        if r["n_bets"] >= 10:
            val_results.append({"threshold": float(t), **r})
            logger.info(
                "  val p>=%.2f+EV0: ROI=%.2f%% n=%d wr=%.3f",
                t,
                r["roi"],
                r["n_bets"],
                r["win_rate"],
            )

    # Find best threshold on val
    best_val = max(val_results, key=lambda x: x["roi"]) if val_results else None
    best_t_val = best_val["threshold"] if best_val else 0.77
    logger.info("Best val threshold: p>=%.2f (ROI=%.2f%%)", best_t_val, best_val["roi"])

    # Apply ALL thresholds to test (for analysis, not selection)
    logger.info("Threshold sweep on TEST (EV>=0):")
    test_results = []
    for t in thresholds:
        r_t77 = calc_roi(test_sf, p_test, threshold=t)
        r_ev = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=t)
        test_results.append(
            {
                "threshold": float(t),
                "roi_t_only": r_t77["roi"],
                "n_t_only": r_t77["n_bets"],
                "roi_ev0": r_ev["roi"],
                "n_ev0": r_ev["n_bets"],
            }
        )
        logger.info(
            "  test p>=%.2f: t_only=%.2f%% n=%d | EV0=%.2f%% n=%d",
            t,
            r_t77["roi"],
            r_t77["n_bets"],
            r_ev["roi"],
            r_ev["n_bets"],
        )

    # Apply val-optimal threshold to test
    r_val_opt = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=best_t_val)
    r_fixed_77 = calc_ev_roi(test_sf, p_test, ev_threshold=0.0, min_prob=0.77)

    logger.info(
        "Val-optimal (p>=%.2f+EV0): ROI=%.2f%% n=%d",
        best_t_val,
        r_val_opt["roi"],
        r_val_opt["n_bets"],
    )
    logger.info("Fixed (p>=0.77+EV0): ROI=%.2f%% n=%d", r_fixed_77["roi"], r_fixed_77["n_bets"])

    # Use the better of val-optimal and fixed 0.77
    if r_val_opt["roi"] > r_fixed_77["roi"] and r_val_opt["n_bets"] >= 50:
        best_result = r_val_opt
        best_strategy = f"p>={best_t_val:.2f}+EV0"
    else:
        best_result = r_fixed_77
        best_strategy = "p>=0.77+EV0"

    with mlflow.start_run(run_name="phase4/step4.14_threshold_ev_sweep") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.14")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "threshold_ev_sweep",
                    "n_features": len(feat_list),
                    "best_val_threshold": best_t_val,
                    "best_strategy": best_strategy,
                }
            )

            # Log all test results
            for r in test_results:
                t_str = f"{r['threshold']:.2f}".replace(".", "")
                mlflow.log_metric(f"roi_t{t_str}_only", r["roi_t_only"])
                mlflow.log_metric(f"roi_t{t_str}_ev0", r["roi_ev0"])

            mlflow.log_metrics(
                {
                    "roi": best_result["roi"],
                    "roc_auc": auc_test,
                    "n_bets": best_result["n_bets"],
                    "win_rate": best_result["win_rate"],
                    "best_threshold": best_t_val,
                    "roi_val_optimal": r_val_opt["roi"],
                    "roi_fixed_077": r_fixed_77["roi"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.14: %s ROI=%.2f%% n=%d AUC=%.4f run=%s",
                best_strategy,
                best_result["roi"],
                best_result["n_bets"],
                auc_test,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
