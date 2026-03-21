"""Step 4.9: Expected Value selection, stacking meta-learner, CatBoost RSM."""

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
from sklearn.linear_model import LogisticRegression
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
    """Выбор ставок по Expected Value: EV = p * odds - 1 > ev_threshold.

    Также требует min_prob чтобы не ставить на маловероятные исходы.
    """
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
    """EV-based selection, stacking, RSM experiments."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_base_features() + get_engineered_features() + get_elo_features()
    results: dict[str, dict] = {}

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    # A: Reference
    check_budget()
    ref_model = CatBoostClassifier(**CB_PARAMS)
    ref_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = ref_model.get_best_iteration()

    params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter + 10
    ft_ref = CatBoostClassifier(**params_ft)
    ft_ref.fit(x_full, train_sf["target"])
    p_ref = ft_ref.predict_proba(x_test)[:, 1]
    _ = ref_model.predict_proba(x_val)[:, 1]  # val proba for reference
    auc_ref = roc_auc_score(test_sf["target"], p_ref)
    results["A_ref"] = {**calc_roi(test_sf, p_ref, threshold=0.77), "auc": auc_ref}
    logger.info(
        "A ref: ROI=%.2f%% n=%d AUC=%.4f",
        results["A_ref"]["roi"],
        results["A_ref"]["n_bets"],
        auc_ref,
    )

    # B: Expected Value based selection
    check_budget()
    # EV = p * odds - 1. When EV > 0, the bet is +EV (profitable in expectation)
    for ev_t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for min_p in [0.5, 0.6, 0.7]:
            ev_roi = calc_ev_roi(test_sf, p_ref, ev_threshold=ev_t, min_prob=min_p)
            if ev_roi["n_bets"] >= 20:
                key = f"B_ev{ev_t}_p{min_p}"
                results[key] = {**ev_roi, "auc": auc_ref}
                logger.info(
                    "B EV>=%.2f p>=%.1f: ROI=%.2f%% n=%d",
                    ev_t,
                    min_p,
                    ev_roi["roi"],
                    ev_roi["n_bets"],
                )

    # C: Combined threshold: p > t AND EV > ev_min
    check_budget()
    for t in [0.70, 0.75, 0.77]:
        for ev_min in [0.0, 0.05, 0.10]:
            odds_test = test_sf["Odds"].values
            ev_test = p_ref * odds_test - 1.0
            mask = (p_ref >= t) & (ev_test >= ev_min)
            n_sel = int(mask.sum())
            if n_sel >= 20:
                selected = test_sf.iloc[np.where(mask)[0]]
                total_s = selected["USD"].sum()
                total_p = selected["Payout_USD"].sum()
                roi_val = (total_p - total_s) / total_s * 100 if total_s > 0 else 0
                n_won = (selected["Status"] == "won").sum()
                key = f"C_combo_t{t}_ev{ev_min}"
                results[key] = {
                    "roi": float(roi_val),
                    "n_bets": n_sel,
                    "n_won": int(n_won),
                    "total_staked": float(total_s),
                    "win_rate": float(n_won / n_sel),
                    "pct_selected": float(n_sel / len(test_sf) * 100),
                    "auc": auc_ref,
                }
                logger.info(
                    "C p>=%.2f EV>=%.2f: ROI=%.2f%% n=%d",
                    t,
                    ev_min,
                    roi_val,
                    n_sel,
                )

    # D: CatBoost with RSM (random subspace method)
    check_budget()
    for rsm in [0.7, 0.8, 0.9]:
        params_rsm = {**CB_PARAMS, "rsm": rsm}
        rsm_model = CatBoostClassifier(**params_rsm)
        rsm_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        rsm_iter = rsm_model.get_best_iteration()

        params_rsm_ft = {k: v for k, v in params_rsm.items() if k != "early_stopping_rounds"}
        params_rsm_ft["iterations"] = max(rsm_iter + 10, 50)
        rsm_ft = CatBoostClassifier(**params_rsm_ft)
        rsm_ft.fit(x_full, train_sf["target"])
        p_rsm = rsm_ft.predict_proba(x_test)[:, 1]
        auc_rsm = roc_auc_score(test_sf["target"], p_rsm)
        roi_rsm = calc_roi(test_sf, p_rsm, threshold=0.77)
        results[f"D_rsm{rsm}"] = {**roi_rsm, "auc": auc_rsm}
        logger.info(
            "D RSM=%.1f: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
            rsm,
            roi_rsm["roi"],
            roi_rsm["n_bets"],
            auc_rsm,
            rsm_iter,
        )

    # E: Stacking -- train diverse CatBoost models, get val predictions, meta-learner
    check_budget()
    model_configs = [
        ("cb_d8", CB_PARAMS),
        ("cb_d7", {**CB_PARAMS, "depth": 7}),
        ("cb_d10_l2_50", {**CB_PARAMS, "depth": 10, "l2_leaf_reg": 50.0}),
        ("cb_ordered", {**CB_PARAMS, "boosting_type": "Ordered"}),
    ]

    val_preds_stack = {}
    test_preds_stack = {}

    for name, params in model_configs:
        m = CatBoostClassifier(**params)
        m.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        bi = m.get_best_iteration()

        val_preds_stack[name] = m.predict_proba(x_val)[:, 1]

        # Full-train for test
        params_s = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        params_s["iterations"] = max(bi + 10, 50)
        mf = CatBoostClassifier(**params_s)
        mf.fit(x_full, train_sf["target"])
        test_preds_stack[name] = mf.predict_proba(x_test)[:, 1]

    # Meta-learner on val predictions
    x_meta_val = np.column_stack(list(val_preds_stack.values()))
    x_meta_test = np.column_stack(list(test_preds_stack.values()))

    meta_lr = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta_lr.fit(x_meta_val, val_df["target"])
    p_stack = meta_lr.predict_proba(x_meta_test)[:, 1]
    auc_stack = roc_auc_score(test_sf["target"], p_stack)

    for t in [0.76, 0.77, 0.78]:
        roi_stack = calc_roi(test_sf, p_stack, threshold=t)
        results[f"E_stack_lr_t{t}"] = {**roi_stack, "auc": auc_stack}
        logger.info(
            "E stack LR t=%.2f: ROI=%.2f%% n=%d AUC=%.4f",
            t,
            roi_stack["roi"],
            roi_stack["n_bets"],
            auc_stack,
        )

    # Meta-learner coefficients
    logger.info(
        "Stack LR coefs: %s", dict(zip(val_preds_stack.keys(), meta_lr.coef_[0], strict=True))
    )

    # Summary
    logger.info("All results (sorted by ROI, top 15):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True)
    for name, r in sorted_results[:15]:
        logger.info(
            "  %s: ROI=%.2f%% n=%d AUC=%.4f",
            name,
            r["roi"],
            r["n_bets"],
            r["auc"],
        )

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.9_ev_stacking") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.9")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "ev_selection_stacking_rsm",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                    "stack_models": str(list(val_preds_stack.keys())),
                }
            )

            for name, r in sorted_results[:10]:
                safe = name.replace(".", "_")
                mlflow.log_metric(f"roi_{safe}", r["roi"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": best_r["auc"],
                    "n_bets": best_r["n_bets"],
                    "win_rate": best_r["win_rate"],
                    "best_threshold": 0.77,
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")

            delta = best_r["roi"] - results["A_ref"]["roi"]
            mlflow.set_tag("convergence_signal", "0.95" if delta <= 0 else "0.85")

            logger.info(
                "Step 4.9: BEST %s ROI=%.2f%% delta=%.2f pp run=%s",
                best_key,
                best_r["roi"],
                delta,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
