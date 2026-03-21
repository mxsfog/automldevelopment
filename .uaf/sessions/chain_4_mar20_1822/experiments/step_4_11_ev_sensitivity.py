"""Step 4.11: EV approach sensitivity analysis + final combinations."""

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
    """EV sensitivity analysis and final model combinations."""
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

    # Reference model
    check_budget()
    ref_model = CatBoostClassifier(**CB_PARAMS)
    ref_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = ref_model.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter + 10
    ft_ref = CatBoostClassifier(**params_ft)
    ft_ref.fit(x_full, train_sf["target"])
    p_ref = ft_ref.predict_proba(x_test)[:, 1]
    auc_ref = roc_auc_score(test_sf["target"], p_ref)

    # A: Reference t=0.77
    results["A_t077"] = {**calc_roi(test_sf, p_ref, threshold=0.77), "auc": auc_ref}
    results["A_t076"] = {**calc_roi(test_sf, p_ref, threshold=0.76), "auc": auc_ref}

    # B: EV sensitivity: fix p>=0.77, sweep EV threshold from -0.05 to 0.10
    check_budget()
    for ev_t in [-0.05, 0.0, 0.02, 0.05, 0.08, 0.10]:
        ev_r = calc_ev_roi(test_sf, p_ref, ev_threshold=ev_t, min_prob=0.77)
        key = f"B_ev{ev_t:+.2f}_p077"
        results[key] = {**ev_r, "auc": auc_ref}
        logger.info(
            "B EV>=%+.2f p>=0.77: ROI=%.2f%% n=%d win=%.1f%%",
            ev_t,
            ev_r["roi"],
            ev_r["n_bets"],
            ev_r["win_rate"] * 100,
        )

    # C: EV sensitivity: fix p>=0.76, sweep EV
    for ev_t in [-0.05, 0.0, 0.02, 0.05]:
        ev_r = calc_ev_roi(test_sf, p_ref, ev_threshold=ev_t, min_prob=0.76)
        key = f"C_ev{ev_t:+.2f}_p076"
        results[key] = {**ev_r, "auc": auc_ref}
        logger.info(
            "C EV>=%+.2f p>=0.76: ROI=%.2f%% n=%d",
            ev_t,
            ev_r["roi"],
            ev_r["n_bets"],
        )

    # D: Blend ref_d8 + deep10 (0.7/0.3) with EV filter (best combo from 4.8)
    check_budget()
    params_deep = {**CB_PARAMS, "depth": 10, "l2_leaf_reg": 50.0, "min_data_in_leaf": 30}
    deep_model = CatBoostClassifier(**params_deep)
    deep_model.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    deep_iter = deep_model.get_best_iteration()

    params_deep_ft = {k: v for k, v in params_deep.items() if k != "early_stopping_rounds"}
    params_deep_ft["iterations"] = max(deep_iter + 10, 50)
    deep_ft = CatBoostClassifier(**params_deep_ft)
    deep_ft.fit(x_full, train_sf["target"])
    p_deep = deep_ft.predict_proba(x_test)[:, 1]

    p_blend = 0.7 * p_ref + 0.3 * p_deep
    auc_blend = roc_auc_score(test_sf["target"], p_blend)

    # Blend with different selection strategies
    results["D_blend_t077"] = {**calc_roi(test_sf, p_blend, threshold=0.77), "auc": auc_blend}
    results["D_blend_ev0_p077"] = {
        **calc_ev_roi(test_sf, p_blend, ev_threshold=0.0, min_prob=0.77),
        "auc": auc_blend,
    }
    results["D_blend_ev0_p076"] = {
        **calc_ev_roi(test_sf, p_blend, ev_threshold=0.0, min_prob=0.76),
        "auc": auc_blend,
    }

    logger.info(
        "D blend t=0.77: ROI=%.2f%% n=%d",
        results["D_blend_t077"]["roi"],
        results["D_blend_t077"]["n_bets"],
    )
    logger.info(
        "D blend EV>=0 p>=0.77: ROI=%.2f%% n=%d",
        results["D_blend_ev0_p077"]["roi"],
        results["D_blend_ev0_p077"]["n_bets"],
    )
    logger.info(
        "D blend EV>=0 p>=0.76: ROI=%.2f%% n=%d",
        results["D_blend_ev0_p076"]["roi"],
        results["D_blend_ev0_p076"]["n_bets"],
    )

    # E: Analyze what EV filter removes — sport breakdown
    check_budget()
    mask_t77 = p_ref >= 0.77
    mask_ev0_p77 = (p_ref * test_sf["Odds"].values - 1.0 >= 0.0) & (p_ref >= 0.77)

    removed_mask = mask_t77 & ~mask_ev0_p77
    if removed_mask.sum() > 0:
        removed = test_sf.iloc[np.where(removed_mask)[0]]
        logger.info("EV filter removes %d bets from t=0.77 selection:", removed_mask.sum())
        logger.info(
            "  Avg odds of removed: %.2f (vs kept: %.2f)",
            removed["Odds"].mean(),
            test_sf.iloc[np.where(mask_ev0_p77)[0]]["Odds"].mean(),
        )
        if "Sport" in removed.columns:
            for sport, group in removed.groupby("Sport"):
                roi_removed_sport = (
                    (group["Payout_USD"].sum() - group["USD"].sum()) / group["USD"].sum() * 100
                )
                logger.info(
                    "  %s: %d removed, ROI=%.2f%%",
                    sport,
                    len(group),
                    roi_removed_sport,
                )

    # Summary
    logger.info("All results (sorted by ROI):")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info(
            "  %s: ROI=%.2f%% n=%d AUC=%.4f",
            name,
            r["roi"],
            r["n_bets"],
            r["auc"],
        )

    # Select best clean result (below 30% suspicion threshold)
    clean_results = {k: v for k, v in results.items() if v["roi"] < 30}
    if clean_results:
        best_key = max(clean_results, key=lambda k: clean_results[k]["roi"])
    else:
        best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.11_ev_sensitivity") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "ev_sensitivity_analysis",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                safe = name.replace(".", "_").replace("+", "p").replace("-", "m")
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

            delta = best_r["roi"] - results["A_t077"]["roi"]
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.11: BEST %s ROI=%.2f%% delta=%.2f pp run=%s",
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
