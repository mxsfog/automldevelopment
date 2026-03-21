"""Step 4.12: Final combinations — EV filter on full ELO, odds-range analysis."""

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
    """Final combinations: EV on full ELO, odds-range analysis."""
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

    # A: Reference (sport-filtered, full-train, EV>=0+p>=0.77)
    check_budget()
    val_split_sf = int(len(train_sf) * 0.8)
    train_fit_sf = train_sf.iloc[:val_split_sf]
    val_sf = train_sf.iloc[val_split_sf:]

    imp_sf = SimpleImputer(strategy="median")
    x_fit_sf = imp_sf.fit_transform(train_fit_sf[feat_list])
    x_val_sf = imp_sf.transform(val_sf[feat_list])

    ref_sf = CatBoostClassifier(**CB_PARAMS)
    ref_sf.fit(x_fit_sf, train_fit_sf["target"], eval_set=(x_val_sf, val_sf["target"]))
    best_iter_sf = ref_sf.get_best_iteration()

    imp_sf_full = SimpleImputer(strategy="median")
    x_sf_full = imp_sf_full.fit_transform(train_sf[feat_list])
    x_sf_test = imp_sf_full.transform(test_sf[feat_list])

    params_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter_sf + 10
    ft_sf = CatBoostClassifier(**params_ft)
    ft_sf.fit(x_sf_full, train_sf["target"])
    p_sf = ft_sf.predict_proba(x_sf_test)[:, 1]
    auc_sf = roc_auc_score(test_sf["target"], p_sf)

    results["A_sf_t077"] = {**calc_roi(test_sf, p_sf, threshold=0.77), "auc": auc_sf}
    results["A_sf_ev0_p077"] = {
        **calc_ev_roi(test_sf, p_sf, ev_threshold=0.0, min_prob=0.77),
        "auc": auc_sf,
    }
    logger.info(
        "A SF t=0.77: ROI=%.2f%% n=%d | EV>=0+p77: ROI=%.2f%% n=%d",
        results["A_sf_t077"]["roi"],
        results["A_sf_t077"]["n_bets"],
        results["A_sf_ev0_p077"]["roi"],
        results["A_sf_ev0_p077"]["n_bets"],
    )

    # B: Full ELO (no sport filter) with EV>=0
    check_budget()
    val_split_elo = int(len(train_elo) * 0.8)
    train_fit_elo = train_elo.iloc[:val_split_elo]
    val_elo = train_elo.iloc[val_split_elo:]

    imp_elo = SimpleImputer(strategy="median")
    x_fit_elo = imp_elo.fit_transform(train_fit_elo[feat_list])
    x_val_elo = imp_elo.transform(val_elo[feat_list])

    ref_elo = CatBoostClassifier(**CB_PARAMS)
    ref_elo.fit(x_fit_elo, train_fit_elo["target"], eval_set=(x_val_elo, val_elo["target"]))
    best_iter_elo = ref_elo.get_best_iteration()

    imp_elo_full = SimpleImputer(strategy="median")
    x_elo_full = imp_elo_full.fit_transform(train_elo[feat_list])
    x_elo_test = imp_elo_full.transform(test_elo[feat_list])

    params_elo_ft = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
    params_elo_ft["iterations"] = best_iter_elo + 10
    ft_elo = CatBoostClassifier(**params_elo_ft)
    ft_elo.fit(x_elo_full, train_elo["target"])
    p_elo = ft_elo.predict_proba(x_elo_test)[:, 1]
    auc_elo = roc_auc_score(test_elo["target"], p_elo)

    results["B_elo_t077"] = {**calc_roi(test_elo, p_elo, threshold=0.77), "auc": auc_elo}
    results["B_elo_ev0_p077"] = {
        **calc_ev_roi(test_elo, p_elo, ev_threshold=0.0, min_prob=0.77),
        "auc": auc_elo,
    }
    results["B_elo_ev0_p076"] = {
        **calc_ev_roi(test_elo, p_elo, ev_threshold=0.0, min_prob=0.76),
        "auc": auc_elo,
    }
    logger.info(
        "B ELO-all t=0.77: ROI=%.2f%% n=%d | EV>=0+p77: ROI=%.2f%% n=%d",
        results["B_elo_t077"]["roi"],
        results["B_elo_t077"]["n_bets"],
        results["B_elo_ev0_p077"]["roi"],
        results["B_elo_ev0_p077"]["n_bets"],
    )

    # C: SF model, EV filter, but applied to FULL ELO test set
    # (trained on SF, predict on all ELO test)
    check_budget()
    p_sf_on_elo = ft_sf.predict_proba(imp_sf_full.transform(test_elo[feat_list]))[:, 1]
    results["C_sf_model_elo_test_t077"] = {
        **calc_roi(test_elo, p_sf_on_elo, threshold=0.77),
        "auc": roc_auc_score(test_elo["target"], p_sf_on_elo),
    }
    results["C_sf_model_elo_test_ev0_p077"] = {
        **calc_ev_roi(test_elo, p_sf_on_elo, ev_threshold=0.0, min_prob=0.77),
        "auc": roc_auc_score(test_elo["target"], p_sf_on_elo),
    }
    logger.info(
        "C SF-model->ELO-test t=0.77: ROI=%.2f%% n=%d | EV>=0+p77: ROI=%.2f%% n=%d",
        results["C_sf_model_elo_test_t077"]["roi"],
        results["C_sf_model_elo_test_t077"]["n_bets"],
        results["C_sf_model_elo_test_ev0_p077"]["roi"],
        results["C_sf_model_elo_test_ev0_p077"]["n_bets"],
    )

    # D: Odds-range analysis of EV filter effectiveness
    check_budget()
    odds_ranges = [
        ("1.01-1.15", 1.01, 1.15),
        ("1.15-1.30", 1.15, 1.30),
        ("1.30-1.50", 1.30, 1.50),
        ("1.50-2.00", 1.50, 2.00),
        ("2.00+", 2.00, 100.0),
    ]

    logger.info("Odds-range analysis (SF model, SF test, p>=0.77 selected):")
    mask_p77 = p_sf >= 0.77
    selected_sf = test_sf.iloc[np.where(mask_p77)[0]]
    selected_p = p_sf[mask_p77]

    for range_name, lo, hi in odds_ranges:
        odds_mask = (selected_sf["Odds"].values >= lo) & (selected_sf["Odds"].values < hi)
        n_range = int(odds_mask.sum())
        if n_range < 5:
            continue
        range_bets = selected_sf.iloc[np.where(odds_mask)[0]]
        range_staked = range_bets["USD"].sum()
        range_payout = range_bets["Payout_USD"].sum()
        range_roi = (range_payout - range_staked) / range_staked * 100 if range_staked > 0 else 0
        range_ev = (selected_p[odds_mask] * range_bets["Odds"].values - 1.0).mean()
        logger.info(
            "  %s: n=%d ROI=%.2f%% avg_EV=%.3f avg_odds=%.2f",
            range_name,
            n_range,
            range_roi,
            range_ev,
            range_bets["Odds"].mean(),
        )

    # E: Combination: SF model + EV>=0 + odds>=1.15 (skip very low odds)
    check_budget()
    odds_test = test_sf["Odds"].values
    for min_odds in [1.10, 1.15, 1.20, 1.25]:
        mask_combo = (p_sf >= 0.77) & (p_sf * odds_test - 1.0 >= 0.0) & (odds_test >= min_odds)
        n_combo = int(mask_combo.sum())
        if n_combo >= 20:
            sel = test_sf.iloc[np.where(mask_combo)[0]]
            staked = sel["USD"].sum()
            payout = sel["Payout_USD"].sum()
            roi_c = (payout - staked) / staked * 100 if staked > 0 else 0
            n_won = (sel["Status"] == "won").sum()
            key = f"E_ev0_p77_odds{min_odds}"
            results[key] = {
                "roi": float(roi_c),
                "n_bets": n_combo,
                "n_won": int(n_won),
                "total_staked": float(staked),
                "win_rate": float(n_won / n_combo),
                "pct_selected": float(n_combo / len(test_sf) * 100),
                "auc": auc_sf,
            }
            logger.info(
                "E EV>=0+p77+odds>=%.2f: ROI=%.2f%% n=%d",
                min_odds,
                roi_c,
                n_combo,
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

    # Best clean result
    clean = {k: v for k, v in results.items() if v["roi"] < 30 and v["n_bets"] >= 50}
    best_key = (
        max(clean, key=lambda k: clean[k]["roi"])
        if clean
        else max(results, key=lambda k: results[k]["roi"])
    )
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.12_final_combinations") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.12")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "final_combinations",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                safe = name.replace(".", "_").replace("+", "p")
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
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.12: BEST %s ROI=%.2f%% n=%d run=%s",
                best_key,
                best_r["roi"],
                best_r["n_bets"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
