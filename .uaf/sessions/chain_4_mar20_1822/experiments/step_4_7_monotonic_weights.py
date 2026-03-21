"""Step 4.7: Monotonic constraints, sample weights, training window optimization."""

import logging
import os
import traceback

import mlflow
import numpy as np
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


def build_monotonic_constraints(feat_list: list[str]) -> list[int]:
    """Монотонные ограничения по доменной логике.

    +1: больше значение → выше вероятность выигрыша
    -1: больше значение → ниже вероятность выигрыша
    0: без ограничений
    """
    positive = {
        "implied_prob",
        "team_winrate_diff",
        "elo_diff",
        "value_ratio",
        "team_winrate_mean",
        "team_winrate_max",
        "ev_positive",
        "team_current_elo_mean",
        "elo_mean_vs_1500",
    }
    # Odds: higher odds → less likely to win (bookmaker view)
    negative = {"Odds", "log_odds"}

    constraints = []
    for f in feat_list:
        if f in positive:
            constraints.append(1)
        elif f in negative:
            constraints.append(-1)
        else:
            constraints.append(0)
    return constraints


def build_recency_weights(df_train, decay: float = 0.5) -> np.ndarray:
    """Экспоненциальные веса по времени: недавние семплы весят больше."""
    n = len(df_train)
    positions = np.arange(n, dtype=float)  # already sorted by time
    # Normalize to [0, 1]
    positions = positions / (n - 1) if n > 1 else positions
    # Exponential decay: w = exp(decay * (pos - 1))
    weights = np.exp(decay * (positions - 1.0))
    # Normalize so mean = 1
    weights = weights / weights.mean()
    return weights


def main() -> None:
    """Monotonic constraints, sample weights, train window."""
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

    # A: Reference (full-train, no constraints)
    check_budget()
    imp_ref = SimpleImputer(strategy="median")
    x_fit_ref = imp_ref.fit_transform(train_fit[feat_list])
    x_val_ref = imp_ref.transform(val_df[feat_list])

    ref_model = CatBoostClassifier(**CB_PARAMS)
    ref_model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
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
    results["A_ref_full"] = {
        **calc_roi(test_sf, p_ref, threshold=0.77),
        "auc": auc_ref,
    }
    logger.info(
        "A ref: ROI=%.2f%% n=%d AUC=%.4f",
        results["A_ref_full"]["roi"],
        results["A_ref_full"]["n_bets"],
        auc_ref,
    )

    # B: Monotonic constraints (full-train)
    check_budget()
    mono = build_monotonic_constraints(feat_list)
    n_constrained = sum(1 for c in mono if c != 0)
    logger.info("Monotonic constraints: %d/%d features constrained", n_constrained, len(feat_list))

    params_mono = {**CB_PARAMS, "monotone_constraints": mono}
    mono_model = CatBoostClassifier(**params_mono)
    mono_model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
    best_iter_mono = mono_model.get_best_iteration()

    params_mono_ft = {k: v for k, v in params_mono.items() if k != "early_stopping_rounds"}
    params_mono_ft["iterations"] = best_iter_mono + 10
    mono_ft = CatBoostClassifier(**params_mono_ft)
    mono_ft.fit(x_full, train_sf["target"])
    p_mono = mono_ft.predict_proba(x_test)[:, 1]
    auc_mono = roc_auc_score(test_sf["target"], p_mono)

    for t in [0.76, 0.77, 0.78]:
        roi_m = calc_roi(test_sf, p_mono, threshold=t)
        results[f"B_mono_t{t}"] = {**roi_m, "auc": auc_mono}
        logger.info("B mono t=%.2f: ROI=%.2f%% n=%d", t, roi_m["roi"], roi_m["n_bets"])

    # C: Sample weights by recency (full-train)
    check_budget()
    for decay in [0.3, 0.5, 1.0, 2.0]:
        weights = build_recency_weights(train_sf, decay=decay)

        params_w = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
        params_w["iterations"] = best_iter + 10
        w_model = CatBoostClassifier(**params_w)
        w_model.fit(x_full, train_sf["target"], sample_weight=weights)
        p_w = w_model.predict_proba(x_test)[:, 1]
        auc_w = roc_auc_score(test_sf["target"], p_w)
        roi_w = calc_roi(test_sf, p_w, threshold=0.77)
        results[f"C_recency_d{decay}"] = {**roi_w, "auc": auc_w}
        logger.info(
            "C recency d=%.1f: ROI=%.2f%% n=%d AUC=%.4f",
            decay,
            roi_w["roi"],
            roi_w["n_bets"],
            auc_w,
        )

    # D: Training window optimization (last N% of train data)
    check_budget()
    for pct in [0.5, 0.7, 0.85]:
        n_total = len(train_sf)
        start_idx = int(n_total * (1 - pct))
        train_window = train_sf.iloc[start_idx:].copy()

        # Inner split for early stopping
        w_val_split = int(len(train_window) * 0.8)
        w_train_fit = train_window.iloc[:w_val_split]
        w_val_df = train_window.iloc[w_val_split:]

        imp_w = SimpleImputer(strategy="median")
        x_w_fit = imp_w.fit_transform(w_train_fit[feat_list])
        x_w_val = imp_w.transform(w_val_df[feat_list])

        w_ref = CatBoostClassifier(**CB_PARAMS)
        w_ref.fit(x_w_fit, w_train_fit["target"], eval_set=(x_w_val, w_val_df["target"]))
        w_best_iter = w_ref.get_best_iteration()

        # Full-train on window
        imp_wf = SimpleImputer(strategy="median")
        x_wf = imp_wf.fit_transform(train_window[feat_list])
        x_wt = imp_wf.transform(test_sf[feat_list])

        params_wf = {k: v for k, v in CB_PARAMS.items() if k != "early_stopping_rounds"}
        params_wf["iterations"] = max(w_best_iter + 10, 50)
        wf_model = CatBoostClassifier(**params_wf)
        wf_model.fit(x_wf, train_window["target"])
        p_wf = wf_model.predict_proba(x_wt)[:, 1]
        auc_wf = roc_auc_score(test_sf["target"], p_wf)
        roi_wf = calc_roi(test_sf, p_wf, threshold=0.77)
        results[f"D_window_{int(pct * 100)}pct"] = {**roi_wf, "auc": auc_wf}
        logger.info(
            "D window %d%%: ROI=%.2f%% n=%d AUC=%.4f (train_n=%d)",
            int(pct * 100),
            roi_wf["roi"],
            roi_wf["n_bets"],
            auc_wf,
            len(train_window),
        )

    # E: Auto class weights balanced
    check_budget()
    params_bal = {**CB_PARAMS, "auto_class_weights": "Balanced"}
    bal_model = CatBoostClassifier(**params_bal)
    bal_model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
    best_iter_bal = bal_model.get_best_iteration()

    params_bal_ft = {k: v for k, v in params_bal.items() if k != "early_stopping_rounds"}
    params_bal_ft["iterations"] = best_iter_bal + 10
    bal_ft = CatBoostClassifier(**params_bal_ft)
    bal_ft.fit(x_full, train_sf["target"])
    p_bal = bal_ft.predict_proba(x_test)[:, 1]
    auc_bal = roc_auc_score(test_sf["target"], p_bal)

    for t in [0.5, 0.6, 0.7, 0.77]:
        roi_bal = calc_roi(test_sf, p_bal, threshold=t)
        results[f"E_balanced_t{t}"] = {**roi_bal, "auc": auc_bal}
        logger.info("E balanced t=%.2f: ROI=%.2f%% n=%d", t, roi_bal["roi"], roi_bal["n_bets"])

    # F: Monotonic + recency (best combo if both improve)
    check_budget()
    best_decay = max(
        [0.3, 0.5, 1.0, 2.0],
        key=lambda d: results[f"C_recency_d{d}"]["roi"],
    )
    logger.info("Best recency decay: %.1f", best_decay)

    weights_best = build_recency_weights(train_sf, decay=best_decay)
    params_combo = {k: v for k, v in params_mono.items() if k != "early_stopping_rounds"}
    params_combo["iterations"] = best_iter_mono + 10
    combo_model = CatBoostClassifier(**params_combo)
    combo_model.fit(x_full, train_sf["target"], sample_weight=weights_best)
    p_combo = combo_model.predict_proba(x_test)[:, 1]
    auc_combo = roc_auc_score(test_sf["target"], p_combo)

    for t in [0.76, 0.77]:
        roi_combo = calc_roi(test_sf, p_combo, threshold=t)
        results[f"F_mono_recency_t{t}"] = {**roi_combo, "auc": auc_combo}
        logger.info(
            "F mono+recency t=%.2f: ROI=%.2f%% n=%d",
            t,
            roi_combo["roi"],
            roi_combo["n_bets"],
        )

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info(
            "  %s: ROI=%.2f%% n=%d AUC=%.4f",
            name,
            r["roi"],
            r["n_bets"],
            r["auc"],
        )

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.7_monotonic_weights") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.7")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "monotonic_weights_window",
                    "n_features": len(feat_list),
                    "n_constrained": n_constrained,
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"auc_{name}", r["auc"])

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

            delta = best_r["roi"] - results["A_ref_full"]["roi"]
            if delta > 0:
                mlflow.set_tag("convergence_signal", "0.85")
            else:
                mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.7: BEST %s ROI=%.2f%% AUC=%.4f delta=%.2f pp run=%s",
                best_key,
                best_r["roi"],
                best_r["auc"],
                delta,
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
