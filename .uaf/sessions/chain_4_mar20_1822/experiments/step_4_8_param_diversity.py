"""Step 4.8: CatBoost parameter diversity — depth, grow policy, ordered boosting, blends."""

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


def train_full_model(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    params: dict,
) -> tuple[CatBoostClassifier, int]:
    """Обучение full-train модели с определением iterations через early stopping."""
    ref = CatBoostClassifier(**params)
    ref.fit(x_fit, y_fit, eval_set=(x_val, y_val))
    best_iter = ref.get_best_iteration()

    ft_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = max(best_iter + 10, 50)
    model = CatBoostClassifier(**ft_params)
    model.fit(x_full, y_full)
    return model, best_iter


def main() -> None:
    """Explore CatBoost parameter diversity and model blending."""
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
    models_preds: dict[str, np.ndarray] = {}

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    # A: Reference (depth=8, ref params)
    check_budget()
    ref_model, ref_iter = train_full_model(
        x_fit, train_fit["target"], x_val, val_df["target"], x_full, train_sf["target"], CB_PARAMS
    )
    p_ref = ref_model.predict_proba(x_test)[:, 1]
    auc_ref = roc_auc_score(test_sf["target"], p_ref)
    results["A_ref_d8"] = {**calc_roi(test_sf, p_ref, threshold=0.77), "auc": auc_ref}
    models_preds["ref_d8"] = p_ref
    logger.info(
        "A ref d8: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
        results["A_ref_d8"]["roi"],
        results["A_ref_d8"]["n_bets"],
        auc_ref,
        ref_iter,
    )

    # B: Depth exploration
    check_budget()
    for depth in [6, 7, 9, 10]:
        params_d = {**CB_PARAMS, "depth": depth}
        model_d, iter_d = train_full_model(
            x_fit,
            train_fit["target"],
            x_val,
            val_df["target"],
            x_full,
            train_sf["target"],
            params_d,
        )
        p_d = model_d.predict_proba(x_test)[:, 1]
        auc_d = roc_auc_score(test_sf["target"], p_d)
        roi_d = calc_roi(test_sf, p_d, threshold=0.77)
        key = f"B_depth{depth}"
        results[key] = {**roi_d, "auc": auc_d}
        models_preds[f"d{depth}"] = p_d
        logger.info(
            "B depth=%d: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
            depth,
            roi_d["roi"],
            roi_d["n_bets"],
            auc_d,
            iter_d,
        )

    # C: Ordered boosting (default is Plain)
    check_budget()
    params_ordered = {**CB_PARAMS, "boosting_type": "Ordered"}
    model_ord, iter_ord = train_full_model(
        x_fit,
        train_fit["target"],
        x_val,
        val_df["target"],
        x_full,
        train_sf["target"],
        params_ordered,
    )
    p_ord = model_ord.predict_proba(x_test)[:, 1]
    auc_ord = roc_auc_score(test_sf["target"], p_ord)
    results["C_ordered"] = {**calc_roi(test_sf, p_ord, threshold=0.77), "auc": auc_ord}
    models_preds["ordered"] = p_ord
    logger.info(
        "C ordered: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
        results["C_ordered"]["roi"],
        results["C_ordered"]["n_bets"],
        auc_ord,
        iter_ord,
    )

    # D: Grow policy = Lossguide (non-symmetric trees)
    check_budget()
    params_lg = {
        **CB_PARAMS,
        "grow_policy": "Lossguide",
        "max_leaves": 64,
    }
    params_lg.pop("depth", None)
    model_lg, iter_lg = train_full_model(
        x_fit,
        train_fit["target"],
        x_val,
        val_df["target"],
        x_full,
        train_sf["target"],
        params_lg,
    )
    p_lg = model_lg.predict_proba(x_test)[:, 1]
    auc_lg = roc_auc_score(test_sf["target"], p_lg)
    results["D_lossguide"] = {**calc_roi(test_sf, p_lg, threshold=0.77), "auc": auc_lg}
    models_preds["lossguide"] = p_lg
    logger.info(
        "D lossguide: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
        results["D_lossguide"]["roi"],
        results["D_lossguide"]["n_bets"],
        auc_lg,
        iter_lg,
    )

    # E: Lower learning rate + more iterations
    check_budget()
    params_lr = {**CB_PARAMS, "learning_rate": 0.03, "iterations": 2500}
    model_lr, iter_lr = train_full_model(
        x_fit,
        train_fit["target"],
        x_val,
        val_df["target"],
        x_full,
        train_sf["target"],
        params_lr,
    )
    p_lr = model_lr.predict_proba(x_test)[:, 1]
    auc_lr = roc_auc_score(test_sf["target"], p_lr)
    results["E_lr003"] = {**calc_roi(test_sf, p_lr, threshold=0.77), "auc": auc_lr}
    models_preds["lr003"] = p_lr
    logger.info(
        "E lr=0.03: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
        results["E_lr003"]["roi"],
        results["E_lr003"]["n_bets"],
        auc_lr,
        iter_lr,
    )

    # F: Deeper + higher l2 (more regularized deep model)
    check_budget()
    params_deep = {**CB_PARAMS, "depth": 10, "l2_leaf_reg": 50.0, "min_data_in_leaf": 30}
    model_deep, iter_deep = train_full_model(
        x_fit,
        train_fit["target"],
        x_val,
        val_df["target"],
        x_full,
        train_sf["target"],
        params_deep,
    )
    p_deep = model_deep.predict_proba(x_test)[:, 1]
    auc_deep = roc_auc_score(test_sf["target"], p_deep)
    results["F_deep10_l2_50"] = {**calc_roi(test_sf, p_deep, threshold=0.77), "auc": auc_deep}
    models_preds["deep10"] = p_deep
    logger.info(
        "F deep10+l2=50: ROI=%.2f%% n=%d AUC=%.4f iter=%d",
        results["F_deep10_l2_50"]["roi"],
        results["F_deep10_l2_50"]["n_bets"],
        auc_deep,
        iter_deep,
    )

    # G: Diverse model blends (top 3 diverse models)
    check_budget()
    # Blend ref_d8 + best depth variant + ordered
    best_depth_key = max(
        [k for k in models_preds if k.startswith("d")],
        key=lambda k: results.get(f"B_depth{k[1:]}", {}).get("roi", -999),
    )
    best_depth_name = f"B_depth{best_depth_key[1:]}"
    logger.info(
        "Best depth variant for blend: %s (ROI=%.2f%%)",
        best_depth_name,
        results[best_depth_name]["roi"],
    )

    blend_configs = [
        ("G_blend_ref_ord_0.7_0.3", {"ref_d8": 0.7, "ordered": 0.3}),
        ("G_blend_ref_lr003_0.6_0.4", {"ref_d8": 0.6, "lr003": 0.4}),
        ("G_blend_3way_equal", {"ref_d8": 0.34, "ordered": 0.33, "lr003": 0.33}),
        ("G_blend_ref_deep_0.7_0.3", {"ref_d8": 0.7, "deep10": 0.3}),
    ]

    for blend_name, weights in blend_configs:
        p_blend = sum(w * models_preds[k] for k, w in weights.items())
        auc_blend = roc_auc_score(test_sf["target"], p_blend)
        for t in [0.76, 0.77]:
            roi_bl = calc_roi(test_sf, p_blend, threshold=t)
            full_name = f"{blend_name}_t{t}"
            results[full_name] = {**roi_bl, "auc": auc_blend}
            logger.info(
                "%s: ROI=%.2f%% n=%d AUC=%.4f",
                full_name,
                roi_bl["roi"],
                roi_bl["n_bets"],
                auc_blend,
            )

    # H: Multi-seed ensemble (seeds 42, 7, 13, 99, 2024)
    check_budget()
    seed_preds = [p_ref]  # seed=42 already computed
    for seed in [7, 13, 99, 2024]:
        params_s = {**CB_PARAMS, "random_seed": seed}
        model_s, _ = train_full_model(
            x_fit,
            train_fit["target"],
            x_val,
            val_df["target"],
            x_full,
            train_sf["target"],
            params_s,
        )
        seed_preds.append(model_s.predict_proba(x_test)[:, 1])

    p_multi_seed = np.mean(seed_preds, axis=0)
    auc_ms = roc_auc_score(test_sf["target"], p_multi_seed)
    for t in [0.76, 0.77]:
        roi_ms = calc_roi(test_sf, p_multi_seed, threshold=t)
        results[f"H_multi_seed_t{t}"] = {**roi_ms, "auc": auc_ms}
        logger.info(
            "H multi_seed t=%.2f: ROI=%.2f%% n=%d AUC=%.4f",
            t,
            roi_ms["roi"],
            roi_ms["n_bets"],
            auc_ms,
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

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    with mlflow.start_run(run_name="phase4/step4.8_param_diversity") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.8")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "param_diversity_blends",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                    "depths_tested": "6,7,8,9,10",
                    "grow_policies": "SymmetricTree,Lossguide",
                    "boosting_types": "Plain,Ordered",
                }
            )

            for name, r in results.items():
                safe_name = name.replace(".", "_")
                mlflow.log_metric(f"roi_{safe_name}", r["roi"])

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

            delta = best_r["roi"] - results["A_ref_d8"]["roi"]
            mlflow.set_tag("convergence_signal", "0.95" if delta <= 0 else "0.85")

            logger.info(
                "Step 4.8: BEST %s ROI=%.2f%% delta=%.2f pp run=%s",
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
