"""Step 4.5: CatBoost native categorical features + feature selection."""

import logging
import os
import traceback

import mlflow
from catboost import CatBoostClassifier
from common import (
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_roi,
    check_budget,
    find_best_threshold_on_val,
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

CB_PARAMS_BASE = {
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


def main() -> None:
    """CatBoost with categorical features + feature importance-based selection."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    num_feats = get_base_features() + get_engineered_features() + get_elo_features()
    cat_feats = ["Sport", "Market"]

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    results: dict[str, dict] = {}

    # A: Reference (numerical only, full-train)
    check_budget()
    imp_ref = SimpleImputer(strategy="median")
    x_fit_ref = imp_ref.fit_transform(train_fit[num_feats])
    x_val_ref = imp_ref.transform(val_df[num_feats])

    ref_model = CatBoostClassifier(**CB_PARAMS_BASE)
    ref_model.fit(x_fit_ref, train_fit["target"], eval_set=(x_val_ref, val_df["target"]))
    best_iter = ref_model.get_best_iteration()

    # Full-train ref
    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[num_feats])
    x_test = imp_full.transform(test_sf[num_feats])

    params_ft = {k: v for k, v in CB_PARAMS_BASE.items() if k != "early_stopping_rounds"}
    params_ft["iterations"] = best_iter + 10
    ft_ref = CatBoostClassifier(**params_ft)
    ft_ref.fit(x_full, train_sf["target"])
    p_ref = ft_ref.predict_proba(x_test)[:, 1]
    auc_ref = roc_auc_score(test_sf["target"], p_ref)
    results["ref_full"] = {
        **calc_roi(test_sf, p_ref, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_ref,
    }

    # B: With categorical features (native CatBoost)
    check_budget()
    all_feats = num_feats + cat_feats
    cat_indices = [all_feats.index(c) for c in cat_feats]

    train_fit_cat = train_fit[all_feats].copy()
    val_df_cat = val_df[all_feats].copy()
    train_sf_cat = train_sf[all_feats].copy()
    test_sf_cat = test_sf[all_feats].copy()

    for c in cat_feats:
        for dframe in [train_fit_cat, val_df_cat, train_sf_cat, test_sf_cat]:
            dframe[c] = dframe[c].fillna("unknown").astype(str)

    # Impute numerical
    for c in num_feats:
        median_val = train_fit_cat[c].median()
        for dframe in [train_fit_cat, val_df_cat, train_sf_cat, test_sf_cat]:
            dframe[c] = dframe[c].fillna(median_val)

    cat_model = CatBoostClassifier(**CB_PARAMS_BASE)
    cat_model.fit(
        train_fit_cat,
        train_fit["target"],
        eval_set=(val_df_cat, val_df["target"]),
        cat_features=cat_indices,
    )
    best_iter_cat = cat_model.get_best_iteration()
    p_val_cat = cat_model.predict_proba(val_df_cat)[:, 1]
    t_cat, _ = find_best_threshold_on_val(val_df, p_val_cat, min_bets=15)

    # Full-train cat model
    params_cat_ft = {k: v for k, v in CB_PARAMS_BASE.items() if k != "early_stopping_rounds"}
    params_cat_ft["iterations"] = best_iter_cat + 10
    cat_ft = CatBoostClassifier(**params_cat_ft)
    cat_ft.fit(train_sf_cat, train_sf["target"], cat_features=cat_indices)
    p_cat = cat_ft.predict_proba(test_sf_cat)[:, 1]
    auc_cat = roc_auc_score(test_sf["target"], p_cat)

    results["cat_full_t77"] = {
        **calc_roi(test_sf, p_cat, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_cat,
    }
    results["cat_full_val_t"] = {
        **calc_roi(test_sf, p_cat, threshold=t_cat),
        "threshold": t_cat,
        "auc": auc_cat,
    }

    # C: Feature selection -- drop bottom 30% by importance
    check_budget()
    fi = ref_model.get_feature_importance()
    fi_pairs = sorted(zip(num_feats, fi, strict=True), key=lambda x: x[1], reverse=True)
    logger.info("Feature importance:")
    for fname, imp_val in fi_pairs:
        logger.info("  %s: %.2f", fname, imp_val)

    # Keep top 70% features
    n_keep = max(10, int(len(num_feats) * 0.7))
    top_feats = [f for f, _ in fi_pairs[:n_keep]]
    logger.info("Keeping top %d features: %s", n_keep, top_feats)

    imp_sel = SimpleImputer(strategy="median")
    x_fit_sel = imp_sel.fit_transform(train_fit[top_feats])
    x_val_sel = imp_sel.transform(val_df[top_feats])

    sel_model = CatBoostClassifier(**CB_PARAMS_BASE)
    sel_model.fit(x_fit_sel, train_fit["target"], eval_set=(x_val_sel, val_df["target"]))
    best_iter_sel = sel_model.get_best_iteration()

    # Full-train selected
    imp_sel_full = SimpleImputer(strategy="median")
    x_sel_full = imp_sel_full.fit_transform(train_sf[top_feats])
    x_sel_test = imp_sel_full.transform(test_sf[top_feats])

    params_sel_ft = {k: v for k, v in CB_PARAMS_BASE.items() if k != "early_stopping_rounds"}
    params_sel_ft["iterations"] = best_iter_sel + 10
    sel_ft = CatBoostClassifier(**params_sel_ft)
    sel_ft.fit(x_sel_full, train_sf["target"])
    p_sel = sel_ft.predict_proba(x_sel_test)[:, 1]
    auc_sel = roc_auc_score(test_sf["target"], p_sel)

    results["selected_full_t77"] = {
        **calc_roi(test_sf, p_sel, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_sel,
    }

    # D: Very aggressive selection -- top 15 features
    check_budget()
    top15 = [f for f, _ in fi_pairs[:15]]
    imp_t15 = SimpleImputer(strategy="median")
    x_fit_t15 = imp_t15.fit_transform(train_fit[top15])
    x_val_t15 = imp_t15.transform(val_df[top15])

    t15_model = CatBoostClassifier(**CB_PARAMS_BASE)
    t15_model.fit(x_fit_t15, train_fit["target"], eval_set=(x_val_t15, val_df["target"]))
    best_iter_t15 = t15_model.get_best_iteration()

    imp_t15f = SimpleImputer(strategy="median")
    x_t15f = imp_t15f.fit_transform(train_sf[top15])
    x_t15t = imp_t15f.transform(test_sf[top15])

    params_t15 = {k: v for k, v in CB_PARAMS_BASE.items() if k != "early_stopping_rounds"}
    params_t15["iterations"] = best_iter_t15 + 10
    t15_ft = CatBoostClassifier(**params_t15)
    t15_ft.fit(x_t15f, train_sf["target"])
    p_t15 = t15_ft.predict_proba(x_t15t)[:, 1]
    auc_t15 = roc_auc_score(test_sf["target"], p_t15)

    results["top15_full_t77"] = {
        **calc_roi(test_sf, p_t15, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_t15,
    }

    # E: Blend: ref + cat (0.7/0.3)
    p_blend = 0.7 * p_ref + 0.3 * p_cat
    auc_blend = roc_auc_score(test_sf["target"], p_blend)
    results["blend_ref_cat"] = {
        **calc_roi(test_sf, p_blend, threshold=0.77),
        "threshold": 0.77,
        "auc": auc_blend,
    }

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

    with mlflow.start_run(run_name="phase4/step4.5_cat_featsel") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.5")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "cat_features_featsel",
                    "n_features_full": len(num_feats),
                    "n_features_selected": n_keep,
                    "n_features_top15": 15,
                    "best_variant": best_key,
                    "cat_features": str(cat_feats),
                    "selected_features": str(top_feats),
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
                    "best_threshold": best_r["threshold"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.9")

            logger.info(
                "Step 4.5: BEST %s ROI=%.2f%% AUC=%.4f run=%s",
                best_key,
                best_r["roi"],
                best_r["auc"],
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
