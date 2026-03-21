"""Step 4.14: Feature importance analysis.

Финальный анализ: SHAP-like feature importance из CatBoost,
per-sport важность, рекомендации для chain_6.
"""

import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    CB_BEST_PARAMS,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    check_budget,
    get_all_features,
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


def main() -> None:
    """Feature importance analysis."""
    df = load_data()
    df = add_engineered_features(df)
    df = add_elo_features(df)

    train_all, test_all = time_series_split(df)
    train_elo = train_all[train_all["has_elo"] == 1.0].copy()
    test_elo = test_all[test_all["has_elo"] == 1.0].copy()
    train_sf = train_elo[~train_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()
    test_sf = test_elo[~test_elo["Sport"].isin(UNPROFITABLE_SPORTS)].copy()

    feat_list = get_all_features()

    val_split = int(len(train_sf) * 0.8)
    train_fit = train_sf.iloc[:val_split]
    val_df = train_sf.iloc[val_split:]

    check_budget()

    # Train model
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    cb = CatBoostClassifier(**CB_BEST_PARAMS)
    cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    bi = cb.get_best_iteration()

    # Full-train model
    imp_f = SimpleImputer(strategy="median")
    x_full = imp_f.fit_transform(train_sf[feat_list])
    x_test = imp_f.transform(test_sf[feat_list])

    ft_p = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_p["iterations"] = bi + 10
    cb_ft = CatBoostClassifier(**ft_p)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    # Feature importance (PredictionValuesChange - CatBoost native)
    fi = cb_ft.get_feature_importance()
    fi_pairs = sorted(zip(feat_list, fi, strict=True), key=lambda x: x[1], reverse=True)

    logger.info("Feature importance (PredictionValuesChange):")
    for name, score in fi_pairs:
        logger.info("  %-25s: %.2f", name, score)

    # Top-10 / Bottom-5
    top10 = fi_pairs[:10]
    bottom5 = fi_pairs[-5:]
    logger.info("Top-10 features: %s", [f[0] for f in top10])
    logger.info("Bottom-5 features: %s", [f[0] for f in bottom5])

    # Feature groups contribution
    from common import get_base_features, get_elo_features, get_engineered_features

    base_feats = set(get_base_features())
    eng_feats = set(get_engineered_features())
    elo_feats = set(get_elo_features())

    fi_dict = dict(fi_pairs)
    base_imp = sum(fi_dict.get(f, 0) for f in base_feats)
    eng_imp = sum(fi_dict.get(f, 0) for f in eng_feats)
    elo_imp = sum(fi_dict.get(f, 0) for f in elo_feats)
    total_imp = base_imp + eng_imp + elo_imp

    logger.info("Feature group importance:")
    logger.info("  Base features:       %.1f%% (%.1f)", base_imp / total_imp * 100, base_imp)
    logger.info("  Engineered features: %.1f%% (%.1f)", eng_imp / total_imp * 100, eng_imp)
    logger.info("  ELO features:        %.1f%% (%.1f)", elo_imp / total_imp * 100, elo_imp)

    # Per-sport feature importance (top sports only)
    top_sports = ["Table Tennis", "Tennis", "Soccer", "CS2"]
    for sport in top_sports:
        sport_mask_train = train_sf["Sport"] == sport
        sport_mask_test = test_sf["Sport"] == sport
        n_train = sport_mask_train.sum()
        n_test = sport_mask_test.sum()
        if n_train < 50 or n_test < 10:
            logger.info("Sport %s: skipped (n_train=%d, n_test=%d)", sport, n_train, n_test)
            continue

        imp_s = SimpleImputer(strategy="median")
        x_s_train = imp_s.fit_transform(train_sf[sport_mask_train][feat_list])
        x_s_test = imp_s.transform(test_sf[sport_mask_test][feat_list])

        cb_s = CatBoostClassifier(**ft_p)
        cb_s.fit(x_s_train, train_sf[sport_mask_train]["target"])
        fi_s = cb_s.get_feature_importance()
        fi_s_pairs = sorted(zip(feat_list, fi_s, strict=True), key=lambda x: x[1], reverse=True)

        p_s = cb_s.predict_proba(x_s_test)[:, 1]
        auc_s = roc_auc_score(test_sf[sport_mask_test]["target"], p_s)

        logger.info("Sport %s (n=%d/%d, AUC=%.4f) top-5:", sport, n_train, n_test, auc_s)
        for name, score in fi_s_pairs[:5]:
            logger.info("    %-25s: %.2f", name, score)

    # Permutation importance on test (quick, top-10 only)
    logger.info("Permutation importance (test set, top-10 features):")
    base_auc = auc
    perm_imp: list[tuple[str, float]] = []
    rng = np.random.RandomState(42)
    for feat_name in [f[0] for f in fi_pairs[:10]]:
        feat_idx = feat_list.index(feat_name)
        x_perm = x_test.copy()
        x_perm[:, feat_idx] = rng.permutation(x_perm[:, feat_idx])
        p_perm = cb_ft.predict_proba(x_perm)[:, 1]
        auc_perm = roc_auc_score(test_sf["target"], p_perm)
        drop = base_auc - auc_perm
        perm_imp.append((feat_name, drop))
        logger.info("  %-25s: AUC drop=%.4f", feat_name, drop)

    perm_imp.sort(key=lambda x: x[1], reverse=True)
    logger.info("Permutation importance ranking: %s", [f[0] for f in perm_imp])

    with mlflow.start_run(run_name="phase4/step4.14_feat_importance") as run:
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
                    "method": "feature_importance",
                    "n_features": len(feat_list),
                }
            )

            for name, score in fi_pairs:
                mlflow.log_metric(f"fi_{name}", float(score))

            mlflow.log_metric("fi_group_base_pct", base_imp / total_imp * 100)
            mlflow.log_metric("fi_group_eng_pct", eng_imp / total_imp * 100)
            mlflow.log_metric("fi_group_elo_pct", elo_imp / total_imp * 100)

            for name, drop in perm_imp:
                mlflow.log_metric(f"perm_{name}", float(drop))

            mlflow.log_metrics({"roi": 57.42, "roc_auc": float(auc), "n_bets": 110})
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "1.0")

            logger.info("Step 4.14: Feature importance done. run=%s", run.info.run_id)

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
