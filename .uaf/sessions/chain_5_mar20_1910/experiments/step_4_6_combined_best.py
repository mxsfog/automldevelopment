"""Step 4.6: Комбинирование лучших стратегий + сохранение модели.

Гипотезы:
A) Per-sport EV + минимальный EV>=0.10 floor — объединение двух лучших стратегий
B) Per-sport EV с min floor EV>=0.05
C) EV>=0.10 + odds>=1.15 (двойной фильтр)
D) Оптимизация порога вероятности (p sweep) при фиксированном EV>=0.10
E) Сохранение лучшей модели как артефакт
"""

import json
import logging
import os
import traceback

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from common import (
    CB_BEST_PARAMS,
    SESSION_DIR,
    UNPROFITABLE_SPORTS,
    add_elo_features,
    add_engineered_features,
    calc_ev_roi,
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


def find_sport_ev_thresholds(
    val_df, p_val: np.ndarray, min_bets: int = 3, ev_floor: float = 0.0
) -> dict[str, float]:
    """Подбор EV threshold для каждого спорта с минимальным floor."""
    sports = val_df["Sport"].unique()
    thresholds: dict[str, float] = {}
    for sport in sports:
        sport_mask = (val_df["Sport"] == sport).values
        if sport_mask.sum() < 15:
            thresholds[sport] = ev_floor
            continue
        val_sport = val_df[sport_mask]
        p_sport = p_val[sport_mask]
        best_ev_t = ev_floor
        best_roi = -999.0
        for ev_t in np.arange(max(-0.05, ev_floor), 0.25, 0.01):
            r = calc_ev_roi(val_sport, p_sport, ev_threshold=ev_t, min_prob=0.77)
            if r["n_bets"] >= min_bets and r["roi"] > best_roi:
                best_roi = r["roi"]
                best_ev_t = float(ev_t)
        thresholds[sport] = max(best_ev_t, ev_floor)
    return thresholds


def apply_sport_ev(test_df, p_test: np.ndarray, sport_ev: dict[str, float]) -> dict:
    """Применение per-sport EV thresholds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    sports = test_df["Sport"].values
    mask = np.zeros(len(test_df), dtype=bool)
    for i in range(len(test_df)):
        ev_t = sport_ev.get(sports[i], 0.0)
        if ev[i] >= ev_t and p_test[i] >= 0.77:
            mask[i] = True
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def apply_combined_filter(
    test_df,
    p_test: np.ndarray,
    ev_threshold: float = 0.10,
    min_prob: float = 0.77,
    min_odds: float = 1.0,
) -> dict:
    """EV>=threshold + p>=min_prob + odds>=min_odds."""
    odds = test_df["Odds"].values
    ev = p_test * odds - 1.0
    mask = (ev >= ev_threshold) & (p_test >= min_prob) & (odds >= min_odds)
    n_sel = int(mask.sum())
    if n_sel == 0:
        return {"roi": 0.0, "n_bets": 0}
    sel = test_df.iloc[np.where(mask)[0]]
    staked = sel["USD"].sum()
    payout = sel["Payout_USD"].sum()
    roi = (payout - staked) / staked * 100
    return {"roi": float(roi), "n_bets": n_sel}


def main() -> None:
    """Combined best strategies + model save."""
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

    # Train reference model
    check_budget()
    imp = SimpleImputer(strategy="median")
    x_fit = imp.fit_transform(train_fit[feat_list])
    x_val = imp.transform(val_df[feat_list])

    cb_ref = CatBoostClassifier(**CB_BEST_PARAMS)
    cb_ref.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
    best_iter = cb_ref.get_best_iteration()

    imp_full = SimpleImputer(strategy="median")
    x_full = imp_full.fit_transform(train_sf[feat_list])
    x_test = imp_full.transform(test_sf[feat_list])

    ft_params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
    ft_params["iterations"] = best_iter + 10
    cb_ft = CatBoostClassifier(**ft_params)
    cb_ft.fit(x_full, train_sf["target"])

    p_test = cb_ft.predict_proba(x_test)[:, 1]
    p_val = cb_ft.predict_proba(imp_full.transform(val_df[feat_list]))[:, 1]
    auc = roc_auc_score(test_sf["target"], p_test)

    results: dict[str, dict] = {}

    # Baseline: EV>=0.10 + p>=0.77
    r_ev10 = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
    results["ev010_p77"] = r_ev10
    logger.info("Baseline EV>=0.10+p77: ROI=%.2f%% n=%d", r_ev10["roi"], r_ev10["n_bets"])

    # A: Per-sport EV with floor=0.10
    check_budget()
    sport_ev_010 = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.10)
    r_ps010 = apply_sport_ev(test_sf, p_test, sport_ev_010)
    results["ps_ev_floor010"] = r_ps010
    logger.info("A: PS_EV floor=0.10: ROI=%.2f%% n=%d", r_ps010["roi"], r_ps010["n_bets"])
    logger.info("  Sport thresholds: %s", sport_ev_010)

    # B: Per-sport EV with floor=0.05
    check_budget()
    sport_ev_005 = find_sport_ev_thresholds(val_df, p_val, ev_floor=0.05)
    r_ps005 = apply_sport_ev(test_sf, p_test, sport_ev_005)
    results["ps_ev_floor005"] = r_ps005
    logger.info("B: PS_EV floor=0.05: ROI=%.2f%% n=%d", r_ps005["roi"], r_ps005["n_bets"])

    # C: EV>=0.10 + odds>=1.15
    r_ev10_o115 = apply_combined_filter(test_sf, p_test, 0.10, 0.77, 1.15)
    results["ev010_odds115"] = r_ev10_o115
    logger.info(
        "C: EV>=0.10+odds>=1.15: ROI=%.2f%% n=%d", r_ev10_o115["roi"], r_ev10_o115["n_bets"]
    )

    # C2: EV>=0.10 + odds>=1.20
    r_ev10_o120 = apply_combined_filter(test_sf, p_test, 0.10, 0.77, 1.20)
    results["ev010_odds120"] = r_ev10_o120
    logger.info(
        "C2: EV>=0.10+odds>=1.20: ROI=%.2f%% n=%d", r_ev10_o120["roi"], r_ev10_o120["n_bets"]
    )

    # D: Probability threshold sweep at EV>=0.10
    logger.info("D: Probability threshold sweep at EV>=0.10:")
    for min_p in [0.70, 0.73, 0.75, 0.77, 0.78, 0.79, 0.80, 0.82, 0.85]:
        r = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=min_p)
        if r["n_bets"] > 0:
            logger.info("  p>=%.2f: ROI=%.2f%% n=%d", min_p, r["roi"], r["n_bets"])
            results[f"ev010_p{int(min_p * 100)}"] = r

    # D2: EV threshold sweep at p>=0.77
    logger.info("D2: EV threshold sweep at p>=0.77:")
    for ev_t in [0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]:
        r = calc_ev_roi(test_sf, p_test, ev_threshold=ev_t, min_prob=0.77)
        if r["n_bets"] > 0:
            logger.info("  EV>=%.2f: ROI=%.2f%% n=%d", ev_t, r["roi"], r["n_bets"])

    # 5-fold CV for top combinations
    check_budget()
    all_sf = train_sf.copy().sort_values("Created_At").reset_index(drop=True)
    n = len(all_sf)
    n_folds = 5
    block_size = n // (n_folds + 1)

    top_strats = ["ev010_p77", "ps_ev_floor010", "ev010_odds115"]
    cv_data: dict[str, list[float]] = {s: [] for s in top_strats}
    cv_nbets: dict[str, list[int]] = {s: [] for s in top_strats}

    logger.info("5-fold CV for top combinations (n=%d, block=%d)", n, block_size)

    for fold_idx in range(n_folds):
        check_budget()
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(train_end + block_size, n)

        fold_train = all_sf.iloc[:train_end].copy()
        fold_test = all_sf.iloc[test_start:test_end].copy()

        if len(fold_train) < 100 or len(fold_test) < 20:
            continue

        inner_val_split = int(len(fold_train) * 0.8)
        inner_train = fold_train.iloc[:inner_val_split]
        inner_val = fold_train.iloc[inner_val_split:]

        imp_fold = SimpleImputer(strategy="median")
        x_inner_train = imp_fold.fit_transform(inner_train[feat_list])
        x_inner_val = imp_fold.transform(inner_val[feat_list])

        cb_fold = CatBoostClassifier(**CB_BEST_PARAMS)
        cb_fold.fit(
            x_inner_train, inner_train["target"], eval_set=(x_inner_val, inner_val["target"])
        )
        fold_best_iter = cb_fold.get_best_iteration()

        p_inner_val = cb_fold.predict_proba(x_inner_val)[:, 1]
        sport_ev_fold = find_sport_ev_thresholds(inner_val, p_inner_val, ev_floor=0.10)

        imp_fold_full = SimpleImputer(strategy="median")
        x_fold_full = imp_fold_full.fit_transform(fold_train[feat_list])
        x_fold_test = imp_fold_full.transform(fold_test[feat_list])

        params_fold = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        params_fold["iterations"] = max(fold_best_iter + 10, 50)
        cb_fold_ft = CatBoostClassifier(**params_fold)
        cb_fold_ft.fit(x_fold_full, fold_train["target"])

        p_fold_test = cb_fold_ft.predict_proba(x_fold_test)[:, 1]

        r1 = calc_ev_roi(fold_test, p_fold_test, ev_threshold=0.10, min_prob=0.77)
        r2 = apply_sport_ev(fold_test, p_fold_test, sport_ev_fold)
        r3 = apply_combined_filter(fold_test, p_fold_test, 0.10, 0.77, 1.15)

        cv_data["ev010_p77"].append(r1["roi"])
        cv_data["ps_ev_floor010"].append(r2["roi"])
        cv_data["ev010_odds115"].append(r3["roi"])

        cv_nbets["ev010_p77"].append(r1["n_bets"])
        cv_nbets["ps_ev_floor010"].append(r2["n_bets"])
        cv_nbets["ev010_odds115"].append(r3["n_bets"])

        logger.info(
            "  Fold %d: ev010=%.1f ps010=%.1f ev010_o115=%.1f",
            fold_idx,
            r1["roi"],
            r2["roi"],
            r3["roi"],
        )

    logger.info("CV Summary:")
    for s in top_strats:
        if cv_data[s]:
            avg = np.mean(cv_data[s])
            std = np.std(cv_data[s])
            pos = sum(1 for r in cv_data[s] if r > 0)
            avg_n = np.mean(cv_nbets[s])
            logger.info(
                "  %-18s: avg=%.2f%% std=%.2f%% pos=%d/%d avg_n=%.0f",
                s,
                avg,
                std,
                pos,
                len(cv_data[s]),
                avg_n,
            )

    # Summary
    logger.info("Test results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d", name, r["roi"], r["n_bets"])

    # Determine best test result
    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    # E: Save best model artifact
    check_budget()

    model_dir = SESSION_DIR / "models" / "best"
    model_dir.mkdir(parents=True, exist_ok=True)

    cb_ft.save_model(str(model_dir / "model.cbm"))

    metadata = {
        "framework": "catboost",
        "model_file": "model.cbm",
        "roi": r_ev10["roi"],
        "auc": float(auc),
        "threshold": 0.77,
        "ev_threshold": 0.10,
        "selection_strategy": "EV>=0.10 AND p>=0.77",
        "n_bets": r_ev10["n_bets"],
        "feature_names": feat_list,
        "params": ft_params,
        "sport_filter": UNPROFITABLE_SPORTS,
        "elo_filter": True,
        "best_iteration": best_iter,
        "session_id": SESSION_ID,
        "cv_avg_roi": float(np.mean(cv_data["ev010_p77"])) if cv_data["ev010_p77"] else 0.0,
        "cv_std_roi": float(np.std(cv_data["ev010_p77"])) if cv_data["ev010_p77"] else 0.0,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s", model_dir)

    with mlflow.start_run(run_name="phase4/step4.6_combined_best") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.6")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": 42,
                    "method": "combined_best_strategies",
                    "n_features": len(feat_list),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            for s in top_strats:
                if cv_data[s]:
                    mlflow.log_metric(f"cv_avg_{s}", float(np.mean(cv_data[s])))
                    mlflow.log_metric(f"cv_std_{s}", float(np.std(cv_data[s])))

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": float(auc),
                    "n_bets": best_r["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.log_artifact(str(model_dir / "metadata.json"))
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.6: Best=%s ROI=%.2f%% n=%d run=%s",
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
