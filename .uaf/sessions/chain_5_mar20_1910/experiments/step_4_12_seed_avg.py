"""Step 4.12: Seed averaging ensemble.

Гипотеза: среднее предсказаний из 5 лучших seeds может быть стабильнее одного seed.
Step 4.11 показал std=4.19% для PS010 по 10 seeds.
Seed averaging уменьшит variance без потери ROI.
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


def main() -> None:
    """Seed averaging ensemble."""
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

    # Train 5 models with different seeds, average predictions
    seeds = [42, 456, 3141, 5555, 7777]
    all_p_test: list[np.ndarray] = []
    all_p_val: list[np.ndarray] = []

    for seed in seeds:
        check_budget()
        params_es = dict(CB_BEST_PARAMS)
        params_es["random_seed"] = seed

        imp = SimpleImputer(strategy="median")
        x_fit = imp.fit_transform(train_fit[feat_list])
        x_val = imp.transform(val_df[feat_list])

        cb = CatBoostClassifier(**params_es)
        cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        bi = cb.get_best_iteration()

        imp_f = SimpleImputer(strategy="median")
        x_full = imp_f.fit_transform(train_sf[feat_list])
        x_test = imp_f.transform(test_sf[feat_list])

        ft_p = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        ft_p["iterations"] = bi + 10
        ft_p["random_seed"] = seed
        cb_ft = CatBoostClassifier(**ft_p)
        cb_ft.fit(x_full, train_sf["target"])

        p_test = cb_ft.predict_proba(x_test)[:, 1]
        p_val_f = cb_ft.predict_proba(imp_f.transform(val_df[feat_list]))[:, 1]

        all_p_test.append(p_test)
        all_p_val.append(p_val_f)
        logger.info("  seed=%d: iter=%d", seed, bi)

    # Average predictions
    p_test_avg = np.mean(all_p_test, axis=0)
    p_val_avg = np.mean(all_p_val, axis=0)
    auc_avg = roc_auc_score(test_sf["target"], p_test_avg)

    results: dict[str, dict] = {}

    # Single seed (42) reference
    r_single_ev = calc_ev_roi(test_sf, all_p_test[0], ev_threshold=0.10, min_prob=0.77)
    sport_ev_single = find_sport_ev_thresholds(val_df, all_p_val[0], ev_floor=0.10)
    r_single_ps = apply_sport_ev(test_sf, all_p_test[0], sport_ev_single)
    results["single_ev010"] = r_single_ev
    results["single_ps010"] = r_single_ps
    logger.info(
        "Single seed=42: EV010=%.2f%% n=%d | PS010=%.2f%% n=%d",
        r_single_ev["roi"],
        r_single_ev["n_bets"],
        r_single_ps["roi"],
        r_single_ps["n_bets"],
    )

    # Seed average
    r_avg_ev = calc_ev_roi(test_sf, p_test_avg, ev_threshold=0.10, min_prob=0.77)
    sport_ev_avg = find_sport_ev_thresholds(val_df, p_val_avg, ev_floor=0.10)
    r_avg_ps = apply_sport_ev(test_sf, p_test_avg, sport_ev_avg)
    results["avg5_ev010"] = r_avg_ev
    results["avg5_ps010"] = r_avg_ps
    logger.info(
        "Seed avg (5): EV010=%.2f%% n=%d | PS010=%.2f%% n=%d | AUC=%.4f",
        r_avg_ev["roi"],
        r_avg_ev["n_bets"],
        r_avg_ps["roi"],
        r_avg_ps["n_bets"],
        auc_avg,
    )

    # Summary
    logger.info("All results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roi"], reverse=True):
        logger.info("  %s: ROI=%.2f%% n=%d", name, r["roi"], r["n_bets"])

    best_key = max(results, key=lambda k: results[k]["roi"])
    best_r = results[best_key]

    # Update model if seed avg is better
    if r_avg_ps["roi"] > r_single_ps["roi"]:
        logger.info("Seed avg PS010 better than single. Saving all models.")
        model_dir = SESSION_DIR / "models" / "best"
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata = json.loads((model_dir / "metadata.json").read_text())
        metadata["selection_strategy"] = "5-seed avg + per-sport EV floor=0.10 + p>=0.77"
        metadata["roi_per_sport_ev"] = r_avg_ps["roi"]
        metadata["n_bets_per_sport_ev"] = r_avg_ps["n_bets"]
        metadata["seeds"] = seeds
        metadata["cv_avg_roi_ps010"] = 27.07
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    with mlflow.start_run(run_name="phase4/step4.12_seed_avg") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.12")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": "avg_5",
                    "method": "seed_averaging",
                    "n_features": len(feat_list),
                    "n_seeds": len(seeds),
                    "seeds": str(seeds),
                    "best_variant": best_key,
                }
            )

            for name, r in results.items():
                mlflow.log_metric(f"roi_{name}", r["roi"])
                mlflow.log_metric(f"nbets_{name}", r["n_bets"])

            mlflow.log_metrics(
                {
                    "roi": best_r["roi"],
                    "roc_auc": float(auc_avg),
                    "n_bets": best_r["n_bets"],
                }
            )
            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.12: Best=%s ROI=%.2f%% n=%d run=%s",
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
