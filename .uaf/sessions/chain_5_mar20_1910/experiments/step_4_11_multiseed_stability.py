"""Step 4.11: Multi-seed stability analysis for best strategy.

Проверка: насколько результат зависит от random seed?
Обучаем модель с 10 разными seed, применяем PS_EV floor=0.10.
Это покажет variance от инициализации модели.
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
    """Multi-seed stability check."""
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

    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4242, 5555, 7777]

    ev010_rois: list[float] = []
    ev010_nbets: list[int] = []
    ps010_rois: list[float] = []
    ps010_nbets: list[int] = []
    aucs: list[float] = []

    for seed in seeds:
        check_budget()
        params = {k: v for k, v in CB_BEST_PARAMS.items() if k != "early_stopping_rounds"}
        params["random_seed"] = seed

        # Get best iteration with early stopping
        params_es = dict(params)
        params_es["iterations"] = 1000
        params_es["early_stopping_rounds"] = 50

        imp = SimpleImputer(strategy="median")
        x_fit = imp.fit_transform(train_fit[feat_list])
        x_val = imp.transform(val_df[feat_list])

        cb = CatBoostClassifier(**params_es)
        cb.fit(x_fit, train_fit["target"], eval_set=(x_val, val_df["target"]))
        bi = cb.get_best_iteration()

        # Full-train
        imp_f = SimpleImputer(strategy="median")
        x_full = imp_f.fit_transform(train_sf[feat_list])
        x_test = imp_f.transform(test_sf[feat_list])

        params["iterations"] = bi + 10
        cb_ft = CatBoostClassifier(**params)
        cb_ft.fit(x_full, train_sf["target"])

        p_test = cb_ft.predict_proba(x_test)[:, 1]
        p_val_full = cb_ft.predict_proba(imp_f.transform(val_df[feat_list]))[:, 1]
        auc = roc_auc_score(test_sf["target"], p_test)

        r_ev010 = calc_ev_roi(test_sf, p_test, ev_threshold=0.10, min_prob=0.77)
        sport_ev = find_sport_ev_thresholds(val_df, p_val_full, ev_floor=0.10)
        r_ps010 = apply_sport_ev(test_sf, p_test, sport_ev)

        ev010_rois.append(r_ev010["roi"])
        ev010_nbets.append(r_ev010["n_bets"])
        ps010_rois.append(r_ps010["roi"])
        ps010_nbets.append(r_ps010["n_bets"])
        aucs.append(auc)

        logger.info(
            "  seed=%d: EV010=%.2f%% n=%d | PS010=%.2f%% n=%d | AUC=%.4f | iter=%d",
            seed,
            r_ev010["roi"],
            r_ev010["n_bets"],
            r_ps010["roi"],
            r_ps010["n_bets"],
            auc,
            bi,
        )

    # Summary
    logger.info("Multi-seed summary (n=%d seeds):", len(seeds))
    logger.info(
        "  EV010: avg=%.2f%% std=%.2f%% min=%.2f%% max=%.2f%%",
        np.mean(ev010_rois),
        np.std(ev010_rois),
        np.min(ev010_rois),
        np.max(ev010_rois),
    )
    logger.info(
        "  PS010: avg=%.2f%% std=%.2f%% min=%.2f%% max=%.2f%%",
        np.mean(ps010_rois),
        np.std(ps010_rois),
        np.min(ps010_rois),
        np.max(ps010_rois),
    )
    logger.info(
        "  AUC:   avg=%.4f std=%.4f",
        np.mean(aucs),
        np.std(aucs),
    )

    with mlflow.start_run(run_name="phase4/step4.11_multiseed") as run:
        mlflow.set_tag("session_id", SESSION_ID)
        mlflow.set_tag("type", "experiment")
        mlflow.set_tag("step", "4.11")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("status", "running")

        try:
            mlflow.log_params(
                {
                    "validation_scheme": "time_series",
                    "seed": "multi",
                    "method": "multiseed_stability",
                    "n_features": len(feat_list),
                    "n_seeds": len(seeds),
                }
            )

            mlflow.log_metrics(
                {
                    "roi": float(np.mean(ps010_rois)),
                    "roi_ev010_avg": float(np.mean(ev010_rois)),
                    "roi_ev010_std": float(np.std(ev010_rois)),
                    "roi_ps010_avg": float(np.mean(ps010_rois)),
                    "roi_ps010_std": float(np.std(ps010_rois)),
                    "roc_auc": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "n_bets": float(np.mean(ps010_nbets)),
                }
            )

            for i, seed in enumerate(seeds):
                mlflow.log_metric(f"ev010_seed{seed}", ev010_rois[i])
                mlflow.log_metric(f"ps010_seed{seed}", ps010_rois[i])

            mlflow.log_artifact(__file__)
            mlflow.set_tag("status", "success")
            mlflow.set_tag("convergence_signal", "0.95")

            logger.info(
                "Step 4.11: EV010 avg=%.2f%% PS010 avg=%.2f%% run=%s",
                np.mean(ev010_rois),
                np.mean(ps010_rois),
                run.info.run_id,
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            mlflow.log_text(traceback.format_exc(), "traceback.txt")
            raise


if __name__ == "__main__":
    main()
