"""Сохранение лучшего pipeline для chain continuation."""

import json
import logging
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    add_sport_market_features,
    load_raw_data,
    prepare_dataset,
    time_series_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

SESSION_DIR = Path(os.environ["UAF_SESSION_DIR"])
SESSION_ID = os.environ["UAF_SESSION_ID"]

FEATURES_BASE = [
    "Odds",
    "USD",
    "Is_Parlay",
    "Outcomes_Count",
    "ML_P_Model",
    "ML_P_Implied",
    "ML_Edge",
    "ML_EV",
    "ML_Winrate_Diff",
    "ML_Rating_Diff",
    "Outcome_Odds",
    "n_outcomes",
    "mean_outcome_odds",
    "max_outcome_odds",
    "min_outcome_odds",
]

FEATURES_ENC = [
    *FEATURES_BASE,
    "Sport_target_enc",
    "Sport_count_enc",
    "Market_target_enc",
    "Market_count_enc",
]


class BestPipeline:
    """Полный пайплайн: feature engineering + 4-model ensemble + blend selection."""

    def __init__(
        self,
        cb: CatBoostClassifier,
        lgbm: LGBMClassifier,
        lr: LogisticRegression,
        xgb: XGBClassifier,
        scaler: StandardScaler,
        sport_target_enc: dict,
        sport_count_enc: dict,
        market_target_enc: dict,
        market_count_enc: dict,
        global_target_mean: float,
        feature_names: list[str],
        blend_alpha: float = 0.5,
    ):
        self.cb = cb
        self.lgbm = lgbm
        self.lr = lr
        self.xgb = xgb
        self.scaler = scaler
        self.sport_target_enc = sport_target_enc
        self.sport_count_enc = sport_count_enc
        self.market_target_enc = market_target_enc
        self.market_count_enc = market_count_enc
        self.global_target_mean = global_target_mean
        self.feature_names = feature_names
        self.blend_alpha = blend_alpha

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применить feature engineering."""
        df = df.copy()
        df["Sport_target_enc"] = df["Sport"].map(self.sport_target_enc).fillna(
            self.global_target_mean
        )
        df["Sport_count_enc"] = df["Sport"].map(self.sport_count_enc).fillna(0)
        df["Market_target_enc"] = df["Market"].map(self.market_target_enc).fillna(
            self.global_target_mean
        )
        df["Market_count_enc"] = df["Market"].map(self.market_count_enc).fillna(0)
        return df[self.feature_names].fillna(0)

    def predict_proba(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Вероятности 4-model ensemble с blend (p_final, p_std)."""
        x = self._build_features(df)
        p_cb = self.cb.predict_proba(x)[:, 1]
        p_lgbm = self.lgbm.predict_proba(x)[:, 1]
        x_s = self.scaler.transform(x)
        p_lr = self.lr.predict_proba(x_s)[:, 1]
        p_xgb = self.xgb.predict_proba(x)[:, 1]

        preds = [p_cb, p_lgbm, p_lr, p_xgb]
        p_model = np.mean(preds, axis=0)
        p_std = np.std(preds, axis=0)

        # Blend with implied probability
        p_implied = 1.0 / df["Odds"].values
        p_final = self.blend_alpha * p_model + (1 - self.blend_alpha) * p_implied

        return p_final, p_std


def main() -> None:
    """Обучить на полном train и сохранить."""
    bets, outcomes, _, _ = load_raw_data()
    df = prepare_dataset(bets, outcomes)
    train, test = time_series_split(df, test_size=0.2)

    # Fit encoders on train
    train_enc, _ = add_sport_market_features(train, train)
    test_enc, _ = add_sport_market_features(test, train_enc)

    # Extract encoding maps
    smooth = 50
    global_mean = train["target"].mean()
    sport_target_enc = {}
    sport_count_enc = dict(train["Sport"].value_counts())
    market_target_enc = {}
    market_count_enc = dict(train["Market"].value_counts())

    for col, enc_dict in [("Sport", sport_target_enc), ("Market", market_target_enc)]:
        means = train.groupby(col)["target"].mean()
        counts = train.groupby(col)["target"].count()
        smoothed = (means * counts + global_mean * smooth) / (counts + smooth)
        enc_dict.update(smoothed.to_dict())

    x_train = train_enc[FEATURES_ENC].fillna(0)
    y_train = train_enc["target"]
    x_test = test_enc[FEATURES_ENC].fillna(0)

    cb = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=0
    )
    cb.fit(x_train, y_train)

    lgbm = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1, min_child_samples=50,
    )
    lgbm.fit(x_train, y_train)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_train_s, y_train)

    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0, min_child_weight=50,
        use_label_encoder=False, eval_metric="logloss",
    )
    xgb.fit(x_train, y_train)

    # Verify with 4m_blend50 strategy
    p_cb = cb.predict_proba(x_test)[:, 1]
    p_lgbm = lgbm.predict_proba(x_test)[:, 1]
    p_lr = lr.predict_proba(x_test_s)[:, 1]
    p_xgb = xgb.predict_proba(x_test)[:, 1]

    p_model = np.mean([p_cb, p_lgbm, p_lr, p_xgb], axis=0)
    p_std = np.std([p_cb, p_lgbm, p_lr, p_xgb], axis=0)
    p_implied = 1.0 / test_enc["Odds"].values
    p_final = 0.5 * p_model + 0.5 * p_implied

    ev = p_final * test_enc["Odds"].values - 1
    mask = (ev >= 0.05) & (p_std <= 0.02)
    n = mask.sum()
    sel = test_enc[mask]
    payout = (sel["target"] * sel["Odds"]).sum()
    roi = (payout - n) / n * 100
    logger.info("Verification: 4m_blend50 ROI=%.2f%% (n=%d)", roi, n)

    # Save pipeline
    pipeline = BestPipeline(
        cb=cb,
        lgbm=lgbm,
        lr=lr,
        xgb=xgb,
        scaler=scaler,
        sport_target_enc=sport_target_enc,
        sport_count_enc=sport_count_enc,
        market_target_enc=market_target_enc,
        market_count_enc=market_count_enc,
        global_target_mean=global_mean,
        feature_names=FEATURES_ENC,
        blend_alpha=0.5,
    )

    model_dir = SESSION_DIR / "models" / "best"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "pipeline.pkl")
    cb.save_model(str(model_dir / "model.cbm"))

    metadata = {
        "framework": "ensemble_cb_lgbm_lr_xgb",
        "model_file": "model.cbm",
        "pipeline_file": "pipeline.pkl",
        "roi": round(roi, 2),
        "n_bets": int(n),
        "feature_names": FEATURES_ENC,
        "selection_method": "4m_blend50_ev005_agree_p02",
        "selection_formula": (
            "p_model = avg(CB, LGBM, LR, XGB); "
            "p_final = 0.5*p_model + 0.5*(1/odds); "
            "EV = p_final*odds - 1; select where EV>=0.05 AND p_std<=0.02"
        ),
        "ensemble": "avg(CatBoost, LightGBM, LogisticRegression, XGBoost)",
        "blend_alpha": 0.5,
        "params": {"iterations": 200, "learning_rate": 0.05, "depth": 6},
        "walkforward_roi": 21.48,
        "walkforward_std": 1.78,
        "walkforward_range": "18.8-24.3%",
        "walkforward_note": (
            "21.48% mean ROI (5 seeds), but driven by extreme odds; "
            "capped at odds<=10: -3.41%"
        ),
        "session_id": SESSION_ID,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved pipeline.pkl + model.cbm + metadata.json")
    logger.info("ROI = %.2f%% (n=%d)", roi, n)


if __name__ == "__main__":
    main()
