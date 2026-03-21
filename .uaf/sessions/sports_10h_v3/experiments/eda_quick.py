"""Быстрая EDA для понимания данных."""

import logging

from common import load_data, time_series_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df = load_data()
    logger.info("Total rows after filtering: %d", len(df))
    logger.info("Target distribution:\n%s", df["target"].value_counts())
    logger.info("Win rate: %.4f", df["target"].mean())
    logger.info("Columns: %s", list(df.columns))
    logger.info("Dtypes:\n%s", df.dtypes)

    logger.info("\nNumeric stats:")
    num_cols = ["Odds", "USD", "Payout_USD", "ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV"]
    for col in num_cols:
        if col in df.columns:
            logger.info(
                "  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, nulls=%d",
                col,
                df[col].mean(),
                df[col].std(),
                df[col].min(),
                df[col].max(),
                df[col].isna().sum(),
            )

    logger.info("\nSport distribution (top 10):")
    if "Sport" in df.columns:
        logger.info("\n%s", df["Sport"].value_counts().head(10))

    logger.info("\nParlay distribution:")
    logger.info("\n%s", df["Is_Parlay"].value_counts())

    logger.info("\nDate range: %s to %s", df["Created_At"].min(), df["Created_At"].max())

    train, test = time_series_split(df)
    logger.info("Train win rate: %.4f", train["target"].mean())
    logger.info("Test win rate: %.4f", test["target"].mean())

    # ROI baseline: all bets
    total_staked = df["USD"].sum()
    total_payout = df["Payout_USD"].sum()
    logger.info(
        "Overall ROI (all bets): %.2f%%", (total_payout - total_staked) / total_staked * 100
    )

    # ROI by sport
    if "Sport" in df.columns:
        logger.info("\nROI by Sport:")
        for sport, g in df.groupby("Sport"):
            s = g["USD"].sum()
            p = g["Payout_USD"].sum()
            if s > 0:
                r = (p - s) / s * 100
                logger.info(
                    "  %s: ROI=%.2f%%, n=%d, WR=%.3f", sport, r, len(g), g["target"].mean()
                )

    # ML features analysis
    logger.info("\nML_P_Model distribution by target:")
    for t in [0, 1]:
        subset = df[df["target"] == t]
        logger.info(
            "  target=%d: ML_P_Model mean=%.4f, ML_Edge mean=%.4f, ML_EV mean=%.4f",
            t,
            subset["ML_P_Model"].mean(),
            subset["ML_Edge"].mean(),
            subset["ML_EV"].mean(),
        )


if __name__ == "__main__":
    main()
