"""EDA: исследование данных спортивных ставок."""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/mnt/d/automl-research/data/sports_betting")


def main() -> None:
    bets = pd.read_csv(DATA_DIR / "bets.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    teams = pd.read_csv(DATA_DIR / "teams.csv")
    elo = pd.read_csv(DATA_DIR / "elo_history.csv")

    logger.info(
        "bets: %s, outcomes: %s, teams: %s, elo: %s",
        bets.shape,
        outcomes.shape,
        teams.shape,
        elo.shape,
    )

    # Распределение Status
    logger.info("Status distribution:\n%s", bets["Status"].value_counts())

    # Исключаем pending, cancelled, error, cashout
    exclude = {"pending", "cancelled", "error", "cashout"}
    bets_filtered = bets[~bets["Status"].isin(exclude)].copy()
    logger.info("After filtering: %d rows", len(bets_filtered))
    logger.info("Status after filter:\n%s", bets_filtered["Status"].value_counts())

    # Target
    bets_filtered["target"] = (bets_filtered["Status"] == "won").astype(int)
    logger.info("Target mean: %.4f", bets_filtered["target"].mean())

    # Time range
    bets_filtered["Created_At"] = pd.to_datetime(bets_filtered["Created_At"])
    logger.info(
        "Time range: %s to %s",
        bets_filtered["Created_At"].min(),
        bets_filtered["Created_At"].max(),
    )

    # Join with outcomes
    outcomes_agg = (
        outcomes.groupby("Bet_ID")
        .agg(
            Sport=("Sport", "first"),
            Market=("Market", "first"),
            Selection=("Selection", "first"),
            Odds_outcome=("Odds", "mean"),
            n_outcomes=("Bet_ID", "count"),
            Start_Time=("Start_Time", "first"),
        )
        .reset_index()
    )
    df = bets_filtered.merge(outcomes_agg, left_on="ID", right_on="Bet_ID", how="left")
    logger.info("After join: %d rows", len(df))

    # Key columns
    logger.info("Is_Parlay distribution:\n%s", df["Is_Parlay"].value_counts())
    logger.info("Sport distribution (top 10):\n%s", df["Sport"].value_counts().head(10))
    logger.info("Market distribution (top 10):\n%s", df["Market"].value_counts().head(10))

    # ML features
    ml_cols = ["ML_P_Model", "ML_P_Implied", "ML_Edge", "ML_EV"]
    logger.info("ML features stats:\n%s", df[ml_cols].describe())
    logger.info("ML_P_Model zero count: %d / %d", (df["ML_P_Model"] == 0).sum(), len(df))

    # ROI baseline
    total_stake = df["USD"].sum()
    total_payout = df["Payout_USD"].sum()
    roi_all = (total_payout - total_stake) / total_stake * 100
    logger.info("Baseline ROI (all bets): %.2f%%", roi_all)
    logger.info("Total stake: $%.2f, total payout: $%.2f", total_stake, total_payout)

    # Odds distribution
    logger.info("Odds stats:\n%s", df["Odds"].describe())

    # Payout_USD for won
    won = df[df["target"] == 1]
    lost = df[df["target"] == 0]
    logger.info(
        "Won: %d (%.1f%%), Lost: %d (%.1f%%)",
        len(won),
        len(won) / len(df) * 100,
        len(lost),
        len(lost) / len(df) * 100,
    )
    logger.info("Avg USD stake: $%.2f", df["USD"].mean())
    logger.info("Median USD stake: $%.2f", df["USD"].median())

    # Check for future leakage columns
    logger.info("Columns in bets: %s", list(bets.columns))
    logger.info("Payout_USD non-zero for lost: %d", (lost["Payout_USD"] > 0).sum())

    # Null check
    logger.info(
        "Nulls in key columns:\n%s",
        df[ml_cols + ["Odds", "USD", "Sport", "Market"]].isnull().sum(),
    )

    # Time series split preview
    df_sorted = df.sort_values("Created_At")
    n = len(df_sorted)
    test_size = int(n * 0.2)
    train = df_sorted.iloc[:-test_size]
    test = df_sorted.iloc[-test_size:]
    logger.info(
        "Train: %d rows [%s to %s], Test: %d rows [%s to %s]",
        len(train),
        train["Created_At"].min(),
        train["Created_At"].max(),
        len(test),
        test["Created_At"].min(),
        test["Created_At"].max(),
    )
    logger.info(
        "Train target mean: %.4f, Test target mean: %.4f",
        train["target"].mean(),
        test["target"].mean(),
    )

    # ROI on test if we bet on everything
    test_roi = (test["Payout_USD"].sum() - test["USD"].sum()) / test["USD"].sum() * 100
    logger.info("Test ROI (all bets): %.2f%%", test_roi)


if __name__ == "__main__":
    main()
