import pandas as pd
from data.processing.common import rolling_zscore


def compute_rate_differential(
    domestic_rate: pd.Series,
    foreign_rate: pd.Series,
) -> pd.Series:
    """Daily interest rate differential (domestic minus foreign, annualised pct)."""
    return (domestic_rate - foreign_rate).rename("rate_differential")


def fx_carry_signal(
    domestic_rate: pd.Series,
    foreign_rate: pd.Series,
    zscore_window: int = 60,
) -> pd.Series:
    """Rolling z-score of rate differential — primary FX carry signal."""
    diff = compute_rate_differential(domestic_rate, foreign_rate)
    return rolling_zscore(diff, window=zscore_window).rename("fx_carry_zscore")