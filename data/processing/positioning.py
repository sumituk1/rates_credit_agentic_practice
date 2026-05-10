import pandas as pd
from data.processing.common import rolling_zscore


def net_speculative_positioning(long_contracts: pd.Series, short_contracts: pd.Series) -> pd.Series:
    """Net speculative positioning = longs - shorts (COT report)."""
    return (long_contracts - short_contracts).rename("net_spec_positioning")


def positioning_zscore(net_position: pd.Series, window: int = 52) -> pd.Series:
    """Rolling z-score of net positioning; 52-week window suits weekly COT frequency."""
    return rolling_zscore(net_position, window=window).rename("positioning_zscore")