import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score with minimum period = window // 2 to avoid NaN-heavy head."""
    min_periods = max(window // 2, 1)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, float("nan"))