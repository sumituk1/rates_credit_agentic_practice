import pandas as pd
from data.processing.common import rolling_zscore


def add_curve_features(df: pd.DataFrame, zscore_window: int = 60) -> pd.DataFrame:
    """Add 2s10s and 5s30s spreads, 5-day momentum, and rolling z-scores."""
    out = df.copy()

    if "us_2y" in out.columns and "us_10y" in out.columns:
        out["us_2s10s"] = out["us_10y"] - out["us_2y"]
        out["us_2s10s_change_5d"] = out["us_2s10s"].diff(5)
        out["us_2s10s_zscore"] = rolling_zscore(out["us_2s10s"], window=zscore_window)

    if "us_5y" in out.columns and "us_30y" in out.columns:
        out["us_5s30s"] = out["us_30y"] - out["us_5y"]
        out["us_5s30s_change_5d"] = out["us_5s30s"].diff(5)
        out["us_5s30s_zscore"] = rolling_zscore(out["us_5s30s"], window=zscore_window)

    return out


def yield_to_approx_returns(yields: pd.Series, duration_years: float) -> pd.Series:
    """Approximate bond returns from yield changes: ret ≈ -duration * Δyield / 100."""
    dy = yields.diff()
    return (-duration_years * dy / 100).rename(f"approx_return_dur{duration_years}")