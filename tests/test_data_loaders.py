import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from data.processing.yield_curve import add_curve_features
from data.processing.fx_carry import fx_carry_signal
from data.processing.common import rolling_zscore


def _mock_yields() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-01", periods=300, freq="B")
    return pd.DataFrame(
        {
            "us_2y": 1.5 + rng.normal(0, 0.1, 300).cumsum() * 0.01,
            "us_5y": 2.0 + rng.normal(0, 0.1, 300).cumsum() * 0.01,
            "us_10y": 2.5 + rng.normal(0, 0.1, 300).cumsum() * 0.01,
            "us_30y": 3.0 + rng.normal(0, 0.1, 300).cumsum() * 0.01,
        },
        index=idx,
    )


def test_add_curve_features_columns():
    df = _mock_yields()
    out = add_curve_features(df)
    assert "us_2s10s" in out.columns
    assert "us_5s30s" in out.columns
    assert "us_2s10s_zscore" in out.columns
    assert "us_5s30s_zscore" in out.columns


def test_add_curve_features_no_lookahead():
    """2s10s spread is a simple difference — no future information."""
    df = _mock_yields()
    out = add_curve_features(df)
    expected = df["us_10y"] - df["us_2y"]
    pd.testing.assert_series_equal(out["us_2s10s"], expected, check_names=False)


def test_rolling_zscore_length():
    s = pd.Series(range(200), dtype=float)
    z = rolling_zscore(s, window=60)
    assert len(z) == len(s)


def test_fx_carry_signal_shape():
    import numpy as np
    idx = pd.date_range("2015-01-01", periods=200, freq="B")
    dom = pd.Series(2.5 + np.random.default_rng(2).normal(0, 0.05, 200), index=idx)
    fgn = pd.Series(0.5 + np.random.default_rng(3).normal(0, 0.05, 200), index=idx)
    sig = fx_carry_signal(dom, fgn, zscore_window=40)
    assert len(sig) == 200


def test_fred_client_raises_without_key():
    """FRED client must raise a clear error when FRED_API_KEY is missing."""
    import os
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("FRED_API_KEY", None)
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            from data.loaders.fred import get_fred_client
            get_fred_client()