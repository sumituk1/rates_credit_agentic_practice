import numpy as np
import pandas as pd
import pytest

from backtests.metrics import sharpe_ratio, max_drawdown, annualized_return
from backtests.engine import run_simple_backtest, run_long_short_backtest


def _constant_returns(value: float, n: int = 500) -> pd.Series:
    return pd.Series([value] * n)


def test_sharpe_positive_returns():
    r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 500))
    assert sharpe_ratio(r) > 0


def test_sharpe_zero_std():
    assert sharpe_ratio(_constant_returns(0.0)) == 0.0


def test_max_drawdown_flat():
    assert max_drawdown(_constant_returns(0.001)) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_negative():
    r = pd.Series([-0.05] * 10 + [0.01] * 100)
    assert max_drawdown(r) < 0


def test_annualized_return_positive():
    r = _constant_returns(0.001, n=252)
    ann = annualized_return(r)
    assert ann > 0


def test_run_simple_backtest_no_lookahead():
    """Position on day t must be based on signal from day t-1."""
    signal = pd.Series([2.0] * 10, index=range(10))
    returns = pd.Series([0.01] * 10, index=range(10))
    bt = run_simple_backtest(signal, returns, threshold=1.0)
    assert bt["position"].iloc[0] == 0.0, "First position must be 0 (signal shifted)"


def test_run_long_short_backtest_shape():
    rng = np.random.default_rng(0)
    signal = pd.Series(rng.standard_normal(300))
    returns = pd.Series(rng.normal(0.001, 0.01, 300))
    bt = run_long_short_backtest(signal, returns)
    assert set(bt.columns) >= {"signal", "returns", "position", "net_returns", "turnover"}
    assert len(bt) > 0