import pandas as pd
from backtests.metrics import sharpe_ratio, max_drawdown


def sub_period_check(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Split history at midpoint; report Sharpe for each half."""
    mid = len(returns) // 2
    first_half = returns.iloc[:mid]
    second_half = returns.iloc[mid:]
    return {
        "first_half_sharpe": sharpe_ratio(first_half, periods_per_year),
        "second_half_sharpe": sharpe_ratio(second_half, periods_per_year),
        "first_half_drawdown": max_drawdown(first_half),
        "second_half_drawdown": max_drawdown(second_half),
    }


def walk_forward_sharpe(
    signal: pd.Series,
    returns: pd.Series,
    train_window: int = 504,
    test_window: int = 63,
    threshold: float = 1.0,
    transaction_cost_bps: float = 1.0,
) -> pd.Series:
    """Rolling walk-forward Sharpe over out-of-sample test windows."""
    from backtests.engine import run_simple_backtest

    oos_returns = []
    idx = train_window
    while idx + test_window <= len(signal):
        test_sig = signal.iloc[idx : idx + test_window]
        test_ret = returns.iloc[idx : idx + test_window]
        bt = run_simple_backtest(test_sig, test_ret, threshold=threshold,
                                  transaction_cost_bps=transaction_cost_bps)
        oos_returns.append(bt["net_returns"])
        idx += test_window

    if not oos_returns:
        return pd.Series(dtype=float)

    combined = pd.concat(oos_returns)
    return combined


def has_minimum_history(returns: pd.Series, min_days: int = 252) -> bool:
    return len(returns.dropna()) >= min_days