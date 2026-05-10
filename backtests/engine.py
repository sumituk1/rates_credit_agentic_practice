import pandas as pd


def run_simple_backtest(
    signal: pd.Series,
    returns: pd.Series,
    threshold: float = 1.0,
    direction: int = 1,
    transaction_cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Vectorised long-only backtest.

    Positions are determined the day AFTER the signal exceeds threshold
    (signal.shift(1)) to prevent lookahead bias.
    """
    positions = (signal > threshold).astype(float) * direction
    positions = positions.shift(1).fillna(0.0)

    gross_returns = positions * returns

    turnover = positions.diff().abs().fillna(0.0)
    costs = turnover * transaction_cost_bps / 10_000

    net_returns = gross_returns - costs

    return pd.DataFrame(
        {
            "signal": signal,
            "returns": returns,
            "position": positions,
            "gross_returns": gross_returns,
            "costs": costs,
            "net_returns": net_returns,
            "turnover": turnover,
        }
    ).dropna()


def run_long_short_backtest(
    signal: pd.Series,
    returns: pd.Series,
    long_threshold: float = 0.5,
    short_threshold: float = -0.5,
    transaction_cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Long when signal > long_threshold, short when signal < short_threshold."""
    positions = pd.Series(0.0, index=signal.index)
    positions[signal > long_threshold] = 1.0
    positions[signal < short_threshold] = -1.0
    positions = positions.shift(1).fillna(0.0)

    gross_returns = positions * returns
    turnover = positions.diff().abs().fillna(0.0)
    costs = turnover * transaction_cost_bps / 10_000
    net_returns = gross_returns - costs

    return pd.DataFrame(
        {
            "signal": signal,
            "returns": returns,
            "position": positions,
            "gross_returns": gross_returns,
            "costs": costs,
            "net_returns": net_returns,
            "turnover": turnover,
        }
    ).dropna()