import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns.fillna(0)).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return float(cumulative ** (1 / years) - 1)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def summarise(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Full performance summary dict."""
    return {
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year),
        "n_obs": int(returns.dropna().__len__()),
    }