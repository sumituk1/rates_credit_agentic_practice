import pandas as pd
import yfinance as yf


def load_yahoo_series(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Download OHLCV data for a ticker from Yahoo Finance."""
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"Yahoo Finance returned no data for ticker '{ticker}' (start={start}). May be rate-limited or ticker invalid.")
    data.index.name = "date"
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def load_fx_returns(ticker: str, start: str = "2010-01-01") -> pd.Series:
    """Return daily log returns for an FX ticker (e.g. 'EURUSD=X')."""
    import numpy as np
    df = load_yahoo_series(ticker, start=start)
    close = df["Close"].dropna()
    returns = np.log(close / close.shift(1)).dropna()
    returns.name = ticker
    return returns


FX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
}