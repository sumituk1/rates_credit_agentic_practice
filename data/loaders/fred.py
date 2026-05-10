import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()


def get_fred_client() -> Fred:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set. Copy .env.example to .env and add your key.")
    return Fred(api_key=api_key)


def load_us_yields(start: str = "2010-01-01") -> pd.DataFrame:
    """Load US Treasury yields from FRED: 2y, 5y, 10y, 30y."""
    fred = get_fred_client()
    series = {
        "us_2y": "DGS2",
        "us_5y": "DGS5",
        "us_10y": "DGS10",
        "us_30y": "DGS30",
    }
    data = {name: fred.get_series(code, observation_start=start) for name, code in series.items()}
    df = pd.DataFrame(data)
    df.index.name = "date"
    df = df.dropna(how="all")
    return df


def load_fred_series(series_id: str, start: str = "2010-01-01") -> pd.Series:
    """Load an arbitrary FRED series by ID."""
    fred = get_fred_client()
    s = fred.get_series(series_id, observation_start=start)
    s.index.name = "date"
    s.name = series_id
    return s.dropna()