from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class LoadedData:
    """Container for loaded price data plus basic metadata."""

    data: pd.DataFrame
    source: str
    start: pd.Timestamp
    end: pd.Timestamp


def load_csv_prices(path: Path | str, timezone: str = "UTC") -> LoadedData:
    """
    Load OHLCV candles from a CSV file into a standardized DataFrame.

    - Supports a "Local time" column with timezone offsets (e.g., TradingView exports).
    - Normalizes column names to: timestamp, open, high, low, close, volume.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "Local time" in df.columns:
        ts = pd.to_datetime(df["Local time"], dayfirst=True, errors="coerce")
        ts = ts.dt.tz_localize("America/New_York")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("CSV must contain either 'Local time' or 'timestamp' column.")

    # Drop rows that failed to parse to avoid .dt errors downstream.
    bad = ts.isna()
    if bad.any():
        df = df.loc[~bad].copy()
        ts = ts.loc[~bad]

    # Normalize timezone to UTC for consistent joins/backtests.
    ts = ts.dt.tz_convert(timezone)

    df = df.rename(
        columns={
            "Local time": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["timestamp"] = ts

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )
    df = df.tz_convert("UTC")  # store internally as UTC

    return LoadedData(data=df, source=str(path), start=df.index[0], end=df.index[-1])


def resample_bars(
    df: pd.DataFrame,
    timeframe: str,
    how: Optional[str] = None,
) -> pd.DataFrame:
    """
    Resample bar data to a different timeframe.

    Parameters
    ----------
    df : DataFrame with UTC DateTimeIndex and ohlcv columns.
    timeframe : pandas offset alias (e.g., '1T', '5T', '1H', '1D').
    how : optional label to store in debug/logs.
    """
    if not set(REQUIRED_COLUMNS[1:]).issubset(df.columns):
        raise ValueError("DataFrame must contain open, high, low, close, volume columns.")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(timeframe, label="right", closed="right").agg(agg).dropna()
    resampled.index.name = "timestamp"
    return resampled
