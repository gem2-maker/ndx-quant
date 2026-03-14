"""Technical analysis indicators."""

import pandas as pd
import numpy as np

from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, SMA_SHORT, SMA_LONG, EMA_SHORT, EMA_LONG,
)


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD — returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = BB_PERIOD,
    std_dev: float = BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands — returns (upper, middle, lower)."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def volume_weighted_average_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """VWAP — Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a DataFrame with OHLCV columns."""
    df = df.copy()

    # Moving averages
    df["SMA_20"] = sma(df["Close"], SMA_SHORT)
    df["SMA_50"] = sma(df["Close"], SMA_LONG)
    df["EMA_12"] = ema(df["Close"], EMA_SHORT)
    df["EMA_26"] = ema(df["Close"], EMA_LONG)

    # RSI
    df["RSI"] = rsi(df["Close"])

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])

    # Bollinger Bands
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = bollinger_bands(df["Close"])

    # ATR
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        df["ATR"] = atr(df["High"], df["Low"], df["Close"])

    # VWAP
    if all(c in df.columns for c in ["High", "Low", "Close", "Volume"]):
        df["VWAP"] = volume_weighted_average_price(
            df["High"], df["Low"], df["Close"], df["Volume"]
        )

    # Returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    return df
