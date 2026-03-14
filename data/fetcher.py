"""Data fetcher — download and cache price data from Yahoo Finance."""

import os
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import DATA_DIR, DEFAULT_PERIOD, DEFAULT_INTERVAL


class DataFetcher:
    """Fetch and cache stock price data."""

    def __init__(self, cache_dir: str = DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, ticker: str, period: str, interval: str) -> Path:
        key = hashlib.md5(f"{ticker}_{period}_{interval}".encode()).hexdigest()[:12]
        return self.cache_dir / f"{ticker}_{key}.parquet"

    def _is_fresh(self, path: Path, max_age_hours: int = 6) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < max_age_hours * 3600

    def fetch(
        self,
        ticker: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker."""
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        cache_path = self._cache_path(ticker, period, interval)
        if use_cache and self._is_fresh(cache_path):
            return pd.read_parquet(cache_path)

        print(f"  Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            return df

        # Clean up
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        for col in ["Dividends", "Stock Splits"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Cache
        df.to_parquet(cache_path)
        return df

    def fetch_multiple(
        self,
        tickers: list[str],
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers. Returns {ticker: DataFrame}."""
        results = {}
        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{total}] {ticker}")
            try:
                df = self.fetch(ticker, period, interval, use_cache)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                print(f"  ERROR: {e}")
            # Be nice to Yahoo Finance
            if not use_cache:
                time.sleep(0.5)
        return results

    def fetch_ndx(
        self,
        period: str = DEFAULT_PERIOD,
        use_cache: bool = True,
        sector: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch all NDX-100 constituents."""
        from data import get_tickers
        tickers = get_tickers(sector)
        print(f"Fetching {len(tickers)} NDX tickers (period={period})")
        return self.fetch_multiple(tickers, period, use_cache=use_cache)

    def get_benchmark(self, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
        """Fetch QQQ (Nasdaq-100 ETF) as benchmark."""
        return self.fetch("QQQ", period=period)
