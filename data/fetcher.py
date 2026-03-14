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
from data.cache import DataCache


class DataFetcher:
    """Fetch and cache stock price data.

    Uses SQLite cache by default (data/cache/market_data.db).
    Falls back to parquet files if SQLite cache is not available.
    """

    def __init__(self, cache_dir: str = DATA_DIR, use_sqlite: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_sqlite = use_sqlite
        self._db_cache = DataCache() if use_sqlite else None

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

        # Try SQLite cache first
        if use_cache and self._db_cache is not None:
            cached = self._db_cache.get(ticker, period, interval)
            if cached is not None:
                self._db_cache.log_fetch(ticker, period, "sqlite", 0, len(cached), cache_hit=True)
                return cached

        # Fall back to parquet cache
        cache_path = self._cache_path(ticker, period, interval)
        if use_cache and self._is_fresh(cache_path):
            df = pd.read_parquet(cache_path)
            # Migrate to SQLite if available
            if self._db_cache is not None:
                self._db_cache.set(ticker, period, interval, df)
            return df

        t0 = time.time()
        print(f"  Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            if self._db_cache is not None:
                self._db_cache.log_fetch(ticker, period, "yfinance", int((time.time()-t0)*1000), 0, error="empty")
            return df

        # Clean up
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        for col in ["Dividends", "Stock Splits"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        duration_ms = int((time.time() - t0) * 1000)

        # Store in SQLite cache
        if self._db_cache is not None:
            self._db_cache.set(ticker, period, interval, df)
            self._db_cache.log_fetch(ticker, period, "yfinance", duration_ms, len(df))
        else:
            # Fall back to parquet
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
