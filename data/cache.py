"""SQLite-based data cache layer.

Replaces ad-hoc parquet file caching with a proper SQLite database.
Supports multiple tickers, periods, and automatic cache invalidation.
"""

from __future__ import annotations

import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import numpy as np

from config import DATA_DIR

DB_PATH = Path(DATA_DIR) / "market_data.db"

# Cache expiry: re-fetch if older than this
DEFAULT_CACHE_HOURS = 4  # Market data is stale after 4h during trading days
WEEKEND_CACHE_HOURS = 48  # On weekends, 48h is fine


class DataCache:
    """
    SQLite-backed cache for market OHLCV data.

    Schema:
        ticker_data: stores OHLCV as JSON blobs keyed by (ticker, period, interval)
        metadata: tracks fetch timestamps and data quality

    Usage:
        cache = DataCache()
        # Try cache first
        df = cache.get("QQQ", "2y", "1d")
        if df is None:
            df = fetch_from_api(...)
            cache.set("QQQ", "2y", "1d", df)
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    period TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    date_start TEXT NOT NULL,
                    date_end TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    cache_key TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_period
                ON ticker_data(ticker, period, interval)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key
                ON ticker_data(cache_key)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fetched_at
                ON ticker_data(fetched_at)
            """)
            # Metadata table for tracking fetch stats
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    period TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fetch_time TEXT NOT NULL,
                    duration_ms INTEGER,
                    row_count INTEGER,
                    cache_hit INTEGER DEFAULT 0,
                    error TEXT
                )
            """)

    @contextmanager
    def _connect(self):
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _make_cache_key(ticker: str, period: str, interval: str) -> str:
        """Generate a unique cache key."""
        raw = f"{ticker.upper()}:{period}:{interval}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _df_to_json(df: pd.DataFrame) -> str:
        """Serialize DataFrame to JSON string."""
        # Convert index to column for JSON serialization
        df_copy = df.copy()
        df_copy.index.name = "Date"
        df_copy = df_copy.reset_index()
        # Convert dates to ISO format
        if "Date" in df_copy.columns:
            df_copy["Date"] = df_copy["Date"].astype(str)
        # Convert numpy types to Python types
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            df_copy[col] = df_copy[col].astype(float)
        return df_copy.to_json(orient="split", date_format="iso")

    @staticmethod
    def _json_to_df(json_str: str) -> pd.DataFrame:
        """Deserialize DataFrame from JSON string."""
        df = pd.read_json(json_str, orient="split")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        return df

    def _is_stale(self, fetched_at_str: str) -> bool:
        """Check if cached data is stale based on fetch time."""
        fetched_at = datetime.fromisoformat(fetched_at_str)
        now = datetime.now()

        # Determine expiry based on day of week
        weekday = now.weekday()  # 0=Mon, 6=Sun
        if weekday >= 5:  # Weekend
            expiry = timedelta(hours=WEEKEND_CACHE_HOURS)
        else:
            expiry = timedelta(hours=DEFAULT_CACHE_HOURS)

        return (now - fetched_at) > expiry

    def get(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        max_age_hours: int | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data for a ticker.

        Returns None if cache miss or data is stale.
        """
        cache_key = self._make_cache_key(ticker, period, interval)

        with self._connect() as conn:
            row = conn.execute(
                """SELECT data_json, fetched_at, row_count, date_start, date_end
                   FROM ticker_data WHERE cache_key = ?""",
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        data_json, fetched_at, row_count, date_start, date_end = row

        # Check staleness
        if max_age_hours is not None:
            fetched_dt = datetime.fromisoformat(fetched_at)
            if (datetime.now() - fetched_dt) > timedelta(hours=max_age_hours):
                return None
        elif self._is_stale(fetched_at):
            return None

        try:
            df = self._json_to_df(data_json)
            return df
        except Exception:
            return None

    def set(
        self,
        ticker: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
    ) -> int:
        """
        Store data in cache. Returns the number of bytes stored.
        """
        cache_key = self._make_cache_key(ticker, period, interval)
        data_json = self._df_to_json(df)
        size_bytes = len(data_json.encode("utf-8"))

        date_start = str(df.index.min().date()) if not df.empty else ""
        date_end = str(df.index.max().date()) if not df.empty else ""
        now = datetime.now().isoformat()

        with self._connect() as conn:
            # Upsert: replace if exists
            conn.execute(
                """INSERT OR REPLACE INTO ticker_data
                   (ticker, period, interval, data_json, row_count,
                    date_start, date_end, fetched_at, cache_key, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker.upper(), period, interval, data_json, len(df),
                    date_start, date_end, now, cache_key, size_bytes,
                ),
            )

        return size_bytes

    def invalidate(self, ticker: str, period: str | None = None) -> int:
        """Remove cached data. Returns number of rows deleted."""
        with self._connect() as conn:
            if period:
                result = conn.execute(
                    "DELETE FROM ticker_data WHERE ticker = ? AND period = ?",
                    (ticker.upper(), period),
                )
            else:
                result = conn.execute(
                    "DELETE FROM ticker_data WHERE ticker = ?",
                    (ticker.upper(),),
                )
            return result.rowcount

    def clear_all(self) -> int:
        """Clear entire cache. Returns number of rows deleted."""
        with self._connect() as conn:
            result = conn.execute("DELETE FROM ticker_data")
            return result.rowcount

    def log_fetch(
        self,
        ticker: str,
        period: str,
        source: str,
        duration_ms: int,
        row_count: int,
        cache_hit: bool = False,
        error: str | None = None,
    ) -> None:
        """Log a fetch operation for monitoring."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO fetch_log
                   (ticker, period, source, fetch_time, duration_ms, row_count, cache_hit, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker.upper(), period, source, datetime.now().isoformat(),
                    duration_ms, row_count, int(cache_hit), error,
                ),
            )

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM ticker_data").fetchone()[0]
            total_size = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM ticker_data").fetchone()[0]
            tickers = conn.execute("SELECT DISTINCT ticker FROM ticker_data").fetchall()
            oldest = conn.execute("SELECT MIN(fetched_at) FROM ticker_data").fetchone()[0]
            newest = conn.execute("SELECT MAX(fetched_at) FROM ticker_data").fetchone()[0]
            fetch_count = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
            cache_hits = conn.execute("SELECT COUNT(*) FROM fetch_log WHERE cache_hit = 1").fetchone()[0]

        return {
            "entries": total,
            "size_mb": total_size / (1024 * 1024),
            "tickers": [t[0] for t in tickers],
            "oldest_fetch": oldest,
            "newest_fetch": newest,
            "total_fetches": fetch_count,
            "cache_hits": cache_hits,
            "hit_rate": f"{cache_hits / fetch_count:.1%}" if fetch_count > 0 else "N/A",
            "db_path": str(self.db_path),
        }

    def format_stats(self) -> str:
        """Format stats as readable string."""
        s = self.stats()
        lines = [
            "Data Cache Statistics",
            "=" * 40,
            f"  Entries:      {s['entries']}",
            f"  Size:         {s['size_mb']:.2f} MB",
            f"  Tickers:      {', '.join(s['tickers']) if s['tickers'] else '(empty)'}",
            f"  Oldest fetch: {s['oldest_fetch'] or 'N/A'}",
            f"  Newest fetch: {s['newest_fetch'] or 'N/A'}",
            f"  Total fetches: {s['total_fetches']}",
            f"  Cache hits:   {s['cache_hits']}",
            f"  Hit rate:     {s['hit_rate']}",
            f"  DB path:      {s['db_path']}",
        ]
        return "\n".join(lines)

    def compact(self) -> int:
        """Remove stale entries. Returns count of removed entries."""
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                "DELETE FROM ticker_data WHERE fetched_at < ?",
                (cutoff,),
            )
            # Also compact the database
            conn.execute("VACUUM")
            return result.rowcount
