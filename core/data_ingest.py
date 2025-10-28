#!/usr/bin/env python3
"""
Enhanced Data Ingestion Module (YFinance-only)

- Uses yfinance for equities/crypto/commodities.
- Robust caching to parquet when pyarrow/fastparquet is available,
  otherwise falls back to CSV.
- Retry/backoff and rate limiting for Yahoo requests.
- Normalizes DataFrames to a uniform schema.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("data_ingest")

# -------------------------
# Parquet engine detection
# -------------------------
PARQUET_ENGINE: Optional[str] = None
try:
    import pyarrow  # noqa
    PARQUET_ENGINE = "pyarrow"
    logger.info("Parquet engine available: pyarrow")
except Exception:
    try:
        import fastparquet  # noqa
        PARQUET_ENGINE = "fastparquet"
        logger.info("Parquet engine available: fastparquet")
    except Exception:
        PARQUET_ENGINE = None
        logger.info("No parquet engine found; will fall back to CSV for caching")

# -------------------------
# Known delisted / replaced tickers (extend as needed)
# -------------------------
KNOWN_DELISTED = {
    "HDFC.NS",  # HDFC merged into HDFC Bank and HDFC.NS was delisted (July 2023)
}

# -------------------------
# Symbol Mapper
# -------------------------
class SymbolMapper:
    """Maps common symbols to Yahoo Finance formats and detects asset types."""

    CRYPTO_MAPPING = {
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "BNB/USD": "BNB-USD",
        "XRP/USD": "XRP-USD",
        "ADA/USD": "ADA-USD",
        "DOGE/USD": "DOGE-USD",
        "SOL/USD": "SOL-USD",
        "MATIC/USD": "MATIC-USD",
    }

    COMMODITIES_MAPPING = {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "CRUDE_OIL": "CL=F",
        "NATURAL_GAS": "NG=F",
        "COPPER": "HG=F",
        "XAU/USD": "GC=F",
        "XAG/USD": "SI=F",
    }

    @classmethod
    def to_yahoo_format(cls, symbol: str) -> str:
        sym = symbol.upper().strip()
        # direct mapping first
        if sym in cls.CRYPTO_MAPPING:
            return cls.CRYPTO_MAPPING[sym]
        if sym in cls.COMMODITIES_MAPPING:
            return cls.COMMODITIES_MAPPING[sym]
        # Already a Yahoo-style symbol? Return as-is
        return sym

    @classmethod
    def detect_asset_type(cls, symbol: str) -> str:
        s = symbol.upper().strip()
        # crypto: explicit mapping keys or common suffixes
        if s in cls.CRYPTO_MAPPING or "-USD" in s or "/USD" in s:
            return "crypto"
        # commodity: mapping keys or yahoo futures tokens (contains '=')
        if s in cls.COMMODITIES_MAPPING or "=" in s or s.startswith("GC") or s.startswith("SI") or "XAU" in s or "XAG" in s:
            return "commodity"
        return "equity"

# -------------------------
# DataIngestion
# -------------------------
class DataIngestion:
    """Data ingestion using yfinance with caching and normalization."""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting per source
        self.last_request_time: Dict[str, float] = {}
        self.min_request_interval: Dict[str, float] = {"yahoo": 0.5}

    def _wait_for_rate_limit(self, source: str):
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            required = self.min_request_interval.get(source, 1.0)
            if elapsed < required:
                to_wait = required - elapsed
                logger.debug("Rate limit: sleeping %.3fs for source %s", to_wait, source)
                time.sleep(to_wait)
        self.last_request_time[source] = time.time()

    @staticmethod
    def _sanitize_symbol_for_filename(symbol: str) -> str:
        """Make a simple filesystem-safe filename for a symbol."""
        s = symbol.upper().strip()
        # allow alnum, dot, hyphen, equals; replace others with underscore
        return re.sub(r"[^A-Z0-9\.\-=]", "_", s)

    def _normalize_dataframe(self, df: pd.DataFrame, original_symbol: str, source: str) -> Optional[pd.DataFrame]:
        """Normalize dataframe: lower-case columns, add symbol/source/fetch_timestamp/date."""
        if df is None or df.empty:
            return None

        # Ensure it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                return None

        df = df.copy()

        # Standardize common column names mapping
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Adjusted Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=column_mapping)
        # lower-case remaining columns
        df.columns = [c.lower() for c in df.columns]

        # If index is DatetimeIndex and 'date' not present -> reset index
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            # some dataframes come with 'date' or 'Date'
            elif "date" not in df.columns and "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})

        # Ensure 'date' column exists
        if "date" not in df.columns:
            # attempt to find a datetime-like column
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df = df.rename(columns={c: "date"})
                    break

        # Try to coerce date to datetime
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
            except Exception:
                # if conversion fails, drop the column — risky but prevents crash
                logger.warning("Could not parse 'date' column for %s; dropping it", original_symbol)
                df = df.drop(columns=["date"], errors="ignore")

        # Add adj_close if missing
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        # Add symbol/source/fetch_timestamp
        df["symbol"] = original_symbol.upper().strip()
        df["source"] = source
        df["fetch_timestamp"] = pd.Timestamp(datetime.now())

        # Reorder columns if date exists
        cols = list(df.columns)
        if "date" in cols:
            # Move date to first column
            cols.insert(0, cols.pop(cols.index("date")))
            df = df[cols]

        return df

    # -------------------------
    # Low-level Yahoo fetch
    # -------------------------
    def fetch_yahoo_finance(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a symbol from Yahoo Finance with retries and backoff."""
        yahoo_symbol = SymbolMapper.to_yahoo_format(symbol)
        yahoo_symbol = yahoo_symbol.upper().strip()

        if yahoo_symbol in KNOWN_DELISTED:
            logger.warning("Symbol %s is known delisted/replaced. Skipping fetch.", yahoo_symbol)
            return None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Fetching %s from Yahoo Finance (attempt %d/%d)", yahoo_symbol, attempt, max_retries)
                self._wait_for_rate_limit("yahoo")

                ticker = yf.Ticker(yahoo_symbol)
                # Use history which is robust for a single symbol
                df = ticker.history(period=period, interval=interval, auto_adjust=False)

                if df is None or getattr(df, "empty", True):
                    logger.warning("No data returned for %s (attempt %d/%d)", yahoo_symbol, attempt, max_retries)
                    if attempt < max_retries:
                        wait_time = attempt * 6
                        logger.info("Waiting %ds before retrying %s", wait_time, yahoo_symbol)
                        time.sleep(wait_time)
                    continue

                # Reset index and normalize
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                normalized = self._normalize_dataframe(df, symbol, "yahoo_finance")
                if normalized is not None:
                    logger.info("Fetched %d rows for %s", len(normalized), yahoo_symbol)
                    # Save to cache
                    try:
                        self._save_to_cache(symbol, normalized)
                    except Exception as e:
                        logger.warning("Could not cache %s: %s", symbol, e)
                    return normalized
                else:
                    logger.warning("Normalization produced empty result for %s", yahoo_symbol)
                    return None

            except Exception as e:
                logger.exception("Yahoo Finance error for %s: %s", yahoo_symbol, e)
                if attempt < max_retries:
                    wait_time = attempt * 6
                    logger.info("Waiting %ds before retrying %s after exception", wait_time, yahoo_symbol)
                    time.sleep(wait_time)

        logger.error("Failed to fetch %s after %d attempts", yahoo_symbol, max_retries)
        return None

    # -------------------------
    # High-level asset fetchers
    # -------------------------
    def fetch_crypto_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        yahoo_sym = SymbolMapper.to_yahoo_format(symbol)
        df = self.fetch_yahoo_finance(yahoo_sym, period=period, interval=interval)
        # Label source specifically
        if df is not None:
            df["source"] = "yahoo_crypto"
        return df

    def fetch_commodity_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        yahoo_sym = SymbolMapper.to_yahoo_format(symbol)
        df = self.fetch_yahoo_finance(yahoo_sym, period=period, interval=interval)
        if df is not None:
            df["source"] = "yahoo_commodity"
        return df

    def fetch_auto(self, symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        asset_type = SymbolMapper.detect_asset_type(symbol)
        logger.info("Detected %s as %s", symbol, asset_type)
        if asset_type == "crypto":
            return self.fetch_crypto_data(symbol, period, interval)
        elif asset_type == "commodity":
            return self.fetch_commodity_data(symbol, period, interval)
        else:
            return self.fetch_yahoo_finance(symbol, period, interval)

    # -------------------------
    # Multi-symbol fetch with caching
    # -------------------------
    def _cache_paths_for_symbol(self, symbol: str) -> Dict[str, Path]:
        safe = self._sanitize_symbol_for_filename(symbol)
        base = self.cache_dir / safe
        return {
            "parquet": base.with_suffix(".parquet"),
            "csv": base.with_suffix(".csv"),
        }

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        paths = self._cache_paths_for_symbol(symbol)
        # prefer parquet if available and engine exists
        if PARQUET_ENGINE and paths["parquet"].exists():
            try:
                df = pd.read_parquet(paths["parquet"], engine=PARQUET_ENGINE)
                logger.info("Loaded %s from parquet cache", symbol)
                return df
            except Exception as e:
                logger.warning("Failed to read parquet cache for %s: %s", symbol, e)
                # fall through to csv

        if paths["csv"].exists():
            try:
                df = pd.read_csv(paths["csv"], parse_dates=["date"], infer_datetime_format=True)
                logger.info("Loaded %s from CSV cache", symbol)
                return df
            except Exception as e:
                logger.warning("Failed to read CSV cache for %s: %s", symbol, e)

        return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        paths = self._cache_paths_for_symbol(symbol)
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Try parquet first if available
        if PARQUET_ENGINE:
            try:
                df.to_parquet(paths["parquet"], engine=PARQUET_ENGINE, index=False)
                logger.info("Saved %s to parquet cache", symbol)
                return
            except Exception as e:
                logger.warning("Failed to write parquet for %s: %s", symbol, e)

        # Fallback to CSV
        try:
            df.to_csv(paths["csv"], index=False)
            logger.info("Saved %s to CSV cache", symbol)
        except Exception as e:
            logger.exception("Failed to write CSV cache for %s: %s", symbol, e)

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        period: str = "6mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            sym_upper = symbol.upper().strip()
            # Try to load from cache
            if not force_refresh:
                cached = self._load_from_cache(sym_upper)
                if cached is not None and not cached.empty:
                    results[sym_upper] = cached
                    continue

            # Fetch from Yahoo
            df = self.fetch_auto(sym_upper, period=period, interval=interval)
            if df is None or df.empty:
                logger.warning("No usable data for %s; skipping", sym_upper)
                continue

            # Save to cache (parquet if possible, else CSV)
            try:
                self._save_to_cache(sym_upper, df)
            except Exception as e:
                logger.warning("Could not cache %s: %s", sym_upper, e)

            results[sym_upper] = df

        return results


# -------------------------
# Module test / example usage
# -------------------------
if __name__ == "__main__":
    ingestion = DataIngestion()

    test_symbols = [
        "AAPL",
        "BTC/USD",
        "ETH/USD",
        "XAU/USD",
        "XAG/USD",
    ]

    print("Testing Enhanced Data Ingestion")
    print("=" * 60)

    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        df = ingestion.fetch_auto(symbol, period="1mo", interval="1d")
        if df is not None and not df.empty:
            print(f"Success: {len(df)} rows fetched")
            print(f"  Source: {df['source'].iloc[0] if 'source' in df.columns else 'unknown'}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        else:
            print(f"Failed to fetch {symbol}")

    print("\n" + "=" * 60)
    print("Testing complete!")
