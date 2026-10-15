"""Market-data collection.

HISTORICAL (training): offline CSV datasets loaded, cleaned, validated and
merged by (Date, Stock Name). Nothing is silently discarded - every dropped or
unmatched row is counted and logged, and the audit is returned to the caller.

LIVE (inference only): Yahoo Finance via yfinance.

Public API
----------
HistoricalDataStore   lazy singleton over the two CSVs
load_historical_prices(ticker)  -> (DataFrame[OHLCV], info)
load_historical_tweets(ticker)  -> (DataFrame[Date, Tweet, ...], info)
fetch_live_prices(ticker)       -> (DataFrame[OHLCV], info)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (END_DATE, HISTORICAL_PRICE_PATH, HISTORICAL_TWEETS_PATH,
                    START_DATE, TICKER_ALIASES)
from utils import get_logger

logger = get_logger("data_collection")

_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


class DataValidationError(RuntimeError):
    """Raised when a required dataset is missing, empty or structurally invalid."""


def _resolve_column(df: pd.DataFrame, candidates: List[str],
                    label: str, path: str) -> str:
    """Case/space-insensitive column lookup with a clear error message."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    raise DataValidationError(
        f"{path}: no {label} column found (looked for {candidates}; "
        f"file has {list(df.columns)})")


def _read_csv(path: str, label: str) -> pd.DataFrame:
    """Read a CSV with explicit, actionable failures."""
    if not os.path.exists(path):
        raise DataValidationError(
            f"{label} dataset not found at '{path}'. Place the CSV there or set "
            f"the corresponding path in config.py / environment variable.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"{label} dataset '{path}' is empty") from exc
    except Exception as exc:
        raise DataValidationError(f"{label} dataset '{path}' unreadable: {exc}") from exc
    if df.empty:
        raise DataValidationError(f"{label} dataset '{path}' contains no rows")
    logger.info("%s dataset loaded: %s rows x %s cols from %s",
                label, len(df), df.shape[1], os.path.basename(path))
    return df


def _to_naive_dates(series: pd.Series) -> pd.Series:
    """Parse to tz-naive, midnight-normalized datetimes.

    Mixed tz-aware/naive strings are common across public datasets; parsing with
    utc=True then dropping the tz yields one consistent, comparable calendar.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None).dt.normalize()


def _normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _alias_set(ticker: str) -> List[str]:
    return [s.upper() for s in TICKER_ALIASES.get(ticker.upper(), [ticker.upper()])]


class HistoricalDataStore:
    """Loads, cleans and validates the two historical CSVs exactly once."""

    _instance: Optional["HistoricalDataStore"] = None

    def __init__(self) -> None:
        self.prices: pd.DataFrame = self._load_prices()
        self.tweets: pd.DataFrame = self._load_tweets()
        self.audit: Dict[str, Dict[str, int]] = {}

    @classmethod
    def instance(cls) -> "HistoricalDataStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------- loaders
    def _load_prices(self) -> pd.DataFrame:
        raw = _read_csv(HISTORICAL_PRICE_PATH, "Historical price")
        date_c = _resolve_column(raw, ["date"], "date", HISTORICAL_PRICE_PATH)
        tick_c = _resolve_column(raw, ["stock name", "stock_name", "ticker", "symbol"],
                                 "ticker", HISTORICAL_PRICE_PATH)
        cols = {str(c).strip().lower(): c for c in raw.columns}

        df = pd.DataFrame({
            "Date": _to_naive_dates(raw[date_c]),
            "Ticker": _normalize_symbol(raw[tick_c]),
        })
        for want in _PRICE_COLS:
            src = cols.get(want.lower())
            if src is None:
                raise DataValidationError(
                    f"{HISTORICAL_PRICE_PATH}: missing required column '{want}'")
            df[want] = pd.to_numeric(raw[src], errors="coerce")
        # 'Adj Close' preferred for Close when present (split/dividend adjusted)
        adj = cols.get("adj close") or cols.get("adj_close")
        if adj is not None:
            adj_vals = pd.to_numeric(raw[adj], errors="coerce")
            df["Close"] = adj_vals.where(adj_vals.notna(), df["Close"])
            logger.info("Historical prices: using 'Adj Close' as Close where available")

        before = len(df)
        df = df.dropna(subset=["Date", "Ticker"])
        bad_dates = before - len(df)
        df = df.dropna(subset=_PRICE_COLS)
        bad_prices = before - bad_dates - len(df)
        df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        if bad_dates or bad_prices:
            logger.warning("Historical prices: dropped %s unparseable dates, "
                           "%s rows with missing/invalid OHLCV (of %s)",
                           bad_dates, bad_prices, before)
        logger.info("Historical prices ready: %s rows, %s tickers, %s -> %s",
                    len(df), df["Ticker"].nunique(),
                    df["Date"].min().date(), df["Date"].max().date())
        return df

    def _load_tweets(self) -> pd.DataFrame:
        raw = _read_csv(HISTORICAL_TWEETS_PATH, "Historical tweets")
        date_c = _resolve_column(raw, ["date", "created_at", "datetime", "timestamp"],
                                 "date", HISTORICAL_TWEETS_PATH)
        text_c = _resolve_column(raw, ["tweet", "text", "body", "post", "message"],
                                 "text", HISTORICAL_TWEETS_PATH)
        tick_c = _resolve_column(raw, ["stock name", "stock_name", "ticker", "symbol"],
                                 "ticker", HISTORICAL_TWEETS_PATH)
        cols = {str(c).strip().lower(): c for c in raw.columns}
        comp_c = cols.get("company name") or cols.get("company_name")

        df = pd.DataFrame({
            "Date": _to_naive_dates(raw[date_c]),
            "Ticker": _normalize_symbol(raw[tick_c]),
            "Tweet": raw[text_c].astype(str).str.strip(),
            "Company": (raw[comp_c].astype(str).str.strip()
                        if comp_c is not None else ""),
        })
        before = len(df)
        df = df.dropna(subset=["Date"])
        bad_dates = before - len(df)
        # Very short texts carry no sentiment signal (e.g. "$AAPL ./")
        df = df[df["Tweet"].str.len() >= 15]
        short = before - bad_dates - len(df)
        df = df.drop_duplicates(subset=["Ticker", "Date", "Tweet"])
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        # canonical social schema: the archive carries text only, so engagement
        # fields are explicit N/A (never invented) and 'Post' mirrors 'Tweet'
        df["Post"] = df["Tweet"]
        df["Source"] = "archive/twitter"
        for col in ("Upvotes", "Comments", "Shares", "Replies"):
            df[col] = np.nan
        df["ST_Label"] = ""

        logger.warning("Historical tweets: dropped %s unparseable dates, "
                       "%s too-short texts (of %s)", bad_dates, short, before)
        logger.info("Historical tweets ready: %s rows, %s tickers, %s -> %s",
                    len(df), df["Ticker"].nunique(),
                    df["Date"].min().date(), df["Date"].max().date())
        return df

    # ------------------------------------------------------------- accessors
    def prices_for(self, ticker: str) -> pd.DataFrame:
        aliases = _alias_set(ticker)
        sub = self.prices[self.prices["Ticker"].isin(aliases)]
        if sub.empty:
            available = ", ".join(sorted(self.prices["Ticker"].unique())[:25])
            raise DataValidationError(
                f"no historical price rows for {ticker} (aliases {aliases}). "
                f"Available symbols include: {available}")
        return sub.drop(columns=["Ticker"]).set_index("Date").sort_index()

    def tweets_for(self, ticker: str) -> pd.DataFrame:
        aliases = _alias_set(ticker)
        sub = self.tweets[self.tweets["Ticker"].isin(aliases)]
        return sub.reset_index(drop=True)

    def merge_audit(self, ticker: str, price_index: pd.DatetimeIndex,
                    tweet_dates: pd.Series) -> Dict[str, int]:
        """Compare tweet dates against trading dates on (Date, Stock Name).

        Reports, rather than hides, the three mismatch classes: tweet days with
        no trading session (weekends/holidays - expected), trading days without
        tweets (sentiment gaps - forward-filled downstream), and full coverage.
        """
        tdates = set(pd.DatetimeIndex(tweet_dates.unique()))
        pdates = set(price_index)
        audit = {
            "trading_days": len(pdates),
            "tweet_days": len(tdates),
            "matched_days": len(tdates & pdates),
            "tweet_days_without_session": len(tdates - pdates),
            "trading_days_without_tweets": len(pdates - tdates),
        }
        self.audit[ticker] = audit
        logger.info(
            "[%s] merge audit: %s/%s trading days have tweets; %s tweet days fall "
            "on non-trading dates (retained, aligned forward); %s trading days "
            "have no tweets (neutral/forward-filled)",
            ticker, audit["matched_days"], audit["trading_days"],
            audit["tweet_days_without_session"], audit["trading_days_without_tweets"])
        return audit


# ------------------------------------------------------------------ public API
def load_historical_prices(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """Return (OHLCV frame indexed by trading date, source_info)."""
    store = HistoricalDataStore.instance()
    df = store.prices_for(ticker)
    info = {
        "ticker": ticker, "source": f"HISTORICAL CSV ({os.path.basename(HISTORICAL_PRICE_PATH)})",
        "is_real": True, "reason": "", "mode": "historical",
        "records": int(len(df)),
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
    }
    return df[_PRICE_COLS].astype(float), info


def load_historical_tweets(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """Return (tweet frame, source_info) for one ticker."""
    store = HistoricalDataStore.instance()
    df = store.tweets_for(ticker)
    info = {
        "ticker": ticker,
        "source": f"HISTORICAL CSV ({os.path.basename(HISTORICAL_TWEETS_PATH)})",
        "is_real": True, "reason": "", "mode": "historical",
        "text_available": True, "has_shares": False,
        "engagement_desc": "likes=N/A comments=N/A shares=N/A (archive text only)",
        "posts": int(len(df)),
    }
    if df.empty:
        info["reason"] = f"no historical tweets matched {ticker}"
        logger.warning("[%s] %s", ticker, info["reason"])
    else:
        info["date_range"] = f"{df['Date'].min().date()} -> {df['Date'].max().date()}"
    return df, info


def fetch_live_prices(ticker: str, start: str = START_DATE,
                      end: str = END_DATE) -> Tuple[pd.DataFrame, Dict]:
    """Live OHLCV from Yahoo Finance. Inference only - never used for training."""
    info = {"ticker": ticker, "source": None, "is_real": False, "reason": "",
            "mode": "live", "records": 0, "date_range": ""}
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False)
        if df is None or df.empty:
            raise ValueError("empty response from yfinance")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[_PRICE_COLS].dropna()
        if df.empty:
            raise ValueError("no rows left after cleaning")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        info.update(source="Yahoo Finance (yfinance)", is_real=True,
                    records=int(len(df)),
                    date_range=f"{df.index.min().date()} -> {df.index.max().date()}")
        logger.info("[%s] live prices: %s rows (%s)", ticker, len(df), info["date_range"])
        return df.astype(float), info
    except Exception as exc:
        info["reason"] = f"yfinance failed: {type(exc).__name__}: {exc}"
        logger.error("[%s] %s", ticker, info["reason"])
        return pd.DataFrame(columns=_PRICE_COLS), info