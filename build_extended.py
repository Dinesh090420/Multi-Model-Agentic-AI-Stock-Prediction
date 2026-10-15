"""One-time builder for the EXTENDED 2015-2020 study.

Writes three new files alongside your existing ones. Nothing already in data/
is modified, so Phase 1 and Phase 2 on the 2021-22 window stay reproducible.

  data/stock_yfinance_2015_2020.csv   prices  (yfinance, same schema as yours)
  data/stock_tweets_2015_2020.csv     social  (Kaggle omermetinn dump, joined)
  data/historical_news_2015_2020.csv  news    (your Alpaca files, re-filtered)

WHY 2015-2020 AND NOT 2018-2024
The Kaggle tweet corpus ends in 2020 and your existing one starts 2021-09, so
any window past 2020 leaves social coverage full of holes - which would quietly
turn E2 and E4 into copies of E1. 2015-2020 is the longest window where prices,
news AND social all have real coverage, so all four experiments stay valid.

The two tweet corpora are NOT concatenated. They are different collections;
splicing them would put a source discontinuity inside the training data. Run
2015-2020 and 2021-2022 as two separate studies instead.

Run once:  python build_extended.py
Then:      OUTPUT_PATH=results_ext \\
           HISTORICAL_PRICE_PATH=data/stock_yfinance_2015_2020.csv \\
           HISTORICAL_TWEETS_PATH=data/stock_tweets_2015_2020.csv \\
           HISTORICAL_NEWS_PATH=data/historical_news_2015_2020.csv \\
           python main.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from config import DATA_DIR

START, END = "2015-01-01", "2020-12-31"
TICKERS = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
           "GOOGL": "Alphabet Inc.", "GOOG": "Alphabet Inc."}

PRICE_OUT = os.path.join(DATA_DIR, "stock_yfinance_2015_2020.csv")
TWEET_OUT = os.path.join(DATA_DIR, "stock_tweets_2015_2020.csv")
NEWS_OUT = os.path.join(DATA_DIR, "historical_news_2015_2020.csv")

TWEET_SRC = os.path.join(DATA_DIR, "Tweet.csv")
MAP_SRC = os.path.join(DATA_DIR, "Company_Tweet.csv")
ALPACA = {"AAPL": "AAPL_Alpaca_NLP.csv", "MSFT": "MSFT_Alpaca_NLP.csv",
          "GOOG": "GOOG_Alpaca_NLP.csv"}

# Cap tweets kept per ticker per day. config.MAX_TWEETS_PER_DAY is 40, so the
# pipeline discards anything beyond that anyway - keeping more only makes
# FinBERT scoring slower for no change in the result. Matched to that cap.
# Deterministic: earliest tweets of each day, so the choice never varies.
from config import MAX_TWEETS_PER_DAY
PER_DAY_CAP = MAX_TWEETS_PER_DAY or 40
CHUNK = 500_000


def _col(df, *names):
    """Case-insensitive column lookup."""
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


# ------------------------------------------------------------------ prices
def build_prices() -> bool:
    """Pull 2015-2020 OHLCV from Yahoo in the exact schema of your existing CSV."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed - skipping prices")
        return False
    frames = []
    for t in ("AAPL", "MSFT", "GOOGL"):
        df = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"  {t}: EMPTY response from yfinance - retry later")
            return False
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Stock Name"] = t
        frames.append(df[["Date", "Open", "High", "Low", "Close",
                          "Adj Close", "Volume", "Stock Name"]])
        print(f"  {t}: {len(df)} trading days "
              f"{df['Date'].min().date()} -> {df['Date'].max().date()}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(PRICE_OUT, index=False)
    print(f"  -> {PRICE_OUT}  ({len(out)} rows)")
    return True


# ------------------------------------------------------------------ tweets
def build_tweets() -> bool:
    """Join Tweet.csv to Company_Tweet.csv and emit your Date/Tweet/Stock Name schema."""
    for path in (TWEET_SRC, MAP_SRC):
        if not os.path.exists(path):
            print(f"  MISSING {path}")
            return False

    mp = pd.read_csv(MAP_SRC)
    id_c, tk_c = _col(mp, "tweet_id", "id"), _col(mp, "ticker_symbol", "ticker", "symbol")
    if not (id_c and tk_c):
        print(f"  {MAP_SRC}: need tweet_id + ticker columns; found {list(mp.columns)}")
        return False
    mp[tk_c] = mp[tk_c].astype(str).str.strip().str.upper()
    mp = mp[mp[tk_c].isin(TICKERS)]
    want = dict(zip(mp[id_c], mp[tk_c]))
    print(f"  mapping: {len(want):,} tweet ids for {sorted(set(mp[tk_c]))}")

    kept, scanned = [], 0
    for chunk in pd.read_csv(TWEET_SRC, chunksize=CHUNK, on_bad_lines="skip",
                             low_memory=False):
        scanned += len(chunk)
        cid = _col(chunk, "tweet_id", "id")
        cdate = _col(chunk, "post_date", "date", "created_at", "timestamp")
        cbody = _col(chunk, "body", "text", "tweet", "content")
        if not (cid and cdate and cbody):
            print(f"  {TWEET_SRC}: need id/date/body; found {list(chunk.columns)}")
            return False
        sub = chunk[chunk[cid].isin(want)]
        if sub.empty:
            continue
        raw = sub[cdate]
        # post_date is a UNIX epoch in this dataset; fall back to string parsing
        when = (pd.to_datetime(raw, unit="s", errors="coerce")
                if pd.api.types.is_numeric_dtype(raw)
                else pd.to_datetime(raw, errors="coerce", utc=True).dt.tz_localize(None))
        kept.append(pd.DataFrame({
            "Date": when,
            "Tweet": sub[cbody].astype(str).str.strip(),
            "Stock Name": sub[cid].map(want),
        }))
        print(f"    scanned {scanned:,} | kept {sum(map(len, kept)):,}", end="\r")

    if not kept:
        print(f"\n  no matching tweets found (scanned {scanned:,})")
        return False
    df = pd.concat(kept, ignore_index=True).dropna(subset=["Date"])
    df = df[(df["Date"] >= START) & (df["Date"] <= END)]
    df = df[df["Tweet"].str.len() >= 15]
    df["Company Name"] = df["Stock Name"].map(TICKERS)
    df = df.drop_duplicates(["Stock Name", "Date", "Tweet"])
    df["_day"] = df["Date"].dt.normalize()
    before = len(df)
    df = (df.sort_values(["Stock Name", "Date"])
            .groupby(["Stock Name", "_day"], group_keys=False).head(PER_DAY_CAP))
    df = df.drop(columns="_day")[["Date", "Tweet", "Stock Name", "Company Name"]]
    df.to_csv(TWEET_OUT, index=False)
    print(f"\n  capped {before:,} -> {len(df):,} rows at {PER_DAY_CAP}/ticker/day")
    for t, n in df["Stock Name"].value_counts().items():
        days = df[df["Stock Name"] == t]["Date"].dt.normalize().nunique()
        print(f"    {t:<6} {n:>8,} tweets on {days:>5} distinct days")
    print(f"  -> {TWEET_OUT}")
    return True


# ------------------------------------------------------------------ news
def build_news() -> bool:
    """Re-filter the Alpaca files to 2015-2020 (they already span 2015-2024)."""
    frames = []
    for ticker, fname in ALPACA.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING {path}")
            continue
        d = pd.read_csv(path)
        when = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        text = d["headline"].astype(str).str.strip()
        ok = when.notna() & when.between(START, END) & (text.str.len() > 0)
        frames.append(pd.DataFrame({"date": when[ok], "ticker": ticker,
                                    "headline": text[ok],
                                    "source": "Alpaca/Benzinga", "url": ""}))
        print(f"  {ticker}: {len(d):>7,} rows -> {int(ok.sum()):>7,} in 2015-2020")
    if not frames:
        return False
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(["ticker", "date", "headline"])
             .sort_values(["ticker", "date"]).reset_index(drop=True))
    out.to_csv(NEWS_OUT, index=False)
    print(f"  -> {NEWS_OUT}  ({len(out):,} unique articles)")
    return True


def main() -> int:
    print(f"Building the extended {START} -> {END} dataset\n")
    print("PRICES");  ok_p = build_prices()
    print("\nTWEETS");  ok_t = build_tweets()
    print("\nNEWS");    ok_n = build_news()

    print("\n" + "=" * 62)
    for label, ok, path in (("prices", ok_p, PRICE_OUT), ("tweets", ok_t, TWEET_OUT),
                            ("news", ok_n, NEWS_OUT)):
        print(f"  [{'OK' if ok else 'FAILED'}] {label:<8} {path if ok else ''}")
    if not (ok_p and ok_t and ok_n):
        print("\n  Fix the failures above before running the extended study.")
        return 1
    print("\n  Run the extended study with:\n")
    print(f"    OUTPUT_PATH=results_ext \\\n"
          f"    HISTORICAL_PRICE_PATH={PRICE_OUT} \\\n"
          f"    HISTORICAL_TWEETS_PATH={TWEET_OUT} \\\n"
          f"    HISTORICAL_NEWS_PATH={NEWS_OUT} \\\n"
          f"    python main.py\n")
    print("  Your 2021-22 results stay untouched under results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
