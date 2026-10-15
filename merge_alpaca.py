"""One-time merge: the three Alpaca news CSVs -> data/historical_news.csv

Completely OFFLINE. No FNSPID, no Hugging Face API, no NewsAPI, no Alpha
Vantage, no StockTwits - the output contains only rows that exist in the three
downloaded files. Nothing is invented, back-filled, forward-filled or
synthesised: a trading day with no article simply has none.

Input   data/AAPL_Alpaca_NLP.csv, data/MSFT_Alpaca_NLP.csv, data/GOOG_Alpaca_NLP.csv
        (columns: headline, date - 'headline' holds the article text)
Output  data/historical_news.csv  (columns: date, ticker, headline, source, url)

The ticker comes from the filename, since the source files carry no ticker
column. GOOG stays GOOG - config.TICKER_ALIASES already maps it to GOOGL.

Run once:  python merge_alpaca.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from config import DATA_DIR, HISTORICAL_PRICE_PATH

FILES = {"AAPL": "AAPL_Alpaca_NLP.csv",
         "MSFT": "MSFT_Alpaca_NLP.csv",
         "GOOG": "GOOG_Alpaca_NLP.csv"}
OUT_PATH = os.path.join(DATA_DIR, "historical_news.csv")
COLUMNS = ["date", "ticker", "headline", "source", "url"]


def price_window():
    """Exact date range of the historical YFinance dataset - never hard-coded."""
    px = pd.read_csv(HISTORICAL_PRICE_PATH,
                     usecols=lambda c: str(c).strip().lower() == "date")
    d = pd.to_datetime(px.iloc[:, 0], errors="coerce", utc=True).dt.tz_localize(None)
    return d.min().normalize(), d.max().normalize()


def main() -> int:
    lo, hi = price_window()
    print(f"price window: {lo.date()} -> {hi.date()}\n")

    frames, missing = [], []
    for ticker, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            missing.append(path)
            print(f"{ticker}: MISSING {path}")
            continue
        df = pd.read_csv(path)
        for need in ("date", "headline"):
            if need not in df.columns:
                print(f"{ticker}: ERROR - no '{need}' column; found {list(df.columns)}")
                return 2
        when = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        text = df["headline"].astype(str).str.strip()
        ok = when.notna() & when.between(lo, hi) & (text.str.len() > 0) & (text != "nan")
        frames.append(pd.DataFrame({"date": when[ok], "ticker": ticker,
                                    "headline": text[ok],           # text preserved as-is
                                    "source": "Alpaca/Benzinga", "url": ""}))
        print(f"{ticker}: {len(df):>6} source rows -> {int(ok.sum()):>6} rows inside price window")

    if not frames:
        print(f"\nno input files found under {DATA_DIR}; expected: "
              + ", ".join(FILES.values()))
        return 1
    if missing:
        print(f"\nWARNING: {len(missing)} input file(s) missing - the merge continues "
              f"with what is present, so those tickers will have no news.")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    out = (merged.drop_duplicates(subset=["ticker", "date", "headline"])
                 .sort_values(["ticker", "date"]).reset_index(drop=True))[COLUMNS]
    print(f"\nDuplicates removed: {before - len(out)}")
    print(f"Final historical news rows: {len(out)}")
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWritten to:\n{OUT_PATH}")

    # ---------------------------------------------------------------- validation
    print("\n" + "=" * 62)
    print("VALIDATION")
    print("=" * 62)
    if not os.path.exists(OUT_PATH):
        print("FAIL: output file was not created")
        return 3
    chk = pd.read_csv(OUT_PATH)
    d = pd.to_datetime(chk["date"], errors="coerce")
    checks = [
        ("file exists", True),
        (f"columns are {COLUMNS}", list(chk.columns) == COLUMNS),
        ("no null dates", int(d.isna().sum()) == 0),
        ("no null/empty headlines",
         int(chk["headline"].isna().sum()) == 0
         and int((chk["headline"].astype(str).str.strip() == "").sum()) == 0),
        ("no duplicate ticker/date/headline",
         not chk.duplicated(["ticker", "date", "headline"]).any()),
        ("all dates inside the price window", bool(d.between(lo, hi).all())),
    ]
    for label, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")

    print(f"\n  date range: {d.min().date()} -> {d.max().date()}")
    print(f"  {'ticker':<8}{'articles':>10}{'distinct dates':>16}")
    for t in FILES:
        sub = chk[chk["ticker"] == t]
        print(f"  {t:<8}{len(sub):>10}{d[sub.index].nunique() if len(sub) else 0:>16}")
    print("\n  Source: the three offline CSVs only. No live API (NewsAPI, Alpha")
    print("  Vantage, StockTwits, Hugging Face) was contacted, and no article was")
    print("  invented, back-filled or forward-filled.")
    return 0 if all(p for _, p in checks) else 4


if __name__ == "__main__":
    sys.exit(main())
