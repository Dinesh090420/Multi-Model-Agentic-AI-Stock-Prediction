"""LIVE data collection (inference only) - financial news and social messages.

Training never calls this module; historical text comes from the offline CSVs
in data_collection.py.

Design: every source is a small subclass of `BaseCollector`, so adding Reddit,
SEC filings or another vendor later means writing one class and registering it -
no change to the agents, feature fusion, models or reports.

    NewsAPICollector      -> Date, Source, Credibility, Headline
    StockTwitsCollector   -> Date, Source, Post, Upvotes, Replies, Comments,
                             Shares, ST_Label
"""
from __future__ import annotations

import abc
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from config import (DEFAULT_CREDIBILITY, HTTP_BACKOFF, HTTP_RETRIES, HTTP_TIMEOUT,
                    NEWS_API_KEY, NEWSAPI_URL, SOURCE_CREDIBILITY,
                    STOCKTWITS_SYMBOL_URL, STOCKTWITS_SCRAPE_SCROLLS,
                    STOCKTWITS_PAGE_TIMEOUT_MS, STOCKTWITS_HEADLESS)
from utils import get_logger

logger = get_logger("live_sources")

_HEADERS = {"User-Agent": "stock-research-agent/2.0"}
MIN_TEXT_LEN = 15


def credibility_for(source_name: str) -> float:
    """Editorial-credibility weight for a news outlet (unknown -> default)."""
    return SOURCE_CREDIBILITY.get(str(source_name).strip().lower(), DEFAULT_CREDIBILITY)


def _request_json(url: str, params: Optional[Dict] = None) -> Dict:
    """GET JSON with bounded retries and exponential backoff.

    Retries transient failures (network, 5xx, 429). Client errors other than 429
    fail fast - retrying a 401 or 403 cannot succeed.
    """
    last: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS,
                                timeout=HTTP_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code} (retryable)")
            resp.raise_for_status()
            return resp.json() or {}
        except (requests.HTTPError, requests.RequestException, ValueError) as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in (None, 429) or (status is not None and status >= 500)
            if not retryable or attempt == HTTP_RETRIES:
                break
            wait = HTTP_BACKOFF ** attempt
            logger.warning("request failed (%s); retry %s/%s in %.1fs",
                           exc, attempt + 1, HTTP_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError(f"request to {url} failed: {last}")


class BaseCollector(abc.ABC):
    """Contract every live source implements."""

    name: str = "base"
    kind: str = "generic"          # "news" or "social"

    @abc.abstractmethod
    def available(self) -> Tuple[bool, str]:
        """(usable, reason) - e.g. missing credentials."""

    @abc.abstractmethod
    def collect(self, ticker: str) -> pd.DataFrame:
        """Fetch records for one ticker; raise on failure."""

    def safe_collect(self, ticker: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Never raises: returns (frame or None, reason)."""
        ok, reason = self.available()
        if not ok:
            logger.warning("[%s] %s unavailable: %s", ticker, self.name, reason)
            return None, reason
        try:
            df = self.collect(ticker)
            if df is None or df.empty:
                return None, f"{self.name} returned no usable records"
            logger.info("[%s] %s: %s records", ticker, self.name, len(df))
            return df, ""
        except Exception as exc:
            reason = f"{self.name} failed: {type(exc).__name__}: {exc}"
            logger.error("[%s] %s", ticker, reason)
            return None, reason


class NewsAPICollector(BaseCollector):
    """Recent financial headlines from newsapi.org (free tier: last ~30 days)."""

    name = "NewsAPI"
    kind = "news"

    def available(self) -> Tuple[bool, str]:
        return (bool(NEWS_API_KEY), "" if NEWS_API_KEY else "NEWSAPI_KEY not set")

    def collect(self, ticker: str) -> pd.DataFrame:
        data = _request_json(NEWSAPI_URL, {
            "q": ticker, "language": "en", "sortBy": "publishedAt",
            "pageSize": 100, "apiKey": NEWS_API_KEY})
        if data.get("status") != "ok":
            raise ValueError(data.get("code") or data.get("message") or "bad response")
        rows: List[Dict] = []
        for art in data.get("articles") or []:
            title = (art.get("title") or "").strip()
            desc = (art.get("description") or "").strip()
            src = ((art.get("source") or {}).get("name") or "unknown")
            when = pd.to_datetime(art.get("publishedAt"), errors="coerce", utc=True)
            if not title or pd.isna(when):
                continue
            rows.append({
                "Date": when.tz_localize(None).normalize(),
                "Source": src,
                "Credibility": credibility_for(src),
                "Headline": (title + ". " + desc).strip(),
            })
        if not rows:
            raise ValueError("no usable articles")
        return (pd.DataFrame(rows)
                .drop_duplicates(subset=["Date", "Headline"])
                .sort_values("Date").reset_index(drop=True))


class StockTwitsCollector(BaseCollector):
    """Live social messages scraped from StockTwits symbol pages via Playwright.

    Uses the public web page https://stocktwits.com/symbol/{ticker} (no API key).
    The page is a React infinite-scroll feed, so the collector loads it in a
    headless browser, scrolls to materialize messages, and extracts each one.

    Extracts: message text, username, timestamp, Bullish/Bearish tag, likes,
    replies - mapped to the SAME schema the rest of the pipeline already
    expects, so FinBERT and every downstream stage are unchanged.

    NOTE: scraping may conflict with StockTwits' Terms of Service; this path is
    provided per project requirement. Playwright must be installed:
        pip install playwright && python -m playwright install chromium
    """

    name = "StockTwits"
    kind = "social"

    def available(self) -> Tuple[bool, str]:
        try:
            import playwright  # noqa: F401
        except ImportError:
            return (False, "playwright not installed "
                    "(pip install playwright && python -m playwright install chromium)")
        return True, ""

    def collect(self, ticker: str) -> pd.DataFrame:
        raw = self._scrape(ticker)
        rows: List[Dict] = []
        for item in raw:
            body = (item.get("body") or "").strip()
            when = pd.to_datetime(item.get("created_at"), errors="coerce", utc=True)
            if len(body) < MIN_TEXT_LEN:
                continue
            if pd.isna(when):
                when = pd.Timestamp.utcnow()          # fall back to fetch time
            rows.append({
                "Date": when.tz_localize(None).normalize(),
                "Ticker": ticker.upper(),
                "Source": "stocktwits",
                "Username": item.get("username", ""),
                "Post": body,
                "Upvotes": float(item.get("likes", 0) or 0),       # engagement magnitude
                "Replies": float(item.get("replies", 0) or 0),
                "Comments": float(item.get("replies", 0) or 0),    # alias for weighting
                "Shares": np.nan,                                  # not shown -> N/A
                "ST_Label": item.get("sentiment", ""),
            })
        if not rows:
            raise ValueError("no messages scraped (layout change or bot block?)")
        return (pd.DataFrame(rows)
                .drop_duplicates(subset=["Date", "Post"])
                .sort_values("Date").reset_index(drop=True))

    def _scrape(self, ticker: str) -> List[Dict]:
        """Render the symbol page and pull message objects out of the DOM.

        Runs entirely in-function so no other module needs to know about
        Playwright. Returns a list of plain dicts (body, username, created_at,
        sentiment, likes, replies).
        """
        from playwright.sync_api import sync_playwright

        url = STOCKTWITS_SYMBOL_URL.format(ticker=ticker)
        # Selectors are defined by data-testid attributes StockTwits uses on the
        # message stream; kept in one place so a layout change is a one-line fix.
        js_extract = """
        () => {
            const out = [];
            const items = document.querySelectorAll('[data-testid="message"], article');
            items.forEach(el => {
                const q = (sel) => { const n = el.querySelector(sel); return n ? n.innerText.trim() : ""; };
                const body = q('[data-testid="message-body"]') ||
                             q('div[class*="RichTextMessage"]') || el.innerText.trim();
                const user = q('[data-testid="username"]') || q('a[href^="/"]');
                const time = (() => {
                    const t = el.querySelector('time');
                    return t ? (t.getAttribute('datetime') || t.innerText.trim()) : "";
                })();
                let sentiment = "";
                const label = (el.innerText.match(/\b(Bullish|Bearish)\b/i) || [])[0];
                if (label) sentiment = label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();
                const nums = Array.from(el.querySelectorAll('[data-testid="like-count"], [data-testid="reply-count"]'))
                                  .map(n => parseInt(n.innerText.replace(/[^0-9]/g, ""), 10) || 0);
                out.push({
                    body: body, username: user, created_at: time,
                    sentiment: sentiment,
                    likes: nums.length > 0 ? nums[0] : 0,
                    replies: nums.length > 1 ? nums[1] : 0
                });
            });
            return out;
        }
        """
        items: List[Dict] = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=STOCKTWITS_HEADLESS)
            try:
                page = browser.new_context(
                    user_agent=_HEADERS["User-Agent"]).new_page()
                page.set_default_timeout(STOCKTWITS_PAGE_TIMEOUT_MS)
                page.goto(url, wait_until="domcontentloaded")
                try:
                    page.wait_for_selector('[data-testid="message"], article',
                                           timeout=STOCKTWITS_PAGE_TIMEOUT_MS)
                except Exception:
                    logger.warning("[%s] StockTwits: message container not found "
                                   "(page layout may have changed)", ticker)
                for _ in range(STOCKTWITS_SCRAPE_SCROLLS):   # materialize infinite scroll
                    page.mouse.wheel(0, 4000)
                    page.wait_for_timeout(800)
                items = page.evaluate(js_extract) or []
            finally:
                browser.close()
        # de-dup on body while preserving order
        seen, unique = set(), []
        for it in items:
            key = (it.get("body") or "").strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(it)
        logger.info("[%s] StockTwits scrape: %s messages from %s",
                    ticker, len(unique), url)
        return unique


NEWS_COLLECTORS: List[BaseCollector] = [NewsAPICollector()]
SOCIAL_COLLECTORS: List[BaseCollector] = [StockTwitsCollector()]


def register_collector(collector: BaseCollector) -> None:
    """Plug in a future source (Reddit, SEC, macro) without touching the agents."""
    target = SOCIAL_COLLECTORS if collector.kind == "social" else NEWS_COLLECTORS
    target.append(collector)
    logger.info("registered %s collector: %s", collector.kind, collector.name)


def _collect_all(collectors: List[BaseCollector], ticker: str,
                 dedupe_on: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    frames, used, reasons = [], [], []
    for col in collectors:
        df, reason = col.safe_collect(ticker)
        if df is not None:
            frames.append(df)
            used.append(f"{col.name} ({len(df)})")
        elif reason:
            reasons.append(reason)
    if not frames:
        return pd.DataFrame(), used, reasons
    merged = (pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=dedupe_on)
                .sort_values("Date").reset_index(drop=True))
    return merged, used, reasons


def fetch_live_news(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """Live news for one ticker from every registered news collector."""
    df, used, reasons = _collect_all(NEWS_COLLECTORS, ticker, ["Date", "Headline"])
    info = {"ticker": ticker, "mode": "live",
            "source": " + ".join(used) if used else "unavailable",
            "is_real": bool(used), "reason": "; ".join(reasons),
            "articles": int(len(df))}
    return df, info


def fetch_live_social(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """Live social messages for one ticker from every registered social collector."""
    df, used, reasons = _collect_all(SOCIAL_COLLECTORS, ticker, ["Date", "Post"])
    info = {"ticker": ticker, "mode": "live",
            "source": " + ".join(used) if used else "unavailable",
            "is_real": bool(used), "reason": "; ".join(reasons),
            "text_available": True, "has_shares": False,
            "engagement_desc": "likes=yes replies=yes shares=N/A",
            "posts": int(len(df))}
    return df, info


def social_engagement_metrics(scored: pd.DataFrame) -> Dict[str, float]:
    """Live social summary fused into the report and dashboard.

    Expects the FinBERT-scored frame from sentiment_analysis.analyze_social.
    """
    if scored is None or scored.empty:
        return {k: 0.0 for k in (
            "Avg_Sentiment", "Bullish_Pct", "Bearish_Pct", "Positive_Count",
            "Negative_Count", "Neutral_Count", "Total_Likes", "Avg_Likes",
            "Total_Replies", "Avg_Replies", "Message_Count")}

    labels = scored["FinBERT_Label"].astype(str)
    n = float(len(scored))
    likes = pd.to_numeric(scored.get("Upvotes", pd.Series(dtype=float)),
                          errors="coerce").fillna(0.0)
    replies = pd.to_numeric(scored.get("Replies", pd.Series(dtype=float)),
                            errors="coerce").fillna(0.0)
    tagged = scored.get("ST_Label", pd.Series([""] * len(scored))).astype(str).str.lower()
    n_tagged = float((tagged != "").sum())

    pos = float((labels == "Bullish").sum())
    neg = float((labels == "Bearish").sum())
    return {
        "Avg_Sentiment": float(pd.to_numeric(scored["FinBERT_Score"],
                                             errors="coerce").fillna(0.0).mean()),
        # author-declared tags when present, else FinBERT class share
        "Bullish_Pct": float(100.0 * ((tagged == "bullish").sum() / n_tagged)
                             if n_tagged else 100.0 * pos / n),
        "Bearish_Pct": float(100.0 * ((tagged == "bearish").sum() / n_tagged)
                             if n_tagged else 100.0 * neg / n),
        "Positive_Count": pos,
        "Negative_Count": neg,
        "Neutral_Count": float((labels == "Neutral").sum()),
        "Total_Likes": float(likes.sum()),
        "Avg_Likes": float(likes.mean()),
        "Total_Replies": float(replies.sum()),
        "Avg_Replies": float(replies.mean()),
        "Message_Count": n,
    }