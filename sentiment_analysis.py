"""FinBERT financial sentiment: batched inference, persistent cache, daily aggregates.

Model: ProsusAI/finbert (configurable path only; no generic sentiment model is
ever substituted).

Score convention: +confidence for Positive, -confidence for Negative, 0.0 for
Neutral, so the daily mean is a signed sentiment in [-1, 1].

Daily aggregation (per Date, per ticker) produces:
    Avg_Sentiment, Positive_Count, Negative_Count, Neutral_Count,
    Avg_Confidence, Text_Count, Positive_Ratio, Negative_Ratio,
    Neutral_Ratio, Sentiment_Trend
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (CACHE_DIR, DEVICE, FINBERT_BATCH_SIZE, FINBERT_MAX_LENGTH,
                    MODEL_PATH, NEWS_WEIGHT, SENTIMENT_AGG_FEATURES,
                    SENTIMENT_CACHE_PATH, SOCIAL_WEIGHT, USE_SENTIMENT_CACHE)
from utils import get_logger

logger = get_logger("sentiment")

_LABEL_MAP = {"positive": "Bullish", "neutral": "Neutral", "negative": "Bearish"}
_EPS = 1e-9

AGG_COLUMNS = ["Avg_Sentiment", "Positive_Count", "Negative_Count",
               "Neutral_Count", "Avg_Confidence", "Text_Count",
               "Positive_Ratio", "Negative_Ratio", "Neutral_Ratio",
               "Sentiment_Trend"]


class FinBERTScorer:
    """Thread-safe singleton wrapper around the FinBERT pipeline.

    Two layers of caching keep repeated runs cheap:
      * in-memory dict for the current process
      * JSON file on disk (config.SENTIMENT_CACHE_PATH) across runs
    Texts are keyed by SHA1 so the cache stays compact.
    """

    _instance: Optional["FinBERTScorer"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._pipe = None
        self._cache: Dict[str, List] = {}
        self._dirty = False
        self._load_cache()

    @classmethod
    def instance(cls) -> "FinBERTScorer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------- cache
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()

    def _load_cache(self) -> None:
        if not USE_SENTIMENT_CACHE:
            return
        try:
            if os.path.exists(SENTIMENT_CACHE_PATH):
                with open(SENTIMENT_CACHE_PATH, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
                logger.info("FinBERT cache loaded: %s entries", len(self._cache))
        except Exception as exc:                      # corrupt cache must not stop a run
            logger.warning("FinBERT cache unreadable (%s); starting empty", exc)
            self._cache = {}

    def save_cache(self) -> None:
        if not (USE_SENTIMENT_CACHE and self._dirty):
            return
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = SENTIMENT_CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
            os.replace(tmp, SENTIMENT_CACHE_PATH)     # atomic
            self._dirty = False
            logger.info("FinBERT cache saved: %s entries", len(self._cache))
        except Exception as exc:
            logger.warning("could not save FinBERT cache: %s", exc)

    # ------------------------------------------------------------- model
    def _pipeline(self):
        if self._pipe is None:
            from transformers import pipeline
            device = 0 if DEVICE == "cuda" else -1     # HF: -1 = CPU
            logger.info("loading FinBERT (%s) on %s", MODEL_PATH,
                        "cuda:0" if device == 0 else "cpu")
            self._pipe = pipeline("sentiment-analysis", model=MODEL_PATH,
                                  truncation=True, max_length=FINBERT_MAX_LENGTH,
                                  device=device)
        return self._pipe

    # ------------------------------------------------------------- scoring
    def score(self, texts: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Return (labels, signed_scores, confidences) aligned to `texts`.

        Deduplicates, consults the cache, and runs FinBERT in batches on the
        remainder. On model failure the batch degrades to Neutral/0.0 rather
        than aborting the pipeline - the failure is logged, never hidden.
        """
        texts = [str(t) for t in texts]
        if not texts:
            return [], np.array([]), np.array([])

        uniq = list(dict.fromkeys(texts))
        pending = [t for t in uniq if self._key(t) not in self._cache]
        logger.info("FinBERT: %s texts (%s unique, %s cached, %s to score)",
                    len(texts), len(uniq), len(uniq) - len(pending), len(pending))

        if pending:
            try:
                pipe = self._pipeline()
                for i in range(0, len(pending), FINBERT_BATCH_SIZE):
                    chunk = pending[i:i + FINBERT_BATCH_SIZE]
                    results = pipe(chunk, batch_size=FINBERT_BATCH_SIZE)
                    for text, res in zip(chunk, results):
                        label = str(res["label"]).lower()
                        conf = float(res["score"])
                        signed = conf if label == "positive" else (
                            -conf if label == "negative" else 0.0)
                        self._cache[self._key(text)] = [label, signed, conf]
                    self._dirty = True
            except Exception as exc:
                logger.error("FinBERT scoring failed (%s: %s); affected texts "
                             "default to Neutral/0.0", type(exc).__name__, exc)
                for text in pending:
                    self._cache.setdefault(self._key(text), ["neutral", 0.0, 0.0])

        labels, scores, confs = [], [], []
        for text in texts:
            label, signed, conf = self._cache.get(self._key(text),
                                                  ["neutral", 0.0, 0.0])
            labels.append(label)
            scores.append(float(signed))
            confs.append(float(conf))
        return labels, np.asarray(scores, dtype=float), np.asarray(confs, dtype=float)


def score_texts(texts: Iterable[str]) -> Tuple[List[str], np.ndarray]:
    """Backwards-compatible helper: (labels, signed scores)."""
    labels, scores, _ = FinBERTScorer.instance().score(list(texts))
    return labels, scores


# ------------------------------------------------------------------ aggregation
def aggregate_daily(df: pd.DataFrame, score_col: str = "FinBERT_Score",
                    conf_col: str = "FinBERT_Confidence",
                    label_col: str = "FinBERT_Label",
                    weight_col: Optional[str] = None) -> pd.DataFrame:
    """Vectorized per-day aggregation of scored texts.

    `weight_col` (optional) applies engagement or credibility weighting to the
    mean; counts and ratios always reflect raw text counts.
    """
    if df.empty:
        return pd.DataFrame(columns=AGG_COLUMNS)

    work = df.copy()
    work["_day"] = pd.DatetimeIndex(work["Date"]).normalize()
    pos_labels = work[label_col].astype(str).str.lower()
    work["_pos"] = pos_labels.isin(["positive", "bullish"]).astype(int)
    work["_neg"] = pos_labels.isin(["negative", "bearish"]).astype(int)
    work["_neu"] = 1 - work["_pos"] - work["_neg"]

    if weight_col and weight_col in work.columns:
        w = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        work["_w"] = np.where(w.to_numpy() > 0, w.to_numpy(), 1.0)
    else:
        work["_w"] = 1.0
    work["_ws"] = work["_w"] * pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)

    grp = work.groupby("_day", sort=True)
    out = pd.DataFrame({
        "Avg_Sentiment": grp["_ws"].sum() / (grp["_w"].sum() + _EPS),
        "Positive_Count": grp["_pos"].sum().astype(float),
        "Negative_Count": grp["_neg"].sum().astype(float),
        "Neutral_Count": grp["_neu"].sum().astype(float),
        "Avg_Confidence": grp[conf_col].mean() if conf_col in work.columns
        else grp["_w"].mean() * 0.0,
        "Text_Count": grp.size().astype(float),
    })
    total = out["Text_Count"].replace(0, np.nan)
    out["Positive_Ratio"] = (out["Positive_Count"] / total).fillna(0.0)
    out["Negative_Ratio"] = (out["Negative_Count"] / total).fillna(0.0)
    out["Neutral_Ratio"] = (out["Neutral_Count"] / total).fillna(0.0)
    # Daily trend = change vs the previous COVERED day (past-only, no look-ahead)
    out["Sentiment_Trend"] = out["Avg_Sentiment"].diff().fillna(0.0)
    out.index.name = "Date"
    return out[AGG_COLUMNS]


# ------------------------------------------------------------------ news
def analyze_news(news_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Score news headlines; daily sentiment is credibility-weighted."""
    df = news_df.copy()
    if df.empty:
        empty = pd.Series(dtype=float, name="News_Sentiment")
        return (pd.DataFrame(columns=["FinBERT_Label", "FinBERT_Score",
                                      "FinBERT_Confidence"]),
                empty, pd.DataFrame(columns=AGG_COLUMNS))

    labels, scores, confs = FinBERTScorer.instance().score(df["Headline"].tolist())
    df["FinBERT_Label"] = [lbl.capitalize() for lbl in labels]   # Positive/Neutral/Negative
    df["FinBERT_Score"] = scores
    df["FinBERT_Confidence"] = confs
    if "Credibility" not in df.columns:
        df["Credibility"] = 1.0

    agg = aggregate_daily(df, weight_col="Credibility")
    daily = agg["Avg_Sentiment"].rename("News_Sentiment")
    return df, daily, agg


# ------------------------------------------------------------------ social / tweets
def _engagement_weights(df: pd.DataFrame) -> np.ndarray:
    """log-damped engagement, winsorized at the 90th percentile.

    Missing fields are ignored (N/A) rather than imputed; a row with no
    engagement information receives a neutral weight of 1.0. Historical archive
    text has no engagement at all, so every row weighs equally - which is the
    correct, assumption-free treatment.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0)
    total = np.zeros(n, dtype=float)
    for col in ("Upvotes", "Comments", "Shares"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
            total += np.log1p(vals.to_numpy(dtype=float))
    total[total <= 0] = 1.0
    cap = float(np.percentile(total, 90))
    return np.minimum(total, cap if cap > 0 else 1.0)


def analyze_social(social_df: pd.DataFrame,
                   info: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Score social/archive texts with FinBERT and aggregate per day.

    Accepts either a 'Post' column (live StockTwits) or 'Tweet' (historical
    archive). Labels are mapped to Bullish/Neutral/Bearish so existing reports
    and distribution plots are unchanged.
    """
    df = social_df.copy()
    if df.empty:
        empty = pd.Series(dtype=float, name="Social_Sentiment")
        return (pd.DataFrame(columns=["FinBERT_Label", "FinBERT_Score",
                                      "FinBERT_Confidence", "Engagement_Weight"]),
                empty, pd.DataFrame(columns=AGG_COLUMNS))

    text_col = "Post" if "Post" in df.columns else (
        "Tweet" if "Tweet" in df.columns else None)
    if text_col is None:
        raise ValueError("social frame needs a 'Post' or 'Tweet' text column")

    labels, scores, confs = FinBERTScorer.instance().score(df[text_col].tolist())
    df["FinBERT_Label"] = [_LABEL_MAP.get(lbl, "Neutral") for lbl in labels]
    df["FinBERT_Score"] = scores
    df["FinBERT_Confidence"] = confs
    df["Engagement_Weight"] = _engagement_weights(df)

    agg = aggregate_daily(df, weight_col="Engagement_Weight")
    daily = agg["Avg_Sentiment"].rename("Social_Sentiment")
    return df, daily, agg


# ------------------------------------------------------------------ fusion
def _tz_naive(series: pd.Series) -> pd.Series:
    """Drop any timezone so daily series align with the tz-naive price index."""
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series = series.copy()
        series.index = series.index.tz_localize(None)
    return series


def build_daily_sentiment(news_daily: pd.Series, social_daily: pd.Series,
                          index: pd.DatetimeIndex,
                          social_agg: Optional[pd.DataFrame] = None,
                          news_agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Align sentiment to trading dates and fuse.

    Sentiment is a persisting state: forward-filled across uncovered days, then
    neutral 0.0 where still unknown - strictly past-only, no look-ahead and no
    fabricated values. Combined sentiment renormalizes its weights over the
    sources that actually have coverage, so an absent channel does not dilute
    the signal toward zero.
    """
    index = pd.DatetimeIndex(index).tz_localize(None) if getattr(
        index, "tz", None) is not None else pd.DatetimeIndex(index)

    news = (_tz_naive(news_daily).reindex(index).ffill().fillna(0.0)
            if news_daily is not None and len(news_daily)
            else pd.Series(0.0, index=index))
    social = (_tz_naive(social_daily).reindex(index).ffill().fillna(0.0)
              if social_daily is not None and len(social_daily)
              else pd.Series(0.0, index=index))

    has_news = bool(news_daily is not None and len(news_daily))
    has_social = bool(social_daily is not None and len(social_daily))
    wn, ws = (NEWS_WEIGHT if has_news else 0.0), (SOCIAL_WEIGHT if has_social else 0.0)
    if wn + ws <= 0:
        combined = pd.Series(0.0, index=index)
    else:
        combined = (wn * news + ws * social) / (wn + ws)

    out = pd.DataFrame({
        "News_Sentiment": news.to_numpy(dtype=float),
        "Social_Sentiment": social.to_numpy(dtype=float),
        "Combined_Sentiment": combined.to_numpy(dtype=float),
    }, index=index)

    # ---- richer aggregate features (social preferred, news as fallback)
    src = social_agg if (social_agg is not None and not social_agg.empty) else news_agg
    if src is not None and not src.empty:
        agg = src.copy()
        agg.index = pd.DatetimeIndex(agg.index)
        if agg.index.tz is not None:
            agg.index = agg.index.tz_localize(None)
        agg = agg.reindex(index)
        counts = agg["Text_Count"].fillna(0.0)
        ratios = agg[["Positive_Ratio", "Negative_Ratio", "Neutral_Ratio",
                      "Avg_Confidence"]].ffill().fillna(0.0)
        out["Sent_Pos_Ratio"] = ratios["Positive_Ratio"].to_numpy()
        out["Sent_Neg_Ratio"] = ratios["Negative_Ratio"].to_numpy()
        out["Sent_Neu_Ratio"] = ratios["Neutral_Ratio"].to_numpy()
        out["Sent_Confidence"] = ratios["Avg_Confidence"].to_numpy()
        # volume z-score over a trailing window (past-only)
        roll = counts.rolling(20, min_periods=5)
        out["Sent_Volume_Z"] = ((counts - roll.mean()) / (roll.std() + _EPS)
                                ).fillna(0.0).to_numpy()
        out["Sent_Trend"] = (agg["Sentiment_Trend"].ffill().fillna(0.0).to_numpy())
    else:
        for col in SENTIMENT_AGG_FEATURES:
            out[col] = 0.0

    return out


def save_cache() -> None:
    """Persist the FinBERT cache (call once at the end of a run)."""
    FinBERTScorer.instance().save_cache()