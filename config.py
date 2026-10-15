"""Central configuration for the Multi-Model Agentic AI stock prediction system.

Two strictly separated data regimes:

  HISTORICAL (training + evaluation) - offline CSV datasets ONLY. No API is
      called during training, so runs are reproducible and leakage-free.
          stock_tweets.csv         Date, Tweet, Stock Name, Company Name
          stock_yfinance_data.csv  Date, Open, High, Low, Close, Adj Close,
                                   Volume, Stock Name

  LIVE (inference only) - Yahoo Finance + NewsAPI + StockTwits, used solely to
      build the current feature window for a live BUY/SELL/HOLD signal from the
      already-trained models.

Secrets are read from environment variables. Never commit real keys here.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, List

# ---------------------------------------------------------------- paths
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")

HISTORICAL_TWEETS_PATH: str = os.getenv(
    "HISTORICAL_TWEETS_PATH", os.path.join(DATA_DIR, "stock_tweets.csv"))
HISTORICAL_PRICE_PATH: str = os.getenv(
    "HISTORICAL_PRICE_PATH", os.path.join(DATA_DIR, "stock_yfinance_data.csv"))

RESULTS_DIR: str = os.getenv("OUTPUT_PATH", os.path.join(BASE_DIR, "results"))
OUTPUT_PATH: str = RESULTS_DIR                       # alias required by spec
LOG_PATH: str = os.getenv("LOG_PATH", os.path.join(RESULTS_DIR, "logs"))
CACHE_DIR: str = os.path.join(BASE_DIR, ".cache")

# ---------------------------------------------------------------- universe
TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL"]

# Dataset symbols that denote the same company as a project ticker.
TICKER_ALIASES: Dict[str, List[str]] = {
    "AAPL": ["AAPL"],
    "MSFT": ["MSFT"],
    "GOOGL": ["GOOGL", "GOOG"],          # Kaggle dataset labels Alphabet "GOOG"
}

# ---------------------------------------------------------------- live window
START_DATE: str = "2018-01-01"
END_DATE: str = (date.today() + timedelta(days=1)).isoformat()   # yfinance end is exclusive

# ---------------------------------------------------------------- sequence / target
SEQUENCE_LENGTH: int = 30      # look-back window (trading days)
HORIZON: int = int(os.getenv("HORIZON", "1"))   # 1 = next day, 5 = one week ahead

TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
# test = remaining 0.15

MIN_SEQUENCES: int = 60        # skip a ticker below this many usable windows

# ---------------------------------------------------------------- model / training
EPOCHS: int = 35
BATCH_SIZE: int = 32
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 2
DROPOUT: float = 0.2
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-5
EARLY_STOP_PATIENCE: int = 7
LR_SCHED_FACTOR: float = 0.5
LR_SCHED_PATIENCE: int = 3
GRAD_CLIP_NORM: float = 1.0    # stabilizes recurrent training
SEED: int = 42

DEVICE_PREFERENCE: str = os.getenv("DEVICE", "auto")
USE_AMP: bool = True           # mixed precision, applied on CUDA only


def _resolve_device(preference: str) -> str:
    """Pick the best available torch device; falls back to CPU safely."""
    if preference and preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE: str = _resolve_device(DEVICE_PREFERENCE)

DO_HPO: bool = True
HPO_EPOCHS: int = 10
HPO_GRID: List[Dict[str, float]] = [
    {"hidden_size": 64, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
    {"hidden_size": 96, "num_layers": 2, "dropout": 0.3, "lr": 5e-4, "batch_size": 64},
]

WF_SPLITS: int = 3
WF_EPOCHS: int = 8

# ---------------------------------------------------------------- decision thresholds
# BUY/SELL/HOLD band on the ACTUAL next-day return. HOLD is |r| < band.
# Settable so the same run can be repeated across bands for a sensitivity
# analysis:  SIGNAL_BAND=0.015 python phase2.py
# Report every band you run, not the one that scores best - choosing a band
# after seeing test results is selection on the test set.
SIGNAL_BAND: float = float(os.getenv("SIGNAL_BAND", "0.005"))
BUY_THRESHOLD: float = SIGNAL_BAND
SELL_THRESHOLD: float = -SIGNAL_BAND
THRESH_QUANTILES = (0.30, 0.35, 0.40)
MIN_SIGNAL_SUPPORT: int = 5

# ---------------------------------------------------------------- feature groups
PRICE_FEATURES: List[str] = ["Open", "High", "Low", "Close", "Volume"]
TECH_FEATURES: List[str] = [
    "RSI", "MACD", "MACD_Signal", "SMA_20", "SMA_50", "EMA_20",
    "BB_Upper", "BB_Mid", "BB_Lower", "ATR_14", "OBV", "VWAP_20",
]
EW_FEATURES: List[str] = ["EW_Trend", "EW_Confidence", "EW_Fib_Level"]

# Names kept exactly so existing reports, plots and CSVs stay unchanged.
SENTIMENT_FEATURES: List[str] = ["News_Sentiment", "Social_Sentiment",
                                 "Combined_Sentiment"]
# Richer daily aggregates produced by SentimentAgent (all past-only).
SENTIMENT_AGG_FEATURES: List[str] = [
    "Sent_Pos_Ratio", "Sent_Neg_Ratio", "Sent_Neu_Ratio",
    "Sent_Confidence", "Sent_Volume_Z", "Sent_Trend",
]
REGIME_FEATURES: List[str] = ["Regime_Bull", "Regime_Bear", "Regime_Sideways",
                              "Regime_HighVol", "Regime_LowVol"]
DERIVED_FEATURES: List[str] = ["Ret_1", "Ret_5", "Vol_10", "Px_SMA20", "BB_PctB",
                               "MACD_Hist", "Vol_Z", "Sent_Mom"]

NEWS_WEIGHT: float = 0.6
SOCIAL_WEIGHT: float = 0.4

# ---------------------------------------------------------------- sentiment engine
FINBERT_MODEL: str = "ProsusAI/finbert"
MODEL_PATH: str = os.getenv("MODEL_PATH", FINBERT_MODEL)   # local dir or HF id
FINBERT_BATCH_SIZE: int = 32
FINBERT_MAX_LENGTH: int = 256
SENTIMENT_CACHE_PATH: str = os.path.join(CACHE_DIR, "finbert_cache.json")
USE_SENTIMENT_CACHE: bool = True
# Deterministic per-day cap on historical tweets: keeps every trading day
# represented while bounding FinBERT runtime. Set to 0 to disable.
MAX_TWEETS_PER_DAY: int = 40

# ---------------------------------------------------------------- live sources
LIVE_PREDICTION_ENABLED: bool = True
# The NewsAPI key is read ONLY from the environment (or a git-ignored .env).
# It is deliberately NOT stored in this file, so it cannot be committed.
# Set it with:  export NEWSAPI_KEY="your_key"   (see .env.example)
NEWS_API_KEY: str = os.getenv("NEWSAPI_KEY", os.getenv("NEWS_API_KEY", "")).strip()
NEWSAPI_KEY: str = NEWS_API_KEY                      # backwards-compatible alias
# StockTwits live social sentiment via Playwright scraping of the public
# symbol pages (no API key). {ticker} is filled from TICKERS automatically.
STOCKTWITS_SYMBOL_URL: str = "https://stocktwits.com/symbol/{ticker}"
STOCKTWITS_SCRAPE_SCROLLS: int = 5          # infinite-scroll passes per page
STOCKTWITS_PAGE_TIMEOUT_MS: int = 20000     # per-page load/selector timeout
STOCKTWITS_HEADLESS: bool = True            # set False to watch the browser
NEWSAPI_URL: str = "https://newsapi.org/v2/everything"
HTTP_TIMEOUT: int = 10
HTTP_RETRIES: int = 2
HTTP_BACKOFF: float = 1.5

SOURCE_CREDIBILITY: Dict[str, float] = {
    "reuters": 1.0, "bloomberg": 1.0, "the wall street journal": 0.95,
    "financial times": 0.95, "cnbc": 0.8, "marketwatch": 0.75,
    "the motley fool": 0.6, "seeking alpha": 0.6, "yahoo entertainment": 0.5,
    "business insider": 0.6, "benzinga": 0.55,
}
DEFAULT_CREDIBILITY: float = 0.5

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")