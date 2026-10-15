"""Multi-Model Agentic AI System for Stock Market Prediction - entry point.

Two phases share this entry point; nothing was removed to add the second.

  PHASE 1 (default)   python main.py
      Four independently trained BiLSTM experiments (phase1.py):
      YFinance only / +NewsAPI / +Social / full fusion.
      One experiment at a time:  EXPERIMENT=yfinance_only python main.py

  LEGACY / PHASE 2    PHASE=legacy python main.py
      The original pipeline, unchanged: five agents per ticker ->
      LSTM + GRU + Transformer -> ensemble -> decision -> report,
      then the reduced four-section results output.

Training always uses the offline datasets. Live Yahoo Finance + NewsAPI +
StockTwits are used only for the live inference block, in both phases.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List

from agents import (DataAgent, DecisionAgent, PredictionAgent, ReportAgent,
                    SentimentAgent)
from config import (DEVICE, HISTORICAL_PRICE_PATH, HISTORICAL_TWEETS_PATH,
                    RESULTS_DIR, TICKER_ALIASES, TICKERS)
from data_collection import DataValidationError, HistoricalDataStore
from sentiment_analysis import save_cache
from utils import configure_logging, ensure_dirs, get_logger, rpath, set_seed
import visualize as viz

logger = get_logger("main")

PHASE: str = os.getenv("PHASE", "1").strip().lower()


def process_ticker(ticker: str) -> Dict:
    """Run the five agents for one ticker and persist its artefacts."""
    bundle = DataAgent().run(ticker)
    bundle = SentimentAgent().run(bundle)
    bundle = PredictionAgent().run(bundle)
    bundle = DecisionAgent().run(bundle)
    bundle = ReportAgent().run(bundle)

    viz.per_ticker_plots(bundle)
    report_path = rpath("reports", f"{ticker}_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(bundle["report_text"])
    return bundle


def _preflight() -> bool:
    """Validate datasets once, up front, with actionable errors."""
    try:
        store = HistoricalDataStore.instance()
    except DataValidationError as exc:
        logger.error("dataset validation failed: %s", exc)
        logger.error("expected files:\n  %s\n  %s",
                     HISTORICAL_PRICE_PATH, HISTORICAL_TWEETS_PATH)
        return False
    available = set(store.prices["Ticker"].unique())
    for ticker in TICKERS:
        aliases = set(TICKER_ALIASES.get(ticker, [ticker]))
        if not (aliases & available):
            logger.warning("[%s] not present in the price dataset (aliases %s); "
                           "it will be skipped", ticker, sorted(aliases))
    return True


def run_legacy() -> int:
    """The original multi-model pipeline. Unchanged behaviour."""
    logger.info("LEGACY PIPELINE | LSTM + GRU + Transformer + Ensemble")
    if not _preflight():
        return 2

    results: List[Dict] = []
    for ticker in TICKERS:
        logger.info("----- Processing %s -----", ticker)
        try:
            results.append(process_ticker(ticker))
        except Exception as exc:            # one bad ticker must not stop the run
            logger.error("[%s] skipped: %s: %s", ticker, type(exc).__name__, exc)
            logger.debug("traceback:\n%s", traceback.format_exc())

    save_cache()
    if not results:
        logger.error("no tickers processed successfully - aborting")
        return 1

    viz.print_report(results)
    logger.info("Outputs under %s (plots + metrics)", RESULTS_DIR)
    return 0


def main() -> int:
    configure_logging()
    ensure_dirs()
    set_seed()
    logger.info("Multi-Model Agentic AI - Stock Market Prediction")
    logger.info("Universe: %s | device: %s | phase: %s",
                ", ".join(TICKERS), DEVICE, PHASE)

    if PHASE in ("1", "phase1", "one"):
        import phase1                      # imported lazily so legacy runs stay light
        if not _preflight():
            return 2
        return phase1.main()
    return run_legacy()


if __name__ == "__main__":
    sys.exit(main())
