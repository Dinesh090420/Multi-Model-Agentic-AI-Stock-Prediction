"""Shared infrastructure: logging, reproducibility and results-folder management."""
from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np

from config import CACHE_DIR, LOG_LEVEL, LOG_PATH, RESULTS_DIR, SEED

_SUBDIRS = ("plots", "metrics", "tables", "reports")
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_configured = False


def ensure_dirs() -> None:
    """Create results/{plots,metrics,tables,reports}, logs and cache dirs."""
    for sub in _SUBDIRS:
        os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def configure_logging(level: Optional[str] = None) -> str:
    """Attach console + rotating-per-run file handlers. Idempotent.

    Returns the path of the log file for this run.
    """
    global _configured
    ensure_dirs()
    log_file = os.path.join(
        LOG_PATH, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    if _configured:
        return log_file

    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO))
    fmt = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(fh)

    # third-party noise
    for noisy in ("urllib3", "yfinance", "transformers", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Module-level logger; configures logging on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


def set_seed(seed: int = SEED) -> None:
    """Make runs reproducible across random / numpy / torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:                                   # torch optional at import
        pass


def rpath(sub: str, name: str) -> str:
    """Path inside a results subfolder, e.g. rpath('plots', 'AAPL_x.png')."""
    return os.path.join(RESULTS_DIR, sub, name)