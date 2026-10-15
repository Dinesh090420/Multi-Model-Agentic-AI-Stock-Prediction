"""Technical indicators and heuristic Elliott-Wave-inspired features.

All indicators are strictly backward-looking (rolling / ewm / diff on past data),
so they introduce no look-ahead bias when used as features at time t.
"""
import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12/26) + signal (9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Moving averages
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # Bollinger Bands (20, 2 sigma)
    df["BB_Mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    high, low, vol = df["High"], df["Low"], df["Volume"]

    # ATR (14) - Wilder's true range, smoothed. Volatility scale in price units.
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(),
                            (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # OBV - cumulative signed volume; a running measure of accumulation.
    direction = np.sign(close.diff()).fillna(0.0)
    df["OBV"] = (direction * vol).fillna(0.0).cumsum()

    # Rolling 20-day VWAP from the typical price (past-only window).
    typical = (high + low + close) / 3.0
    pv = (typical * vol).rolling(20).sum()
    vv = vol.rolling(20).sum()
    df["VWAP_20"] = pv / vv.replace(0.0, np.nan)

    return df


def add_elliott_wave_features(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic Elliott-Wave-INSPIRED features (NOT a formal wave detector).

    Approximates, over a trailing window ending at t:
      - EW_Trend      : +1 impulse-up / -1 corrective-down vs window mean
      - EW_Confidence : trend strength (distance from mean, normalized by range)
      - EW_Fib_Level  : nearest Fibonacci retracement level of the current price
    Uses only past/current data (window [t-w, t]) -> no look-ahead.
    """
    df = df.copy()
    close = df["Close"].values
    n = len(close)
    trend = np.zeros(n)
    confidence = np.zeros(n)
    fib = np.zeros(n)
    fib_ref = [0.236, 0.382, 0.5, 0.618, 0.786]
    w = 20

    for i in range(w, n):
        seg = close[i - w:i + 1]
        lo, hi = seg.min(), seg.max()
        rng = hi - lo + 1e-10
        mean = seg.mean()
        trend[i] = 1.0 if close[i] > mean else -1.0
        confidence[i] = abs(close[i] - mean) / rng
        retrace = (hi - close[i]) / rng
        fib[i] = min(fib_ref, key=lambda x: abs(x - retrace))

    df["EW_Trend"] = trend
    df["EW_Confidence"] = confidence
    df["EW_Fib_Level"] = fib
    return df