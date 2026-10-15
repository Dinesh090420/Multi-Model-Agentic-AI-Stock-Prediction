"""Feature fusion + leakage-free sequence construction.

Fuses: price + technical indicators + Elliott-wave + sentiment + regime one-hot
into one feature matrix, builds look-back sequences, and scales using scalers
fit on the TRAINING ROWS ONLY (no leakage from val/test).

Target: next-day (HORIZON) close price.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import (PRICE_FEATURES, TECH_FEATURES, EW_FEATURES,
                    SENTIMENT_FEATURES, SENTIMENT_AGG_FEATURES, REGIME_FEATURES,
                    DERIVED_FEATURES, SEQUENCE_LENGTH, HORIZON, TRAIN_FRAC,
                    VAL_FRAC, MIN_SEQUENCES)
from utils import get_logger

logger = get_logger("features")

FEATURE_COLUMNS = (PRICE_FEATURES + TECH_FEATURES + EW_FEATURES +
                   SENTIMENT_FEATURES + SENTIMENT_AGG_FEATURES +
                   REGIME_FEATURES + DERIVED_FEATURES)

_EPS = 1e-9


def _add_derived_features(df):
    """Engineered fusion features - all strictly past-only (no look-ahead)."""
    close, vol = df["Close"], df["Volume"]
    df["Ret_1"] = close.pct_change()
    df["Ret_5"] = close.pct_change(5)
    df["Vol_10"] = df["Ret_1"].rolling(10).std()
    df["Px_SMA20"] = close / df["SMA_20"] - 1.0
    df["BB_PctB"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + _EPS)
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["Vol_Z"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + _EPS)
    df["Sent_Mom"] = df["Combined_Sentiment"].diff()
    return df


def build_feature_frame(stock_df, sentiment_daily, regime_df):
    """Fuse price + indicators + Elliott-wave + sentiment + regime + derived features.

    Sentiment and regime one-hot are attached first (derived Sent_Mom needs sentiment),
    then engineered features are computed, then warm-up NaN rows are dropped.
    """
    df = stock_df.copy()
    for c in SENTIMENT_FEATURES + SENTIMENT_AGG_FEATURES:
        if c in sentiment_daily.columns:
            df[c] = sentiment_daily[c].reindex(df.index).fillna(0.0).to_numpy()
        else:                       # source without aggregates -> neutral column
            df[c] = 0.0
    for c in REGIME_FEATURES:
        df[c] = regime_df[c].reindex(df.index).fillna(0.0).to_numpy()
    df = _add_derived_features(df)
    rows_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    validate_feature_frame(df, rows_before)
    return df


def validate_feature_frame(df, rows_before=None):
    """Fail loudly on structural problems; warn on constant/near-constant columns.

    Catches the failure modes that silently degrade a model: missing columns,
    residual NaN/inf, non-monotonic dates (breaks chronological splitting) and
    zero-variance features (which can never receive gradient).
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    if df.empty:
        raise ValueError("feature frame is empty after warm-up/NaN removal")
    if not df.index.is_monotonic_increasing:
        raise ValueError("feature frame index is not chronologically sorted")
    if df.index.has_duplicates:
        raise ValueError("feature frame index contains duplicate dates")
    block = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(block).all():
        raise ValueError("feature frame still contains NaN/inf after cleaning")
    if rows_before is not None and rows_before > len(df):
        logger.info("feature fusion: %s warm-up/NaN rows dropped (%s -> %s)",
                    rows_before - len(df), rows_before, len(df))
    constant = [c for c in FEATURE_COLUMNS if float(df[c].std()) < 1e-12]
    if constant:
        logger.warning("constant features (no variance -> zero gradient, "
                       "importance is unmeasurable): %s", constant)
    return True


def make_dataset(feature_df, feature_columns=None, seq_len=None):
    """Return a dict of arrays + train-only scalers, split-ready.

    Windows end at index i (features known up to i). The TARGET is the next-day
    RETURN r = (Close[i+HORIZON] - Close[i]) / Close[i] - a stationary, bounded
    target that behaves far better than raw price levels and makes directional
    accuracy meaningful. Predicted price is recovered as current_close*(1+return).

    Feature scaler (MinMax) and return scaler (Standard) are fit on rows / targets
    from the TRAINING portion only - no leakage from val/test.
    """
    cols = list(feature_columns) if feature_columns else FEATURE_COLUMNS
    L = int(seq_len or SEQUENCE_LENGTH)          # tunable look-back window
    feats = feature_df[cols].to_numpy(dtype=float)
    close_raw = feature_df["Close"].to_numpy(dtype=float)
    dates_index = feature_df.index
    T = len(feature_df)

    start_i = L - 1
    valid = list(range(start_i, T - HORIZON))
    N = len(valid)
    if N < MIN_SEQUENCES:
        raise ValueError(
            f"insufficient data: only {N} usable sequences (need >= "
            f"{MIN_SEQUENCES}). Reduce SEQUENCE_LENGTH or supply more history.")

    n_train = int(N * TRAIN_FRAC)
    n_val = int(N * VAL_FRAC)
    logger.info("dataset: %s sequences -> train %s / val %s / test %s "
                "(chronological, no shuffling)",
                N, n_train, n_val, N - n_train - n_val)

    # rows usable for fitting the feature scaler = up to the last TRAIN window end
    last_train_i = valid[n_train - 1]
    cutoff = last_train_i + 1
    feat_scaler = MinMaxScaler().fit(feats[:cutoff])
    feats_s = feat_scaler.transform(feats)

    cur, fut, prev, dates = [], [], [], []
    for i in valid:
        cur.append(close_raw[i])
        fut.append(close_raw[i + HORIZON])
        prev.append(close_raw[i - 1])
        dates.append(dates_index[i + HORIZON])
    cur = np.asarray(cur, dtype=float)
    fut = np.asarray(fut, dtype=float)
    prev = np.asarray(prev, dtype=float)
    dates = np.asarray(dates)
    actual_return = (fut - cur) / cur

    # return target scaled with a StandardScaler fit on TRAIN returns only
    ret_scaler = StandardScaler().fit(actual_return[:n_train].reshape(-1, 1))
    y = ret_scaler.transform(actual_return.reshape(-1, 1)).ravel().astype(np.float32)

    X = np.asarray([feats_s[i - L + 1: i + 1] for i in valid],
                   dtype=np.float32)

    return {
        "X": X, "y": y,
        "current_close": cur, "future_close": fut, "prev_close": prev,
        "dates": dates, "actual_return": actual_return,
        "n_train": n_train, "n_val": n_val, "N": N,
        "feat_scaler": feat_scaler, "ret_scaler": ret_scaler,
        "feature_columns": cols, "seq_len": L,
        "input_size": X.shape[2],
    }


def split_bounds(data):
    """Return (train_slice, val_slice, test_slice) over the sequence axis."""
    N, n_train, n_val = data["N"], data["n_train"], data["n_val"]
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, N)
    return tr, va, te