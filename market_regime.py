"""Explicit market-regime modelling - leakage-free.

Two independent axes, each computed only from PAST/CURRENT data (no look-ahead):
  Trend      : Bull / Bear / Sideways   from trailing 20-day momentum
  Volatility : HighVol / LowVol         from 20-day realized vol vs its trailing median

Output columns (attached to the stock frame and fed to the models as REAL features):
  Regime_Bull, Regime_Bear, Regime_Sideways, Regime_HighVol, Regime_LowVol  (one-hot 0/1)
  Trend_Regime, Vol_Regime (strings), Regime_Label ("Bull / HighVol")

One-hot encoding is used (not ordinal integers) so the model does not treat the
categories as a meaningful continuous scale.
"""
import pandas as pd
from config import REGIME_FEATURES


def detect_regime(df, mom_window=20, vol_window=20, vol_lookback=120, mom_thr=0.02):
    close = df["Close"]
    ret = close.pct_change()

    momentum = close.pct_change(mom_window)          # trailing return, past-only
    vol = ret.rolling(vol_window).std()              # realized vol, past-only
    vol_thr = vol.rolling(vol_lookback, min_periods=vol_window).median()  # adaptive, past-only

    trend = pd.Series("Sideways", index=df.index)
    trend[momentum > mom_thr] = "Bull"
    trend[momentum < -mom_thr] = "Bear"

    volr = pd.Series("LowVol", index=df.index)
    volr[vol > vol_thr] = "HighVol"

    out = pd.DataFrame(0, index=df.index, columns=REGIME_FEATURES, dtype=float)
    out.loc[trend == "Bull", "Regime_Bull"] = 1.0
    out.loc[trend == "Bear", "Regime_Bear"] = 1.0
    out.loc[trend == "Sideways", "Regime_Sideways"] = 1.0
    out.loc[volr == "HighVol", "Regime_HighVol"] = 1.0
    out.loc[volr == "LowVol", "Regime_LowVol"] = 1.0

    out["Trend_Regime"] = trend
    out["Vol_Regime"] = volr
    out["Regime_Label"] = trend.str.cat(volr, sep=" / ")
    return out


def regime_trend_score(trend_label):
    """Map trend regime to a numeric tilt for cross-sectional ranking."""
    return {"Bull": 1.0, "Sideways": 0.0, "Bear": -1.0}.get(trend_label, 0.0)
