# indicators.py
import numpy as np
import pandas as pd

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/length, adjust=False).mean()
    rd = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs(),], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def bullish_engulf(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    prev_bear = pc < po
    curr_bull = c > o
    engulf = (o <= pc) & (c >= po)
    return (prev_bear & curr_bull & engulf).astype(int)

def bearish_engulf(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    prev_bull = pc > po
    curr_bear = c < o
    engulf = (o >= pc) & (c <= po)
    return (prev_bull & curr_bear & engulf).astype(int)

def swing_levels(df: pd.DataFrame, lookback: int = 3):
    highs, lows = df["high"], df["low"]
    w = 2 * lookback + 1
    swing_high = highs.rolling(w, center=True).max()
    swing_low = lows.rolling(w, center=True).min()
    is_sh = highs == swing_high
    is_sl = lows == swing_low
    resistance = highs.where(is_sh).ffill()
    support = lows.where(is_sl).ffill()
    return support, resistance
