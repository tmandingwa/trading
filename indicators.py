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
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def sma(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic(df: pd.DataFrame, k_len: int = 14, d_len: int = 3, smooth: int = 3):
    low_min = df["low"].rolling(k_len, min_periods=k_len).min()
    high_max = df["high"].rolling(k_len, min_periods=k_len).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    k_smooth = k.rolling(smooth, min_periods=smooth).mean()
    d = k_smooth.rolling(d_len, min_periods=d_len).mean()
    return k_smooth, d

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low).abs(),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr_ = tr.rolling(length, min_periods=length).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(length, min_periods=length).mean() / (atr_ + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(length, min_periods=length).mean() / (atr_ + 1e-12)

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    return dx.rolling(length, min_periods=length).mean()

# --- Bill Williams core ---

def alligator(df: pd.DataFrame):
    # Median price
    m = (df["high"] + df["low"]) / 2.0
    # Simple approximation using EMA (SMMA would be closer; EMA works well in practice)
    jaw = ema(m, 13).shift(8)
    teeth = ema(m, 8).shift(5)
    lips = ema(m, 5).shift(3)
    return jaw, teeth, lips

def fractals(df: pd.DataFrame):
    # Classic 2-bar fractal
    h, l = df["high"], df["low"]
    fh = (h.shift(2) < h.shift(1)) & (h.shift(1) < h) & (h > h.shift(-1)) & (h.shift(-1) > h.shift(-2))
    fl = (l.shift(2) > l.shift(1)) & (l.shift(1) > l) & (l < l.shift(-1)) & (l.shift(-1) < l.shift(-2))
    fractal_high = h.where(fh)
    fractal_low = l.where(fl)
    return fractal_low.ffill(), fractal_high.ffill()

def awesome_oscillator(df: pd.DataFrame):
    m = (df["high"] + df["low"]) / 2.0
    return sma(m, 5) - sma(m, 34)

# Existing ones you already had
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
