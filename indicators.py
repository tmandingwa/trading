# indicators.py
import numpy as np
import pandas as pd

# ------------------------
# Core indicators (existing)
# ------------------------

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

# ------------------------
# Bill Williams (existing)
# ------------------------

def alligator(df: pd.DataFrame):
    m = (df["high"] + df["low"]) / 2.0
    jaw = ema(m, 13).shift(8)
    teeth = ema(m, 8).shift(5)
    lips = ema(m, 5).shift(3)
    return jaw, teeth, lips

def fractals(df: pd.DataFrame):
    h, l = df["high"], df["low"]
    fh = (h.shift(2) < h.shift(1)) & (h.shift(1) < h) & (h > h.shift(-1)) & (h.shift(-1) > h.shift(-2))
    fl = (l.shift(2) > l.shift(1)) & (l.shift(1) > l) & (l < l.shift(-1)) & (l.shift(-1) < l.shift(-2))
    fractal_high = h.where(fh)
    fractal_low = l.where(fl)
    return fractal_low.ffill(), fractal_high.ffill()

def awesome_oscillator(df: pd.DataFrame):
    m = (df["high"] + df["low"]) / 2.0
    return sma(m, 5) - sma(m, 34)

# ------------------------
# NEW: Bollinger Bands + StdDev regime
# ------------------------

def bollinger(close: pd.Series, length: int = 20, n_std: float = 2.0):
    mid = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / (mid.abs() + 1e-12)
    return mid, upper, lower, sd, width

def rolling_zscore(x: pd.Series, length: int = 50) -> pd.Series:
    m = x.rolling(length, min_periods=length).mean()
    s = x.rolling(length, min_periods=length).std(ddof=0)
    return (x - m) / (s + 1e-12)

# ------------------------
# NEW: Adaptive MA (KAMA)
# ------------------------

def kama(close: pd.Series, er_len: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average.
    - er_len: efficiency ratio window
    - fast/slow: smoothing constants bounds
    """
    change = close.diff(er_len).abs()
    volatility = close.diff().abs().rolling(er_len, min_periods=er_len).sum()
    er = change / (volatility + 1e-12)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = pd.Series(index=close.index, dtype=float)
    out.iloc[:er_len] = np.nan

    # seed
    first = close.iloc[er_len]
    out.iloc[er_len] = first

    for i in range(er_len + 1, len(close)):
        prev = out.iloc[i - 1]
        if not np.isfinite(prev):
            prev = close.iloc[i - 1]
        out.iloc[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)

    return out

# ------------------------
# NEW: Parabolic SAR
# ------------------------

def psar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    Parabolic SAR (classic). Returns SAR series.
    """
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    sar = np.full(n, np.nan, dtype=float)
    if n < 3:
        return pd.Series(sar, index=df.index)

    # initial trend guess
    up = df["close"].iloc[1] > df["close"].iloc[0]
    ep = high[0] if up else low[0]
    af = af_step
    sar[0] = low[0] if up else high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        # basic SAR step
        sar_i = prev_sar + af * (ep - prev_sar)

        # clamp with prior 2 lows/highs
        if up:
            if i >= 2:
                sar_i = min(sar_i, low[i - 1], low[i - 2])
            else:
                sar_i = min(sar_i, low[i - 1])
        else:
            if i >= 2:
                sar_i = max(sar_i, high[i - 1], high[i - 2])
            else:
                sar_i = max(sar_i, high[i - 1])

        # check reversal
        if up:
            if low[i] < sar_i:
                up = False
                sar_i = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            if high[i] > sar_i:
                up = True
                sar_i = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        sar[i] = sar_i

    return pd.Series(sar, index=df.index)

# ------------------------
# Existing patterns (kept)
# ------------------------

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
