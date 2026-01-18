# candle_agg.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Optional

TF_TO_PANDAS = {
    "M1": "1min",
    "M5": "5min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "H6": "6h",
}

def floor_ts(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    return ts.floor(TF_TO_PANDAS[tf])

@dataclass
class Candle:
    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

class CandleAggregator:
    """
    Aggregates mid prices to OHLC candles for a timeframe.
    Stores only CLOSED candles in self.closed.
    update() returns:
      - None if candle not closed
      - Candle if candle was just closed (so caller can act on close events)
    """
    def __init__(self, tf: str, max_candles: int = 800):
        self.tf = tf
        self.max_candles = max_candles
        self.current: Optional[Candle] = None
        self.closed: list[Candle] = []

    def seed_from_ohlc_df(self, df: pd.DataFrame):
        self.closed = []
        if df is None or df.empty:
            self.current = None
            return

        for ts, row in df.iterrows():
            self.closed.append(Candle(
                start=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            ))

        if len(self.closed) > self.max_candles:
            self.closed = self.closed[-self.max_candles:]
        self.current = None

    def update(self, ts: pd.Timestamp, price: float) -> Optional[Candle]:
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        start = floor_ts(ts, self.tf)

        if self.current is None:
            self.current = Candle(start=start, open=price, high=price, low=price, close=price)
            return None

        # new candle started => close the previous one
        if start > self.current.start:
            just_closed = self.current
            self.closed.append(just_closed)
            if len(self.closed) > self.max_candles:
                self.closed = self.closed[-self.max_candles:]
            self.current = Candle(start=start, open=price, high=price, low=price, close=price)
            return just_closed

        # update current candle
        self.current.high = max(self.current.high, price)
        self.current.low = min(self.current.low, price)
        self.current.close = price
        return None

    def to_df(self) -> pd.DataFrame:
        if not self.closed:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        idx = [c.start for c in self.closed]
        df = pd.DataFrame(
            {
                "open": [c.open for c in self.closed],
                "high": [c.high for c in self.closed],
                "low": [c.low for c in self.closed],
                "close": [c.close for c in self.closed],
            },
            index=pd.DatetimeIndex(idx, tz="UTC"),
        )
        return df.sort_index()
