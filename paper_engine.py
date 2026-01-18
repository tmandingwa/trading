# paper_engine.py
from __future__ import annotations
import sqlite3
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

@dataclass
class PaperPosition:
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    units: float
    reason: str

@dataclass
class PaperTrade:
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    sl: float
    tp: float
    units: float
    pnl_usd: float
    outcome: str
    reason: str

class PaperEngine:
    def __init__(self, instrument: str, tf: str, db_path: str = "trades.db", max_trades: int = 200):
        self.instrument = instrument
        self.tf = tf
        self.db_path = db_path
        self.position: Optional[PaperPosition] = None
        self.trades: List[PaperTrade] = []
        self.max_trades = max_trades
        self.last_candle_time: Optional[pd.Timestamp] = None
        self.balance_start: float = 1000.0
        self.balance: float = 1000.0

    def _get_conn(self):
        # Ensure directory exists before connecting
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def load_from_db(self):
        """Restores engine state from the database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT balance, balance_start FROM engine_state WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            row = cursor.fetchone()
            if row:
                self.balance, self.balance_start = row

            cursor.execute("SELECT side, entry_time, entry, sl, tp, units, reason FROM open_positions WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            row = cursor.fetchone()
            if row:
                self.position = PaperPosition(
                    side=row[0], entry_time=pd.to_datetime(row[1]),
                    entry=row[2], sl=row[3], tp=row[4], units=row[5], reason=row[6]
                )

            cursor.execute("""
                SELECT side, entry_time, exit_time, entry, exit_px, sl, tp, units, pnl_usd, outcome, reason 
                FROM trade_history WHERE instrument=? AND tf=? ORDER BY exit_time DESC LIMIT ?
            """, (self.instrument, self.tf, self.max_trades))
            rows = cursor.fetchall()
            self.trades = [PaperTrade(*r) for r in reversed(rows)]

    def _save_state(self):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO engine_state (instrument, tf, balance, balance_start)
                VALUES (?, ?, ?, ?)
            """, (self.instrument, self.tf, self.balance, self.balance_start))

    def reset(self, balance: float):
        self.position = None
        self.trades = []
        self.last_candle_time = None
        self.balance_start = float(balance)
        self.balance = float(balance)
        with self._get_conn() as conn:
            conn.execute("DELETE FROM open_positions WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            conn.execute("DELETE FROM trade_history WHERE instrument=? AND tf=?", (self.instrument, self.tf))
        self._save_state()

    @staticmethod
    def compute_units_quote_usd(balance: float, risk_pct: float, entry: float, sl: float) -> float:
        risk_usd = float(balance) * (float(risk_pct) / 100.0)
        stop_dist = abs(float(entry) - float(sl))
        return float(max(risk_usd / stop_dist, 0.0)) if stop_dist > 0 else 0.0

    @staticmethod
    def units_to_lots(units: float) -> float:
        return float(units) / 100000.0

    def update_on_new_closed_candle(self, candles: pd.DataFrame):
        if candles is None or len(candles) < 2: return
        last_ts = candles.index[-1]
        if self.last_candle_time is not None and last_ts == self.last_candle_time: return
        self.last_candle_time = last_ts
        if self.position is None: return
        
        last = candles.iloc[-1]
        pos = self.position
        hi, lo = float(last["high"]), float(last["low"])
        hit_sl = (lo <= pos.sl) if pos.side == "BUY" else (hi >= pos.sl)
        hit_tp = (hi >= pos.tp) if pos.side == "BUY" else (lo <= pos.tp)

        outcome, exit_px = None, None
        if hit_sl: outcome, exit_px = "SL", pos.sl
        elif hit_tp: outcome, exit_px = "TP", pos.tp

        if outcome:
            pnl = (exit_px - pos.entry) * pos.units if pos.side == "BUY" else (pos.entry - exit_px) * pos.units
            new_trade = PaperTrade(
                side=pos.side, entry_time=str(pos.entry_time), exit_time=str(last_ts),
                entry=float(pos.entry), exit=float(exit_px), sl=float(pos.sl), tp=float(pos.tp),
                units=float(pos.units), pnl_usd=float(pnl), outcome=outcome, reason=pos.reason
            )
            self.trades.append(new_trade)
            self.balance += float(pnl)
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO trade_history (instrument, tf, side, entry_time, exit_time, entry, exit_px, sl, tp, units, pnl_usd, outcome, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (self.instrument, self.tf, new_trade.side, new_trade.entry_time, new_trade.exit_time, 
                      new_trade.entry, new_trade.exit, new_trade.sl, new_trade.tp, new_trade.units, 
                      new_trade.pnl_usd, new_trade.outcome, new_trade.reason))
                conn.execute("DELETE FROM open_positions WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            self._save_state()
            self.position = None

    def maybe_enter(self, state: Dict[str, Any], candles: pd.DataFrame, balance: float, risk_pct: float):
        if self.position is not None or not state or not state.get("ok"): return
        side = state.get("side")
        if side not in ("BUY", "SELL"): return
        plan = state.get("plan", {})
        entry, sl, tp = plan.get("entry"), plan.get("sl"), plan.get("tp")
        if entry is None or sl is None or tp is None: return

        units = self.compute_units_quote_usd(balance=balance, risk_pct=risk_pct, entry=entry, sl=sl)
        if units <= 0: return

        ts = candles.index[-1]
        self.position = PaperPosition(side=side, entry_time=ts, entry=entry, sl=sl, tp=tp, units=float(units), reason=str(state.get("reason", "")))
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO open_positions (instrument, tf, side, entry_time, entry, sl, tp, units, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (self.instrument, self.tf, self.position.side, str(self.position.entry_time), 
                  self.position.entry, self.position.sl, self.position.tp, self.position.units, self.position.reason))

    def summary(self) -> Dict[str, Any]:
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        pnl_total = sum(t.pnl_usd for t in self.trades)
        return {
            "balance_start": float(self.balance_start), "balance": float(self.balance),
            "pnl_total": float(pnl_total), "trades": int(n), "wins": int(wins),
            "win_rate": (wins / n * 100.0) if n else 0.0,
            "open_position": (self.position.__dict__ if self.position else None),
            "trade_log": [t.__dict__ for t in self.trades[-20:]],
        }