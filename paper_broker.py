# paper_broker.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import math
import time

@dataclass
class Position:
    instrument: str
    tf: str
    side: str               # BUY / SELL
    units: float
    entry: float
    sl: float
    tp: float
    opened_ts: str
    opened_bar_ts: str

@dataclass
class Trade:
    instrument: str
    tf: str
    side: str
    units: float
    entry: float
    exit: float
    sl: float
    tp: float
    pnl: float
    opened_ts: str
    closed_ts: str
    opened_bar_ts: str
    closed_bar_ts: str
    outcome: str           # TP / SL / MANUAL

class PaperBroker:
    def __init__(self, start_balance: float, risk_pct: float, max_units: float, max_log: int = 200):
        self.start_balance = float(start_balance)
        self.balance = float(start_balance)
        self.equity = float(start_balance)
        self.risk_pct = float(risk_pct)
        self.max_units = float(max_units)
        self.max_log = int(max_log)

        # key: (instrument, tf)
        self.open_pos: Dict[Tuple[str, str], Position] = {}
        self.trades: List[Trade] = []

        self.last_price: Dict[str, float] = {}

    def _calc_units(self, entry: float, sl: float) -> float:
        risk_usd = self.balance * self.risk_pct
        dist = abs(entry - sl)
        if dist <= 0:
            return 0.0
        units = risk_usd / dist  # USD account assumption (works well for *_USD and XAU_USD)
        units = min(units, self.max_units)
        return float(max(units, 0.0))

    def on_tick(self, instrument: str, mid: float):
        self.last_price[instrument] = float(mid)
        # update equity (mark-to-market)
        eq = self.balance
        for (inst, tf), pos in self.open_pos.items():
            if inst != instrument:
                continue
            px = float(mid)
            if pos.side == "BUY":
                eq += (px - pos.entry) * pos.units
            else:
                eq += (pos.entry - px) * pos.units
        self.equity = float(eq)

        # check SL/TP hits
        to_close = []
        for key, pos in self.open_pos.items():
            if pos.instrument != instrument:
                continue
            px = float(mid)

            if pos.side == "BUY":
                if px <= pos.sl:
                    to_close.append((key, px, "SL"))
                elif px >= pos.tp:
                    to_close.append((key, px, "TP"))
            else:  # SELL
                if px >= pos.sl:
                    to_close.append((key, px, "SL"))
                elif px <= pos.tp:
                    to_close.append((key, px, "TP"))

        for key, exit_px, outcome in to_close:
            self._close_position(key, exit_px, outcome)

    def maybe_open_on_signal(self, instrument: str, tf: str, state: Dict[str, Any]):
        """
        Called on candle-close events.
        Opens trade if:
          - PAPER enabled in caller
          - state.side is BUY/SELL
          - no open position for (instrument, tf)
        """
        key = (instrument, tf)
        if key in self.open_pos:
            return

        if not state.get("ok"):
            return
        side = state.get("side", "NO_TRADE")
        if side not in ("BUY", "SELL"):
            return

        plan = state.get("plan") or {}
        entry = plan.get("entry")
        sl = plan.get("sl")
        tp = plan.get("tp")
        if entry is None or sl is None or tp is None:
            return

        units = self._calc_units(float(entry), float(sl))
        if units <= 0:
            return

        now_ts = str(time.time())
        pos = Position(
            instrument=instrument,
            tf=tf,
            side=side,
            units=units,
            entry=float(entry),
            sl=float(sl),
            tp=float(tp),
            opened_ts=now_ts,
            opened_bar_ts=state.get("timestamp", ""),
        )
        self.open_pos[key] = pos

    def _close_position(self, key: Tuple[str, str], exit_px: float, outcome: str):
        pos = self.open_pos.pop(key, None)
        if pos is None:
            return

        if pos.side == "BUY":
            pnl = (float(exit_px) - pos.entry) * pos.units
        else:
            pnl = (pos.entry - float(exit_px)) * pos.units

        self.balance += float(pnl)
        self.equity = self.balance

        tr = Trade(
            instrument=pos.instrument,
            tf=pos.tf,
            side=pos.side,
            units=pos.units,
            entry=pos.entry,
            exit=float(exit_px),
            sl=pos.sl,
            tp=pos.tp,
            pnl=float(pnl),
            opened_ts=pos.opened_ts,
            closed_ts=str(time.time()),
            opened_bar_ts=pos.opened_bar_ts,
            closed_bar_ts="",  # we can fill later if you want bar time on close
            outcome=outcome,
        )
        self.trades.append(tr)
        if len(self.trades) > self.max_log:
            self.trades = self.trades[-self.max_log:]

    def snapshot(self) -> Dict[str, Any]:
        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = sum(1 for t in self.trades if t.pnl <= 0)
        total = len(self.trades)
        win_rate = (wins / total) if total else 0.0

        return {
            "start_balance": self.start_balance,
            "balance": self.balance,
            "equity": self.equity,
            "risk_pct": self.risk_pct,
            "max_units": self.max_units,
            "open_positions": [asdict(p) for p in self.open_pos.values()],
            "trades": [asdict(t) for t in self.trades[-50:]],  # send last 50
            "stats": {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "net_pnl": self.balance - self.start_balance,
            },
        }
