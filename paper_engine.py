# paper_engine.py
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
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

    # ✅ NEW: ML probability at entry time
    model_prob: Optional[float] = None

    # trade management state
    r_dist: float = 0.0
    be_moved: bool = False
    mfv: float = 0.0
    mfa: float = 0.0


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

    # ✅ NEW: ML probability captured at entry
    model_prob: Optional[float] = None


class PaperEngine:
    """
    Paper trading engine with:
      - one open position per (instrument, tf)
      - cooldown after SL
      - break-even at +1R
      - trailing after +1.5R
      - sqlite persistence + schema auto-ensure
      - ✅ ML model_prob stored per trade
    """

    def __init__(self, instrument: str, tf: str, db_path: str = "trades.db", max_trades: int = 200):
        self.instrument = instrument
        self.tf = tf
        self.db_path = db_path
        self.max_trades = max_trades

        self.position: Optional[PaperPosition] = None
        self.trades: List[PaperTrade] = []
        self.last_candle_time: Optional[pd.Timestamp] = None

        self.balance_start: float = 1000.0
        self.balance: float = 1000.0

        self.cooldown_until: Optional[pd.Timestamp] = None

        with self._get_conn() as conn:
            self._ensure_schema(conn)

    # -----------------------------
    # DB helpers / schema
    # -----------------------------
    def _get_conn(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    @staticmethod
    def _table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in rows]

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engine_state (
                instrument TEXT NOT NULL,
                tf TEXT NOT NULL,
                balance REAL NOT NULL,
                balance_start REAL NOT NULL,
                cooldown_until TEXT,
                PRIMARY KEY (instrument, tf)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                instrument TEXT NOT NULL,
                tf TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                units REAL NOT NULL,
                reason TEXT,

                -- ✅ NEW
                model_prob REAL,

                r_dist REAL,
                be_moved INTEGER,
                mfv REAL,
                mfa REAL,
                PRIMARY KEY (instrument, tf)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                tf TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry REAL NOT NULL,
                exit_px REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                units REAL NOT NULL,
                pnl_usd REAL NOT NULL,
                outcome TEXT NOT NULL,
                reason TEXT,

                -- ✅ NEW
                model_prob REAL
            )
        """)

        # lightweight migrations
        cols = self._table_cols(conn, "engine_state")
        if "cooldown_until" not in cols:
            conn.execute("ALTER TABLE engine_state ADD COLUMN cooldown_until TEXT")

        cols = self._table_cols(conn, "open_positions")
        for col, ddl in [
            ("r_dist", "ALTER TABLE open_positions ADD COLUMN r_dist REAL"),
            ("be_moved", "ALTER TABLE open_positions ADD COLUMN be_moved INTEGER"),
            ("mfv", "ALTER TABLE open_positions ADD COLUMN mfv REAL"),
            ("mfa", "ALTER TABLE open_positions ADD COLUMN mfa REAL"),
            ("model_prob", "ALTER TABLE open_positions ADD COLUMN model_prob REAL"),
        ]:
            if col not in cols:
                conn.execute(ddl)

        cols = self._table_cols(conn, "trade_history")
        if "model_prob" not in cols:
            conn.execute("ALTER TABLE trade_history ADD COLUMN model_prob REAL")

        conn.commit()

    def load_from_db(self):
        with self._get_conn() as conn:
            self._ensure_schema(conn)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT balance, balance_start, cooldown_until FROM engine_state WHERE instrument=? AND tf=?",
                (self.instrument, self.tf),
            )
            row = cursor.fetchone()
            if row:
                self.balance, self.balance_start = float(row[0]), float(row[1])
                self.cooldown_until = pd.to_datetime(row[2]) if row[2] else None

            cursor.execute("""
                SELECT side, entry_time, entry, sl, tp, units, reason, model_prob,
                       r_dist, be_moved, mfv, mfa
                FROM open_positions WHERE instrument=? AND tf=?
            """, (self.instrument, self.tf))
            row = cursor.fetchone()
            if row:
                self.position = PaperPosition(
                    side=row[0],
                    entry_time=pd.to_datetime(row[1]),
                    entry=float(row[2]),
                    sl=float(row[3]),
                    tp=float(row[4]),
                    units=float(row[5]),
                    reason=str(row[6] or ""),
                    model_prob=(float(row[7]) if row[7] is not None else None),
                    r_dist=float(row[8] or abs(float(row[2]) - float(row[3]))),
                    be_moved=bool(int(row[9] or 0)),
                    mfv=float(row[10] or float(row[2])),
                    mfa=float(row[11] or float(row[2])),
                )

            cursor.execute("""
                SELECT side, entry_time, exit_time, entry, exit_px, sl, tp, units, pnl_usd, outcome, reason, model_prob
                FROM trade_history
                WHERE instrument=? AND tf=?
                ORDER BY exit_time DESC
                LIMIT ?
            """, (self.instrument, self.tf, self.max_trades))
            rows = cursor.fetchall()
            self.trades = [PaperTrade(*r) for r in reversed(rows)]

    def _save_state(self):
        with self._get_conn() as conn:
            self._ensure_schema(conn)
            conn.execute("""
                INSERT OR REPLACE INTO engine_state (instrument, tf, balance, balance_start, cooldown_until)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.instrument,
                self.tf,
                float(self.balance),
                float(self.balance_start),
                str(self.cooldown_until) if self.cooldown_until is not None else None
            ))
            conn.commit()

    def reset(self, balance: float):
        self.position = None
        self.trades = []
        self.last_candle_time = None
        self.balance_start = float(balance)
        self.balance = float(balance)
        self.cooldown_until = None

        with self._get_conn() as conn:
            self._ensure_schema(conn)
            conn.execute("DELETE FROM open_positions WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            conn.execute("DELETE FROM trade_history WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            conn.execute("DELETE FROM engine_state WHERE instrument=? AND tf=?", (self.instrument, self.tf))
            conn.commit()

        self._save_state()

    # -----------------------------
    # Risk sizing helpers
    # -----------------------------
    @staticmethod
    def compute_units_quote_usd(balance: float, risk_pct: float, entry: float, sl: float) -> float:
        risk_usd = float(balance) * (float(risk_pct) / 100.0)
        stop_dist = abs(float(entry) - float(sl))
        return float(max(risk_usd / stop_dist, 0.0)) if stop_dist > 0 else 0.0

    @staticmethod
    def units_to_lots(units: float) -> float:
        return float(units) / 100000.0

    # -----------------------------
    # Timeframe helpers
    # -----------------------------
    def _tf_minutes(self) -> int:
        tf = str(self.tf).upper().strip()
        if tf.startswith("M"):
            return int(tf.replace("M", ""))
        if tf.startswith("H"):
            return int(tf.replace("H", "")) * 60
        if tf.startswith("D"):
            return 24 * 60
        return 15

    def _set_cooldown_after_sl(self, last_ts: pd.Timestamp, cooldown_bars: int):
        mins = self._tf_minutes()
        self.cooldown_until = pd.to_datetime(last_ts) + pd.Timedelta(minutes=int(mins * cooldown_bars))

    def cooldown_active(self, ts: pd.Timestamp) -> bool:
        if self.cooldown_until is None:
            return False
        return pd.to_datetime(ts) < pd.to_datetime(self.cooldown_until)

    def build_strategy_cfg(self) -> Dict[str, Any]:
        return {
            "COOLDOWN_UNTIL": str(self.cooldown_until) if self.cooldown_until is not None else None
        }

    # -----------------------------
    # Trade management rules
    # -----------------------------
    def _manage_open_position(self, candles: pd.DataFrame, plan: Optional[Dict[str, Any]] = None) -> None:
        if self.position is None or candles is None or len(candles) < 2:
            return

        pos = self.position
        last = candles.iloc[-1]
        hi, lo = float(last["high"]), float(last["low"])

        be_at = 1.0
        trail_after = 1.5
        trail_lock_r = 1.0

        if isinstance(plan, dict):
            be_at = float(plan.get("breakeven_at_r", be_at))
            trail_after = float(plan.get("trail_after_r", trail_after))
            trail_lock_r = float(plan.get("trail_lock_r", trail_lock_r))

        R = float(pos.r_dist) if pos.r_dist and pos.r_dist > 0 else abs(float(pos.entry) - float(pos.sl))
        if R <= 0:
            return

        if pos.side == "BUY":
            pos.mfv = max(pos.mfv if pos.mfv else pos.entry, hi)
            move_r = (pos.mfv - pos.entry) / R

            if (not pos.be_moved) and move_r >= be_at:
                pos.sl = max(pos.sl, pos.entry)
                pos.be_moved = True

            if move_r >= trail_after:
                new_sl = pos.entry + max((move_r - trail_lock_r), 0.0) * R
                pos.sl = max(pos.sl, new_sl)

        else:
            pos.mfa = min(pos.mfa if pos.mfa else pos.entry, lo)
            move_r = (pos.entry - pos.mfa) / R

            if (not pos.be_moved) and move_r >= be_at:
                pos.sl = min(pos.sl, pos.entry)
                pos.be_moved = True

            if move_r >= trail_after:
                new_sl = pos.entry - max((move_r - trail_lock_r), 0.0) * R
                pos.sl = min(pos.sl, new_sl)

        with self._get_conn() as conn:
            self._ensure_schema(conn)
            conn.execute("""
                INSERT OR REPLACE INTO open_positions
                  (instrument, tf, side, entry_time, entry, sl, tp, units, reason, model_prob,
                   r_dist, be_moved, mfv, mfa)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.instrument, self.tf, pos.side, str(pos.entry_time),
                float(pos.entry), float(pos.sl), float(pos.tp), float(pos.units),
                str(pos.reason),
                float(pos.model_prob) if pos.model_prob is not None else None,
                float(pos.r_dist),
                int(1 if pos.be_moved else 0),
                float(pos.mfv if pos.mfv else pos.entry),
                float(pos.mfa if pos.mfa else pos.entry),
            ))
            conn.commit()

    # -----------------------------
    # Candle updates (exit logic)
    # -----------------------------
    def update_on_new_closed_candle(self, candles: pd.DataFrame, plan: Optional[Dict[str, Any]] = None):
        if candles is None or len(candles) < 2:
            return

        last_ts = candles.index[-1]
        if self.last_candle_time is not None and last_ts == self.last_candle_time:
            return
        self.last_candle_time = last_ts

        if self.position is None:
            return

        self._manage_open_position(candles=candles, plan=plan)

        last = candles.iloc[-1]
        pos = self.position
        hi, lo = float(last["high"]), float(last["low"])

        hit_sl = (lo <= pos.sl) if pos.side == "BUY" else (hi >= pos.sl)
        hit_tp = (hi >= pos.tp) if pos.side == "BUY" else (lo <= pos.tp)

        outcome, exit_px = None, None
        if hit_sl:
            outcome, exit_px = "SL", float(pos.sl)
        elif hit_tp:
            outcome, exit_px = "TP", float(pos.tp)

        if outcome:
            pnl = (exit_px - pos.entry) * pos.units if pos.side == "BUY" else (pos.entry - exit_px) * pos.units

            new_trade = PaperTrade(
                side=pos.side,
                entry_time=str(pos.entry_time),
                exit_time=str(last_ts),
                entry=float(pos.entry),
                exit=float(exit_px),
                sl=float(pos.sl),
                tp=float(pos.tp),
                units=float(pos.units),
                pnl_usd=float(pnl),
                outcome=outcome,
                reason=pos.reason,
                model_prob=(float(pos.model_prob) if pos.model_prob is not None else None),
            )

            self.trades.append(new_trade)
            self.balance += float(pnl)

            if outcome == "SL":
                self._set_cooldown_after_sl(last_ts=pd.to_datetime(last_ts), cooldown_bars=3)

            with self._get_conn() as conn:
                self._ensure_schema(conn)
                conn.execute("""
                    INSERT INTO trade_history
                      (instrument, tf, side, entry_time, exit_time, entry, exit_px, sl, tp, units,
                       pnl_usd, outcome, reason, model_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.instrument, self.tf,
                    new_trade.side, new_trade.entry_time, new_trade.exit_time,
                    new_trade.entry, new_trade.exit,
                    new_trade.sl, new_trade.tp,
                    new_trade.units, new_trade.pnl_usd,
                    new_trade.outcome, new_trade.reason,
                    float(new_trade.model_prob) if new_trade.model_prob is not None else None
                ))
                conn.execute("DELETE FROM open_positions WHERE instrument=? AND tf=?", (self.instrument, self.tf))
                conn.commit()

            self.position = None
            self._save_state()

    # -----------------------------
    # Entry logic
    # -----------------------------
    def maybe_enter(self, state: Dict[str, Any], candles: pd.DataFrame, balance: float, risk_pct: float):
        if candles is None or len(candles) < 2:
            return

        ts = candles.index[-1]

        if self.position is not None:
            return
        if self.cooldown_active(ts):
            return
        if not state or not state.get("ok"):
            return

        side = state.get("side")
        if side not in ("BUY", "SELL"):
            return

        plan = state.get("plan", {}) or {}
        entry, sl, tp = plan.get("entry"), plan.get("sl"), plan.get("tp")
        if entry is None or sl is None or tp is None:
            return

        units = self.compute_units_quote_usd(
            balance=float(balance),
            risk_pct=float(risk_pct),
            entry=float(entry),
            sl=float(sl)
        )
        if units <= 0:
            return

        r_dist = abs(float(entry) - float(sl))
        if r_dist <= 0:
            return

        # ✅ model probability captured here
        model_prob = state.get("model_prob", None)
        if model_prob is not None:
            try:
                model_prob = float(model_prob)
            except Exception:
                model_prob = None

        pos = PaperPosition(
            side=str(side),
            entry_time=pd.to_datetime(ts),
            entry=float(entry),
            sl=float(sl),
            tp=float(tp),
            units=float(units),
            reason=str(state.get("reason", "")),
            model_prob=model_prob,
            r_dist=float(r_dist),
            be_moved=False,
            mfv=float(entry),
            mfa=float(entry),
        )
        self.position = pos

        with self._get_conn() as conn:
            self._ensure_schema(conn)
            conn.execute("""
                INSERT OR REPLACE INTO open_positions
                  (instrument, tf, side, entry_time, entry, sl, tp, units, reason, model_prob,
                   r_dist, be_moved, mfv, mfa)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.instrument, self.tf, pos.side, str(pos.entry_time),
                pos.entry, pos.sl, pos.tp, pos.units, pos.reason,
                float(pos.model_prob) if pos.model_prob is not None else None,
                pos.r_dist, int(0), pos.mfv, pos.mfa
            ))
            conn.commit()

        self._save_state()

    # -----------------------------
    # Optional cooldown based on strategy meta
    # -----------------------------
    def apply_cooldown_from_state_meta(self, state: Dict[str, Any], last_exit_ts: pd.Timestamp):
        try:
            bars = int(((state.get("meta") or {}).get("cooldown_bars_recommendation", 3)))
        except Exception:
            bars = 3
        self._set_cooldown_after_sl(pd.to_datetime(last_exit_ts), bars)
        self._save_state()

    # -----------------------------
    # Summary
    # -----------------------------
    def summary(self) -> Dict[str, Any]:
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        pnl_total = sum(t.pnl_usd for t in self.trades)
        return {
            "balance_start": float(self.balance_start),
            "balance": float(self.balance),
            "pnl_total": float(pnl_total),
            "trades": int(n),
            "wins": int(wins),
            "win_rate": (wins / n * 100.0) if n else 0.0,
            "open_position": (self.position.__dict__ if self.position else None),
            "cooldown_until": (str(self.cooldown_until) if self.cooldown_until is not None else None),
            "trade_log": [t.__dict__ for t in self.trades[-20:]],
        }
