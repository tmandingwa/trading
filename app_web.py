# app_web.py
import asyncio
import json
import sqlite3
import os
import time
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect
from starlette.staticfiles import StaticFiles

from config import (
    OANDA_TOKEN, OANDA_ACCOUNT_ID, OANDA_ENV,
    DEFAULT_INSTRUMENT, DEFAULT_TF,
    EMA_FAST, EMA_SLOW, RSI_LEN, RSI_BUY_MAX, RSI_SELL_MIN,
    ATR_LEN, SL_ATR, TP_RR, USE_ENGULFING, USE_SR, SWING_LOOKBACK,
    SR_TOL_ATR, MIN_CONDITIONS_TO_TRADE, SEED_CANDLES,
    SUPPORTED_INSTRUMENTS, SUPPORTED_TFS,
)

from candle_agg import CandleAggregator, TF_TO_PANDAS
from oanda_stream import price_stream, OandaStreamError
from oanda_history import fetch_candles
from strategy_rules import compute_state
from paper_engine import PaperEngine

# ✅ NEW: ML loader (expects you created ml_model.py)
from ml_model import load_models_from_dir

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "index.html")

print("[boot] app_web.py imported ✅", flush=True)
print("[boot] BASE_DIR =", BASE_DIR, flush=True)
print("[boot] TEMPLATES_DIR exists? ", os.path.exists(TEMPLATES_DIR), flush=True)
print("[boot] INDEX_TEMPLATE exists? ", os.path.exists(INDEX_TEMPLATE_PATH), flush=True)
print("[boot] PORT env =", os.getenv("PORT"), flush=True)
print("[boot] OANDA_ENV =", OANDA_ENV, flush=True)
print("[boot] OANDA_TOKEN set? ", "YES" if (OANDA_TOKEN and len(OANDA_TOKEN) > 20) else "NO", flush=True)
print("[boot] OANDA_ACCOUNT_ID set? ", "YES" if (OANDA_ACCOUNT_ID and len(OANDA_ACCOUNT_ID) > 5) else "NO", flush=True)

DB_PATH = os.getenv("DB_PATH", "/data/trades.db")

STREAM_RETRY_SECONDS = float(os.getenv("STREAM_RETRY_SECONDS", "5"))
STREAM_HEARTBEAT_EVERY = int(os.getenv("STREAM_HEARTBEAT_EVERY", "200"))
FAIL_FAST_IF_NO_OANDA = os.getenv("FAIL_FAST_IF_NO_OANDA", "0") == "1"

AUTO_ENGINE_ALL = os.getenv("AUTO_ENGINE_ALL", "1") == "1"
AUTO_ENGINE_SLEEP = float(os.getenv("AUTO_ENGINE_SLEEP", "0.8"))
AUTO_BALANCE = float(os.getenv("AUTO_BALANCE", "1000"))
AUTO_RISK_PCT = float(os.getenv("AUTO_RISK_PCT", "1.0"))

GLOBAL_TRADE_HISTORY_LIMIT = int(os.getenv("GLOBAL_TRADE_HISTORY_LIMIT", "5000"))
GLOBAL_OPEN_LIMIT = int(os.getenv("GLOBAL_OPEN_LIMIT", "200"))
GLOBAL_CLOSED_LIMIT = int(os.getenv("GLOBAL_CLOSED_LIMIT", "5000"))

# ✅ NEW: ML artifacts directory + models cache
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", os.path.join(BASE_DIR, "artifacts"))
ML_MODELS: Dict[str, Any] = {}

STREAM_TASK: asyncio.Task | None = None
ENGINE_TASK: asyncio.Task | None = None

ENGINE_LOCKS: Dict[Tuple[str, str], asyncio.Lock] = {}
LAST_ENGINE_CANDLE: Dict[Tuple[str, str], int] = {}  # last processed CLOSED candle time (ns)


def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        # Base tables (keep yours, but add missing cols safely)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engine_state (
                instrument TEXT,
                tf TEXT,
                balance REAL,
                balance_start REAL,
                cooldown_until TEXT,
                PRIMARY KEY (instrument, tf)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                instrument TEXT,
                tf TEXT,
                side TEXT,
                entry_time TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                units REAL,
                reason TEXT,
                model_prob REAL,
                PRIMARY KEY (instrument, tf)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT,
                tf TEXT,
                side TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry REAL,
                exit_px REAL,
                sl REAL,
                tp REAL,
                units REAL,
                pnl_usd REAL,
                outcome TEXT,
                reason TEXT,
                model_prob REAL
            )
        """)

        # Lightweight migrations for older DBs
        def _cols(table: str) -> List[str]:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return [r[1] for r in rows]

        # engine_state: cooldown_until
        cols = _cols("engine_state")
        if "cooldown_until" not in cols:
            conn.execute("ALTER TABLE engine_state ADD COLUMN cooldown_until TEXT")

        # open_positions: model_prob
        cols = _cols("open_positions")
        if "model_prob" not in cols:
            conn.execute("ALTER TABLE open_positions ADD COLUMN model_prob REAL")

        # trade_history: model_prob
        cols = _cols("trade_history")
        if "model_prob" not in cols:
            conn.execute("ALTER TABLE trade_history ADD COLUMN model_prob REAL")

        conn.commit()


def db_fetchone(query: str, params: tuple = ()):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(query, params)
        return cur.fetchone()


def db_fetchall(query: str, params: tuple = ()):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall()


app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("[boot] Static mounted at /static ✅", flush=True)
else:
    print("[boot] No static/ folder (OK)", flush=True)

SUPPORTED_TFS = [tf for tf in SUPPORTED_TFS if tf in TF_TO_PANDAS]

AGGS = {
    inst: {tf: CandleAggregator(tf=tf, max_candles=800) for tf in SUPPORTED_TFS}
    for inst in SUPPORTED_INSTRUMENTS
}

CFG = {
    "EMA_FAST": EMA_FAST,
    "EMA_SLOW": EMA_SLOW,
    "RSI_LEN": RSI_LEN,
    "RSI_BUY_MAX": RSI_BUY_MAX,
    "RSI_SELL_MIN": RSI_SELL_MIN,
    "ATR_LEN": ATR_LEN,
    "SL_ATR": SL_ATR,
    "TP_RR": TP_RR,
    "USE_ENGULFING": USE_ENGULFING,
    "USE_SR": USE_SR,
    "SWING_LOOKBACK": SWING_LOOKBACK,
    "SR_TOL_ATR": SR_TOL_ATR,
    "MIN_CONDITIONS_TO_TRADE": MIN_CONDITIONS_TO_TRADE,
}

PAPER = {
    inst: {tf: PaperEngine(instrument=inst, tf=tf, db_path=DB_PATH) for tf in SUPPORTED_TFS}
    for inst in SUPPORTED_INSTRUMENTS
}

for inst in SUPPORTED_INSTRUMENTS:
    for tf in SUPPORTED_TFS:
        ENGINE_LOCKS[(inst, tf)] = asyncio.Lock()


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        dt = (time.time() - t0) * 1000
        print(f"[http] {request.method} {request.url.path} -> {response.status_code} ({dt:.1f}ms)", flush=True)
        return response
    except Exception as e:
        dt = (time.time() - t0) * 1000
        print(f"[http][ERR] {request.method} {request.url.path} crashed after {dt:.1f}ms: {repr(e)}", flush=True)
        raise


def get_global_metrics_from_db(limit_closed: int = 25, limit_open: int = 25) -> Dict[str, Any]:
    open_rows = db_fetchall(
        """
        SELECT instrument, tf, side, entry_time, entry, sl, tp, units, reason, model_prob
        FROM open_positions
        ORDER BY entry_time DESC
        LIMIT ?
        """,
        (limit_open,),
    )
    open_positions = []
    for r in open_rows:
        open_positions.append({
            "instrument": r[0], "tf": r[1], "side": r[2],
            "entry_time": r[3], "entry": r[4], "sl": r[5],
            "tp": r[6], "units": r[7], "reason": r[8],
            "model_prob": r[9],
        })

    open_count_row = db_fetchone("SELECT COUNT(*) FROM open_positions")
    open_count = int(open_count_row[0]) if open_count_row else 0

    closed_count_row = db_fetchone("SELECT COUNT(*) FROM trade_history WHERE exit_time IS NOT NULL AND exit_time != ''")
    closed_count = int(closed_count_row[0]) if closed_count_row else 0

    wins_row = db_fetchone(
        "SELECT COUNT(*) FROM trade_history WHERE exit_time IS NOT NULL AND outcome IN ('WIN','TP','SUCCESS','PROFIT')"
    )
    losses_row = db_fetchone(
        "SELECT COUNT(*) FROM trade_history WHERE exit_time IS NOT NULL AND outcome IN ('LOSS','SL','FAIL','FAILED','STOP')"
    )
    wins = int(wins_row[0]) if wins_row else 0
    losses = int(losses_row[0]) if losses_row else 0

    if (wins + losses) == 0:
        wins2_row = db_fetchone("SELECT COUNT(*) FROM trade_history WHERE exit_time IS NOT NULL AND pnl_usd > 0")
        losses2_row = db_fetchone("SELECT COUNT(*) FROM trade_history WHERE exit_time IS NOT NULL AND pnl_usd <= 0")
        wins = int(wins2_row[0]) if wins2_row else 0
        losses = int(losses2_row[0]) if losses2_row else 0

    pnl_row = db_fetchone("SELECT COALESCE(SUM(pnl_usd), 0) FROM trade_history")
    pnl_total = float(pnl_row[0]) if pnl_row else 0.0

    closed_rows = db_fetchall(
        """
        SELECT instrument, tf, side, entry_time, exit_time, entry, exit_px, sl, tp, units, pnl_usd, outcome, reason, model_prob
        FROM trade_history
        WHERE exit_time IS NOT NULL AND exit_time != ''
        ORDER BY exit_time DESC
        LIMIT ?
        """,
        (limit_closed,),
    )
    closed_trades = []
    for r in closed_rows:
        closed_trades.append({
            "instrument": r[0], "tf": r[1], "side": r[2],
            "entry_time": r[3], "exit_time": r[4],
            "entry": r[5], "exit_px": r[6],
            "sl": r[7], "tp": r[8], "units": r[9],
            "pnl_usd": r[10], "outcome": r[11], "reason": r[12],
            "model_prob": r[13],
        })

    total_finished = wins + losses
    win_rate = (wins / total_finished * 100.0) if total_finished else 0.0

    return {
        "open_trades": open_count,
        "closed_trades": closed_count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "pnl_total": round(pnl_total, 2),
        "open_positions": open_positions,
        "closed_history": closed_trades,
    }


def build_global_timeline(limit: int = 5000) -> List[Dict[str, Any]]:
    open_rows = db_fetchall(
        """
        SELECT instrument, tf, side, entry_time, entry, sl, tp, units, reason, model_prob
        FROM open_positions
        ORDER BY entry_time DESC
        LIMIT ?
        """,
        (GLOBAL_OPEN_LIMIT,),
    )
    open_items = []
    for r in open_rows:
        open_items.append({
            "status": "OPEN",
            "instrument": r[0], "tf": r[1], "side": r[2],
            "entry_time": r[3], "exit_time": None,
            "entry": r[4], "sl": r[5], "tp": r[6],
            "units": r[7], "pnl_usd": None, "outcome": None,
            "reason": r[8],
            "model_prob": r[9],
        })

    closed_rows = db_fetchall(
        """
        SELECT instrument, tf, side, entry_time, exit_time, entry, exit_px, sl, tp, units, pnl_usd, outcome, reason, model_prob
        FROM trade_history
        WHERE exit_time IS NOT NULL AND exit_time != ''
        ORDER BY exit_time DESC
        LIMIT ?
        """,
        (GLOBAL_CLOSED_LIMIT,),
    )
    closed_items = []
    for r in closed_rows:
        closed_items.append({
            "status": "CLOSED",
            "instrument": r[0], "tf": r[1], "side": r[2],
            "entry_time": r[3], "exit_time": r[4],
            "entry": r[5], "exit_px": r[6],
            "sl": r[7], "tp": r[8], "units": r[9],
            "pnl_usd": r[10], "outcome": r[11], "reason": r[12],
            "model_prob": r[13],
        })

    def _ts_key(x: Dict[str, Any]) -> str:
        return str(x.get("exit_time") or x.get("entry_time") or "")

    all_items = open_items + closed_items
    all_items.sort(key=_ts_key, reverse=True)
    return all_items[:limit]


def get_global_paper_stats():
    total_trades, total_wins, total_pnl, global_history = 0, 0, 0.0, []
    for inst in SUPPORTED_INSTRUMENTS:
        for tf in SUPPORTED_TFS:
            summary = PAPER[inst][tf].summary()
            total_trades += summary.get("trades", 0)
            total_wins += summary.get("wins", 0)
            total_pnl += summary.get("pnl_total", 0.0)
            for t in summary.get("trade_log", []):
                t_enriched = t.copy()
                t_enriched.update({"instrument": inst, "tf": tf})
                global_history.append(t_enriched)

    global_history.sort(key=lambda x: x.get("exit_time", ""), reverse=True)

    dbm = get_global_metrics_from_db(limit_closed=25, limit_open=25)
    timeline = build_global_timeline(limit=GLOBAL_TRADE_HISTORY_LIMIT)

    return {
        "trades": int(total_trades),
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades else 0,
        "pnl": round(float(total_pnl), 2),
        "history": global_history[:15],

        "open_trades": dbm["open_trades"],
        "closed_trades": dbm["closed_trades"],
        "wins": dbm["wins"],
        "losses": dbm["losses"],
        "win_rate_db": dbm["win_rate"],
        "pnl_total_db": dbm["pnl_total"],
        "open_positions": dbm["open_positions"],
        "closed_history": dbm["closed_history"],

        "timeline": timeline,
    }


@app.get("/ping")
async def ping():
    return PlainTextResponse("pong")


@app.get("/healthz")
async def healthz():
    return JSONResponse({
        "ok": True,
        "service": "trading",
        "port": os.getenv("PORT"),
        "oanda_env": OANDA_ENV,
        "templates_dir_exists": os.path.exists(TEMPLATES_DIR),
        "index_template_exists": os.path.exists(INDEX_TEMPLATE_PATH),
        "auto_engine_all": AUTO_ENGINE_ALL,
        "global_trade_history_limit": GLOBAL_TRADE_HISTORY_LIMIT,
        "artifacts_dir": ARTIFACTS_DIR,
        "ml_models_loaded": list(ML_MODELS.keys()),
    })


@app.on_event("startup")
async def on_startup():
    global STREAM_TASK, ENGINE_TASK, ML_MODELS
    print("[startup] on_startup() entered ✅", flush=True)

    init_db()

    # ✅ Load ML models once
    try:
        ML_MODELS = load_models_from_dir(ARTIFACTS_DIR)
        print(f"[ml] loaded models from {ARTIFACTS_DIR}: {list(ML_MODELS.keys())}", flush=True)
    except Exception as e:
        ML_MODELS = {}
        print(f"[ml][WARN] could not load models: {repr(e)}", flush=True)

    if (not OANDA_TOKEN or not OANDA_ACCOUNT_ID) and FAIL_FAST_IF_NO_OANDA:
        raise RuntimeError("Missing OANDA_TOKEN or OANDA_ACCOUNT_ID (set Railway Variables).")

    for inst in SUPPORTED_INSTRUMENTS:
        for tf in SUPPORTED_TFS:
            try:
                hist = await fetch_candles(inst, tf, count=SEED_CANDLES)
                AGGS[inst][tf].seed_from_ohlc_df(hist)
                PAPER[inst][tf].load_from_db()
                print(f"[seed] {inst} {tf}: {len(hist)} candles", flush=True)
            except Exception as e:
                print(f"[seed][WARN] {inst} {tf} failed: {repr(e)}", flush=True)

    STREAM_TASK = asyncio.create_task(stream_loop())
    print("[startup] Live stream task started ✅", flush=True)

    if AUTO_ENGINE_ALL:
        ENGINE_TASK = asyncio.create_task(engine_loop_all())
        print("[startup] Auto-engine (ALL pairs/TFs) started ✅", flush=True)


async def stream_loop():
    instruments = ",".join(SUPPORTED_INSTRUMENTS)
    tick_count = 0

    while True:
        try:
            if not OANDA_TOKEN or not OANDA_ACCOUNT_ID:
                print(f"[stream][WARN] Missing OANDA creds. retry in {STREAM_RETRY_SECONDS}s…", flush=True)
                await asyncio.sleep(STREAM_RETRY_SECONDS)
                continue

            print(f"[stream] connecting… instruments={instruments}", flush=True)

            async for tick in price_stream(instruments):
                inst, mid, ts = tick["instrument"], tick["mid"], tick["time"]

                if inst in AGGS:
                    for tf, agg in AGGS[inst].items():
                        agg.update(ts, mid)

                tick_count += 1
                if tick_count % STREAM_HEARTBEAT_EVERY == 0:
                    print(f"[stream] heartbeat ticks={tick_count} last={inst} mid={mid}", flush=True)

        except (OandaStreamError,) as e:
            print(f"[stream][WARN] stream error: {repr(e)} — reconnecting in {STREAM_RETRY_SECONDS}s…", flush=True)
            await asyncio.sleep(STREAM_RETRY_SECONDS)
        except Exception as e:
            print(f"[stream][WARN] unexpected error: {repr(e)} — reconnecting in {STREAM_RETRY_SECONDS}s…", flush=True)
            await asyncio.sleep(STREAM_RETRY_SECONDS)


def df_to_candles_payload(df: pd.DataFrame, limit: int = 200):
    if df is None or df.empty:
        return []
    d = df.tail(limit).copy().sort_index()
    return [
        {
            "t": int(ts.value // 1_000_000),  # ms for JS
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
        }
        for ts, row in d.iterrows()
    ]


async def process_engine_once(inst: str, tf: str, balance: float, risk_pct: float):
    """
    Engine uses CLOSED candles ONLY (no forming candle).
    UI will use include_current=True separately.
    """
    key = (inst, tf)
    lock = ENGINE_LOCKS.get(key)
    if lock is None:
        ENGINE_LOCKS[key] = asyncio.Lock()
        lock = ENGINE_LOCKS[key]

    async with lock:
        agg = AGGS[inst][tf]
        df = agg.to_df(include_current=False)  # ✅ CLOSED ONLY for engine
        if df is None or df.empty or len(df) < 50:
            return None, df

        try:
            latest_closed_ts_ns = int(df.index[-1].value)
        except Exception:
            latest_closed_ts_ns = int(time.time() * 1e9)

        pe = PAPER[inst][tf]

        # ✅ Build per-call cfg (TF + cooldown + ML)
        cfg2 = dict(CFG)
        cfg2["TF"] = tf
        cfg2.update(pe.build_strategy_cfg())

        cfg2["ML_ENABLED"] = True
        cfg2["ML_MODELS"] = ML_MODELS
        # Optional overrides:
        # cfg2["ML_BUY_TH"] = 0.54
        # cfg2["ML_SELL_TH"] = 0.46

        last = LAST_ENGINE_CANDLE.get(key)
        if last is not None and latest_closed_ts_ns <= last:
            state = compute_state(df, cfg2)
            return state, df

        LAST_ENGINE_CANDLE[key] = latest_closed_ts_ns

        state = compute_state(df, cfg2)

        # ✅ Pass plan so BE/Trail uses the plan settings (if any)
        plan = (state or {}).get("plan") if isinstance(state, dict) else None
        pe.update_on_new_closed_candle(df, plan=plan)
        pe.maybe_enter(state, df, balance, risk_pct)

        return state, df


async def engine_loop_all():
    print("[engine] engine_loop_all started ✅", flush=True)
    while True:
        try:
            for inst in SUPPORTED_INSTRUMENTS:
                for tf in SUPPORTED_TFS:
                    await process_engine_once(inst, tf, AUTO_BALANCE, AUTO_RISK_PCT)
        except Exception as e:
            print("[engine][WARN] loop error:", repr(e), flush=True)

        await asyncio.sleep(AUTO_ENGINE_SLEEP)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        if os.path.exists(INDEX_TEMPLATE_PATH):
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "instruments": SUPPORTED_INSTRUMENTS,
                    "tfs": SUPPORTED_TFS,
                    "default_instrument": DEFAULT_INSTRUMENT,
                    "default_tf": DEFAULT_TF,
                },
            )

        html = f"""
        <html><body style="font-family:Arial; padding:24px;">
        <h2>FX TA Dashboard (Fallback)</h2>
        <p>templates/index.html not found.</p>
        <p><a href="/healthz">/healthz</a> | <a href="/ping">/ping</a></p>
        <pre>BASE_DIR={BASE_DIR}\nTEMPLATES_DIR={TEMPLATES_DIR}\nINDEX_TEMPLATE_PATH={INDEX_TEMPLATE_PATH}</pre>
        </body></html>
        """
        return HTMLResponse(content=html, status_code=200)

    except Exception as e:
        return HTMLResponse(content=f"<h3>UI failed but backend running</h3><pre>{repr(e)}</pre>", status_code=200)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sub = {"instrument": DEFAULT_INSTRUMENT, "tf": DEFAULT_TF}
    user_cfg = {"balance": 1000.0, "risk_pct": 1.0}

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.25)
                data = json.loads(msg)

                sub.update({k: data[k] for k in ["instrument", "tf"] if k in data})
                user_cfg.update({k: float(data[k]) for k in ["balance", "risk_pct"] if k in data})

                if data.get("reset_paper"):
                    PAPER[sub["instrument"]][sub["tf"]].reset(balance=user_cfg["balance"])

            except asyncio.TimeoutError:
                pass

            inst, tf = sub["instrument"], sub["tf"]

            if inst not in AGGS or tf not in AGGS[inst]:
                await ws.send_text(json.dumps({
                    "subscription": sub,
                    "error": f"Invalid subscription: {inst}/{tf}",
                    "global_stats": get_global_paper_stats(),
                }, default=str))
                await asyncio.sleep(0.5)
                continue

            # Engine state (closed candles)
            state, df_closed = await process_engine_once(inst, tf, user_cfg["balance"], user_cfg["risk_pct"])

            agg = AGGS[inst][tf]
            df_display = agg.to_df(include_current=True)  # ✅ UI sees forming candle
            live_mid = agg.last_mid()

            if df_display is None or df_display.empty:
                await ws.send_text(json.dumps({
                    "subscription": sub,
                    "error": f"No candles yet for {inst}/{tf}",
                    "global_stats": get_global_paper_stats(),
                }, default=str))
                await asyncio.sleep(0.5)
                continue

            # ✅ Make displayed "close" be live mid (not last closed)
            if state and isinstance(state, dict) and live_mid is not None:
                state = state.copy()
                state["close"] = float(live_mid)

            pe = PAPER[inst][tf]

            sizing = {"units": None, "lots": None}
            if state and state.get("ok") and state.get("side") in ("BUY", "SELL"):
                p = state.get("plan", {})
                if p.get("entry") and p.get("sl"):
                    u = PaperEngine.compute_units_quote_usd(
                        user_cfg["balance"], user_cfg["risk_pct"], p["entry"], p["sl"]
                    )
                    sizing = {"units": float(u), "lots": float(PaperEngine.units_to_lots(u))}

            await ws.send_text(json.dumps({
                "subscription": sub,
                "state": state,
                "sizing": sizing,
                "candles": df_to_candles_payload(df_display),  # ✅ includes current candle
                "paper": pe.summary(),
                "global_stats": get_global_paper_stats(),
                "params": CFG,
            }, default=str))

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
