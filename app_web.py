# app_web.py
import asyncio
import json
import sqlite3
import os
import time
from typing import Dict, Any

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
)

from candle_agg import CandleAggregator, TF_TO_PANDAS
from oanda_stream import price_stream, OandaStreamError
from oanda_history import fetch_candles
from strategy_rules import compute_state
from paper_engine import PaperEngine

# ============================================================
# PATHS (make templates/static robust on Railway)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "index.html")

# ============================================================
# BOOT LOGS (helps Railway debugging)
# ============================================================
print("[boot] app_web.py imported ✅", flush=True)
print("[boot] BASE_DIR =", BASE_DIR, flush=True)
print("[boot] TEMPLATES_DIR exists? ", os.path.exists(TEMPLATES_DIR), flush=True)
print("[boot] INDEX_TEMPLATE exists? ", os.path.exists(INDEX_TEMPLATE_PATH), flush=True)
print("[boot] PORT env =", os.getenv("PORT"), flush=True)
print("[boot] OANDA_ENV =", OANDA_ENV, flush=True)
print("[boot] OANDA_TOKEN set? ", "YES" if (OANDA_TOKEN and len(OANDA_TOKEN) > 20) else "NO", flush=True)
print("[boot] OANDA_ACCOUNT_ID set? ", "YES" if (OANDA_ACCOUNT_ID and len(OANDA_ACCOUNT_ID) > 5) else "NO", flush=True)

# Default path for local development. Change this to /data/trades.db on Railway.
DB_PATH = os.getenv("DB_PATH", "/data/trades.db")

# Stream robustness
STREAM_RETRY_SECONDS = float(os.getenv("STREAM_RETRY_SECONDS", "5"))
STREAM_HEARTBEAT_EVERY = int(os.getenv("STREAM_HEARTBEAT_EVERY", "200"))  # ticks
FAIL_FAST_IF_NO_OANDA = os.getenv("FAIL_FAST_IF_NO_OANDA", "0") == "1"


def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS engine_state (instrument TEXT, tf TEXT, balance REAL, balance_start REAL, PRIMARY KEY (instrument, tf))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS open_positions (instrument TEXT, tf TEXT, side TEXT, entry_time TEXT, entry REAL, sl REAL, tp REAL, units REAL, reason TEXT, PRIMARY KEY (instrument, tf))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_history (id INTEGER PRIMARY KEY AUTOINCREMENT, instrument TEXT, tf TEXT, side TEXT, entry_time TEXT, exit_time TEXT, entry REAL, exit_px REAL, sl REAL, tp REAL, units REAL, pnl_usd REAL, outcome TEXT, reason TEXT)"
        )


app = FastAPI()

# Templates (absolute directory)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Static (optional) - only mount if folder exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("[boot] Static mounted at /static ✅", flush=True)
else:
    print("[boot] No static/ folder (OK)", flush=True)


SUPPORTED_INSTRUMENTS = ["EUR_USD", "GBP_USD", "XAU_USD"]
SUPPORTED_TFS = list(TF_TO_PANDAS.keys())

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

STREAM_TASK: asyncio.Task | None = None


# ============================================================
# SIMPLE REQUEST LOGGER (CRITICAL for Railway routing debug)
# ============================================================
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


def get_global_paper_stats():
    total_trades, total_wins, total_pnl, global_history = 0, 0, 0.0, []
    for inst in SUPPORTED_INSTRUMENTS:
        for tf in SUPPORTED_TFS:
            summary = PAPER[inst][tf].summary()
            total_trades += summary["trades"]
            total_wins += summary["wins"]
            total_pnl += summary["pnl_total"]
            for t in summary["trade_log"]:
                t_enriched = t.copy()
                t_enriched.update({"instrument": inst, "tf": tf})
                global_history.append(t_enriched)
    global_history.sort(key=lambda x: x.get("exit_time", ""), reverse=True)
    return {
        "trades": total_trades,
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades else 0,
        "pnl": round(total_pnl, 2),
        "history": global_history[:15],
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
    })


@app.get("/readyz")
async def readyz():
    try:
        inst = DEFAULT_INSTRUMENT if DEFAULT_INSTRUMENT in SUPPORTED_INSTRUMENTS else SUPPORTED_INSTRUMENTS[0]
        tf = DEFAULT_TF if DEFAULT_TF in SUPPORTED_TFS else SUPPORTED_TFS[0]
        df = AGGS[inst][tf].to_df()
        ok = df is not None and len(df) >= 50
        return JSONResponse({"ready": bool(ok), "candles": int(len(df)) if df is not None else 0})
    except Exception as e:
        return JSONResponse({"ready": False, "error": repr(e)}, status_code=500)


@app.on_event("startup")
async def on_startup():
    global STREAM_TASK
    print("[startup] on_startup() entered ✅", flush=True)

    init_db()

    if (not OANDA_TOKEN or not OANDA_ACCOUNT_ID) and FAIL_FAST_IF_NO_OANDA:
        raise RuntimeError("Missing OANDA_TOKEN or OANDA_ACCOUNT_ID (set Railway Variables).")

    # Seed history for each instrument/tf
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


async def stream_loop():
    instruments = ",".join(SUPPORTED_INSTRUMENTS)
    tick_count = 0

    while True:
        try:
            if not OANDA_TOKEN or not OANDA_ACCOUNT_ID:
                msg = "Missing OANDA_TOKEN or OANDA_ACCOUNT_ID (set env vars)."
                print(f"[stream][WARN] {msg} retrying in {STREAM_RETRY_SECONDS}s…", flush=True)
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


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Serve UI if template exists. Otherwise return a fallback HTML page
    (so Railway always shows something on /).
    """
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

        # Fallback HTML: proves routing works even if templates missing
        html = f"""
        <html>
          <head><title>FX TA Dashboard</title></head>
          <body style="font-family:Arial; padding:24px;">
            <h2>FX TA Dashboard (Fallback)</h2>
            <p><b>Status:</b> Running ✅</p>
            <p><b>Note:</b> templates/index.html was not found in this deployment.</p>
            <p>Check: <a href="/healthz">/healthz</a> | <a href="/readyz">/readyz</a> | <a href="/ping">/ping</a></p>
            <pre>BASE_DIR={BASE_DIR}
TEMPLATES_DIR={TEMPLATES_DIR}
INDEX_TEMPLATE_PATH={INDEX_TEMPLATE_PATH}</pre>
          </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=200)

    except Exception as e:
        return HTMLResponse(
            content=f"<h3>UI failed but backend running</h3><pre>{repr(e)}</pre>",
            status_code=200
        )


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
                }))
                await asyncio.sleep(0.5)
                continue

            df = AGGS[inst][tf].to_df()
            state = compute_state(df, CFG)

            pe = PAPER[inst][tf]
            pe.update_on_new_closed_candle(df)
            pe.maybe_enter(state, df, user_cfg["balance"], user_cfg["risk_pct"])

            sizing = {"units": None, "lots": None}
            if state.get("ok") and state.get("side") in ("BUY", "SELL"):
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
                "candles": df_to_candles_payload(df),
                "paper": pe.summary(),
                "global_stats": get_global_paper_stats(),
                "params": CFG,
            }, default=str))

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
