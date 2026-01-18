# app_web.py
import asyncio
import json
import sqlite3
import os
from typing import Dict, Any
import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from config import (
    OANDA_TOKEN, OANDA_ACCOUNT_ID, DEFAULT_INSTRUMENT, DEFAULT_TF,
    EMA_FAST, EMA_SLOW, RSI_LEN, RSI_BUY_MAX, RSI_SELL_MIN,
    ATR_LEN, SL_ATR, TP_RR, USE_ENGULFING, USE_SR, SWING_LOOKBACK, 
    SR_TOL_ATR, MIN_CONDITIONS_TO_TRADE, SEED_CANDLES,
)
from candle_agg import CandleAggregator, TF_TO_PANDAS
from oanda_stream import price_stream
from oanda_history import fetch_candles
from strategy_rules import compute_state
from paper_engine import PaperEngine

# Default path for local development. Change this to /data/trades.db on Railway.
DB_PATH = "/data/trades.db"

def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS engine_state (instrument TEXT, tf TEXT, balance REAL, balance_start REAL, PRIMARY KEY (instrument, tf))")
        conn.execute("CREATE TABLE IF NOT EXISTS open_positions (instrument TEXT, tf TEXT, side TEXT, entry_time TEXT, entry REAL, sl REAL, tp REAL, units REAL, reason TEXT, PRIMARY KEY (instrument, tf))")
        conn.execute("CREATE TABLE IF NOT EXISTS trade_history (id INTEGER PRIMARY KEY AUTOINCREMENT, instrument TEXT, tf TEXT, side TEXT, entry_time TEXT, exit_time TEXT, entry REAL, exit_px REAL, sl REAL, tp REAL, units REAL, pnl_usd REAL, outcome TEXT, reason TEXT)")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SUPPORTED_INSTRUMENTS = ["EUR_USD", "GBP_USD", "XAU_USD"]
SUPPORTED_TFS = list(TF_TO_PANDAS.keys())

AGGS = {inst: {tf: CandleAggregator(tf=tf, max_candles=800) for tf in SUPPORTED_TFS} for inst in SUPPORTED_INSTRUMENTS}
CFG = {"EMA_FAST": EMA_FAST, "EMA_SLOW": EMA_SLOW, "RSI_LEN": RSI_LEN, "RSI_BUY_MAX": RSI_BUY_MAX, "RSI_SELL_MIN": RSI_SELL_MIN, "ATR_LEN": ATR_LEN, "SL_ATR": SL_ATR, "TP_RR": TP_RR, "USE_ENGULFING": USE_ENGULFING, "USE_SR": USE_SR, "SWING_LOOKBACK": SWING_LOOKBACK, "SR_TOL_ATR": SR_TOL_ATR, "MIN_CONDITIONS_TO_TRADE": MIN_CONDITIONS_TO_TRADE}
PAPER = {inst: {tf: PaperEngine(instrument=inst, tf=tf, db_path=DB_PATH) for tf in SUPPORTED_TFS} for inst in SUPPORTED_INSTRUMENTS}

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
    global_history.sort(key=lambda x: x["exit_time"], reverse=True)
    return {"trades": total_trades, "win_rate": round(total_wins/total_trades*100, 1) if total_trades else 0, "pnl": round(total_pnl, 2), "history": global_history[:15]}

@app.on_event("startup")
async def on_startup():
    init_db()
    for inst in SUPPORTED_INSTRUMENTS:
        for tf in SUPPORTED_TFS:
            try:
                hist = await fetch_candles(inst, tf, count=SEED_CANDLES)
                AGGS[inst][tf].seed_from_ohlc_df(hist)
                PAPER[inst][tf].load_from_db()
            except Exception as e: print(f"[seed][WARN] {inst} {tf} failed: {e}")
    asyncio.create_task(stream_loop())

async def stream_loop():
    instruments = ",".join(SUPPORTED_INSTRUMENTS)
    async for tick in price_stream(instruments):
        inst, mid, ts = tick["instrument"], tick["mid"], tick["time"]
        if inst in AGGS:
            for tf, agg in AGGS[inst].items(): agg.update(ts, mid)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "instruments": SUPPORTED_INSTRUMENTS, "tfs": SUPPORTED_TFS, "default_instrument": DEFAULT_INSTRUMENT, "default_tf": DEFAULT_TF})

def df_to_candles_payload(df: pd.DataFrame, limit: int = 200):
    if df is None or df.empty: return []
    d = df.tail(limit).copy().sort_index()
    return [{"t": int(ts.value // 1_000_000), "o": float(row["open"]), "h": float(row["high"]), "l": float(row["low"]), "c": float(row["close"])} for ts, row in d.iterrows()]

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sub, user_cfg = {"instrument": DEFAULT_INSTRUMENT, "tf": DEFAULT_TF}, {"balance": 1000.0, "risk_pct": 1.0}
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.25)
                data = json.loads(msg)
                sub.update({k: data[k] for k in ["instrument", "tf"] if k in data})
                user_cfg.update({k: float(data[k]) for k in ["balance", "risk_pct"] if k in data})
                if data.get("reset_paper"): PAPER[sub["instrument"]][sub["tf"]].reset(balance=user_cfg["balance"])
            except asyncio.TimeoutError: pass
            inst, tf = sub["instrument"], sub["tf"]
            df = AGGS[inst][tf].to_df()
            state = compute_state(df, CFG)
            pe = PAPER[inst][tf]
            pe.update_on_new_closed_candle(df)
            pe.maybe_enter(state, df, user_cfg["balance"], user_cfg["risk_pct"])
            sizing = {"units": None, "lots": None}
            if state.get("ok") and state.get("side") in ("BUY", "SELL"):
                p = state.get("plan", {})
                if p.get("entry") and p.get("sl"):
                    u = PaperEngine.compute_units_quote_usd(user_cfg["balance"], user_cfg["risk_pct"], p["entry"], p["sl"])
                    sizing = {"units": float(u), "lots": float(PaperEngine.units_to_lots(u))}
            await ws.send_text(json.dumps({"subscription": sub, "state": state, "sizing": sizing, "candles": df_to_candles_payload(df), "paper": pe.summary(), "global_stats": get_global_paper_stats(), "params": CFG}, default=str))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect: pass