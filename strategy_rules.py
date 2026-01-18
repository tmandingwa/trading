# strategy_rules.py
import numpy as np
import pandas as pd
from typing import Dict, Any
from indicators import rsi, atr, bullish_engulf, bearish_engulf, swing_levels

def compute_state(candles: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if candles is None or len(candles) == 0:
        return {"ok": False, "msg": "No candles yet"}

    min_needed = max(cfg["EMA_SLOW"] + 5, cfg["ATR_LEN"] + 5, cfg["RSI_LEN"] + 5, 60)
    if len(candles) < min_needed:
        return {"ok": False, "msg": f"Not enough candles yet ({len(candles)}/{min_needed})"}

    df = candles.copy()

    df["ema_fast"] = df["close"].ewm(span=cfg["EMA_FAST"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg["EMA_SLOW"], adjust=False).mean()
    df["rsi"] = rsi(df["close"], cfg["RSI_LEN"])
    df["atr"] = atr(df, cfg["ATR_LEN"])

    if cfg["USE_ENGULFING"]:
        df["bull_engulf"] = bullish_engulf(df)
        df["bear_engulf"] = bearish_engulf(df)
    else:
        df["bull_engulf"] = 0
        df["bear_engulf"] = 0

    if cfg["USE_SR"]:
        df["support"], df["resistance"] = swing_levels(df, cfg["SWING_LOOKBACK"])
    else:
        df["support"] = np.nan
        df["resistance"] = np.nan

    last = df.iloc[-1]
    ts = df.index[-1]

    trend_up = bool(last["ema_fast"] > last["ema_slow"])
    trend_down = bool(last["ema_fast"] < last["ema_slow"])

    rsi_ok_buy = bool(last["rsi"] <= cfg["RSI_BUY_MAX"])
    rsi_ok_sell = bool(last["rsi"] >= cfg["RSI_SELL_MIN"])

    bull_eng = bool(last["bull_engulf"] == 1)
    bear_eng = bool(last["bear_engulf"] == 1)

    near_support = False
    near_resistance = False
    if cfg["USE_SR"] and np.isfinite(last["atr"]) and np.isfinite(last["support"]) and np.isfinite(last["resistance"]):
        tol = cfg["SR_TOL_ATR"] * float(last["atr"])
        near_support = bool(abs(float(last["close"]) - float(last["support"])) <= tol)
        near_resistance = bool(abs(float(last["close"]) - float(last["resistance"])) <= tol)

    buy_conditions = {
        "EMA trend up": trend_up,
        "RSI ok buy": rsi_ok_buy,
        "Bull engulfing": (bull_eng if cfg["USE_ENGULFING"] else True),
        "Near support": (near_support if cfg["USE_SR"] else True),
    }
    sell_conditions = {
        "EMA trend down": trend_down,
        "RSI ok sell": rsi_ok_sell,
        "Bear engulfing": (bear_eng if cfg["USE_ENGULFING"] else True),
        "Near resistance": (near_resistance if cfg["USE_SR"] else True),
    }

    buy_score = sum(bool(v) for v in buy_conditions.values())
    sell_score = sum(bool(v) for v in sell_conditions.values())

    side = "NO_TRADE"
    score = 0
    conds: Dict[str, Any] = {}
    reasons = []

    if buy_score >= cfg["MIN_CONDITIONS_TO_TRADE"] and buy_score > sell_score:
        side = "BUY"
        score = buy_score
        conds = buy_conditions
        reasons = [k for k, v in buy_conditions.items() if v]
    elif sell_score >= cfg["MIN_CONDITIONS_TO_TRADE"] and sell_score > buy_score:
        side = "SELL"
        score = sell_score
        conds = sell_conditions
        reasons = [k for k, v in sell_conditions.items() if v]
    else:
        conds = {"BUY_score": buy_score, "SELL_score": sell_score}

    entry_proxy = float(last["close"])
    atrv = float(last["atr"]) if np.isfinite(last["atr"]) else np.nan

    plan = {"entry": None, "sl": None, "tp": None, "sl_atr": cfg["SL_ATR"], "tp_rr": cfg["TP_RR"]}
    if side in ("BUY", "SELL") and np.isfinite(atrv):
        sl_dist = cfg["SL_ATR"] * atrv
        if side == "BUY":
            sl = entry_proxy - sl_dist
            tp = entry_proxy + cfg["TP_RR"] * (entry_proxy - sl)
        else:
            sl = entry_proxy + sl_dist
            tp = entry_proxy - cfg["TP_RR"] * (sl - entry_proxy)
        plan = {"entry": entry_proxy, "sl": float(sl), "tp": float(tp), "sl_atr": cfg["SL_ATR"], "tp_rr": cfg["TP_RR"]}

    reason_str = ""
    if side != "NO_TRADE" and reasons:
        reason_str = f"{side} because " + " + ".join(reasons)

    return {
        "ok": True,
        "timestamp": str(ts),
        "close": float(last["close"]),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]) if np.isfinite(last["atr"]) else None,
        "support": float(last["support"]) if np.isfinite(last["support"]) else None,
        "resistance": float(last["resistance"]) if np.isfinite(last["resistance"]) else None,
        "side": side,
        "score": int(score),
        "min_required": int(cfg["MIN_CONDITIONS_TO_TRADE"]),
        "conditions": conds,
        "reason": reason_str,
        "plan": plan,
    }
