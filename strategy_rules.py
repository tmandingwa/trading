# strategy_rules.py
import numpy as np
import pandas as pd
from typing import Dict, Any

from indicators import (
    rsi, atr, ema, macd, stochastic, adx,
    alligator, fractals, awesome_oscillator,
)

def _tf_preset(tf: str) -> Dict[str, Any]:
    # Slightly different “tightness” per TF
    if tf == "M15":
        return {
            "EMA_FAST": 20, "EMA_SLOW": 50,
            "RSI_LEN": 14,
            "RSI_BUY_MAX": 45,    # buy when momentum is not overbought
            "RSI_SELL_MIN": 55,
            "STO_K": 14, "STO_D": 3, "STO_SMOOTH": 3,
            "ADX_LEN": 14, "ADX_MIN": 18,
            "ATR_LEN": 14,
            "SL_ATR": 1.2, "TP_RR": 1.5,
            "MIN_SCORE": 4,
        }
    # M30 default
    return {
        "EMA_FAST": 20, "EMA_SLOW": 50,
        "RSI_LEN": 14,
        "RSI_BUY_MAX": 48,
        "RSI_SELL_MIN": 52,
        "STO_K": 14, "STO_D": 3, "STO_SMOOTH": 3,
        "ADX_LEN": 14, "ADX_MIN": 20,
        "ATR_LEN": 14,
        "SL_ATR": 1.3, "TP_RR": 1.5,
        "MIN_SCORE": 4,
    }

def compute_state(candles: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if candles is None or len(candles) == 0:
        return {"ok": False, "msg": "No candles yet"}

    # infer TF from cfg if provided; else default to M30 preset style
    tf = str(cfg.get("TF", "M30"))
    p = _tf_preset(tf)

    # Need enough bars for slow indicators (AO 34, EMA 50, etc.)
    min_needed = max(p["EMA_SLOW"] + 10, p["ATR_LEN"] + 10, 80)
    if len(candles) < min_needed:
        return {"ok": False, "msg": f"Not enough candles yet ({len(candles)}/{min_needed})"}

    df = candles.copy()

    # --- Trend backbone ---
    df["ema20"] = ema(df["close"], p["EMA_FAST"])
    df["ema50"] = ema(df["close"], p["EMA_SLOW"])
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"], 12, 26, 9)
    df["adx"] = adx(df, p["ADX_LEN"])

    # --- Momentum / timing ---
    df["rsi"] = rsi(df["close"], p["RSI_LEN"])
    df["stoch_k"], df["stoch_d"] = stochastic(df, p["STO_K"], p["STO_D"], p["STO_SMOOTH"])

    # --- Volatility ---
    df["atr"] = atr(df, p["ATR_LEN"])
    df["atr_pct"] = df["atr"] / (df["close"].abs() + 1e-12)

    # --- Bill Williams ---
    df["jaw"], df["teeth"], df["lips"] = alligator(df)
    df["fr_low"], df["fr_high"] = fractals(df)
    df["ao"] = awesome_oscillator(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    ts = df.index[-1]

    # Safety checks
    if not np.isfinite(last["atr"]) or not np.isfinite(last["ema20"]) or not np.isfinite(last["ema50"]):
        return {"ok": False, "msg": "Indicators not ready yet"}

    # --- Regime filters ---
    vol_ok = bool(last["atr_pct"] > 0.00015)   # avoid dead flat periods
    trend_up = bool(last["ema20"] > last["ema50"])
    trend_dn = bool(last["ema20"] < last["ema50"])
    macd_up = bool(last["macd_hist"] > 0)
    macd_dn = bool(last["macd_hist"] < 0)
    adx_ok = bool(np.isfinite(last["adx"]) and last["adx"] >= p["ADX_MIN"])

    # Bill Williams “alignment”
    alli_up = bool(last["lips"] > last["teeth"] > last["jaw"])
    alli_dn = bool(last["lips"] < last["teeth"] < last["jaw"])

    # Momentum confirmations
    rsi_buy_ok = bool(last["rsi"] <= p["RSI_BUY_MAX"])
    rsi_sell_ok = bool(last["rsi"] >= p["RSI_SELL_MIN"])

    stoch_cross_up = bool(prev["stoch_k"] < prev["stoch_d"] and last["stoch_k"] > last["stoch_d"] and last["stoch_k"] < 40)
    stoch_cross_dn = bool(prev["stoch_k"] > prev["stoch_d"] and last["stoch_k"] < last["stoch_d"] and last["stoch_k"] > 60)

    ao_rising = bool(last["ao"] > prev["ao"])
    ao_falling = bool(last["ao"] < prev["ao"])

    # Fractal break entry trigger (clean for M15/M30)
    break_fr_high = bool(np.isfinite(last["fr_high"]) and last["close"] > last["fr_high"])
    break_fr_low  = bool(np.isfinite(last["fr_low"])  and last["close"] < last["fr_low"])

    buy_conditions = {
        "Volatility ok": vol_ok,
        "EMA20>EMA50": trend_up,
        "MACD hist > 0": macd_up,
        "ADX ok": adx_ok,
        "Alligator up": alli_up,
        "Stoch cross up": stoch_cross_up,
        "AO rising": ao_rising,
        "Break fractal high": break_fr_high,
        "RSI ok buy": rsi_buy_ok,
    }

    sell_conditions = {
        "Volatility ok": vol_ok,
        "EMA20<EMA50": trend_dn,
        "MACD hist < 0": macd_dn,
        "ADX ok": adx_ok,
        "Alligator down": alli_dn,
        "Stoch cross down": stoch_cross_dn,
        "AO falling": ao_falling,
        "Break fractal low": break_fr_low,
        "RSI ok sell": rsi_sell_ok,
    }

    buy_score = sum(bool(v) for v in buy_conditions.values())
    sell_score = sum(bool(v) for v in sell_conditions.values())

    side = "NO_TRADE"
    conds: Dict[str, Any] = {"BUY_score": buy_score, "SELL_score": sell_score}
    reasons = []

    if buy_score >= p["MIN_SCORE"] and buy_score > sell_score:
        side = "BUY"
        conds = buy_conditions
        reasons = [k for k, v in buy_conditions.items() if v]
    elif sell_score >= p["MIN_SCORE"] and sell_score > buy_score:
        side = "SELL"
        conds = sell_conditions
        reasons = [k for k, v in sell_conditions.items() if v]

    entry_proxy = float(last["close"])
    atrv = float(last["atr"])

    plan = {"entry": None, "sl": None, "tp": None, "sl_atr": p["SL_ATR"], "tp_rr": p["TP_RR"]}
    if side in ("BUY", "SELL"):
        sl_dist = p["SL_ATR"] * atrv
        if side == "BUY":
            sl = entry_proxy - sl_dist
            tp = entry_proxy + p["TP_RR"] * (entry_proxy - sl)
        else:
            sl = entry_proxy + sl_dist
            tp = entry_proxy - p["TP_RR"] * (sl - entry_proxy)
        plan = {"entry": entry_proxy, "sl": float(sl), "tp": float(tp), "sl_atr": p["SL_ATR"], "tp_rr": p["TP_RR"]}

    reason_str = ""
    if side != "NO_TRADE" and reasons:
        reason_str = f"{side} because " + " + ".join(reasons)

    return {
        "ok": True,
        "timestamp": str(ts),
        "close": float(last["close"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]) if np.isfinite(last["atr"]) else None,
        "side": side,
        "score": int(max(buy_score, sell_score)) if side != "NO_TRADE" else 0,
        "min_required": int(p["MIN_SCORE"]),
        "conditions": conds,
        "reason": reason_str,
        "plan": plan,
    }
