# strategy_rules.py
import numpy as np
import pandas as pd
from typing import Dict, Any

from indicators import (
    rsi, atr, ema, macd, stochastic, adx,
    alligator, fractals, awesome_oscillator,
    bollinger, rolling_zscore, kama, psar
)

def _tf_preset(tf: str) -> Dict[str, Any]:
    if tf == "M15":
        return {
            # Trend + momentum defaults
            "EMA_FAST": 20, "EMA_SLOW": 50,
            "KAMA_ER": 10, "KAMA_FAST": 2, "KAMA_SLOW": 30,
            "RSI_LEN": 14,
            "RSI_BUY_MAX": 48,
            "RSI_SELL_MIN": 52,
            "STO_K": 14, "STO_D": 3, "STO_SMOOTH": 3,
            "ADX_LEN": 14, "ADX_MIN": 18,
            "ATR_LEN": 14,

            # BB / regime
            "BB_LEN": 20, "BB_STD": 2.0,
            "STD_Z_LEN": 60,
            "BB_WIDTH_MIN": 0.0010,    # 0.10% width (tune)
            "STD_Z_MIN": -0.50,        # allow normal volatility and above

            # PSAR
            "PSAR_STEP": 0.02, "PSAR_MAX": 0.2,

            # Risk
            "SL_ATR": 1.2, "TP_RR": 1.5,

            # Decision tightness
            "MIN_SCORE": 5,
        }

    # M30
    return {
        "EMA_FAST": 20, "EMA_SLOW": 50,
        "KAMA_ER": 10, "KAMA_FAST": 2, "KAMA_SLOW": 30,
        "RSI_LEN": 14,
        "RSI_BUY_MAX": 50,
        "RSI_SELL_MIN": 50,
        "STO_K": 14, "STO_D": 3, "STO_SMOOTH": 3,
        "ADX_LEN": 14, "ADX_MIN": 20,
        "ATR_LEN": 14,

        "BB_LEN": 20, "BB_STD": 2.0,
        "STD_Z_LEN": 80,
        "BB_WIDTH_MIN": 0.0012,
        "STD_Z_MIN": -0.35,

        "PSAR_STEP": 0.02, "PSAR_MAX": 0.2,

        "SL_ATR": 1.3, "TP_RR": 1.5,
        "MIN_SCORE": 5,
    }

def compute_state(candles: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if candles is None or len(candles) == 0:
        return {"ok": False, "msg": "No candles yet"}

    tf = str(cfg.get("TF", "M30"))
    p = _tf_preset(tf)

    # Need enough bars (AO 34, EMA 50, BB 20, zscore 60/80)
    min_needed = max(
        p["EMA_SLOW"] + 10,
        p["ATR_LEN"] + 10,
        p["BB_LEN"] + 5,
        p["STD_Z_LEN"] + 5,
        120
    )
    if len(candles) < min_needed:
        return {"ok": False, "msg": f"Not enough candles yet ({len(candles)}/{min_needed})"}

    df = candles.copy()

    # --- Trend backbone ---
    df["ema_fast"] = ema(df["close"], p["EMA_FAST"])
    df["ema_slow"] = ema(df["close"], p["EMA_SLOW"])
    df["kama"] = kama(df["close"], p["KAMA_ER"], p["KAMA_FAST"], p["KAMA_SLOW"])
    df["kama_slope"] = df["kama"].diff()

    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"], 12, 26, 9)
    df["adx"] = adx(df, p["ADX_LEN"])

    # --- Momentum / timing ---
    df["rsi"] = rsi(df["close"], p["RSI_LEN"])
    df["stoch_k"], df["stoch_d"] = stochastic(df, p["STO_K"], p["STO_D"], p["STO_SMOOTH"])

    # --- Volatility ---
    df["atr"] = atr(df, p["ATR_LEN"])
    df["atr_pct"] = df["atr"] / (df["close"].abs() + 1e-12)

    # --- Bollinger + StdDev regime ---
    df["bb_mid"], df["bb_up"], df["bb_lo"], df["bb_sd"], df["bb_width"] = bollinger(
        df["close"], p["BB_LEN"], p["BB_STD"]
    )
    df["sd_z"] = rolling_zscore(df["bb_sd"], p["STD_Z_LEN"])

    # --- PSAR ---
    df["psar"] = psar(df, p["PSAR_STEP"], p["PSAR_MAX"])

    # --- Bill Williams ---
    df["jaw"], df["teeth"], df["lips"] = alligator(df)
    df["fr_low"], df["fr_high"] = fractals(df)
    df["ao"] = awesome_oscillator(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    ts = df.index[-1]

    # Safety checks
    needed_cols = ["atr", "ema_fast", "ema_slow", "kama", "bb_width", "psar"]
    if any((not np.isfinite(last[c])) for c in needed_cols):
        return {"ok": False, "msg": "Indicators not ready yet"}

    # ------------------------
    # Regime filters
    # ------------------------
    # 1) basic “not dead” volatility
    vol_ok = bool(last["atr_pct"] > 0.00015)

    # 2) BB width + stddev zscore says market is “moving enough”
    bb_ok = bool(last["bb_width"] >= p["BB_WIDTH_MIN"])
    std_ok = bool(np.isfinite(last["sd_z"]) and last["sd_z"] >= p["STD_Z_MIN"])

    # 3) trend strength
    adx_ok = bool(np.isfinite(last["adx"]) and last["adx"] >= p["ADX_MIN"])

    regime_ok = bool(vol_ok and (bb_ok or std_ok))  # allow either width OR sd_z to pass

    # ------------------------
    # Trend direction (avoid redundancy)
    # ------------------------
    ema_up = bool(last["ema_fast"] > last["ema_slow"])
    ema_dn = bool(last["ema_fast"] < last["ema_slow"])

    kama_up = bool(last["kama_slope"] > 0)
    kama_dn = bool(last["kama_slope"] < 0)

    # Trend vote: accept EMA or KAMA (don’t require both)
    trend_up = bool(ema_up or kama_up)
    trend_dn = bool(ema_dn or kama_dn)

    # MACD confirmation (kept)
    macd_up = bool(last["macd_hist"] > 0)
    macd_dn = bool(last["macd_hist"] < 0)

    # PSAR direction confirmation (light)
    psar_bull = bool(last["close"] > last["psar"])
    psar_bear = bool(last["close"] < last["psar"])

    # Alligator alignment
    alli_up = bool(last["lips"] > last["teeth"] > last["jaw"])
    alli_dn = bool(last["lips"] < last["teeth"] < last["jaw"])

    # Momentum confirmations
    rsi_buy_ok = bool(last["rsi"] <= p["RSI_BUY_MAX"])
    rsi_sell_ok = bool(last["rsi"] >= p["RSI_SELL_MIN"])

    stoch_cross_up = bool(
        prev["stoch_k"] < prev["stoch_d"] and
        last["stoch_k"] > last["stoch_d"] and
        last["stoch_k"] < 45
    )
    stoch_cross_dn = bool(
        prev["stoch_k"] > prev["stoch_d"] and
        last["stoch_k"] < last["stoch_d"] and
        last["stoch_k"] > 55
    )

    ao_rising = bool(last["ao"] > prev["ao"])
    ao_falling = bool(last["ao"] < prev["ao"])

    # Fractal break trigger
    break_fr_high = bool(np.isfinite(last["fr_high"]) and last["close"] > last["fr_high"])
    break_fr_low  = bool(np.isfinite(last["fr_low"])  and last["close"] < last["fr_low"])

    # Optional BB location filter (prevents buying at extreme tops / selling at extreme bottoms)
    not_buy_extreme = bool(last["close"] < last["bb_up"])   # avoid buying above upper band
    not_sell_extreme = bool(last["close"] > last["bb_lo"])  # avoid selling below lower band

    # ------------------------
    # Conditions
    # ------------------------
    buy_conditions = {
        "Regime ok (vol + BB/Std)": regime_ok,
        "Trend up (EMA or KAMA)": trend_up,
        "MACD hist > 0": macd_up,
        "ADX ok": adx_ok,
        "PSAR bull": psar_bull,
        "Alligator up": alli_up,
        "Stoch cross up": stoch_cross_up,
        "AO rising": ao_rising,
        "Break fractal high": break_fr_high,
        "RSI ok buy": rsi_buy_ok,
        "Not at BB upper extreme": not_buy_extreme,
    }

    sell_conditions = {
        "Regime ok (vol + BB/Std)": regime_ok,
        "Trend down (EMA or KAMA)": trend_dn,
        "MACD hist < 0": macd_dn,
        "ADX ok": adx_ok,
        "PSAR bear": psar_bear,
        "Alligator down": alli_dn,
        "Stoch cross down": stoch_cross_dn,
        "AO falling": ao_falling,
        "Break fractal low": break_fr_low,
        "RSI ok sell": rsi_sell_ok,
        "Not at BB lower extreme": not_sell_extreme,
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

    # Trade plan: ATR based
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

        plan = {
            "entry": float(entry_proxy),
            "sl": float(sl),
            "tp": float(tp),
            "sl_atr": float(p["SL_ATR"]),
            "tp_rr": float(p["TP_RR"]),
        }

    reason_str = ""
    if side != "NO_TRADE" and reasons:
        # keep short so UI doesn't overflow
        reason_str = f"{side} because " + " + ".join(reasons[:6]) + (" ..." if len(reasons) > 6 else "")

    return {
        "ok": True,
        "timestamp": str(ts),
        "close": float(last["close"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]) if np.isfinite(last["atr"]) else None,

        # useful extra telemetry if you want later
        "adx": float(last["adx"]) if np.isfinite(last["adx"]) else None,
        "bb_width": float(last["bb_width"]) if np.isfinite(last["bb_width"]) else None,
        "sd_z": float(last["sd_z"]) if np.isfinite(last["sd_z"]) else None,

        "side": side,
        "score": int(max(buy_score, sell_score)) if side != "NO_TRADE" else 0,
        "min_required": int(p["MIN_SCORE"]),
        "conditions": conds,
        "reason": reason_str,
        "plan": plan,
    }
