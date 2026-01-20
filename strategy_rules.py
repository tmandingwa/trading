# strategy_rules.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from indicators import (
    rsi, atr, ema, macd, stochastic, adx,
    alligator, fractals, awesome_oscillator,
    bollinger, rolling_zscore, kama, psar
)

# ------------------------------------------------------------
# Presets
# ------------------------------------------------------------
def _tf_preset(tf: str) -> Dict[str, Any]:
    """
    Notes on changes:
    - TP_RR bumped to 2.0 so the system can survive lower win-rates.
    - Added BB_MID_BUFFER to avoid entries in the "middle of the range" chop.
    - Added COOLDOWN_BARS recommendation (engine can enforce; optional cfg input supported too).
    - Added HTF gate defaults:
        * M15 is gated by M30 direction
        * M30 is optionally gated by H1 direction (if you pass H1 candles in cfg)
    """
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
            "BB_WIDTH_MIN": 0.0010,     # width threshold
            "STD_Z_MIN": -0.50,

            # Avoid chop: require price to be away from BB mid by a fraction of BB width
            "BB_MID_BUFFER": 0.15,      # 15% of BB width away from mid

            # PSAR
            "PSAR_STEP": 0.02, "PSAR_MAX": 0.2,

            # Risk (updated)
            "SL_ATR": 1.2,
            "TP_RR": 2.0,               # ✅ changed from 1.5 → 2.0

            # Decision tightness
            "MIN_SCORE": 5,

            # Meta recommendations
            "COOLDOWN_BARS": 3,         # recommended after SL (engine can enforce)
            "HTF_GATE_TF": "M30",       # M15 should align with M30
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

        "BB_MID_BUFFER": 0.12,        # slightly softer on M30

        "PSAR_STEP": 0.02, "PSAR_MAX": 0.2,

        # Risk (updated)
        "SL_ATR": 1.3,
        "TP_RR": 2.0,                 # ✅ changed from 1.5 → 2.0

        "MIN_SCORE": 5,

        "COOLDOWN_BARS": 2,
        "HTF_GATE_TF": "H1",          # ✅ optional: gate M30 by H1 if provided in cfg
    }


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _min_needed_from_preset(p: Dict[str, Any]) -> int:
    return max(
        p["EMA_SLOW"] + 10,
        p["ATR_LEN"] + 10,
        p["BB_LEN"] + 5,
        p["STD_Z_LEN"] + 5,
        120
    )

def _coerce_ts(x) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    try:
        if isinstance(x, pd.Timestamp):
            return x
        return pd.to_datetime(x, utc=False, errors="coerce")
    except Exception:
        return None

def _compute_indicators(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    """
    Computes indicators used both for LTF and HTF direction gating.
    Assumes df has columns: open, high, low, close
    """
    out = df.copy()

    # Trend backbone
    out["ema_fast"] = ema(out["close"], p["EMA_FAST"])
    out["ema_slow"] = ema(out["close"], p["EMA_SLOW"])
    out["kama"] = kama(out["close"], p["KAMA_ER"], p["KAMA_FAST"], p["KAMA_SLOW"])
    out["kama_slope"] = out["kama"].diff()

    out["macd"], out["macd_sig"], out["macd_hist"] = macd(out["close"], 12, 26, 9)
    out["adx"] = adx(out, p["ADX_LEN"])

    # Momentum / timing
    out["rsi"] = rsi(out["close"], p["RSI_LEN"])
    out["stoch_k"], out["stoch_d"] = stochastic(out, p["STO_K"], p["STO_D"], p["STO_SMOOTH"])

    # Volatility
    out["atr"] = atr(out, p["ATR_LEN"])
    out["atr_pct"] = out["atr"] / (out["close"].abs() + 1e-12)

    # Bollinger + StdDev regime
    out["bb_mid"], out["bb_up"], out["bb_lo"], out["bb_sd"], out["bb_width"] = bollinger(
        out["close"], p["BB_LEN"], p["BB_STD"]
    )
    out["sd_z"] = rolling_zscore(out["bb_sd"], p["STD_Z_LEN"])

    # PSAR
    out["psar"] = psar(out, p["PSAR_STEP"], p["PSAR_MAX"])

    # Bill Williams
    out["jaw"], out["teeth"], out["lips"] = alligator(out)
    out["fr_low"], out["fr_high"] = fractals(out)
    out["ao"] = awesome_oscillator(out)

    return out

def _trend_direction(last: pd.Series) -> str:
    """
    Returns: "UP" | "DOWN" | "FLAT"
    Uses EMA + KAMA slope + MACD hist for a robust directional bias.
    """
    if not np.isfinite(last.get("ema_fast", np.nan)) or not np.isfinite(last.get("ema_slow", np.nan)):
        return "FLAT"

    ema_up = bool(last["ema_fast"] > last["ema_slow"])
    ema_dn = bool(last["ema_fast"] < last["ema_slow"])

    kama_slope = last.get("kama_slope", np.nan)
    kama_up = bool(np.isfinite(kama_slope) and kama_slope > 0)
    kama_dn = bool(np.isfinite(kama_slope) and kama_slope < 0)

    macd_hist = last.get("macd_hist", np.nan)
    macd_up = bool(np.isfinite(macd_hist) and macd_hist > 0)
    macd_dn = bool(np.isfinite(macd_hist) and macd_hist < 0)

    # 2-of-3 vote
    up_votes = int(ema_up) + int(kama_up) + int(macd_up)
    dn_votes = int(ema_dn) + int(kama_dn) + int(macd_dn)

    if up_votes >= 2 and up_votes > dn_votes:
        return "UP"
    if dn_votes >= 2 and dn_votes > up_votes:
        return "DOWN"
    return "FLAT"

def _apply_htf_gate(
    ltf_side: str,
    htf_dir: str
) -> Tuple[bool, str]:
    """
    Gate LTF signal by HTF direction.
    - If HTF is UP: block SELL
    - If HTF is DOWN: block BUY
    - If HTF is FLAT: allow both (or you can block both; we allow to avoid 0 trades)
    """
    if ltf_side not in ("BUY", "SELL"):
        return True, ""

    if htf_dir == "UP" and ltf_side == "SELL":
        return False, "Blocked by HTF gate (HTF UP, SELL not allowed)"
    if htf_dir == "DOWN" and ltf_side == "BUY":
        return False, "Blocked by HTF gate (HTF DOWN, BUY not allowed)"
    return True, ""


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def compute_state(candles: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      candles: OHLC dataframe for the selected TF (UTC timestamps are fine; index can be datetime)
      cfg:
        - TF: "M15" or "M30" (default "M30")
        Optional (to enable HTF gating):
        - HTF_CANDLES: OHLC dataframe for the higher timeframe
        - HTF_TF: string label for UI/debug (e.g. "M30" or "H1")
        Optional (cooldown enforcement - engine can pass this after SL):
        - COOLDOWN_UNTIL: timestamp (string/datetime) -> if now < this, NO_TRADE
    """
    if candles is None or len(candles) == 0:
        return {"ok": False, "msg": "No candles yet"}

    tf = str(cfg.get("TF", "M30"))
    p = _tf_preset(tf)

    # Cooldown (optional enforcement)
    cooldown_until = _coerce_ts(cfg.get("COOLDOWN_UNTIL"))
    if cooldown_until is not None:
        # current candle timestamp
        now_ts = candles.index[-1]
        try:
            now_ts = pd.to_datetime(now_ts)
        except Exception:
            now_ts = pd.Timestamp.utcnow()

        if pd.notna(cooldown_until) and pd.to_datetime(now_ts) < pd.to_datetime(cooldown_until):
            return {
                "ok": True,
                "timestamp": str(now_ts),
                "close": float(candles["close"].iloc[-1]),
                "rsi": None,
                "atr": None,
                "adx": None,
                "bb_width": None,
                "sd_z": None,
                "side": "NO_TRADE",
                "score": 0,
                "min_required": int(p["MIN_SCORE"]),
                "conditions": {"Cooldown active": True},
                "reason": f"NO_TRADE (cooldown active until {cooldown_until})",
                "plan": {"entry": None, "sl": None, "tp": None, "sl_atr": p["SL_ATR"], "tp_rr": p["TP_RR"]},
                "meta": {
                    "tf": tf,
                    "cooldown_until": str(cooldown_until),
                    "cooldown_bars_recommendation": int(p["COOLDOWN_BARS"]),
                }
            }

    min_needed = _min_needed_from_preset(p)
    if len(candles) < min_needed:
        return {"ok": False, "msg": f"Not enough candles yet ({len(candles)}/{min_needed})"}

    # Compute LTF indicators
    df = _compute_indicators(candles, p)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    ts = df.index[-1]

    # Safety checks
    needed_cols = ["atr", "ema_fast", "ema_slow", "kama", "bb_width", "psar", "bb_mid", "bb_up", "bb_lo"]
    if any((not np.isfinite(last.get(c, np.nan))) for c in needed_cols):
        return {"ok": False, "msg": "Indicators not ready yet"}

    # ------------------------
    # Regime filters
    # ------------------------
    vol_ok = bool(last["atr_pct"] > 0.00015)
    bb_ok = bool(last["bb_width"] >= p["BB_WIDTH_MIN"])
    std_ok = bool(np.isfinite(last["sd_z"]) and last["sd_z"] >= p["STD_Z_MIN"])
    adx_ok = bool(np.isfinite(last["adx"]) and last["adx"] >= p["ADX_MIN"])

    regime_ok = bool(vol_ok and (bb_ok or std_ok))

    # ------------------------
    # Trend direction (LTF)
    # ------------------------
    ema_up = bool(last["ema_fast"] > last["ema_slow"])
    ema_dn = bool(last["ema_fast"] < last["ema_slow"])

    kama_up = bool(last["kama_slope"] > 0)
    kama_dn = bool(last["kama_slope"] < 0)

    trend_up = bool(ema_up or kama_up)
    trend_dn = bool(ema_dn or kama_dn)

    macd_up = bool(last["macd_hist"] > 0)
    macd_dn = bool(last["macd_hist"] < 0)

    psar_bull = bool(last["close"] > last["psar"])
    psar_bear = bool(last["close"] < last["psar"])

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

    break_fr_high = bool(np.isfinite(last["fr_high"]) and last["close"] > last["fr_high"])
    break_fr_low  = bool(np.isfinite(last["fr_low"])  and last["close"] < last["fr_low"])

    # Optional BB extreme filter (kept)
    not_buy_extreme = bool(last["close"] < last["bb_up"])
    not_sell_extreme = bool(last["close"] > last["bb_lo"])

    # ✅ New: avoid entries in the middle of the band (chop zone)
    # buffer = fraction of bb_width, so it adapts to volatility
    bb_mid = float(last["bb_mid"])
    bb_width = float(last["bb_width"])
    mid_buffer = float(p["BB_MID_BUFFER"]) * bb_width
    away_from_mid_buy = bool(last["close"] > (bb_mid + mid_buffer))
    away_from_mid_sell = bool(last["close"] < (bb_mid - mid_buffer))

    # ------------------------
    # Conditions (LTF)
    # ------------------------
    buy_conditions = {
        "Regime ok (vol + BB/Std)": regime_ok,
        "Away from BB mid (anti-chop)": away_from_mid_buy,   # ✅ added
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
        "Away from BB mid (anti-chop)": away_from_mid_sell,  # ✅ added
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

    # ------------------------
    # ✅ HTF Direction Gate
    # - M15 gates by M30 (recommended)
    # - M30 can gate by H1 (recommended if you have H1 candles)
    # Works only if you pass cfg["HTF_CANDLES"].
    # ------------------------
    htf_dir = "NA"
    gate_ok = True
    gate_msg = ""

    htf_candles = cfg.get("HTF_CANDLES", None)
    htf_tf_label = str(cfg.get("HTF_TF", p.get("HTF_GATE_TF", "HTF")))

    if isinstance(htf_candles, pd.DataFrame) and len(htf_candles) > 0:
        # Use the HTF's own preset if known; otherwise reuse current preset.
        # If HTF is H1, using M30 preset is fine for direction bias.
        htf_p = _tf_preset("M30") if htf_tf_label in ("M30", "H1", "H4", "D") else p

        htf_min_needed = _min_needed_from_preset(htf_p)
        if len(htf_candles) >= htf_min_needed:
            hdf = _compute_indicators(htf_candles, htf_p)
            hlast = hdf.iloc[-1]
            htf_dir = _trend_direction(hlast)

            gate_ok, gate_msg = _apply_htf_gate(side, htf_dir)
            if not gate_ok:
                # Block trade
                side = "NO_TRADE"
                # preserve original score info
                conds = {"HTF gate": gate_msg, "HTF_dir": htf_dir, "HTF_TF": htf_tf_label, **conds}
                reasons = []
        else:
            htf_dir = "NOT_READY"

    # Trade plan: ATR based (RR already adjusted to 2.0 in presets)
    entry_proxy = float(last["close"])
    atrv = float(last["atr"])

    plan = {"entry": None, "sl": None, "tp": None, "sl_atr": p["SL_ATR"], "tp_rr": p["TP_RR"]}

    if side in ("BUY", "SELL"):
        sl_dist = float(p["SL_ATR"]) * atrv
        rr = float(p["TP_RR"])

        if side == "BUY":
            sl = entry_proxy - sl_dist
            tp = entry_proxy + rr * (entry_proxy - sl)
        else:
            sl = entry_proxy + sl_dist
            tp = entry_proxy - rr * (sl - entry_proxy)

        plan = {
            "entry": float(entry_proxy),
            "sl": float(sl),
            "tp": float(tp),
            "sl_atr": float(p["SL_ATR"]),
            "tp_rr": float(p["TP_RR"]),
            # Optional guidance for engine (if you implement later):
            "breakeven_at_r": 1.0,     # move SL to entry at +1R
            "trail_after_r": 1.5,      # start trailing after +1.5R
        }

    reason_str = ""
    if side != "NO_TRADE" and reasons:
        reason_str = f"{side} because " + " + ".join(reasons[:6]) + (" ..." if len(reasons) > 6 else "")
    elif side == "NO_TRADE" and gate_msg:
        reason_str = f"NO_TRADE because {gate_msg}"

    # LTF direction telemetry (useful in logs)
    ltf_dir = _trend_direction(last)

    return {
        "ok": True,
        "timestamp": str(ts),
        "close": float(last["close"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]) if np.isfinite(last["atr"]) else None,

        "adx": float(last["adx"]) if np.isfinite(last["adx"]) else None,
        "bb_width": float(last["bb_width"]) if np.isfinite(last["bb_width"]) else None,
        "sd_z": float(last["sd_z"]) if np.isfinite(last["sd_z"]) else None,

        "side": side,
        "score": int(max(buy_score, sell_score)) if side != "NO_TRADE" else 0,
        "min_required": int(p["MIN_SCORE"]),
        "conditions": conds,
        "reason": reason_str,
        "plan": plan,

        # ✅ meta: helps you debug why trades were blocked / allowed
        "meta": {
            "tf": tf,
            "ltf_dir": ltf_dir,
            "htf_tf": htf_tf_label,
            "htf_dir": htf_dir,
            "htf_gate_ok": bool(gate_ok),
            "bb_mid_buffer_frac": float(p["BB_MID_BUFFER"]),
            "cooldown_bars_recommendation": int(p["COOLDOWN_BARS"]),
        },
    }
