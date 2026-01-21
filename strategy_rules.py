# strategy_rules.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from indicators import (
    rsi, atr, ema, macd, stochastic, adx,
    alligator, fractals, awesome_oscillator,
    bollinger, rolling_zscore, kama, psar
)

# ✅ ML
from ml_model import DirectionModelStore


# ------------------------------------------------------------
# Presets
# ------------------------------------------------------------
def _tf_preset(tf: str) -> Dict[str, Any]:
    if tf == "M15":
        return {
            "EMA_FAST": 20, "EMA_SLOW": 50,
            "KAMA_ER": 10, "KAMA_FAST": 2, "KAMA_SLOW": 30,
            "RSI_LEN": 14,
            "RSI_BUY_MAX": 48,
            "RSI_SELL_MIN": 52,
            "STO_K": 14, "STO_D": 3, "STO_SMOOTH": 3,
            "ADX_LEN": 14, "ADX_MIN": 18,
            "ATR_LEN": 14,

            "BB_LEN": 20, "BB_STD": 2.0,
            "STD_Z_LEN": 60,
            "BB_WIDTH_MIN": 0.0010,
            "STD_Z_MIN": -0.50,
            "BB_MID_BUFFER": 0.15,

            "PSAR_STEP": 0.02, "PSAR_MAX": 0.2,

            "SL_ATR": 1.2,
            "TP_RR": 2.0,

            "MIN_SCORE": 5,

            "COOLDOWN_BARS": 3,
            "HTF_GATE_TF": "M30",
        }

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
        "BB_MID_BUFFER": 0.12,

        "PSAR_STEP": 0.02, "PSAR_MAX": 0.2,

        "SL_ATR": 1.3,
        "TP_RR": 2.0,

        "MIN_SCORE": 5,

        "COOLDOWN_BARS": 2,
        "HTF_GATE_TF": "H1",
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
    out = df.copy()

    out["ema_fast"] = ema(out["close"], p["EMA_FAST"])
    out["ema_slow"] = ema(out["close"], p["EMA_SLOW"])
    out["kama"] = kama(out["close"], p["KAMA_ER"], p["KAMA_FAST"], p["KAMA_SLOW"])
    out["kama_slope"] = out["kama"].diff()

    out["macd"], out["macd_sig"], out["macd_hist"] = macd(out["close"], 12, 26, 9)
    out["adx"] = adx(out, p["ADX_LEN"])

    out["rsi"] = rsi(out["close"], p["RSI_LEN"])
    out["stoch_k"], out["stoch_d"] = stochastic(out, p["STO_K"], p["STO_D"], p["STO_SMOOTH"])

    out["atr"] = atr(out, p["ATR_LEN"])
    out["atr_pct"] = out["atr"] / (out["close"].abs() + 1e-12)

    out["bb_mid"], out["bb_up"], out["bb_lo"], out["bb_sd"], out["bb_width"] = bollinger(
        out["close"], p["BB_LEN"], p["BB_STD"]
    )
    out["sd_z"] = rolling_zscore(out["bb_sd"], p["STD_Z_LEN"])

    out["psar"] = psar(out, p["PSAR_STEP"], p["PSAR_MAX"])

    out["jaw"], out["teeth"], out["lips"] = alligator(out)
    out["fr_low"], out["fr_high"] = fractals(out)
    out["ao"] = awesome_oscillator(out)

    return out

def _trend_direction(last: pd.Series) -> str:
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

    up_votes = int(ema_up) + int(kama_up) + int(macd_up)
    dn_votes = int(ema_dn) + int(kama_dn) + int(macd_dn)

    if up_votes >= 2 and up_votes > dn_votes:
        return "UP"
    if dn_votes >= 2 and dn_votes > up_votes:
        return "DOWN"
    return "FLAT"

def _apply_htf_gate(ltf_side: str, htf_dir: str) -> Tuple[bool, str]:
    if ltf_side not in ("BUY", "SELL"):
        return True, ""

    if htf_dir == "UP" and ltf_side == "SELL":
        return False, "Blocked by HTF gate (HTF UP, SELL not allowed)"
    if htf_dir == "DOWN" and ltf_side == "BUY":
        return False, "Blocked by HTF gate (HTF DOWN, BUY not allowed)"
    return True, ""


# ✅ cached model store
_MODEL_STORE: Optional[DirectionModelStore] = None

def _get_model_store(artifacts_dir: str) -> DirectionModelStore:
    global _MODEL_STORE
    if _MODEL_STORE is None or getattr(_MODEL_STORE, "artifacts_dir", None) != artifacts_dir:
        _MODEL_STORE = DirectionModelStore(artifacts_dir=artifacts_dir)
    return _MODEL_STORE


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


# ------------------------------------------------------------
# ✅ ML gating policy (fixes “threshold too high / wrong direction”)
# ------------------------------------------------------------
def _ml_gate_policy(
    side: str,
    p_up: Optional[float],
    buy_th: Optional[float],
    sell_th: Optional[float],
    cfg: Dict[str, Any]
) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """
    Returns: (allowed, message, buy_th_used, sell_th_used)

    Key changes vs your current code:
    1) ML FAILURE DOES NOT BLOCK trades by default (fallback to rules).
    2) Thresholds are clamped into a sane range, so you don't set them too strict accidentally.
    3) Optional “soft gate”: require only a small edge over 0.5 (default 0.02) if thresholds look too extreme.
    """
    # If we don't have a valid probability, don't block (rules-only)
    if p_up is None:
        return True, "ML: no prob (fallback to rules)", buy_th, sell_th

    # Config knobs (can be set in CFG/env and passed down):
    # - ML_STRICT: if True, block when ML says no (old behavior)
    # - ML_EDGE: small edge over 0.5 required if you want soft confirmation
    # - ML_MIN_TH / ML_MAX_TH: clamp thresholds into this band
    ml_strict = bool(cfg.get("ML_STRICT", False))
    ml_edge = _as_float(cfg.get("ML_EDGE", 0.02))  # 0.02 => 0.52/0.48
    ml_min_th = _as_float(cfg.get("ML_MIN_TH", 0.52))
    ml_max_th = _as_float(cfg.get("ML_MAX_TH", 0.65))

    # Normalize/clamp thresholds
    bt = _as_float(buy_th)
    st = _as_float(sell_th)

    # If model did not provide thresholds, build soft thresholds around 0.5
    if bt is None:
        bt = 0.5 + float(ml_edge)
    if st is None:
        st = 0.5 - float(ml_edge)

    # Clamp buy threshold to [ml_min_th, ml_max_th]
    if ml_min_th is not None and ml_max_th is not None:
        bt = float(min(max(bt, ml_min_th), ml_max_th))
        # sell threshold is symmetric; keep it as (1 - bt) unless provided
        # but still clamp to [1-ml_max_th, 1-ml_min_th]
        st_lo = 1.0 - float(ml_max_th)
        st_hi = 1.0 - float(ml_min_th)
        st = float(min(max(st, st_lo), st_hi))

    # Decide
    if side == "BUY":
        ok = bool(p_up >= bt)
        msg = f"ML BUY gate: p_up={p_up:.3f} >= {bt:.3f} -> {'OK' if ok else 'BLOCK'}"
    elif side == "SELL":
        ok = bool(p_up <= st)
        msg = f"ML SELL gate: p_up={p_up:.3f} <= {st:.3f} -> {'OK' if ok else 'BLOCK'}"
    else:
        return True, "ML: no side", bt, st

    # Non-strict mode: if ML blocks, we DO NOT block trades; we just annotate.
    if (not ok) and (not ml_strict):
        return True, f"{msg} (non-strict: allow rules)", bt, st

    return ok, msg, bt, st


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def compute_state(candles: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if candles is None or len(candles) == 0:
        return {"ok": False, "msg": "No candles yet"}

    tf = str(cfg.get("TF", "M30")).upper().strip()
    p = _tf_preset(tf)

    # Cooldown (optional enforcement)
    cooldown_until = _coerce_ts(cfg.get("COOLDOWN_UNTIL"))
    if cooldown_until is not None:
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
                "model_prob": None,
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

    needed_cols = ["atr", "ema_fast", "ema_slow", "kama", "bb_width", "psar", "bb_mid", "bb_up", "bb_lo"]
    if any((not np.isfinite(last.get(c, np.nan))) for c in needed_cols):
        return {"ok": False, "msg": "Indicators not ready yet"}

    # Regime filters
    vol_ok = bool(last["atr_pct"] > 0.00015)
    bb_ok = bool(last["bb_width"] >= p["BB_WIDTH_MIN"])
    std_ok = bool(np.isfinite(last["sd_z"]) and last["sd_z"] >= p["STD_Z_MIN"])
    adx_ok = bool(np.isfinite(last["adx"]) and last["adx"] >= p["ADX_MIN"])
    regime_ok = bool(vol_ok and (bb_ok or std_ok))

    # Trend direction (LTF)
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
    break_fr_low = bool(np.isfinite(last["fr_low"]) and last["close"] < last["fr_low"])

    not_buy_extreme = bool(last["close"] < last["bb_up"])
    not_sell_extreme = bool(last["close"] > last["bb_lo"])

    bb_mid = float(last["bb_mid"])
    bb_width = float(last["bb_width"])
    mid_buffer = float(p["BB_MID_BUFFER"]) * bb_width
    away_from_mid_buy = bool(last["close"] > (bb_mid + mid_buffer))
    away_from_mid_sell = bool(last["close"] < (bb_mid - mid_buffer))

    buy_conditions = {
        "Regime ok (vol + BB/Std)": regime_ok,
        "Away from BB mid (anti-chop)": away_from_mid_buy,
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
        "Away from BB mid (anti-chop)": away_from_mid_sell,
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
    # HTF Direction Gate
    # ------------------------
    htf_dir = "NA"
    gate_ok = True
    gate_msg = ""

    htf_candles = cfg.get("HTF_CANDLES", None)
    htf_tf_label = str(cfg.get("HTF_TF", p.get("HTF_GATE_TF", "HTF")))

    if isinstance(htf_candles, pd.DataFrame) and len(htf_candles) > 0:
        htf_p = _tf_preset("M30") if htf_tf_label in ("M30", "H1", "H4", "D") else p
        htf_min_needed = _min_needed_from_preset(htf_p)

        if len(htf_candles) >= htf_min_needed:
            hdf = _compute_indicators(htf_candles, htf_p)
            hlast = hdf.iloc[-1]
            htf_dir = _trend_direction(hlast)

            gate_ok, gate_msg = _apply_htf_gate(side, htf_dir)
            if not gate_ok:
                side = "NO_TRADE"
                conds = {"HTF gate": gate_msg, "HTF_dir": htf_dir, "HTF_TF": htf_tf_label, **conds}
                reasons = []
        else:
            htf_dir = "NOT_READY"

    # ------------------------
    # ✅ ML Direction Gate (fixed)
    # ------------------------
    artifacts_dir = str(cfg.get("ML_ARTIFACTS_DIR", "artifacts"))
    use_ml_gate = bool(cfg.get("USE_ML_GATE", True))

    model_prob = None
    ml_gate_ok = True
    ml_gate_msg = ""
    ml_buy_th = None
    ml_sell_th = None
    ml_h = None

    if use_ml_gate and side in ("BUY", "SELL"):
        try:
            store = _get_model_store(artifacts_dir=artifacts_dir)
            pred = store.predict_latest(tf=tf, candles=candles)

            if pred.get("ok"):
                model_prob = _as_float(pred.get("p_up"))
                ml_buy_th = _as_float(pred.get("buy_th"))
                ml_sell_th = _as_float(pred.get("sell_th"))
                try:
                    ml_h = int(pred.get("horizon_bars")) if pred.get("horizon_bars") is not None else None
                except Exception:
                    ml_h = None

                ml_gate_ok, ml_gate_msg, bt_used, st_used = _ml_gate_policy(
                    side=side,
                    p_up=model_prob,
                    buy_th=ml_buy_th,
                    sell_th=ml_sell_th,
                    cfg=cfg
                )

                # record the USED thresholds (may be clamped/soft)
                ml_buy_th = bt_used
                ml_sell_th = st_used

                # Only force NO_TRADE if strict policy actually blocks
                if (not ml_gate_ok) and bool(cfg.get("ML_STRICT", False)):
                    side = "NO_TRADE"

            else:
                # ✅ IMPORTANT: do NOT block if ML fails (fallback to rules)
                ml_gate_ok = True
                ml_gate_msg = f"ML predict failed (fallback to rules): {pred.get('msg','unknown')}"

        except Exception as e:
            # ✅ IMPORTANT: do NOT block on exception (fallback to rules)
            ml_gate_ok = True
            ml_gate_msg = f"ML exception (fallback to rules): {type(e).__name__}: {e}"

        # Always include ML telemetry in conditions for debugging
        conds = {"ML": ml_gate_msg, **conds}

    # Trade plan
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
            "breakeven_at_r": 1.0,
            "trail_after_r": 1.5,
        }

    reason_str = ""
    if side != "NO_TRADE" and reasons:
        reason_str = f"{side} because " + " + ".join(reasons[:6]) + (" ..." if len(reasons) > 6 else "")
    elif side == "NO_TRADE" and gate_msg:
        reason_str = f"NO_TRADE because {gate_msg}"

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

        # ✅ p(up) for your trained horizon
        "model_prob": (float(model_prob) if model_prob is not None else None),

        "meta": {
            "tf": tf,
            "ltf_dir": ltf_dir,
            "htf_tf": htf_tf_label,
            "htf_dir": htf_dir,
            "htf_gate_ok": bool(gate_ok),

            # ✅ ML telemetry
            "use_ml_gate": bool(use_ml_gate),
            "ml_gate_ok": bool(ml_gate_ok),
            "ml_horizon_bars": (int(ml_h) if ml_h is not None else None),
            "ml_buy_th": (float(ml_buy_th) if ml_buy_th is not None else None),
            "ml_sell_th": (float(ml_sell_th) if ml_sell_th is not None else None),

            # Policy knobs (visible in UI debugging)
            "ml_strict": bool(cfg.get("ML_STRICT", False)),
            "ml_edge": float(cfg.get("ML_EDGE", 0.02)),
            "ml_min_th": float(cfg.get("ML_MIN_TH", 0.52)),
            "ml_max_th": float(cfg.get("ML_MAX_TH", 0.65)),

            "bb_mid_buffer_frac": float(p["BB_MID_BUFFER"]),
            "cooldown_bars_recommendation": int(p["COOLDOWN_BARS"]),
        },
    }
