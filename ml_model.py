# ml_model.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from joblib import load


# -----------------------------
# Feature helpers (must match training meta["feature_cols"])
# -----------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line = _ema(macd_line, sig)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def _rolling_z(s: pd.Series, n: int) -> pd.Series:
    mu = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - mu) / (sd + 1e-12)


def _tod_sin_cos(idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    t = pd.to_datetime(idx)
    minutes = (t.hour * 60 + t.minute).astype(float)
    ang = 2.0 * np.pi * (minutes / 1440.0)
    return pd.Series(np.sin(ang), index=idx), pd.Series(np.cos(ang), index=idx)


def build_ml_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the SAME feature set used in your training meta files.
    Assumes candles has: open, high, low, close, spread (optional), volume (optional)
    """
    df = candles.copy()

    # Make sure optional columns exist
    if "spread" not in df.columns:
        df["spread"] = 0.0
    if "volume" not in df.columns:
        df["volume"] = 0.0

    close = df["close"].astype(float)

    # Returns
    df["ret1"] = close.pct_change(1)
    df["ret4"] = close.pct_change(4)
    df["ret8"] = close.pct_change(8)
    df["ret16"] = close.pct_change(16)

    # EMAs
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)

    df["ema_diff2050"] = (ema20 - ema50) / (close.abs() + 1e-12)
    df["ema_diff50100"] = (ema50 - ema100) / (close.abs() + 1e-12)

    df["ema20_slope"] = ema20.diff()
    df["ema50_slope"] = ema50.diff()

    # RSI
    rsi14 = _rsi(close, 14)
    df["rsi14"] = rsi14
    df["rsi_slope"] = rsi14.diff()

    # MACD
    macd_line, sig_line, hist = _macd(close, 12, 26, 9)
    df["macd"] = macd_line
    df["macd_sig"] = sig_line
    df["macd_hist"] = hist
    df["macd_cross_up"] = ((macd_line.shift(1) <= sig_line.shift(1)) & (macd_line > sig_line)).astype(int)
    df["macd_cross_dn"] = ((macd_line.shift(1) >= sig_line.shift(1)) & (macd_line < sig_line)).astype(int)

    # ATR
    atr14 = _atr(df, 14)
    df["atr_pct"] = atr14 / (close.abs() + 1e-12)
    df["atr_z"] = _rolling_z(df["atr_pct"], 100)

    # Candle shape
    df["hl_range"] = (df["high"].astype(float) - df["low"].astype(float)) / (close.abs() + 1e-12)
    df["oc_range"] = (df["close"].astype(float) - df["open"].astype(float)) / (close.abs() + 1e-12)
    body = (df["close"].astype(float) - df["open"].astype(float)).abs()
    rng = (df["high"].astype(float) - df["low"].astype(float)).abs() + 1e-12
    df["body_to_range"] = body / rng

    # Bollinger width proxy (using rolling std)
    bb_sd = close.rolling(20).std()
    df["bb_width"] = (2.0 * bb_sd) / (close.abs() + 1e-12)
    df["bb_z"] = _rolling_z(bb_sd, 100)
    df["bb_width_z"] = _rolling_z(df["bb_width"], 100)

    # Spread + volume features
    df["spread_pct"] = df["spread"].astype(float) / (close.abs() + 1e-12)
    df["spread_z"] = _rolling_z(df["spread_pct"], 100)

    vol = df["volume"].astype(float)
    df["vol_z"] = _rolling_z(vol, 100)
    df["vol_chg"] = vol.pct_change(1)

    # Time-of-day
    if isinstance(df.index, pd.DatetimeIndex):
        df["tod_sin"], df["tod_cos"] = _tod_sin_cos(df.index)
    else:
        df["tod_sin"] = 0.0
        df["tod_cos"] = 0.0

    # Simple regime flags
    df["trend_flag"] = (ema20 > ema50).astype(int) - (ema20 < ema50).astype(int)  # +1 up, -1 down
    df["vol_flag"] = (df["atr_pct"] > df["atr_pct"].rolling(200).median()).astype(int)

    return df


# -----------------------------
# Model wrapper
# -----------------------------
@dataclass
class DirectionModelBundle:
    tf: str
    horizon_bars: int
    feature_cols: list
    buy_th: float
    sell_th: float
    model: Any
    meta: Dict[str, Any]


class DirectionModelStore:
    """
    Loads your trained artifacts and provides:
      - p_up for the latest candle
      - gating decision (pass/fail) given a trade side
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self._cache: Dict[str, DirectionModelBundle] = {}

    def _paths_for_tf(self, tf: str) -> Tuple[str, str]:
        tf = tf.upper().strip()
        if tf == "M15":
            job = os.path.join(self.artifacts_dir, "direction_model_M15_H4.joblib")
            jsn = os.path.join(self.artifacts_dir, "direction_model_M15_H4.json")
            return job, jsn
        if tf == "M30":
            job = os.path.join(self.artifacts_dir, "direction_model_M30_H2.joblib")
            jsn = os.path.join(self.artifacts_dir, "direction_model_M30_H2.json")
            return job, jsn
        raise ValueError(f"Unsupported TF for Option A: {tf}")

    def load_for_tf(self, tf: str) -> DirectionModelBundle:
        tf = tf.upper().strip()
        if tf in self._cache:
            return self._cache[tf]

        model_path, meta_path = self._paths_for_tf(tf)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        th = meta.get("thresholds_best", {}) or {}
        buy_th = float(th.get("buy", 0.60))
        sell_th = float(th.get("sell", 0.40))

        bundle = DirectionModelBundle(
            tf=str(meta.get("tf", tf)),
            horizon_bars=int(meta.get("horizon_bars", 0)),
            feature_cols=list(meta.get("feature_cols", [])),
            buy_th=buy_th,
            sell_th=sell_th,
            model=load(model_path),
            meta=meta,
        )
        self._cache[tf] = bundle
        return bundle

    def predict_latest(self, tf: str, candles: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns:
          { ok, p_up, buy_th, sell_th, horizon_bars, used_features, msg }
        """
        bundle = self.load_for_tf(tf)

        if candles is None or len(candles) < 120:
            return {"ok": False, "msg": "Not enough candles for ML features", "p_up": None}

        feats = build_ml_features(candles)
        if len(feats) == 0:
            return {"ok": False, "msg": "ML features not ready", "p_up": None}

        last = feats.iloc[-1]
        cols = bundle.feature_cols
        missing = [c for c in cols if c not in feats.columns]
        if missing:
            return {"ok": False, "msg": f"Missing ML feature cols: {missing[:5]}...", "p_up": None}

        x = last[cols].astype(float).values.reshape(1, -1)

        if not np.all(np.isfinite(x)):
            return {"ok": False, "msg": "Non-finite ML features at latest candle", "p_up": None}

        p_up = float(bundle.model.predict_proba(x)[0, 1])

        return {
            "ok": True,
            "p_up": p_up,
            "buy_th": float(bundle.buy_th),
            "sell_th": float(bundle.sell_th),
            "horizon_bars": int(bundle.horizon_bars),
            "used_features": cols,
        }

    @staticmethod
    def gate(side: str, p_up: Optional[float], buy_th: float, sell_th: float) -> Tuple[bool, str]:
        if side not in ("BUY", "SELL"):
            return True, ""

        if p_up is None or not np.isfinite(p_up):
            return False, "Blocked by ML (no probability)"

        if side == "BUY":
            return (p_up >= buy_th), f"Blocked by ML (BUY needs p_up>={buy_th:.2f}, got {p_up:.3f})"
        else:
            return (p_up <= sell_th), f"Blocked by ML (SELL needs p_up<={sell_th:.2f}, got {p_up:.3f})"


# ============================================================
# ✅ REQUIRED BY app_web.py
# ============================================================
def load_models_from_dir(artifacts_dir: str) -> Dict[str, Any]:
    """
    app_web.py expects this function.

    Returns a dict that contains:
      - "store": DirectionModelStore (loaded/cached)
      - "loaded_tfs": list of TFs that successfully loaded
      - "artifacts_dir": the directory used
    """
    store = DirectionModelStore(artifacts_dir=artifacts_dir)

    loaded: List[str] = []
    for tf in ("M15", "M30"):
        try:
            store.load_for_tf(tf)
            loaded.append(tf)
        except Exception:
            # It's OK if one TF isn't present
            pass

    return {
        "store": store,
        "loaded_tfs": loaded,
        "artifacts_dir": artifacts_dir,
    }


def ml_predict_latest(ml_models: Dict[str, Any], tf: str, candles: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience helper for strategy_rules if you want it.
    """
    store = ml_models.get("store") if isinstance(ml_models, dict) else None
    if store is None:
        return {"ok": False, "msg": "ML store not loaded", "p_up": None}
    return store.predict_latest(tf, candles)
