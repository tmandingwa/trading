# config.py
import os
from dataclasses import dataclass

# -------------------------
# OANDA
# -------------------------
OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")  # practice | live

# ============================================================
# DASHBOARD + ENGINE SCOPE
# ============================================================
# You requested the app to focus ONLY on:
#   - Instrument: EUR_USD
#   - Timeframes: M15, M30
#
# If you later want more pairs/TFs, extend these lists.
SUPPORTED_INSTRUMENTS = ["EUR_USD"]
SUPPORTED_TFS = ["M15", "M30"]

# Dashboard defaults
DEFAULT_INSTRUMENT = "EUR_USD"
DEFAULT_TF = "M15"  # M15, M30

# -------------------------
# Trading rule parameters
# -------------------------
EMA_FAST = 20
EMA_SLOW = 50

RSI_LEN = 14
RSI_BUY_MAX = 35     # buy when RSI <= this (oversold-ish)
RSI_SELL_MIN = 65    # sell when RSI >= this (overbought-ish)

ATR_LEN = 14
SL_ATR = 1.2
TP_RR = 1.5

USE_ENGULFING = True
USE_SR = True
SWING_LOOKBACK = 20
SR_TOL_ATR = 0.35
MIN_CONDITIONS_TO_TRADE = 2

# How many candles to seed for each TF on startup
SEED_CANDLES = 600

# -------------------------
# Server / WS
# -------------------------
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
WS_ENABLED = os.getenv("WS_ENABLED", "1") == "1"
WS_PING_SEC = int(os.getenv("WS_PING_SEC", "25"))

# -------------------------
# Structured config object
# -------------------------
@dataclass
class TAConfig:
    ema_fast: int = EMA_FAST
    ema_slow: int = EMA_SLOW
    rsi_len: int = RSI_LEN
    rsi_buy_max: float = RSI_BUY_MAX
    rsi_sell_min: float = RSI_SELL_MIN
    atr_len: int = ATR_LEN
    sl_atr: float = SL_ATR
    tp_rr: float = TP_RR
    use_engulfing: bool = USE_ENGULFING
    use_sr: bool = USE_SR
    swing_lookback: int = SWING_LOOKBACK
    sr_tol_atr: float = SR_TOL_ATR
    min_conditions_to_trade: int = MIN_CONDITIONS_TO_TRADE


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
