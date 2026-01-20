# config.py
import os
from dataclasses import dataclass

# ============================================================
# OANDA (Used by oanda_stream.py and oanda_history.py)
# ============================================================
OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")

# Accept: practice/live (any casing)
OANDA_ENV = os.getenv("OANDA_ENV", "practice").strip().lower()  # practice | live

# Base URLs
if OANDA_ENV == "live":
    REST_URL = "https://api-fxtrade.oanda.com"
    STREAM_URL = "https://stream-fxtrade.oanda.com"
else:
    REST_URL = "https://api-fxpractice.oanda.com"
    STREAM_URL = "https://stream-fxpractice.oanda.com"

# Backwards-compatible aliases (some files may import these)
OANDA_API_URL = REST_URL
OANDA_STREAM_URL = STREAM_URL

# Optional: allow UI to start without creds
REQUIRE_OANDA_CREDS = os.getenv("REQUIRE_OANDA_CREDS", "1") == "1"
if REQUIRE_OANDA_CREDS:
    if not OANDA_TOKEN:
        raise RuntimeError("Missing env var OANDA_TOKEN")
    if not OANDA_ACCOUNT_ID:
        raise RuntimeError("Missing env var OANDA_ACCOUNT_ID")

# ============================================================
# DASHBOARD + ENGINE SCOPE (your requested scope)
# ============================================================
SUPPORTED_INSTRUMENTS = ["EUR_USD"]
SUPPORTED_TFS = ["M15", "M30"]

DEFAULT_INSTRUMENT = "EUR_USD"
DEFAULT_TF = "M15"

# ============================================================
# Trading rule parameters
# ============================================================
EMA_FAST = 20
EMA_SLOW = 50

RSI_LEN = 14
RSI_BUY_MAX = 35
RSI_SELL_MIN = 65

ATR_LEN = 14
SL_ATR = 1.2
TP_RR = 1.5

USE_ENGULFING = True
USE_SR = True
SWING_LOOKBACK = 20
SR_TOL_ATR = 0.35
MIN_CONDITIONS_TO_TRADE = 2

SEED_CANDLES = 600

# ============================================================
# Server / WS
# ============================================================
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
WS_ENABLED = os.getenv("WS_ENABLED", "1") == "1"
WS_PING_SEC = int(os.getenv("WS_PING_SEC", "25"))

# ============================================================
# Structured config object
# ============================================================
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
    "OANDA_ENV": OANDA_ENV,
    "REST_URL": REST_URL,
    "STREAM_URL": STREAM_URL,
    "OANDA_API_URL": OANDA_API_URL,
    "OANDA_STREAM_URL": OANDA_STREAM_URL,

    "SUPPORTED_INSTRUMENTS": SUPPORTED_INSTRUMENTS,
    "SUPPORTED_TFS": SUPPORTED_TFS,
    "DEFAULT_INSTRUMENT": DEFAULT_INSTRUMENT,
    "DEFAULT_TF": DEFAULT_TF,

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

    "SEED_CANDLES": SEED_CANDLES,

    "SERVER_HOST": SERVER_HOST,
    "SERVER_PORT": SERVER_PORT,
    "WS_ENABLED": WS_ENABLED,
    "WS_PING_SEC": WS_PING_SEC,
}
