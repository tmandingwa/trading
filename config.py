# config.py
import os

OANDA_ENV = os.getenv("OANDA_ENV", "PRACTICE").upper()

if OANDA_ENV == "LIVE":
    REST_URL = "https://api-fxtrade.oanda.com"
    STREAM_URL = "https://stream-fxtrade.oanda.com"
else:
    REST_URL = "https://api-fxpractice.oanda.com"
    STREAM_URL = "https://stream-fxpractice.oanda.com"

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "").strip()
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()

# Dashboard defaults
DEFAULT_INSTRUMENT = "EUR_USD"
DEFAULT_TF = "M1"  # M1, M5, M30, H1, H4, H6

# Strategy params
MIN_CONDITIONS_TO_TRADE = 2

EMA_FAST = 20
EMA_SLOW = 50

RSI_LEN = 14
RSI_BUY_MAX = 60
RSI_SELL_MIN = 40

ATR_LEN = 14
SL_ATR = 1.2
TP_RR = 1.8

USE_ENGULFING = True
USE_SR = True
SWING_LOOKBACK = 3
SR_TOL_ATR = 0.5

# History seed
SEED_CANDLES = 400

# ============================================================
# PAPER TRADING + RISK SIZING
# ============================================================
PAPER_ENABLED = True
START_BALANCE_USD = 10000.0

# Risk sizing: position units = (balance * risk_pct) / abs(entry - SL)
RISK_PCT = 0.005  # 0.5% risk per trade
MAX_UNITS = 200000  # safety cap on units

# One position per instrument+tf
ALLOW_MULTIPLE_POSITIONS_PER_STREAM = True

# How many trades to keep in memory (sent to UI)
MAX_TRADE_LOG = 200
