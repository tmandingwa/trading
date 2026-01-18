# oanda_history.py
import httpx
import pandas as pd
from config import REST_URL, OANDA_TOKEN

TF_TO_OANDA = {
    "M1": "M1",
    "M5": "M5",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "H6": "H6",
}

async def fetch_candles(instrument: str, tf: str, count: int = 400) -> pd.DataFrame:
    """
    Fetch MID candles from OANDA REST and return CLOSED candles:
    index UTC: candle start
    cols: open,high,low,close
    """
    if not OANDA_TOKEN:
        raise RuntimeError("Missing OANDA_TOKEN env var")

    url = f"{REST_URL}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {
        "granularity": TF_TO_OANDA[tf],
        "count": int(count),
        "price": "M",  # mid candles
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

    rows = []
    for c in data.get("candles", []):
        if not c.get("complete", False):
            continue
        t = pd.to_datetime(c["time"], utc=True)
        m = c["mid"]
        rows.append({
            "time": t,
            "open": float(m["o"]),
            "high": float(m["h"]),
            "low": float(m["l"]),
            "close": float(m["c"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = df.set_index("time").sort_index()
    return df
