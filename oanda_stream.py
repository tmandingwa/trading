# oanda_stream.py
from __future__ import annotations
import json
import pandas as pd
import httpx
from typing import AsyncIterator, Dict, Any

from config import STREAM_URL, OANDA_TOKEN, OANDA_ACCOUNT_ID

class OandaStreamError(Exception):
    pass

async def price_stream(instruments: str) -> AsyncIterator[Dict[str, Any]]:
    """
    instruments: "EUR_USD,GBP_USD,XAU_USD"
    Yields: {time, instrument, bid, ask, mid}
    """
    if not OANDA_TOKEN or not OANDA_ACCOUNT_ID:
        raise OandaStreamError("Missing OANDA_TOKEN or OANDA_ACCOUNT_ID (set env vars).")

    url = f"{STREAM_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/pricing/stream"
    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    params = {"instruments": instruments}

    # no read timeout; OANDA stream is long-lived
    timeout = httpx.Timeout(None, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url, headers=headers, params=params) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                snippet = raw[:400]
                raise OandaStreamError(f"Stream HTTP {resp.status_code}: {snippet!r}")

            async for line in resp.aiter_lines():
                if not line:
                    continue

                # OANDA sometimes sends HEARTBEAT / other types; ignore safely
                try:
                    msg = json.loads(line)
                except Exception:
                    continue

                if msg.get("type") != "PRICE":
                    continue

                bids = msg.get("bids", [])
                asks = msg.get("asks", [])
                if not bids or not asks:
                    continue

                try:
                    bid = float(bids[0]["price"])
                    ask = float(asks[0]["price"])
                except Exception:
                    continue

                mid = 0.5 * (bid + ask)

                yield {
                    "time": pd.to_datetime(msg.get("time"), utc=True, errors="coerce"),
                    "instrument": msg.get("instrument"),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                }
