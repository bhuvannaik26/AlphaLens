"""
AlphaLens — Data Provider v3 (FIXED)
=====================================
Priority chain — stops at first success:
  1. Finnhub       — 60 req/min free, covers NSE India  ✅ BEST FOR DEMO
  2. Alpha Vantage — 25 req/day free, covers NSE India  ✅ BACKUP
  3. Simulation    — realistic OHLCV, always works      ✅ NEVER CRASHES UI

Setup:
  In your .env file add:
    FINNHUB_TOKEN=your_token_here        # free at finnhub.io
    ALPHA_VANTAGE_KEY=your_key_here      # free at alphavantage.co

How to get keys (both free, no credit card):
  Finnhub      → https://finnhub.io/register
  Alpha Vantage → https://www.alphavantage.co/support/#api-key

Install: pip install requests pandas numpy python-dotenv
(No yfinance needed!)
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── API keys from .env ────────────────────────────────────────────────────────
FINNHUB_TOKEN      = os.getenv("FINNHUB_TOKEN", "")
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── In-memory cache (5 min TTL) ───────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL    = 300   # seconds


def _cache_valid(entry: dict) -> bool:
    return (datetime.now() - entry["ts"]).seconds < CACHE_TTL


# ── Realistic simulation (FINAL fallback — zero network, always works) ────────

SEED_PRICES = {
    "RELIANCE":   2850, "TCS":        3920, "INFY":       1580,
    "HDFCBANK":   1720, "WIPRO":       480, "TATAMOTORS":  960,
    "SBIN":        780, "ADANIENT":   2640, "BAJFINANCE": 7100,
    "ICICIBANK":  1240, "HINDUNILVR": 2560, "MARUTI":     9800,
}


def _simulate(ticker: str) -> pd.DataFrame:
    """
    Generate 90 days of realistic OHLCV data.
    Seeded by ticker name so same ticker always gives same data (demo-stable).
    """
    base  = SEED_PRICES.get(ticker.upper(), 1000 + (abs(hash(ticker)) % 3000))
    seed  = abs(hash(ticker)) % 99999
    np.random.seed(seed)

    rows   = []
    price  = float(base)
    today  = datetime.now()

    for i in range(90, 0, -1):
        d = today - timedelta(days=i)
        if d.weekday() >= 5:          # skip Saturday / Sunday
            continue
        # realistic daily change: small drift + occasional jump
        chg   = np.random.normal(0.0004, 0.014)
        o     = round(price, 2)
        c     = round(price * (1 + chg), 2)
        h     = round(max(o, c) * (1 + abs(np.random.normal(0, 0.005))), 2)
        l     = round(min(o, c) * (1 - abs(np.random.normal(0, 0.005))), 2)
        v     = int(np.random.randint(500_000, 15_000_000))
        rows.append({"Date": d, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
        price = c

    df = pd.DataFrame(rows).set_index("Date")
    df.index = pd.DatetimeIndex(df.index)
    df._is_simulated = True
    return df


# ── Source 1: Finnhub ─────────────────────────────────────────────────────────
# Docs: https://finnhub.io/docs/api/stock-candles
# NSE symbol format: NSE:RELIANCE  (Finnhub uses exchange prefix)
# Free tier: 60 req/min — perfect for hackathon demos

def _fetch_finnhub(ticker: str) -> pd.DataFrame | None:
    if not FINNHUB_TOKEN:
        return None

    symbol = f"NSE:{ticker.upper()}"
    end    = int(datetime.now().timestamp())
    start  = int((datetime.now() - timedelta(days=100)).timestamp())

    url = (
        f"https://finnhub.io/api/v1/stock/candle"
        f"?symbol={symbol}&resolution=D&from={start}&to={end}"
        f"&token={FINNHUB_TOKEN}"
    )

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()

        # Finnhub returns {"s": "no_data"} when symbol not found
        if data.get("s") != "ok":
            print(f"[Finnhub] {symbol}: status={data.get('s')} — trying Alpha Vantage")
            return None

        df = pd.DataFrame({
            "Open":   data["o"],
            "High":   data["h"],
            "Low":    data["l"],
            "Close":  data["c"],
            "Volume": data["v"],
        }, index=pd.to_datetime(data["t"], unit="s"))

        df.index.name = "Date"
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        if len(df) < 20:
            print(f"[Finnhub] {symbol}: only {len(df)} rows — insufficient")
            return None

        print(f"[Finnhub] {symbol} ✓  {len(df)} rows")
        return df

    except requests.exceptions.Timeout:
        print(f"[Finnhub] {ticker}: timeout")
        return None
    except Exception as e:
        print(f"[Finnhub] {ticker}: {e}")
        return None


# ── Source 2: Alpha Vantage ───────────────────────────────────────────────────
# Docs: https://www.alphavantage.co/documentation/#time-series-daily
# NSE symbol format: RELIANCE.BSE  (Alpha Vantage uses .BSE suffix for Indian)
# Free tier: 25 req/day — use sparingly, only when Finnhub fails

def _fetch_alpha_vantage(ticker: str) -> pd.DataFrame | None:
    if not ALPHA_VANTAGE_KEY:
        return None

    # Alpha Vantage uses .BSE for Indian stocks on free tier
    symbol = f"{ticker.upper()}.BSE"
    url    = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}"
        f"&outputsize=compact"        # last 100 days — saves quota
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    try:
        resp = requests.get(url, timeout=12)
        data = resp.json()

        # Rate limit hit
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information", "")
            print(f"[AlphaVantage] Rate limit: {msg[:80]}")
            return None

        ts = data.get("Time Series (Daily)")
        if not ts:
            print(f"[AlphaVantage] {symbol}: no time series in response")
            return None

        rows = []
        for date_str, vals in sorted(ts.items()):
            rows.append({
                "Date":   pd.to_datetime(date_str),
                "Open":   float(vals["1. open"]),
                "High":   float(vals["2. high"]),
                "Low":    float(vals["3. low"]),
                "Close":  float(vals["4. close"]),
                "Volume": int(vals["5. volume"]),
            })

        df = pd.DataFrame(rows).set_index("Date").sort_index()

        # Keep last 90 trading days
        cutoff = datetime.now() - timedelta(days=100)
        df     = df[df.index >= pd.Timestamp(cutoff)]

        if len(df) < 20:
            print(f"[AlphaVantage] {symbol}: only {len(df)} rows")
            return None

        print(f"[AlphaVantage] {symbol} ✓  {len(df)} rows")
        return df

    except requests.exceptions.Timeout:
        print(f"[AlphaVantage] {ticker}: timeout")
        return None
    except Exception as e:
        print(f"[AlphaVantage] {ticker}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex.  NEVER raises — always returns valid data.

    Check df._is_simulated (True/False) to know if it's live or demo data.
    """
    key = ticker.upper()

    # 1. Cache hit
    if key in _cache and _cache_valid(_cache[key]):
        print(f"[cache] {key} ✓")
        return _cache[key]["data"]

    # 2. Finnhub (primary — 60 req/min free)
    df = _fetch_finnhub(key)
    if df is not None:
        _cache[key] = {"data": df, "ts": datetime.now()}
        return df

    time.sleep(0.3)   # small pause between sources

    # 3. Alpha Vantage (backup — 25 req/day free)
    df = _fetch_alpha_vantage(key)
    if df is not None:
        _cache[key] = {"data": df, "ts": datetime.now()}
        return df

    # 4. Simulation (always works — never breaks the demo)
    print(f"[simulate] {key} — all live sources failed or unconfigured")
    df = _simulate(key)
    _cache[key] = {"data": df, "ts": datetime.now()}
    return df


def get_current_price(ticker: str) -> float:
    """Returns latest closing price."""
    df = get_stock_data(ticker)
    return round(float(df["Close"].iloc[-1]), 2)


def get_data_source(ticker: str) -> str:
    """Returns 'finnhub', 'alphavantage', or 'simulated' for UI display."""
    key = ticker.upper()
    if key in _cache:
        df = _cache[key]["data"]
        if getattr(df, "_is_simulated", False):
            return "simulated"
        return "live"
    return "unknown"
