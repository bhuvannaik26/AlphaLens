"""
AlphaLens — Signal Detection Agent v3
=======================================
Detects: Breakout · RSI Oversold/Overbought · EMA Cross · Volume Spike · Trend
"""

import numpy as np
from agents.data_provider import get_stock_data

# ── Stock universe ─────────────────────────────────────────────────────────────
NSE_STOCKS = {
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "WIPRO":      "WIPRO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "SBIN":       "SBIN.NS",
    "ADANIENT":   "ADANIENT.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ICICIBANK":  "ICICIBANK.NS",
}


# ── Technical indicators ───────────────────────────────────────────────────────

def compute_rsi(prices: list, period: int = 14) -> float:
    """
    Wilder's RSI — standard 14-period implementation.
    Returns 50.0 if not enough data.
    """
    if len(prices) < period + 1:
        return 50.0

    deltas   = np.diff(prices)
    gains    = np.where(deltas > 0,  deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_ema(prices: list, period: int) -> list:
    """
    Exponential Moving Average.
    Returns list of same length as prices.
    """
    if not prices:
        return []
    k   = 2 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append((p - ema[-1]) * k + ema[-1])
    return ema


def compute_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    Average True Range — measures volatility.
    Useful for confidence scoring.
    """
    if len(highs) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i]  - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )
        trs.append(tr)
    return round(float(np.mean(trs[-period:])), 2)


# ── Main detection ─────────────────────────────────────────────────────────────

def detect(ticker_name: str) -> dict:
    """
    Runs full signal detection for one NSE ticker.
    Always returns a valid dict — never raises.

    Returns:
        ticker, symbol, current_price, price_change_5d,
        rsi, ema20, ema50, vol_ratio, atr,
        primary_signal, signals, confidence,
        closes, volumes, dates, data_source
    """
    symbol = ticker_name.upper() + ".NS"

    try:
        # get_stock_data() always returns a valid DataFrame (never a dict)
        hist = get_stock_data(ticker_name)

        if hist.empty or len(hist) < 20:
            return _neutral(ticker_name, symbol, "Insufficient data")

        # ── Extract arrays ─────────────────────────────────────────────────
        closes  = [float(x) for x in hist["Close"].tolist()]
        volumes = [float(x) for x in hist["Volume"].tolist()]
        highs   = [float(x) for x in hist["High"].tolist()]
        lows    = [float(x) for x in hist["Low"].tolist()]
        dates   = [str(d.date()) for d in hist.index]

        signals    = []
        confidence = 50

        # ── 1. RSI ────────────────────────────────────────────────────────
        rsi = compute_rsi(closes)

        if rsi < 35:
            signals.append({
                "type":   "RSI_OVERSOLD",
                "detail": f"RSI={rsi} — heavily oversold, mean-reversion likely",
            })
            confidence += 15
        elif rsi > 68:
            signals.append({
                "type":   "RSI_OVERBOUGHT",
                "detail": f"RSI={rsi} — overbought, watch for pullback",
            })
            confidence -= 5

        # ── 2. EMA Cross ──────────────────────────────────────────────────
        ema20 = compute_ema(closes, 20)
        ema50 = compute_ema(closes, 50) if len(closes) >= 50 else compute_ema(closes, 20)

        if len(ema20) >= 2 and len(ema50) >= 2:
            golden = ema20[-1] > ema50[-1] and ema20[-2] <= ema50[-2]
            death  = ema20[-1] < ema50[-1] and ema20[-2] >= ema50[-2]

            if golden:
                signals.append({
                    "type":   "EMA_BULLISH_CROSS",
                    "detail": f"EMA20 (₹{round(ema20[-1],2)}) crossed above EMA50 (₹{round(ema50[-1],2)}) — golden cross",
                })
                confidence += 20
            elif death:
                signals.append({
                    "type":   "EMA_BEARISH_CROSS",
                    "detail": f"EMA20 (₹{round(ema20[-1],2)}) crossed below EMA50 (₹{round(ema50[-1],2)}) — death cross",
                })
                confidence -= 10

        # ── 3. Volume Spike ───────────────────────────────────────────────
        recent_vols = volumes[-20:] if len(volumes) >= 20 else volumes
        avg_vol     = float(np.mean(recent_vols)) if recent_vols else 1.0
        last_vol    = volumes[-1] if volumes else 0.0
        vol_ratio   = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0

        if vol_ratio > 1.8:
            signals.append({
                "type":   "VOLUME_SPIKE",
                "detail": f"{vol_ratio:.1f}x average volume — strong institutional participation",
            })
            confidence += 15
        elif vol_ratio > 1.3:
            signals.append({
                "type":   "VOLUME_ELEVATED",
                "detail": f"{vol_ratio:.1f}x average volume — elevated interest",
            })
            confidence += 5

        # ── 4. Breakout ───────────────────────────────────────────────────
        lookback     = highs[-20:] if len(highs) >= 20 else highs
        recent_high  = max(lookback)
        recent_low   = min(lows[-20:]) if len(lows) >= 20 else min(lows)

        if closes[-1] >= recent_high * 0.99:
            signals.append({
                "type":   "BREAKOUT",
                "detail": f"Price ₹{round(closes[-1],2)} at/near 20-day high ₹{round(recent_high,2)}",
            })
            confidence += 18

        elif closes[-1] <= recent_low * 1.01:
            signals.append({
                "type":   "BREAKDOWN",
                "detail": f"Price ₹{round(closes[-1],2)} near 20-day low ₹{round(recent_low,2)}",
            })
            confidence -= 12

        # ── 5. 5-Day Trend ────────────────────────────────────────────────
        pct_5d = 0.0
        if len(closes) >= 6:
            pct_5d = round(((closes[-1] - closes[-6]) / closes[-6]) * 100, 2)

        if pct_5d > 3:
            signals.append({
                "type":   "UPTREND",
                "detail": f"+{pct_5d}% momentum over 5 sessions — buyers in control",
            })
            confidence += 10
        elif pct_5d < -3:
            signals.append({
                "type":   "DOWNTREND",
                "detail": f"{pct_5d}% decline over 5 sessions — sellers dominant",
            })
            confidence -= 10

        # ── ATR for volatility context ─────────────────────────────────────
        atr = compute_atr(highs, lows, closes)

        # ── Clamp confidence ──────────────────────────────────────────────
        confidence = int(max(20, min(95, confidence)))

        # ── Data source tag ───────────────────────────────────────────────
        is_sim = getattr(hist, "_is_simulated", False)

        return {
            "ticker":          ticker_name,
            "symbol":          symbol,
            "current_price":   round(closes[-1], 2),
            "price_change_5d": pct_5d,
            "rsi":             rsi,
            "ema20":           round(ema20[-1], 2),
            "ema50":           round(ema50[-1], 2),
            "vol_ratio":       vol_ratio,
            "atr":             atr,
            "primary_signal":  signals[0]["type"] if signals else "NEUTRAL",
            "signals":         signals,
            "confidence":      confidence,
            # Send last 60 candles to frontend (enough for chart)
            "closes":          [round(c, 2) for c in closes[-60:]],
            "volumes":         [int(v)      for v in volumes[-60:]],
            "highs":           [round(h, 2) for h in highs[-60:]],
            "lows":            [round(l, 2) for l in lows[-60:]],
            "dates":           dates[-60:],
            "data_source":     "simulated" if is_sim else "live",
        }

    except Exception as e:
        print(f"[detect] ERROR for {ticker_name}: {e}")
        return _neutral(ticker_name, symbol, str(e))


def _neutral(ticker_name: str, symbol: str, reason: str = "") -> dict:
    """
    Safe neutral fallback — used when detection pipeline fails.
    Returns enough fields to keep the UI rendering correctly.
    """
    return {
        "ticker":          ticker_name,
        "symbol":          symbol,
        "current_price":   0.0,
        "price_change_5d": 0.0,
        "rsi":             50.0,
        "ema20":           0.0,
        "ema50":           0.0,
        "vol_ratio":       1.0,
        "atr":             0.0,
        "primary_signal":  "NEUTRAL",
        "signals":         [],
        "confidence":      50,
        "closes":          [],
        "volumes":         [],
        "highs":           [],
        "lows":            [],
        "dates":           [],
        "data_source":     "error",
        "error":           reason,
    }
