"""
AlphaLens — Reasoning Agent v7 (FINAL FIX)
============================================
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_MODEL    = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
MAX_RETRIES = 3
TIMEOUT_SEC = 40   # free tier is slow — never use < 25s


# ── Prompt Builder — /api/signals deep analysis ───────────────────────────────

def build_prompt(stock_name: str, signal_data: dict, question: str = None) -> str:
    sig     = signal_data
    price   = sig.get("current_price", 0)
    rsi     = sig.get("rsi", 50)
    ema20   = sig.get("ema20", 0)
    ema50   = sig.get("ema50", 0)
    pct_5d  = sig.get("price_change_5d", 0)
    vol_r   = sig.get("vol_ratio", 1.0)
    atr     = sig.get("atr", 0)
    primary = sig.get("primary_signal", "NEUTRAL")
    conf    = sig.get("confidence", 50)
    src     = sig.get("data_source", "simulated")

    signals_text = "\n".join(
        f"  • {s['type']}: {s['detail']}"
        for s in sig.get("signals", [])
    ) or "  • No strong technical signals detected"

    ema_align = "bullish (EMA20 above EMA50)" if ema20 > ema50 else "bearish (EMA20 below EMA50)"
    rsi_zone  = "oversold zone" if rsi < 35 else "overbought zone" if rsi > 68 else "neutral zone"

    q = question or (
        f"Analyze {stock_name}: Why is it moving this way? "
        "What is the 2-4 week trend? Should a retail investor buy, hold, or avoid?"
    )

    return (
        f"<s>[INST]\n"
        f"You are AlphaLens AI — expert stock analyst for Indian retail investors (NSE/BSE).\n"
        f"Use the exact numbers below. Be specific with ₹ values. No generic disclaimers.\n\n"
        f"━━ LIVE DATA ({src.upper()}) ━━━━━━━━━━━━━━━━━━━━\n"
        f"Stock        : {stock_name}\n"
        f"Price        : ₹{price}  |  5D: {'+' if pct_5d>=0 else ''}{pct_5d}%\n"
        f"RSI (14)     : {rsi} — {rsi_zone}\n"
        f"EMA          : {ema_align} (EMA20 ₹{ema20} / EMA50 ₹{ema50})\n"
        f"Volume Ratio : {vol_r}x average\n"
        f"ATR          : ₹{atr}\n"
        f"Signal       : {primary} ({conf}% confidence)\n\n"
        f"━━ DETECTED SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{signals_text}\n\n"
        f"QUESTION: {q}\n\n"
        f"Answer using actual ₹ numbers. Include: trend reason, outlook, "
        f"action (BUY/HOLD/WATCH/AVOID), stop-loss at ₹{round(price - 1.5*atr, 2)}. "
        f"Under 140 words.\n"
        f"[/INST]"
    )


# ── Chat Prompt Builder — /api/chat conversational Q&A ────────────────────────

def build_chat_prompt(question: str, signal_data: dict = None) -> str:
    if not signal_data or signal_data.get("current_price", 0) == 0:
        return (
            f"<s>[INST]\n"
            f"You are AlphaLens AI — smart stock market assistant for Indian retail investors (NSE/BSE).\n"
            f"Answer clearly. Use ₹ for prices. Be specific. Under 120 words.\n"
            f"User question: {question}\n"
            f"[/INST]"
        )

    sig     = signal_data
    price   = sig.get("current_price", 0)
    rsi     = sig.get("rsi", 50)
    ema20   = sig.get("ema20", 0)
    ema50   = sig.get("ema50", 0)
    pct_5d  = sig.get("price_change_5d", 0)
    vol_r   = sig.get("vol_ratio", 1.0)
    atr     = sig.get("atr", 0)
    ticker  = sig.get("ticker", "this stock")
    primary = sig.get("primary_signal", "NEUTRAL")
    conf    = sig.get("confidence", 50)

    stop_loss = round(price - 1.5 * atr, 2) if atr > 0 else round(price * 0.95, 2)
    t1        = round(price * 1.05, 2)
    t2        = round(price * 1.10, 2)
    ema_bias  = "bullish" if ema20 > ema50 else "bearish"
    rsi_ctx   = "oversold" if rsi < 35 else "overbought" if rsi > 68 else "neutral"

    return (
        f"<s>[INST]\n"
        f"You are AlphaLens AI — expert stock analyst for Indian retail investors.\n"
        f"Answer the SPECIFIC question using the live data below. Be direct.\n\n"
        f"━━ {ticker} LIVE DATA ━━━━━━━━━━━━━━━━━━━━━\n"
        f"Price    : ₹{price}  |  5D: {'+' if pct_5d>=0 else ''}{pct_5d}%\n"
        f"RSI      : {rsi} ({rsi_ctx})\n"
        f"EMA Bias : {ema_bias} — EMA20 ₹{ema20} vs EMA50 ₹{ema50}\n"
        f"Volume   : {vol_r}x avg  |  ATR: ₹{atr}\n"
        f"Stop-loss: ₹{stop_loss}  |  Targets: ₹{t1} / ₹{t2}\n"
        f"Signal   : {primary} ({conf}% confidence)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Question: {question}\n\n"
        f"Answer specifically. Use actual ₹ values. Under 120 words. Direct answer first.\n"
        f"[/INST]"
    )


# ── LLM Query — question is param 3 ──────────────────────────────────────────

def query_llm(prompt: str, signal_data: dict = None, question: str = "") -> str:
    """
    Calls Mistral-7B via HuggingFace Inference API.

    FIX: question is now param 3 and passed into _rule_based_fallback
    so the fallback routes to the correct intent handler.

    FIX: Removed strict text validation — any non-empty HF response is returned.
    Old code discarded valid HF responses when they lacked SUMMARY/RECOMMENDATION.
    """
    token = HF_TOKEN.strip()
    if not token or token in ("", "your_hf_token_here"):
        print("[LLM] No HF_TOKEN — using rule-based fallback")
        return _rule_based_fallback(signal_data, question)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   380,
            "temperature":      0.72,
            "top_p":            0.92,
            "do_sample":        True,
            "return_full_text": False,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[LLM] Attempt {attempt}/{MAX_RETRIES} → {HF_MODEL}")
            resp = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SEC,
            )

            # ── 200 SUCCESS ───────────────────────────────────────────────
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "").strip()
                    text = _clean(text)
                    if text:  # accept ANY non-empty response from HF
                        print(f"[LLM] ✓ HF response received ({len(text)} chars)")
                        return text
                print(f"[LLM] Empty response from HF on attempt {attempt}")

            # ── 503 MODEL LOADING ─────────────────────────────────────────
            elif resp.status_code == 503:
                try:
                    estimated = float(resp.json().get("estimated_time", 20))
                except Exception:
                    estimated = 20.0
                wait = min(estimated, 35.0)
                print(f"[LLM] 503 model loading — waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue

            # ── 429 RATE LIMITED ──────────────────────────────────────────
            elif resp.status_code == 429:
                print("[LLM] 429 rate limit — waiting 15s")
                time.sleep(15)
                continue

            # ── 401 BAD TOKEN ─────────────────────────────────────────────
            elif resp.status_code == 401:
                print("[LLM] 401 Unauthorized — HF_TOKEN is invalid. Check .env file.")
                print(f"[LLM] Token starts with: {token[:8]}...")
                return _rule_based_fallback(signal_data, question)

            else:
                print(f"[LLM] HTTP {resp.status_code} on attempt {attempt}: {resp.text[:200]}")

            time.sleep(3)

        except requests.exceptions.Timeout:
            print(f"[LLM] Timeout on attempt {attempt} (>{TIMEOUT_SEC}s)")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"[LLM] Connection error on attempt {attempt}")
            time.sleep(3)
        except Exception as e:
            print(f"[LLM] Unexpected error on attempt {attempt}: {e}")
            time.sleep(2)

    print("[LLM] All attempts failed — using rule-based fallback")
    return _rule_based_fallback(signal_data, question)


# ── Question-Aware Rule-Based Fallback ────────────────────────────────────────

def _rule_based_fallback(signal_data: dict = None, question: str = "") -> str:
    """
    Routes question to one of 10 intent handlers.
    Each handler uses real signal_data numbers — nothing is hardcoded/static.
    No HF_TOKEN disclaimer shown here — UI handles that separately via data source badge.
    """
    if not signal_data or signal_data.get("current_price", 0) == 0:
        return (
            "No stock data loaded yet. Select a stock and click Analyze "
            "to get a full technical analysis."
        )

    sig     = signal_data
    ticker  = sig.get("ticker", "This stock")
    price   = sig.get("current_price", 0)
    rsi     = sig.get("rsi", 50)
    ema20   = sig.get("ema20", 0)
    ema50   = sig.get("ema50", 0)
    pct_5d  = sig.get("price_change_5d", 0)
    vol_r   = sig.get("vol_ratio", 1.0)
    atr     = sig.get("atr", 0)
    primary = sig.get("primary_signal", "NEUTRAL")
    conf    = sig.get("confidence", 50)

    # ── Derived values ────────────────────────────────────────────────────
    stop_1x  = round(price - 1.0 * atr, 2) if atr > 0 else round(price * 0.97, 2)
    stop_15x = round(price - 1.5 * atr, 2) if atr > 0 else round(price * 0.95, 2)
    stop_2x  = round(price - 2.0 * atr, 2) if atr > 0 else round(price * 0.93, 2)
    t1       = round(price * 1.05, 2)
    t2       = round(price * 1.08, 2)
    t3       = round(price * 1.12, 2)
    t4       = round(price * 1.15, 2)
    ema_bias = "bullish" if ema20 > ema50 else "bearish"
    ema_spread = round(abs(((ema20 - ema50) / ema50) * 100), 2) if ema50 > 0 else 0
    is_bull  = primary in ("BREAKOUT", "EMA_BULLISH_CROSS", "RSI_OVERSOLD", "UPTREND", "VOLUME_SPIKE")
    is_bear  = primary in ("EMA_BEARISH_CROSS", "DOWNTREND", "BREAKDOWN")
    atr_pct  = round((atr / price) * 100, 2) if price > 0 else 0
    rr_ratio = round((t1 - price) / (price - stop_15x), 2) if (price - stop_15x) > 0 else 0

    q = question.lower().strip()

    # ── 1. EXIT / SELL ────────────────────────────────────────────────────
    if any(w in q for w in ["exit", "sell", "should i exit", "should i sell",
                             "book profit", "close position", "get out"]):
        if is_bull:
            return (
                f"📊 EXIT ANALYSIS — {ticker} @ ₹{price}\n\n"
                f"Signal: {primary} ({conf}% confidence) — currently BULLISH.\n\n"
                f"• RSI {rsi} — {'overbought, consider partial booking' if rsi > 68 else 'not overbought, trend intact'}\n"
                f"• EMA20 ₹{ema20} is {ema_spread}% above EMA50 ₹{ema50} — uptrend intact\n"
                f"• 5-day momentum: {'+' if pct_5d>=0 else ''}{pct_5d}%\n\n"
                f"RECOMMENDATION: {'PARTIAL BOOK — RSI overbought, book 30-50% and trail stop.' if rsi > 68 else 'HOLD — bullish trend intact, no exit signal yet.'}\n"
                f"Trailing stop: ₹{stop_15x}. Full exit only if price closes below ₹{stop_2x}."
            )
        else:
            return (
                f"📊 EXIT ANALYSIS — {ticker} @ ₹{price}\n\n"
                f"Signal: {primary} ({conf}% confidence) — BEARISH.\n\n"
                f"• RSI {rsi} — {'oversold, selling may be exhausted' if rsi < 35 else 'still in downward momentum'}\n"
                f"• EMA20 ₹{ema20} is {ema_spread}% BELOW EMA50 ₹{ema50} — bearish alignment\n"
                f"• 5-day decline: {pct_5d}%\n\n"
                f"RECOMMENDATION: EXIT NOW — bearish signal at {conf}% confidence.\n"
                f"Cut losses. Stop reference: ₹{stop_1x}."
            )

    # ── 2. ENTRY / BUY ────────────────────────────────────────────────────
    elif any(w in q for w in ["enter", "buy", "invest", "should i", "good time",
                               "right time", "worth", "purchase", "accumulate"]):
        if is_bull:
            return (
                f"📈 ENTRY ANALYSIS — {ticker} @ ₹{price}\n\n"
                f"Signal: {primary} ({conf}% confidence) — BULLISH setup.\n\n"
                f"• RSI {rsi} — {'neutral, room to run ✓' if rsi < 60 else 'elevated, entry risk higher ⚠'}\n"
                f"• EMA bias: {ema_bias} — EMA20 ₹{ema20} vs EMA50 ₹{ema50} ({ema_spread}% spread)\n"
                f"• Volume: {vol_r}x avg — {'confirmed ✓' if vol_r > 1.3 else 'needs volume confirmation ⚠'}\n\n"
                f"ENTRY PLAN:\n"
                f"  Entry zone : ₹{round(price*0.99,2)} – ₹{price}\n"
                f"  Stop-loss  : ₹{stop_15x} (risk: {round(((price-stop_15x)/price)*100,1)}%)\n"
                f"  Target 1   : ₹{t1} (+5%)   Target 2: ₹{t2} (+8%)\n"
                f"  R/R Ratio  : {rr_ratio}:1 {'✓ favorable' if rr_ratio >= 1.5 else '— wait for better setup'}"
            )
        else:
            return (
                f"📉 ENTRY ANALYSIS — {ticker} @ ₹{price}\n\n"
                f"Signal: {primary} ({conf}% confidence) — BEARISH. Avoid entry.\n\n"
                f"• RSI {rsi} — {'oversold, watch for reversal' if rsi < 35 else 'still falling'}\n"
                f"• EMA20 ₹{ema20} BELOW EMA50 ₹{ema50} by {ema_spread}% — no buy signal\n\n"
                f"RECOMMENDATION: AVOID — wait for:\n"
                f"  ✗ Signal to flip bullish\n"
                f"  ✗ RSI recovery above 40\n"
                f"  ✓ Re-evaluate if price stabilizes above ₹{round(price*1.03,2)}"
            )

    # ── 3. STOP LOSS ──────────────────────────────────────────────────────
    elif any(w in q for w in ["stop loss", "stop-loss", "stoploss", "cut loss",
                               " sl ", "where to exit if wrong", "maximum loss"]):
        r1 = round(((price - stop_1x)  / price) * 100, 1)
        r2 = round(((price - stop_15x) / price) * 100, 1)
        r3 = round(((price - stop_2x)  / price) * 100, 1)
        return (
            f"🛡 STOP-LOSS LEVELS — {ticker} @ ₹{price}\n\n"
            f"ATR (daily volatility): ₹{atr} = {atr_pct}% per session\n\n"
            f"  Conservative : ₹{stop_1x}  (1× ATR, -{r1}%) — intraday/short trades\n"
            f"  Standard     : ₹{stop_15x} (1.5× ATR, -{r2}%) — ✓ recommended\n"
            f"  Aggressive   : ₹{stop_2x}  (2× ATR, -{r3}%) — swing trades\n\n"
            f"Signal: {primary} ({conf}% conf)\n"
            f"RSI {rsi} — {'near exhaustion zone, stop may be close to reversal' if rsi < 35 else 'no RSI conflict with stop levels'}"
        )

    # ── 4. TARGET / UPSIDE ────────────────────────────────────────────────
    elif any(w in q for w in ["target", "upside", "how high", "price target",
                               "expected return", "how much", "potential", "profit"]):
        return (
            f"🎯 PRICE TARGETS — {ticker} @ ₹{price}\n\n"
            f"Based on {primary} signal ({conf}% confidence):\n\n"
            f"  Target 1 (+5%) : ₹{t1}  — 1–2 weeks\n"
            f"  Target 2 (+8%) : ₹{t2}  — 2–3 weeks\n"
            f"  Target 3 (+12%): ₹{t3}  — 3–5 weeks\n"
            f"  Target 4 (+15%): ₹{t4}  — extended bull run\n\n"
            f"Stop reference: ₹{stop_15x}  →  R/R at T1 = {rr_ratio}:1\n\n"
            f"{'✅ Targets realistic — EMA bullish, momentum positive' if is_bull else '⚠ Targets less reliable — current signal is bearish'}"
        )

    # ── 5. RISK ───────────────────────────────────────────────────────────
    elif any(w in q for w in ["risk level", "risky", "volatil", "safe to",
                               "how risky", "dangerous", "atr", "position size"]):
        pos = round(100 / (atr_pct * 2), 1) if atr_pct > 0 else 0
        return (
            f"⚠ RISK ANALYSIS — {ticker} @ ₹{price}\n\n"
            f"ATR (volatility): ₹{atr} = {atr_pct}% of price per session\n"
            f"Risk level: {'🔴 HIGH' if atr_pct > 3 else '🟡 MODERATE' if atr_pct > 1.5 else '🟢 LOW'}\n\n"
            f"Position sizing (2% account risk rule):\n"
            f"  ₹1,00,000 portfolio → max ≈ ₹{int(pos*1000):,} in this trade\n"
            f"  At ₹{price}/share → approx {int(pos*1000/price) if price>0 else 0} shares\n\n"
            f"Signal: {primary} ({conf}% conf)\n"
            f"Volume {vol_r}x avg — {'high velocity, both up & down' if vol_r > 1.8 else 'normal volume risk'}"
        )

    # ── 6. RSI ────────────────────────────────────────────────────────────
    elif any(w in q for w in ["rsi", "overbought", "oversold", "relative strength"]):
        if rsi < 35:
            explain = f"RSI {rsi} = OVERSOLD (below 35). NSE large-caps bounce 60-70% of the time from this zone. Watch for a reversal candle before entering."
        elif rsi > 68:
            explain = f"RSI {rsi} = OVERBOUGHT (above 68). Rally may be overextended — 3–8% pullback possible. Partial profit booking advisable."
        elif rsi > 55:
            explain = f"RSI {rsi} = BULLISH ZONE (55-68). Momentum is healthy and not overextended — typically a good trending range."
        else:
            explain = f"RSI {rsi} = NEUTRAL (40-55). No strong bias from RSI alone. Use EMA alignment ({ema_bias}) and volume ({vol_r}x) for direction."
        return (
            f"📊 RSI ANALYSIS — {ticker}\n\n"
            f"{explain}\n\n"
            f"  RSI      : {rsi}\n"
            f"  EMA bias : {ema_bias} (EMA20 ₹{ema20} vs EMA50 ₹{ema50})\n"
            f"  Signal   : {primary} ({conf}% conf)\n"
            f"  5D change: {'+' if pct_5d>=0 else ''}{pct_5d}%"
        )

    # ── 7. VOLUME ─────────────────────────────────────────────────────────
    elif any(w in q for w in ["volume", "institutional", "fii", "dii",
                               "participation", "liquidity", "traded"]):
        if vol_r > 1.8:
            explain = f"Volume {vol_r}x above average — SIGNIFICANT spike. Institutional desks (FIIs/DIIs) likely involved. High-volume moves sustain longer."
        elif vol_r > 1.3:
            explain = f"Volume {vol_r}x above average — ELEVATED. Growing interest. Watch for escalation to 2x+ to confirm the {primary} signal strongly."
        else:
            explain = f"Volume only {vol_r}x average — BELOW CONFIRMATION THRESHOLD. The {primary} signal is weaker without volume. Wait for high-volume session."
        return (
            f"📊 VOLUME ANALYSIS — {ticker} @ ₹{price}\n\n"
            f"{explain}\n\n"
            f"  Volume ratio : {vol_r}x 20-day average\n"
            f"  Signal       : {primary} ({conf}% conf)\n"
            f"  5D change    : {'+' if pct_5d>=0 else ''}{pct_5d}%\n"
            f"  RSI          : {rsi}\n\n"
            f"{'Move HAS volume backing ✓' if vol_r > 1.5 else 'Move LACKS volume backing — treat with caution ⚠'}"
        )

    # ── 8. TREND / OUTLOOK ────────────────────────────────────────────────
    elif any(w in q for w in ["trend", "outlook", "next week", "direction",
                               "momentum", "going forward", "forecast"]):
        short  = f"Short-term (1–2 wks): {'Bullish — '+primary+' signal active' if is_bull else 'Bearish — '+primary+' in effect'}"
        medium = f"Medium-term (3–6 wks): {'Positive above ₹'+str(stop_15x) if ema20>ema50 else 'Cautious — EMA20 below EMA50 = downward pressure'}"
        return (
            f"📈 TREND ANALYSIS — {ticker} @ ₹{price}\n\n"
            f"{short}\n{medium}\n\n"
            f"  5D momentum : {'+' if pct_5d>=0 else ''}{pct_5d}%\n"
            f"  RSI         : {rsi}\n"
            f"  EMA bias    : {ema_bias} ({ema_spread}% spread EMA20 vs EMA50)\n"
            f"  Volume      : {vol_r}x avg — {'confirming' if vol_r>1.3 else 'unconfirmed'}\n\n"
            f"Key level: ₹{stop_15x} — {'Bull case intact above this' if is_bull else 'Bear case continues below this'}."
        )

    # ── 9. EMA ────────────────────────────────────────────────────────────
    elif any(w in q for w in ["ema", "crossover", "golden cross", "death cross",
                               "moving average", "ema20", "ema50"]):
        if primary == "EMA_BULLISH_CROSS":
            cross = "🟢 GOLDEN CROSS just occurred — EMA20 crossed above EMA50. Historically precedes 6–10% gains over 3–6 weeks in NSE large-caps."
        elif primary == "EMA_BEARISH_CROSS":
            cross = "🔴 DEATH CROSS just occurred — EMA20 crossed below EMA50. Often precedes 5–8% further downside over 2–4 weeks."
        elif ema20 > ema50:
            cross = f"EMA20 ₹{ema20} is {ema_spread}% above EMA50 ₹{ema50} — bullish structure. No fresh crossover but uptrend intact."
        else:
            cross = f"EMA20 ₹{ema20} is {ema_spread}% below EMA50 ₹{ema50} — bearish structure. Avoid new long positions."
        return (
            f"📊 EMA ANALYSIS — {ticker} @ ₹{price}\n\n"
            f"{cross}\n\n"
            f"  EMA 20 : ₹{ema20}\n"
            f"  EMA 50 : ₹{ema50}\n"
            f"  Spread : {ema_spread}% {'above (bullish)' if ema20>ema50 else 'below (bearish)'}\n"
            f"  RSI    : {rsi}\n"
            f"  Signal : {primary} ({conf}% conf)"
        )

    # ── 10. COMPARE / NIFTY ───────────────────────────────────────────────
    elif any(w in q for w in ["nifty", "compare", "benchmark", "index",
                               "relative", "sector", "vs market"]):
        return (
            f"📊 RELATIVE PERFORMANCE — {ticker} @ ₹{price}\n\n"
            f"  5D change    : {'+' if pct_5d>=0 else ''}{pct_5d}% (Nifty avg ~1–2%/week)\n"
            f"  Volume       : {vol_r}x avg (>1.5x = institutional interest)\n"
            f"  Signal       : {primary} ({conf}% conf)\n"
            f"  EMA bias     : {ema_bias}\n"
            f"  RSI          : {rsi} (healthy market: 45–65)\n\n"
            f"{'✅ '+ticker+' showing relative strength — bullish EMA, positive momentum' if (is_bull and pct_5d>0) else '⚠ '+ticker+' underperforming — wait for relative strength before entry'}"
        )

    # ── DEFAULT — Full stock summary ──────────────────────────────────────
    else:
        if pct_5d > 3:
            trend = f"{ticker} gained {pct_5d}% over 5 sessions — strong buying momentum."
        elif pct_5d < -3:
            trend = f"{ticker} fell {abs(pct_5d)}% over 5 sessions — selling pressure dominant."
        else:
            trend = f"{ticker} is consolidating ({pct_5d:+.2f}% over 5 sessions)."

        rsi_line = (
            f"RSI {rsi} — oversold, reversal possible." if rsi < 35 else
            f"RSI {rsi} — overbought, pullback risk." if rsi > 68 else
            f"RSI {rsi} — {'bullish zone' if rsi > 55 else 'neutral territory'}."
        )

        ema_line = (
            f"EMA20 ₹{ema20} is {ema_spread}% above EMA50 ₹{ema50} — bullish structure."
            if ema20 > ema50
            else f"EMA20 ₹{ema20} is {ema_spread}% below EMA50 ₹{ema50} — bearish alignment."
        )

        vol_line = (
            f"Volume {vol_r}x avg — strong institutional participation." if vol_r > 1.8 else
            f"Volume {vol_r}x avg — moderate interest." if vol_r > 1.3 else
            f"Volume {vol_r}x avg — lacks confirmation."
        )

        if is_bull and conf >= 60:
            action = (
                f"ACTION — WATCH/BUY: {conf}% confidence {primary.replace('_',' ')} signal. "
                f"Upside targets: ₹{t1}–₹{t2}. Stop-loss: ₹{stop_15x}. R/R = {rr_ratio}:1."
            )
        elif is_bear:
            action = (
                f"ACTION — AVOID: {primary.replace('_',' ')} at {conf}% confidence. "
                f"Avoid new entries. Stop reference: ₹{stop_15x}."
            )
        else:
            action = (
                f"ACTION — WATCH: Mixed signals at {conf}% confidence. "
                f"Wait for RSI/EMA confirmation. Key support: ₹{stop_15x}."
            )

        return f"{trend}\n\n{rsi_line} {ema_line}\n\n{vol_line}\n\n{action}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip leaked instruction tokens."""
    for marker in ["[/INST]", "[INST]", "</s>", "<s>"]:
        while marker in text:
            text = text.split(marker)[-1]
    return re.sub(r"^\s*[-–—•]\s*", "", text.strip()).strip()
