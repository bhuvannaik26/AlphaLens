"""
AlphaLens — Flask Server v7 (FINAL FIX)
=========================================
"""

import os
import traceback
import requests
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

from agents.data_provider import get_stock_data
from agents import (
    detect, NSE_STOCKS,
    build_prompt, build_chat_prompt, query_llm,
    explain,
)

load_dotenv()
app = Flask(__name__, static_folder="static")

FINNHUB_TOKEN     = os.getenv("FINNHUB_TOKEN", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")


# ── CORS ───────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── GLOBAL ERROR HANDLERS — always JSON, never HTML ───────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "path": request.path}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": str(e), "type": type(e).__name__}), 500


# ── PIPELINE — question flows all the way through ─────────────────────────────
def _full_pipeline(ticker: str) -> dict:
    """
    SYSTEM reasoning — NO user question allowed
    """
    signal_data = detect(ticker)

    # 🔒 PURE SYSTEM PROMPT (NO question)
    prompt      = build_prompt(ticker, signal_data)
    llm_insight = query_llm(prompt, signal_data, "")   # empty question

    explanation = explain(signal_data, llm_insight)

    return {
        "ticker":      ticker,
        "signal_data": signal_data,
        "explanation": explanation,
    }

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    finnhub_ok = bool(FINNHUB_TOKEN and FINNHUB_TOKEN not in ("", "your_finnhub_token_here"))
    av_ok      = bool(ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY not in ("", "your_alphavantage_key_here"))
    hf_ok      = bool(HF_TOKEN and HF_TOKEN not in ("", "your_hf_token_here"))
    return jsonify({
        "status": "ok",
        "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "services": {
            "finnhub":       "✅ configured" if finnhub_ok else "⚠ not set",
            "alpha_vantage": "✅ configured" if av_ok      else "⚠ not set",
            "huggingface":   "✅ configured" if hf_ok      else "⚠ not set — rule-based fallback active",
            "news_api":      "✅ configured" if NEWS_API_KEY else "ℹ not set — mock headlines",
        },
        "active_data_source": "finnhub" if finnhub_ok else "alphavantage" if av_ok else "simulation",
        "mode": "live" if (finnhub_ok or av_ok) else "simulation",
    })


@app.route("/api/stocks/list")
def stocks_list():
    return jsonify({"stocks": list(NSE_STOCKS.keys())})


@app.route("/api/stock")
def get_stock():
    ticker = request.args.get("ticker", "RELIANCE").strip().upper()
    try:
        hist = get_stock_data(ticker)
        closes  = [round(float(c), 2) for c in hist["Close"].tolist()]
        volumes = [int(float(v))      for v in hist["Volume"].tolist()]
        highs   = [round(float(h), 2) for h in hist["High"].tolist()]
        lows    = [round(float(l), 2) for l in hist["Low"].tolist()]
        opens   = [round(float(o), 2) for o in hist["Open"].tolist()]
        dates   = [str(d.date())      for d in hist.index]
        pct_5d  = 0.0
        if len(closes) >= 6:
            pct_5d = round(((closes[-1] - closes[-6]) / closes[-6]) * 100, 2)
        is_sim = getattr(hist, "_is_simulated", False)
        return jsonify({
            "ticker":          ticker,
            "symbol":          ticker + ".NS",
            "company_name":    ticker,
            "sector":          "NSE Listed",
            "current_price":   closes[-1] if closes else 0,
            "price_change_5d": pct_5d,
            "closes":          closes,
            "volumes":         volumes,
            "highs":           highs,
            "lows":            lows,
            "opens":           opens,
            "dates":           dates,
            "data_source":     "simulated" if is_sim else "live",
            "total_rows":      len(closes),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/api/signals")
def get_signals():
    ticker = request.args.get("ticker", "RELIANCE").strip().upper()

    try:
        result = _full_pipeline(ticker)   # ❌ REMOVE question
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500

@app.route("/api/signals/all")
def get_all_signals():
    opportunities = []
    for name in list(NSE_STOCKS.keys())[:6]:
        try:
            sig = detect(name)
            opportunities.append({
                "ticker":          name,
                "primary_signal":  sig.get("primary_signal", "NEUTRAL"),
                "confidence":      sig.get("confidence", 50),
                "current_price":   sig.get("current_price", 0),
                "price_change_5d": sig.get("price_change_5d", 0),
                "rsi":             sig.get("rsi", 50),
                "vol_ratio":       sig.get("vol_ratio", 1.0),
                "data_source":     sig.get("data_source", "unknown"),
            })
        except Exception as e:
            print(f"[/api/signals/all] {name}: {e}")
    opportunities.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify({"opportunities": opportunities})


@app.route("/api/news")
def get_news():
    ticker = request.args.get("ticker", "market").strip()
    if NEWS_API_KEY:
        try:
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={ticker}+NSE+India+stock"
                f"&sortBy=publishedAt&pageSize=5&language=en"
                f"&apiKey={NEWS_API_KEY}"
            )
            r = requests.get(url, timeout=10)
            articles = r.json().get("articles", [])
            if articles:
                return jsonify({
                    "news": [{"title": a.get("title",""), "source": a.get("source",{}).get("name",""),
                              "url": a.get("url","#"), "published": a.get("publishedAt","")}
                             for a in articles[:5]],
                    "source": "live",
                })
        except Exception as e:
            print(f"[/api/news] {e}")

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return jsonify({
        "news": [
            {"title": f"FIIs raise stake in {ticker} amid strong Q3 earnings",         "source": "ET Markets",     "url": "#", "published": now},
            {"title": "RBI holds rates — markets rally on dovish commentary",           "source": "Moneycontrol",   "url": "#", "published": now},
            {"title": f"{ticker} near 52-week high on heavy institutional volumes",     "source": "BSE India",      "url": "#", "published": now},
            {"title": "Sensex, Nifty climb 0.8% on positive global cues",              "source": "LiveMint",       "url": "#", "published": now},
            {"title": "SEBI tightens F&O rules: key implications for retail investors", "source": "Economic Times", "url": "#", "published": now},
        ],
        "source": "mock",
    })


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body     = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    ticker   = (body.get("ticker") or "").strip().upper() or None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # ── STOCK CHAT ─────────────────────────────
        if ticker and ticker in NSE_STOCKS:
            signal_data = detect(ticker)

            # 💬 CHAT ONLY (NO explainability here)
            prompt   = build_chat_prompt(question, signal_data)
            response = query_llm(prompt, signal_data, question)

            return jsonify({
                "response": response,
                "signal_data": signal_data
            })

        # ── GENERAL CHAT ───────────────────────────
        prompt = build_chat_prompt(question)

        return jsonify({
            "response": query_llm(prompt, None, question),
            "signal_data": None
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hf_set = bool(HF_TOKEN and HF_TOKEN not in ("", "your_hf_token_here"))
    fh_set = bool(FINNHUB_TOKEN and FINNHUB_TOKEN not in ("", "your_finnhub_token_here"))
    print("\n" + "="*56)
    print("  🔭  AlphaLens — Explainable Market Intelligence v7")
    print("="*56)
    print(f"  Dashboard  →  http://localhost:5000")
    print(f"  Health     →  http://localhost:5000/api/health")
    print(f"  Finnhub    →  {'✅ configured' if fh_set else '⚠  not set (simulation mode)'}")
    print(f"  HuggingFace→  {'✅ configured — Mistral-7B active' if hf_set else '⚠  not set — rule-based fallback'}")
    print("="*56 + "\n")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
