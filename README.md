# AlphaLens — Explainable Market Intelligence
AlphaLens is an AI-powered stock analysis platform designed for Indian retail investors.
It combines technical indicators, explainable AI, and conversational intelligence to provide transparent and actionable insights.
---
## Key Features

* **Technical Signal Detection**
  * RSI, EMA (20/50), Volume, ATR
* **Explainable AI Reasoning**
  * Every recommendation is justified with data
* **AI Chat Interface**
  * Ask anything about a stock (entry, exit, targets, risk)
* **Opportunity Radar**
  * Scans top NSE stocks for high-confidence setups
* **News Integration**
  * Market context with latest headlines
* **Hybrid AI System**
  * Mistral-7B (HuggingFace) + rule-based fallback

---
## System Architecture

```text
Data Provider
   ↓
Signal Detection Agent
   ↓
Reasoning Agent (LLM + Rule Engine)
   ↓
Explainability Agent
   ↓
Flask API
   ↓
Frontend Dashboard
```

---

## Tech Stack

* **Backend:** Python, Flask
* **AI Model:** Mistral-7B (HuggingFace Inference API)
* **Data Sources:** Finnhub, Alpha Vantage, NewsAPI
* **Frontend:** HTML, CSS, JavaScript

---

## Setup Instructions
### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/alphalens.git
cd alphalens
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` file

```env
FINNHUB_TOKEN=your_key
ALPHA_VANTAGE_KEY=your_key
HF_TOKEN=your_key
NEWS_API_KEY=your_key
```

### 5. Run the app

```bash
python app.py
```

Open:

```
http://localhost:5000
```
---

## How It Works
* Detects technical signals from market data
* Uses AI + rules to generate reasoning
* Converts outputs into explainable insights
* Provides chat-based interaction for users
---

## Disclaimer

This project is for educational purposes only.
It does not provide financial advice. Always consult a qualified advisor before investing.
---

