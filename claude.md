# CLAUDE.md — MarketPulse

Financial sentiment hub powered by free RSS news feeds. Two pages:
1. **MarketPulse** — search any ticker for sentiment breakdown, AI verdict, 7-day trend (top 50 by mentions)
2. **Trading Bot** — autonomous paper-trading scalp bot powered by financial-mcp-server

No API keys required. `ANTHROPIC_API_KEY` is optional (enables AI verdicts; static fallback without it).

---

## Architecture

```
Free News RSS (Google News + Yahoo Finance + CNBC + MarketWatch)
        ↓
Keyword/emoji labeling functions (16 LFs) → confidence-weighted vote
        ↓
TF-IDF + LogReg classifies all posts (auto-trains when ≥200 labeled)
        ↓
Posts + per-ticker summaries saved to SQLite (data/marketpulse.db)
        ↓
Home page: top 50 tickers by mentions, search → AI verdict → dialog popup
        ↓
Trading Bot: MCP Client connects to financial-mcp-server via SSE
  → autonomous scalp bot (continuous cycles, score-based entry/exit, rotation)
  → portfolio state in data/financial_mcp.db (managed by MCP server)
```

---

## Project Structure

```
MarketPulse/
├── start.bat                          # One-command launcher (MCP server + Streamlit)
├── app/
│   ├── MarketPulse.py                 # Home page: search bar + top 50 grid + ticker detail dialog
│   ├── pipeline_runner.py             # refresh_pipeline(), get_ticker_cache(), load_model()
│   ├── pages/
│   │   └── 2_Trading_Bot.py           # Market regime + VIX + ticker analysis + autonomous bot control
│   └── components/
│       ├── charts.py                  # Plotly chart components (bar, trend)
│       ├── styles.py                  # Dark theme colors, animations, and CSS
│       └── trading_charts.py          # Candlestick, score gauge, CFTC bars
├── src/
│   ├── ingestion/
│   │   ├── base.py                    # Abstract base ingester + REQUIRED_COLUMNS schema
│   │   ├── news.py                    # Free RSS ingester (always available, no API key)
│   │   └── manager.py                # Orchestrates news RSS ingestion
│   ├── labeling/
│   │   ├── functions.py               # 16 keyword/emoji/structural labeling functions
│   │   └── aggregator.py             # Confidence-weighted vote aggregation
│   ├── models/
│   │   └── pipeline.py               # TF-IDF + LogReg training and inference
│   ├── extraction/
│   │   ├── ticker_extractor.py        # Cashtag, bare ticker, company name extraction
│   │   └── normalizer.py             # Canonical company name normalization
│   ├── analysis/
│   │   └── ticker_sentiment.py        # Per-ticker aggregation from labeled posts
│   ├── investor/
│   │   ├── mcp_client.py             # MCP client — SSE connection to financial-mcp-server
│   │   └── bot_engine.py             # Autonomous scalp trading bot (BotState + BotEngine)
│   ├── storage/
│   │   └── db.py                      # SQLite read/write (data/marketpulse.db)
│   ├── agent/
│   │   └── briefing.py               # Claude AI verdict — one API call per search
│   └── utils/
│       ├── config.py                  # YAML config loader
│       ├── logger.py                  # Structured logging
│       └── cache.py                   # API response cache
├── scripts/
│   └── run_pipeline.py                # CLI: full pipeline end-to-end
├── config/
│   └── default.yaml                   # RSS sources, model hyperparameters, MCP server config
├── tests/                             # pytest test suite (210 tests)
├── data/
│   ├── marketpulse.db                 # SQLite — sentiment data (gitignored)
│   ├── financial_mcp.db               # SQLite — portfolio state, managed by MCP server (gitignored)
│   └── models/                        # Trained model artifacts (gitignored)
└── requirements.txt
```

---

## Trading Bot (`src/investor/bot_engine.py`)

Autonomous paper-trading bot built on probability math — thinks like a casino, not a gambler. Runs continuous cycles (no timer — immediate rescan with 5s pause between cycles).

**Cycle:** detect regime → check VIX → retry pending sells → smart exits → scan universe → score candidates → enter/rotate positions → recompute stats → snapshot portfolio

**Quant Decision Engine:**
- **Expected Value (EV):** `EV = (WinRate × AvgWin) - (LossRate × AvgLoss)` — computed from actual closed trades every cycle. Only sizes up when EV is positive.
- **Kelly Criterion:** half-Kelly position sizing from real win/loss statistics. Replaces fixed percentage tiers. Adapts as the bot learns its own edge.
- **Risk of Ruin:** probability of total account loss, computed every cycle. If >5%, position sizes auto-halve to protect the account.
- **Variance tracking:** standard deviation of trade P&L. Positions with unrealized loss >2σ are cut as outliers.
- **Conviction scaling:** higher MCP scores → proportionally more capital (score/100 × base risk).
- **Hard 2% cap:** never risk more than 2% of portfolio per trade, regardless of Kelly.
- **Conservative bootstrap:** 1% per trade until 10+ closed trades provide enough data for Kelly (Law of Large Numbers).

**Smart exits (4 triggers):**
1. Signal reversal — score dropped 30%+ from entry or below 40
2. Profit taking — score fading 15%+ while position is green (lock gains)
3. Momentum stall — held 10+ cycles with no score improvement (free slot)
4. Outlier loss — unrealized loss exceeds 2 standard deviations (cut variance)

**Position rotation:** When at 20 max positions, sells weakest holding if a new candidate scores 10+ points higher.

**Dashboard:** `@st.fragment(run_every=1)` — live-updating panel with:
- Portfolio value, P&L, open positions count
- Edge Statistics: EV/trade, win rate, R:R ratio, half-Kelly %, risk of ruin, streak, σ
- Open positions table with real-time P&L
- Activity log streaming buys/sells as they happen

---

## Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=...       # Optional — enables AI verdict (static fallback without it)
```

News RSS is free — no keys needed.

---

## Running

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY if you want AI verdicts
start.bat                     # starts MCP server + Streamlit on localhost:8501
```

---

## Key Design Decisions

- **RSS-only primary source** — no API keys required for core functionality
- **Two clean pages** — MarketPulse (sentiment) and Trading Bot (MCP-powered)
- **Data separation** — home page uses only RSS/SQLite; trading bot uses only MCP server
- **MCP = data layer, bot_engine = decision layer** — the MCP server provides market intelligence; the bot's math determines when to buy/sell and how much to risk
- **Probability over prediction** — position sizing from actual trade statistics (Kelly Criterion), not gut feel or fixed percentages
- **Many small bets** — 2% max risk per trade, conservative 1% until edge is proven over 10+ trades
- **Ticker detail as dialog** — `@st.dialog` popup, not a separate page
- **Live bot panel** — `@st.fragment(run_every=1)` for real-time updates without page flicker
- **Paper trading** — all trades go through MCP execute_buy/execute_sell on financial_mcp.db
