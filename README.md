# MarketPulse

Financial sentiment dashboard + autonomous paper-trading bot. Two pages:

1. **MarketPulse** — scours free RSS news feeds, analyzes sentiment across 60 tickers, shows top 50 by mentions with AI-powered verdicts
2. **Trading Bot** — autonomous scalp bot powered by [financial-mcp-server](https://github.com/arnavbhat1/financial-mcp), makes buy/sell decisions using probability math (Expected Value, Kelly Criterion, Risk of Ruin)

No API keys required. Works out of the box.

## Quick Start

```bash
pip install -r requirements.txt
pip install financial-mcp-server   # the MCP market-data + paper-trading server (PyPI)
cp .env.example .env               # optionally add ANTHROPIC_API_KEY for AI verdicts
start.bat                          # starts MCP server (SSE) + Streamlit on localhost:8501
```

> The Trading Bot talks to [`financial-mcp-server`](https://pypi.org/project/financial-mcp-server/)
> over **SSE on port 8520**. `start.bat` launches it with `financial-mcp --transport sse`
> (the server defaults to stdio, so the flag is required). The app will also
> auto-start the server in SSE mode on first connect.

### Recording a demo

```bash
python scripts/seed_demo.py    # warm-start the Edge panel with illustrative trade history
start.bat                      # then open the Trading Bot page and click Start Bot
```

See [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) for the full record-the-video checklist.

## What You Get

**MarketPulse page:**
- 100+ tickers auto-discovered from RSS, **auto-refreshed in the background**
- Browsable grid: market-mood bar, filter + sentiment + sort, momentum arrows & 7-day sparklines
- Click any ticker → dialog with sentiment split, **price line**, AI verdict, trend, clickable headlines
- Search bar for any ticker → inline briefing card

**Trading Bot page:**
- Market regime detection (BULL/BEAR/SIDEWAYS) + VIX analysis
- Ticker analysis: candlestick, score gauge, fundamentals, momentum, smart money, **Catalysts (SEC filings · insider trades · search trends)**
- **News-sentiment → score bridge**: MarketPulse RSS sentiment tilts the bot's scores
- Autonomous bot: click Start → it continuously scans, buys, sells, rotates positions
- Live dashboard: portfolio value, **equity curve**, P&L, Edge Statistics, **risk/exposure panel**, activity log
- **Configurable params**: capital, max positions, risk cap, min score, sentiment toggle

## Architecture

```
Free News RSS (Google News + Yahoo Finance + CNBC + MarketWatch)
        ↓
16 labeling functions → sentiment classification → SQLite
        ↓
MarketPulse: top 50 grid + search + AI verdict (Claude)
        ↓
Trading Bot: MCP client → financial-mcp-server (market data + paper trading)
        ↓
Bot Engine: Kelly Criterion sizing, EV tracking, smart exits, position rotation
```

**Data separation:** MarketPulse uses only RSS/SQLite. Trading Bot uses only the MCP server. They don't share data sources.

## Trading Bot Math

The bot thinks like a casino, not a gambler:

- **Expected Value:** `EV = (WinRate × AvgWin) - (LossRate × AvgLoss)` — only sizes up when EV is positive
- **Kelly Criterion:** half-Kelly position sizing from actual trade statistics
- **Risk of Ruin:** auto-halves positions if account destruction probability exceeds 5%
- **2% hard cap:** never risks more than 2% of portfolio per trade
- **Conservative bootstrap:** 1% per trade until 10+ closed trades prove the edge
- **5 exit triggers:** hard stop (-5%), trailing stop (-3% from peak), signal reversal, profit taking, outlier loss (>2σ)
- **Position rotation:** sells weakest holding when a stronger candidate appears
- **Dynamic universe:** each cycle scans MCP anomaly/volume/gap movers + top news-mentioned tickers (capped at 25), then scores them live
- **Persistent ledger:** trade history survives restarts in `data/bot_state.json`

## API Keys

| Key | Required? | What it does |
|-----|-----------|-------------|
| `ANTHROPIC_API_KEY` | Optional | AI verdict on ticker briefing cards. Falls back to static text without it. |

News RSS is free — no keys needed. The MCP server uses yfinance (free) for market data.

## Tech Stack

Python 3.9+ · Streamlit · scikit-learn · Plotly · Anthropic SDK · SQLite · feedparser · yfinance · financial-mcp-server · pandas
