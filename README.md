# MarketPulse

**Live financial-sentiment hub + autonomous paper-trading bot, powered by free data and a custom [MCP](https://modelcontextprotocol.io) server.**

🔴 **Live demo:** https://navs-marketpulse.streamlit.app/

Two pages:

1. **MarketPulse** — scrapes free RSS news, classifies sentiment, and auto-discovers 100+ tickers from a 200+ company universe. Browsable mood grid, momentum sparklines, and AI-powered verdicts.
2. **Trading Bot** — an autonomous paper-trading bot powered by [`financial-mcp-server`](https://pypi.org/project/financial-mcp-server/). It makes buy/sell decisions with probability math (Expected Value, Kelly Criterion, Risk of Ruin) and even lets the news-sentiment page tilt its scores.

No API keys required. `ANTHROPIC_API_KEY` is optional (enables Claude AI verdicts; static fallback without it).

---

## Quick start

```powershell
pip install -r requirements.txt        # includes financial-mcp-server + curl_cffi
cp .env.example .env                    # optional: add ANTHROPIC_API_KEY for AI verdicts
.\start.ps1                             # PowerShell (recommended) — opens Chrome, clean Ctrl-C
```

`start.bat` works too (cmd). Both launch the MCP server and the dashboard on http://localhost:8501.

> **How the two halves connect:** the Trading Bot talks to [`financial-mcp-server`](https://pypi.org/project/financial-mcp-server/)
> over **SSE on port 8520**. The server defaults to stdio, so it's launched with
> `financial-mcp --transport sse`. The app also auto-starts it in SSE mode on first connect.

### Recording a demo

```powershell
python scripts/seed_demo.py     # warm-start the bot's Edge panel with illustrative history
.\start.ps1                      # open the Trading Bot page → Start Bot
```

See [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) for the full record-the-video checklist.

---

## What you get

**MarketPulse page**
- 100+ tickers auto-discovered from RSS (200+ company universe), **auto-refreshed in the background**
- Browsable grid: market-mood bar, filter + sentiment + sort (mentions / heating-up / bullish / bearish / A–Z)
- Per-card momentum arrow (Δ vs prior day) + 7-day sentiment sparkline
- Click any ticker → dialog with sentiment split, **price line**, AI verdict, 7-day trend, clickable headlines
- Search bar → inline briefing card for any ticker

**Trading Bot page**
- Market regime (BULL / BEAR / SIDEWAYS / …) + VIX analysis
- Ticker analysis: candlestick, composite score gauge, fundamentals, momentum, smart money, and a **Catalysts** panel (SEC filings · insider filings · search trends)
- **News-sentiment → score bridge:** MarketPulse RSS sentiment tilts the bot's MCP scores
- Autonomous bot: click **Start** → it scans, scores, sizes with Kelly math, buys/sells, and rotates positions
- Live dashboard: portfolio value, **equity curve**, P&L, Edge Statistics, **risk/exposure panel**, streaming activity log
- **Configurable params:** starting capital, max positions, risk cap, min score, sentiment toggle

---

## Architecture

```
Free News RSS (Google News + Yahoo Finance + CNBC + MarketWatch)
        ↓
16 labeling functions → confidence-weighted vote → TF-IDF + LogReg → SQLite
        ↓
MarketPulse: auto-refreshed mood grid + search + Claude AI verdict
        ↓                                    ↘  (sentiment bridge)
Trading Bot ──→ financial-mcp-server (33 MCP tools: prices, fundamentals,
   (MCP client over SSE)               SEC, macro, regime, paper trading)
        ↓
Bot Engine: Kelly sizing · EV / Risk-of-Ruin · smart exits · rotation · equity curve
```

**Data separation:** the sentiment page uses only RSS/SQLite; the bot uses the MCP server for market data and paper trading. The one deliberate link between them is the sentiment→score bridge.

---

## The MCP server

The bot is driven by [`financial-mcp-server`](https://pypi.org/project/financial-mcp-server/)
([source](https://github.com/arnavbhatia1/FinancialMCP)) — an MCP server exposing **33 tools**
(prices, fundamentals, SEC EDGAR, FRED macro, CFTC positioning, market-regime/VIX, anomaly
scanners, and paper trading). It needs **no API keys** (yfinance, SEC, CFTC, Treasury, Google
Trends) and speaks standard MCP, so the same tools work in Claude Desktop, Cursor, or any agent.

```bash
pip install financial-mcp-server
```

---

## Trading Bot math

The bot thinks like a casino, not a gambler:

- **Expected Value:** `EV = (WinRate × AvgWin) − (LossRate × AvgLoss)` — only sizes up when EV is positive
- **Kelly Criterion:** half-Kelly sizing from the bot's *actual* win/loss statistics
- **Risk of Ruin:** auto-halves positions if the account-destruction probability exceeds 5%
- **2% hard cap:** never risks more than 2% of the portfolio per trade
- **Conservative bootstrap:** 1% per trade until 10+ closed trades prove the edge
- **5 exit triggers:** hard stop (−5%), trailing stop (−3% from peak), signal reversal, profit taking, outlier loss (>2σ)
- **Position rotation:** sells the weakest holding when a stronger candidate appears
- **Dynamic universe:** each cycle draws from MCP anomaly/volume/gap movers + top news-mentioned tickers (a pool up to 250), shuffles, and scores 25/cycle — so it rotates across everything the news surfaces
- **Persistent ledger:** trade history + equity curve survive restarts in `data/bot_state.json`

---

## Deploy it free

Deploys to **[Streamlit Community Cloud](https://share.streamlit.io)** from this repo — main file `app/MarketPulse.py`. `requirements.txt` pulls `financial-mcp-server`, and the app auto-starts it in-container over SSE.

> **Note:** Yahoo/yfinance throttles datacenter IPs. The MCP server (v0.1.10+) impersonates a
> browser via `curl_cffi` to work around this, but live market data can still be intermittent on
> shared cloud hosts. For a flawless recording, run locally (home IP) and share the cloud link as
> the "try it live" CTA.

---

## Tests & CI

Two guards keep broken code off the deployed app:

- **Pre-push hook** (`.githooks/pre-push`) runs the suite before every push; a failure blocks it. Enable once per clone:
  ```bash
  git config core.hooksPath .githooks
  ```
- **GitHub Actions** (`.github/workflows/ci.yml`) runs on every push/PR to `main`: the full offline suite **plus a headless page-render smoke** (`scripts/render_smoke.py`) that boots both Streamlit pages and fails on any load-time error.

```bash
pytest -q                        # offline unit tests
python scripts/render_smoke.py   # boots both pages headless; non-zero on any render error
```

> **Bulletproof option:** point Streamlit Cloud at a `release` branch and only fast-forward
> `main → release` when CI is green — then a broken commit can never auto-deploy. (Mark the CI
> check "required" under branch protection to enforce it on PRs.)

## Config & keys

| Key | Required? | What it does |
|-----|-----------|--------------|
| `ANTHROPIC_API_KEY` | Optional | Claude AI verdict on ticker cards. Static fallback without it. |

Everything else is free — no keys needed.

## Tech stack

Python 3.10+ · Streamlit · scikit-learn · Plotly · Anthropic SDK · SQLite · feedparser · yfinance · `financial-mcp-server` (MCP) · pandas
