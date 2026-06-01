# CLAUDE.md — MarketPulse

Financial sentiment hub powered by free RSS news feeds. Two pages:
1. **MarketPulse** — auto-refreshing sentiment hub: search any ticker (sentiment, AI verdict, 7-day trend, price line), plus a sortable/filterable grid of 100+ news-discovered tickers with momentum sparklines and a market-mood bar
2. **Trading Bot** — autonomous paper-trading bot powered by financial-mcp-server: equity curve, configurable params, live MCP catalysts (SEC/insider/trends), risk panel, and a news-sentiment→score bridge

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
Home page: auto-refreshed mood grid (100+ tickers, sort/filter), search → AI verdict → dialog
        ↓
Trading Bot: MCP Client connects to financial-mcp-server via SSE
  → autonomous bot (continuous cycles, score-based entry/exit, rotation, equity curve)
  → news-sentiment → score bridge; portfolio state in data/financial_mcp.db (MCP server)
```

---

## Project Structure

```
MarketPulse/
├── start.ps1                          # Launcher (recommended) — opens Chrome, clean Ctrl-C
├── start.bat                          # Launcher (cmd) — MCP server + Streamlit
├── app/
│   ├── MarketPulse.py                 # Home: search + sortable/filterable mood grid + detail dialog
│   ├── auto_refresh.py                # Background scheduler — keeps sentiment data fresh
│   ├── pipeline_runner.py             # refresh_pipeline(), get_ticker_cache(), load_model()
│   ├── pages/
│   │   └── 2_Trading_Bot.py           # Regime/VIX + ticker analysis + catalysts + bot control
│   └── components/
│       ├── charts.py                  # Plotly components (bar, trend, price line)
│       ├── styles.py                  # Dark theme colors, animations, and CSS
│       └── trading_charts.py          # Candlestick, score gauge, CFTC bars, equity curve
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
│   │   ├── ticker_universe.py         # Bundled 200+ symbol→company universe (source of truth)
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

Autonomous paper-trading bot built on probability math — thinks like a casino, not a gambler. Runs continuous cycles with a 60s pause between them (MCP scores need time to update). On MCP failure it backs off exponentially (10s→300s) and auto-stops after 10 consecutive failures.

**Cycle:** detect regime → check VIX → retry pending sells → build dynamic universe → **one batched `scan_universe`** over held + candidate symbols → smart exits → score/enter/rotate positions → recompute stats → snapshot portfolio (also persists ledger)

**Dynamic universe:** each cycle is built from MCP scanners (anomalies + volume leaders + gap movers — event-driven catalysts) plus the news-mentioned tickers from the RSS `ticker_cache` — a pool of up to 250 symbols. The pool is shuffled and the top `SCORE_CANDIDATES_LIMIT` (25) are scored per cycle, so over successive cycles the bot rotates across every ticker the news surfaces. The scanners' rich payloads (volume ratio, gap %, anomaly score) are distilled into a per-symbol **catalyst bonus** (`_catalyst_bonus`) that tilts event-driven names — data the bot previously fetched and discarded. The scanners re-download from yfinance un-cached (~9s), so their results are reused for `SCANNER_TTL` (150s) via a small TTL cache — the main lever that drops steady-state cycles from ~25s to ~13–17s.

**Scoring — the right tool is `scan_universe`, not per-ticker `analyze_ticker`.** A single batched `scan_universe` call ranks the cycle's candidates *and* held positions against real peers (momentum/valuation percentiles + sector medians computed across the whole batch) and returns each symbol's price (financial-mcp ≥0.1.11). Per-ticker `analyze_ticker` scores each name against a peer set of one, so every percentile collapses to ~50 and scores barely differentiate (measured σ≈2.7 vs batch σ≈10) — it can't tell a good ticker from a bad one. The batch call is also ~0.34s/symbol in one round-trip, so it's both more accurate and faster than the old per-candidate loop. The exit engine reads price+score from the same batch; only a held name missing from the batch falls back to `analyze_ticker`.

**Score blend:** a candidate's final score = batch `scan_universe` composite + **sentiment tilt** + **catalyst bonus**. Sentiment tilt (when the bridge is enabled, default) = `(bullish_ratio - bearish_ratio) × 10`, clamped to ±10, letting the news-sentiment page influence what the bot trades (shown in the buy reason, e.g. "score 78 · news +6"). The catalyst bonus (±10, `_catalyst_bonus`) is *directional*: an up-gap on heavy volume lifts a name, a down-gap penalises it (the bot is long-only — don't buy falling knives).

**Quant Decision Engine:**
- **Expected Value (EV):** `EV = (WinRate × AvgWin) - (LossRate × AvgLoss)` — computed from actual closed trades every cycle. Only sizes up when EV is positive.
- **Kelly Criterion:** half-Kelly position sizing from real win/loss statistics. Replaces fixed percentage tiers. Adapts as the bot learns its own edge.
- **Risk of Ruin:** probability of total account loss, computed every cycle. If >5%, position sizes auto-halve to protect the account.
- **Variance tracking:** standard deviation of trade P&L. Positions with unrealized loss >2σ are cut as outliers.
- **Regime adaptation:** `detect_market_regime` actually drives trading (not just logged). The regime scales position size (`risk_factor`: BULL 1.0 → SIDEWAYS 0.85 → HIGH_VOLATILITY 0.6 → BEAR 0.5 → CRASH 0.0) and raises the entry bar (`min_score_bump`: 0 → 3 → 6 → 10 → 100), so the bot presses in strength, shrinks in weakness, and halts new longs in a crash while exits keep running.
- **Conviction scaling:** higher MCP scores → proportionally more capital (score/100 × base risk).
- **Hard 2% cap:** never risk more than 2% of portfolio per trade, regardless of Kelly.
- **Conservative bootstrap:** 1% per trade until 10+ closed trades provide enough data for Kelly (Law of Large Numbers).
- **Single-share floor:** when a Kelly allocation rounds to zero shares, buy one share if a single share still fits the per-trade cap — so the bot can take positions in pricier names instead of only penny stocks. The default account is **$100,000** (standard paper size): at 2%/trade that's ~$2k per position, enough for real share counts across all price ranges (a $10k account can't hold a $300–600 mega-cap inside a 2% cap).

**Smart exits (5 triggers, checked in order):**
1. Hard stop — price dropped 5% from entry (non-negotiable capital protection)
2. Trailing stop — price dropped 3% from peak since entry (lock gains after a run-up)
3. Signal reversal — score dropped 30%+ from entry or below 40
4. Profit taking — score fading 15%+ while position is green (lock gains)
5. Outlier loss — unrealized loss exceeds 2 standard deviations (cut variance)

**Re-entry gate:** a ticker sold recently must score 15+ points above its sell score to be re-bought (prevents churn); sell history expires after 24h.

**Position rotation:** When at 20 max positions, sells weakest holding if a new candidate scores 10+ points higher.

**Persistence & warm start:** `_snapshot_portfolio` writes the portfolio id + realized trade log to `data/bot_state.json` each cycle; `load_state()` restores it once on page load so the bot resumes warm after a Streamlit restart. On a *cold* start (no prior/seeded trades and `warm_start` on, the default), `BotEngine.start()` calls `seed_warm_start()` to populate an illustrative ledger so the Edge panel and equity curve are alive immediately while the first live cycle fills real positions. The seed scales to the chosen capital (`build_seed_trade_log`/`build_seed_equity_curve`, shared with `scripts/seed_demo.py`).

**Configurable params:** the "⚙ Bot Settings" panel exposes starting capital (edit while stopped), max positions, min score, max risk/trade %, the sentiment-bridge toggle, and the warm-start toggle. These live on `BotState` (`max_positions`, `min_score`, `max_risk_per_trade`, `starting_capital`, `sentiment_bridge`, `warm_start`) and are read by the cycle functions.

**Dashboard:** `@st.fragment(run_every=1)` — live-updating panel with:
- Portfolio value, P&L, open positions count
- **Equity curve** (portfolio value over time, persisted in `equity_curve`)
- Edge Statistics: EV/trade, win rate, R:R ratio, half-Kelly %, risk of ruin, streak, σ
- Open positions table with real-time P&L
- **Risk & Exposure** expander (MCP `check_risk`: stress score, scenario drawdowns, sector allocation)
- Activity log streaming buys/sells as they happen

**Ticker Analysis** (above the bot) shows live MCP tools per ticker: candlestick, composite score gauge, news-adjusted score (the bridge), fundamentals/momentum/smart-money cards, and a **Catalysts** expander (`get_sec_filings`, `get_insider_trades`, `get_search_trends`). All per-ticker MCP calls are `st.cache_data`-cached so flipping the chart period stays instant.

---

## MarketPulse page (`app/MarketPulse.py`)

- **Background auto-refresh** (`app/auto_refresh.py`) — a daemon thread runs the RSS→sentiment pipeline every 3 min so data stays fresh without clicking Refresh; the sidebar shows live status + label coverage. Manual "Refresh Data" still forces an immediate run.
- **Browsable grid** — market-mood bar (bullish/bearish split across all tickers), filter-by-text, sentiment filter, and sort (mentions / heating-up / bullish / bearish / A–Z). Each card shows a momentum arrow (Δ vs prior day) and a 7-day sentiment sparkline.
- **Ticker detail** (search or card → `@st.dialog`) — sentiment split bar, **price line** (sentiment-vs-price story), AI verdict, 7-day trend, and clickable headlines.
- **Ticker universe** — `src/extraction/ticker_universe.py` holds 200+ symbol→company mappings; the extractor recognizes any of them, so the grid tracks 100+ news-discovered tickers (not a fixed list).

---

## Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=...       # Optional — enables AI verdict (static fallback without it)
```

News RSS is free — no keys needed.

---

## Running

```powershell
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY if you want AI verdicts
.\start.ps1                   # recommended: opens Chrome, clean Ctrl-C (or start.bat)
```

Deploy: Streamlit Community Cloud, main file `app/MarketPulse.py`. Cloud-host yfinance is
throttled — the MCP server (v0.1.10+) impersonates a browser via `curl_cffi` to mitigate it;
record demos locally for reliability. See `DEMO_RUNBOOK.md`.

---

## Key Design Decisions

- **RSS-only primary source** — no API keys required for core functionality
- **Two clean pages** — MarketPulse (sentiment) and Trading Bot (MCP-powered)
- **Data separation** — home page uses only RSS/SQLite; trading bot uses only MCP server
- **MCP = data layer, bot_engine = decision layer** — the MCP server provides market intelligence; the bot's math determines when to buy/sell and how much to risk
- **SSE transport (critical)** — `financial-mcp` (the published `financial-mcp-server` package) defaults to **stdio**; the app connects over **SSE on :8520**, so it must be launched with `--transport sse`. Both `start.bat` and `mcp_client._start_mcp_server()` pass this flag. Without it, the Trading Bot page shows "MCP server not running."
- **Probability over prediction** — position sizing from actual trade statistics (Kelly Criterion), not gut feel or fixed percentages
- **Many small bets** — 2% max risk per trade, conservative 1% until edge is proven over 10+ trades
- **Ticker detail as dialog** — `@st.dialog` popup, not a separate page
- **Live bot panel** — `@st.fragment(run_every=1)` for real-time updates without page flicker
- **Paper trading** — all trades go through MCP execute_buy/execute_sell on financial_mcp.db
