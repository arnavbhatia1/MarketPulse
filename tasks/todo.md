# MarketPulse × financial-mcp-server — demo prep

## Shipped this session
- [x] **Transport fix** — server launched with `--transport sse` (mcp_client + start.bat); reuse pre-check.
- [x] **Cycle perf** — capped scoring per cycle, dropped redundant scan_universe pre-rank (~41s cycle).
- [x] **Persistence + seed** — bot_state.json survives restarts; scripts/seed_demo.py warm-starts Edge panel.
- [x] **Activity-log $ rendering bug** — Streamlit LaTeX mangled `$…$`; fixed via HTML entity.
- [x] **Seed realism** — correct HH:MM clock, ≤2% position sizing, realistic P&L.
- [x] **Latency** — cached candlestick fetch + all per-ticker MCP calls on Trading Bot tab;
      parallelized RSS ingestion (ThreadPoolExecutor) → Refresh ~50s → ~13s.
- [x] **Ticker variety** — bundled 200+ ticker universe (src/extraction/ticker_universe.py);
      extractor recognizes any of them by cashtag/bare/company-name. Cache 31 → 107 tickers.
      Home page shows up to 150. Bot pool up to 250, sampled per cycle (trades any news ticker over time).
- [x] All 239 tests green.

## DONE — feature build (243 tests green, live-smoke verified)
### MarketPulse tab
- [x] 1. Background auto-refresh scheduler (app/auto_refresh.py) + sidebar live status
- [x] 2. Sortable / filterable ticker grid + text filter + market-mood bar
- [x] 3. Sentiment momentum arrow (Δ vs prior day) + 7-day CSS sparkline per card
- [x] 4. Price line in the detail card (sentiment-vs-price)
- [x] 6. Label coverage in sidebar + market mood bar
### Trading Bot tab
- [x] 1. Equity-curve chart (BotState.equity_curve, persisted, seeded)
- [x] 2. Configurable params (⚙ Bot Settings: capital, max positions, risk cap, min score, bridge)
- [x] 3. Catalysts panel — get_sec_filings / get_insider_trades / get_search_trends (live-verified)
- [x] 4. Sentiment→score bridge (_rss_sentiment_map tilt, visible in buy reason; UI shows adjusted score)
- [x] 5. Risk & Exposure panel (check_risk: stress, scenario drawdowns, sector allocation)

## Future roadmap (see chat) — split by tab
### MarketPulse tab (RSS / sentiment)
- [ ] Background auto-refresh (scheduler) so data is fresh without manual click
- [ ] Sortable/filterable ticker grid (by sentiment, mentions, Δ) + search-as-you-type
- [ ] Sentiment momentum (Δ vs yesterday), volume sparkline per ticker
- [ ] More sources (Reddit, StockTwits) behind the existing BaseIngester
- [ ] Per-ticker price line from yfinance alongside sentiment
- [ ] Confidence/label-coverage surfaced in UI; model retrain cadence

### Trading Bot tab (MCP)
- [ ] Equity curve chart (snapshots already taken) + drawdown
- [ ] Configurable bot params in UI (capital, max positions, risk cap, min score)
- [ ] Use more MCP tools live (SEC filings, insider trades, search trends) in the thesis
- [ ] Sentiment→score bridge: feed RSS sentiment into score_ticker's `sentiment` arg
- [ ] Sector/risk exposure panel (check_risk, analyze_portfolio already available)
- [ ] Backtest mode / trade blotter export
