# MarketPulse × financial-mcp-server — demo prep

## CURRENT — Trading Bot: faster, better, right MCP tools, promising on start

### Diagnosis (empirically confirmed)
- **Broken scoring**: bot scores each candidate via per-ticker `analyze_ticker`, whose
  percentile/sector ranks collapse to ~50 with a peer set of one. Measured spread
  σ=2.7 (everything ~67) vs `scan_universe` batch σ=10.0 (real 45–77). → buys ~randomly.
- **Slow**: ~25 serial `analyze_ticker` (scoring) + up to 20 serial (exits) per cycle.
- **Wasted data**: scanners batch-fetch volume/gap/anomaly catalysts; bot keeps only symbols.
- **Can't fill on $10k**: 1% sizing (~$76) → `int($76/$200)=0` shares → large-caps never bought.
- **CLAUDE.md wrong**: claims scan_universe "~3s/symbol, times out". Measured 0.34s/symbol.

### Plan
FinancialMCP server (shipped → financial-mcp-server 0.1.11 on PyPI):
- [x] `market_data.get_batch_prices(symbols)` — one `yf_download`, last close per symbol
- [x] `engine.score_universe` — attach `price` to each score row (additive, safe)

MarketPulse bot_engine.py:
- [x] `_score_universe_batch(symbols)` — wrap `scan_universe` → `{sym: {score, price}}`
- [x] `_build_dynamic_universe()` → `(symbols, catalyst_map)`; capture vol/gap/anomaly bonus
- [x] `_scan_candidates()` → `(candidates, catalyst_map)`
- [x] `_score_candidates(candidates, batch_scores, catalyst_map, ...)` — base + sentiment + catalyst
- [x] `_check_exits(pid, batch_scores, ...)` — price/score from batch; analyze_ticker fallback
- [x] `_run_cycle` — ONE scan over held ∪ candidates; feed exits + entry
- [x] `_enter_positions` — sizing floor: 1 share when it still fits the per-trade cap
- [x] Warm + fast start: auto-seed illustrative ledger if cold; $100k default capital

Tests + docs:
- [x] Updated test_bot_engine.py for new signatures (+ batch-path & catalyst tests)
- [x] Updated CLAUDE.md; both suites green (245 MarketPulse + 13 FinancialMCP offline)

### Review (live-verified)
- Scoring spread fixed: per-ticker σ≈2.7 (all ~67) → batch σ≈10 (real 45–79 ranking).
- Full `_run_cycle` cold ≈27s (3 scanners + regime + batch scan of 44 + trades);
  one scan_universe round-trip replaces ~45 serial analyze_ticker calls.
- At $100k it fills **19 positions across all price ranges** (CRWD $782 … F $17),
  deploying ~$35k — was filling only sub-$200 penny names on $10k.
- Warm seed: 14 trades, 64% win, +EV, climbing equity curve — Edge panel never empty.
- Shipped FinancialMCP 0.1.11 first (PyPI build confirmed), then bumped the pin.

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
