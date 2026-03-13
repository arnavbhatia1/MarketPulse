# Trading Bot MCP Integration — Design Spec

**Date:** 2026-03-13
**Status:** Approved
**Feature:** Integrate `financial-mcp-server` as the backend for MarketPulse's trading bot, replacing custom investor modules with MCP tool calls over SSE.

---

## Overview

Replace the custom `src/investor/` modules (~2,000 lines) with a single MCP client that connects to the `financial-mcp-server` (33 tools) over SSE. The MCP server becomes the single source of truth for portfolio state, scoring, risk, and market intelligence. The Streamlit app becomes a pure UI layer.

**Key decisions:**
- MCP client over SSE (`http://localhost:8520/sse`) — proper protocol architecture
- No sentiment in scoring — pure quant (fundamentals + momentum + risk)
- MCP server's SQLite DB (`data/financial_mcp.db`) owns all portfolio/trading state
- No Claude review layer — `run_rebalance` auto-executes trades
- No auth/paywall — single user, direct access
- MarketPulse home page = lite teaser (market regime, top movers, anomalies)
- Full trading terminal = new dedicated page with portfolio management + market intelligence

---

## Section 1: MCP Client Module

A single file `src/investor/mcp_client.py` replaces all 6 existing investor modules.

### Connection Management

- Uses the `mcp` Python SDK (`mcp.client.sse.sse_client` + `mcp.client.session.ClientSession`)
- Connects to `http://localhost:8520/sse` (configurable via `config/default.yaml`)
- Background thread keeps the SSE connection alive for the Streamlit app lifetime
- Sync wrappers submit tool calls to the background thread via a thread-safe queue
- Reconnection: on any exception from `call_tool`, tear down the connection and re-establish

### Client Pattern

The MCP SDK uses async context managers (`sse_client` yields read/write streams, `ClientSession` wraps them). Since Streamlit runs its own event loop, the MCP connection lives in a dedicated background thread running `anyio`.

```python
import json
import threading
import queue
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

_call_queue: queue.Queue = queue.Queue()
_result_queues: dict[int, queue.Queue] = {}
_thread: threading.Thread | None = None
_connected = threading.Event()

def _run_mcp_loop(url: str):
    """Background thread: keeps SSE connection alive, processes tool calls."""
    import anyio

    async def _loop():
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                _connected.set()
                while True:
                    call_id, tool_name, kwargs = _call_queue.get()
                    try:
                        result = await session.call_tool(tool_name, kwargs)
                        parsed = json.loads(result.content[0].text)
                        _result_queues[call_id].put(("ok", parsed))
                    except Exception as e:
                        _result_queues[call_id].put(("error", {"error": str(e)}))

    anyio.run(_loop)

def _ensure_connected():
    """Start background thread if not running."""
    global _thread
    if _thread is None or not _thread.is_alive():
        _connected.clear()
        url = config["mcp_server"]["url"]
        _thread = threading.Thread(target=_run_mcp_loop, args=(url,), daemon=True)
        _thread.start()
        if not _connected.wait(timeout=config["mcp_server"]["timeout"]):
            raise ConnectionError("Could not connect to financial-mcp server")

def call_tool(tool_name: str, **kwargs) -> dict:
    """Sync wrapper — submits tool call to background thread, blocks for result."""
    _ensure_connected()
    call_id = id(threading.current_thread())
    _result_queues[call_id] = queue.Queue()
    _call_queue.put((call_id, tool_name, kwargs))
    status, result = _result_queues[call_id].get(timeout=config["mcp_server"]["timeout"])
    del _result_queues[call_id]
    if status == "error":
        return result
    return result
```

**Reconnection strategy:** If the background thread dies (SSE disconnect, server restart), `_ensure_connected()` detects `not _thread.is_alive()` and spawns a new one. Tool calls that were in-flight get an error response.

### Function Surface

| MCP Client Function | MCP Tool Called | Replaces |
|---|---|---|
| `score_ticker(symbol)` | `score_ticker` | `engine.score_ticker()` |
| `scan_universe(symbols: list[str])` | `scan_universe` (joins to comma-separated string) | `engine.score_universe()` |
| `get_fundamentals(symbol)` | `get_fundamentals` | `market_data.get_fundamentals()` |
| `get_momentum(symbol)` | `get_momentum` | `market_data.get_momentum_signals()` |
| `get_price(symbol)` | `get_price` | `market_data.get_current_price()` |
| `execute_buy(portfolio_id, symbol, shares)` | `execute_buy` | `PaperBroker.execute_buy()` |
| `execute_sell(portfolio_id, symbol, shares)` | `execute_sell` | `PaperBroker.execute_sell()` |
| `create_portfolio(capital, risk_profile, investment_horizon, name)` | `create_portfolio` | `portfolio.create_user_portfolio()` |
| `analyze_portfolio(portfolio_id)` | `analyze_portfolio` | `portfolio.get_portfolio_summary()` |
| `check_risk(portfolio_id)` | `check_risk` | `risk.compute_stress_score()` |
| `run_rebalance(portfolio_id, trigger, symbols)` | `run_rebalance` | `rebalancer.run_rebalance()` |
| `get_holdings(portfolio_id)` | `get_holdings` | (new) |
| `get_trades(portfolio_id, status)` | `get_trades` | (new) |
| `analyze_ticker(symbol)` | `analyze_ticker` | (new) |
| `detect_market_regime()` | `detect_market_regime` | (new) |
| `get_vix_analysis()` | `get_vix_analysis` | (new) |
| `scan_anomalies(symbols: list[str])` | `scan_anomalies` (joins to comma-separated string) | (new) |
| `scan_volume_leaders(symbols: list[str])` | `scan_volume_leaders` (joins to comma-separated string) | (new) |
| `scan_gap_movers(symbols: list[str])` | `scan_gap_movers` (joins to comma-separated string) | (new) |
| `get_smart_money_signal(market)` | `get_smart_money_signal` | (new) |
| `get_futures_positioning(market)` | `get_futures_positioning` | (new) |

### Error Handling

Every function returns a dict. On MCP failure, returns `{"error": "..."}`. UI checks for the `error` key before rendering.

---

## Section 2: MarketPulse Home Page (Lite Teaser)

The existing home page (`app/MarketPulse.py`) gets a **Market Intelligence** section — powered by MCP tools, read-only, no trading.

### What the Lite Teaser Shows

1. **Market Regime Banner** — Full-width card at the top. Regime label (BULL/BEAR/SIDEWAYS/HIGH_VOLATILITY/CRASH), VIX level + fear signal, one-line recommendation. Source: `detect_market_regime()` + `get_vix_analysis()`.

2. **Top Movers Grid** — 6 cards in a 3x2 grid. Ticker, price, daily % change, anomaly badges (52W High, Volume Spike, Gap Up/Down). Source: `scan_anomalies()` + `scan_volume_leaders()` + `scan_gap_movers()`, merged and sorted by anomaly score.

3. **"Powered by financial-mcp" branding** — small badge linking to the full trading terminal page.

### What It Does NOT Show

- No portfolio, holdings, or trade execution
- No fundamentals/momentum score cards
- No smart money / CFTC positioning
- No rebalance controls

### Existing Functionality Preserved

- Search bar + sentiment briefing cards stay as-is
- Market sentiment grid from `ticker_cache` stays
- Pipeline refresh button stays

### Layout

```
┌─────────────────────────────────────────────────┐
│  Market Regime: BULL MARKET  │  VIX: 14.2 Low   │
├─────────────────────────────────────────────────┤
│  [Search bar — existing]                         │
├─────────────────────────────────────────────────┤
│  Top Movers  (6 cards with anomaly badges)       │
│  AAPL +3.2% [52W High]  │  TSLA +6.6%           │
│  NVDA +5.6% [Vol Spike] │  MSFT -2.1% [Vol Spike│
├─────────────────────────────────────────────────┤
│  Market Sentiment Grid — existing from pipeline  │
├─────────────────────────────────────────────────┤
│  ⚡ Powered by financial-mcp → Full Terminal     │
└─────────────────────────────────────────────────┘
```

### MCP Connection Handling

If the MCP server isn't running, the lite section gracefully hides with a subtle "Market intelligence unavailable" note. Existing sentiment features work independently.

---

## Section 3: Full Trading Terminal Page

New page `app/pages/2_Trading_Bot.py` — the complete trading experience.

### Layout — Three Zones

```
┌──────────────────────────────────────────────────────────────┐
│  Powered by financial-mcp    [1D 1W 1M 3M 1Y]    🔍 Search  │
├────────────┬───────────────────────────┬─────────────────────┤
│            │                           │                     │
│  Market    │   Ticker Detail           │   Top Movers        │
│  Regime    │   (selected ticker)       │   6 cards with      │
│  Indicator │   Candlestick chart       │   anomaly badges    │
│  + VIX     │   + annotations           │                     │
│            │                           │                     │
├────────────┴──────────┬────────────────┴─────────────────────┤
│  Fundamentals         │  Momentum          │  Smart Money     │
│  Score: 78/100        │  Score: 85/100     │  CFTC Positioning│
│  PE, EV/EBITDA, etc   │  30D, 90D, Vol, RS │  Commercial/Ret  │
├───────────────────────┴────────────────┴─────────────────────┤
│                    PORTFOLIO SECTION                          │
├──────────────────┬──────────────────┬────────────────────────┤
│  Portfolio Value │  Holdings Table  │  Rebalance Controls    │
│  Daily P&L       │  (sortable)      │  [Rebalance Now]       │
│  Cash remaining  │                  │  Risk profile selector │
├──────────────────┴──────────────────┴────────────────────────┤
│  Trade History (filterable table)                             │
│  Risk Assessment — stress gauge + sector allocation          │
└──────────────────────────────────────────────────────────────┘
```

### Top Zone — Market Intelligence

Loaded on page load, cached with TTL.

- **Market Regime card:** `detect_market_regime()` — regime label, score, recommendation, signal breakdown
- **VIX card:** `get_vix_analysis()` — level, percentile, fear signal, term structure
- **Ticker search:** Select a ticker to populate the detail panel
- **Ticker detail:** `analyze_ticker(symbol)` — price, fundamentals, momentum, composite score
- **Top Movers:** `scan_anomalies()` + `scan_volume_leaders()` + `scan_gap_movers()`

### Middle Zone — Ticker Score Cards

Shown when a ticker is selected.

- **Fundamentals card:** `get_fundamentals(symbol)` — PE, EV/EBITDA, P/B, div yield, market cap. Score from `score_ticker()`.
- **Momentum card:** `get_momentum(symbol)` — 30D/90D return, volatility, relative strength, max drawdown. Score from `score_ticker()`.
- **Smart Money card:** `get_smart_money_signal(market)` — commercial net position, percentile, bullish/bearish signal. `get_futures_positioning(market)` for the bar visualization.

### Bottom Zone — Portfolio Management

- **Portfolio setup:** First-time onboarding — capital slider ($10k-$1M), risk profile radio (`conservative`/`moderate`/`aggressive`), investment horizon dropdown (`short`/`medium`/`long`). Calls `create_portfolio(starting_capital, risk_profile, investment_horizon, name)`.
- **Portfolio summary:** `analyze_portfolio(portfolio_id)` — total value, daily change, holdings with weights/gains, sector allocation, geo allocation, performance metrics.
- **Holdings table:** Sortable by symbol, shares, cost basis, current value, gain/loss %, weight, sector.
- **Rebalance controls:** "Rebalance Now" button → `run_rebalance(portfolio_id)`. Shows results: trades executed, buy/sell signals.
- **Trade history:** `get_trades(portfolio_id)` — filterable by status (executed/proposed/rejected), sortable by date.
- **Risk panel:** `check_risk(portfolio_id)` — stress score gauge, scenario drawdowns (2008/2020/2022), sector allocation bars.

### Candlestick Chart

The MCP server doesn't expose raw OHLC data. For the candlestick chart, call `yfinance` directly from the UI (same dependency, already installed). This is the one place that doesn't go through MCP.

### State Management

- `portfolio_id` stored in `session_state` and persisted to `data/portfolio_id.txt` (survives app restarts)
- Selected ticker in `session_state`
- MCP data cached via `@st.cache_data(ttl=...)` per data type

---

## Section 4: Files Changed & Deleted

### New Files

| File | Purpose |
|---|---|
| `src/investor/mcp_client.py` | MCP client — all tool calls, connection management, sync wrappers |
| `app/pages/2_Trading_Bot.py` | Full trading terminal page |
| `app/components/trading_charts.py` | Candlestick chart, score gauges, stress gauge, CFTC bars, anomaly badge cards |

### Modified Files

| File | Change |
|---|---|
| `app/MarketPulse.py` | Add lite market intelligence section (regime banner, top movers, "powered by" link). Remove auth gate: delete `show_login_form`/`auth_guard` import, remove login/register form block, remove sidebar user info/sign out block, remove guest mode logic. |
| `app/components/styles.py` | New CSS for trading terminal cards, regime banner, anomaly badges, score gauges |
| `config/default.yaml` | Remove entire `investor:` config block. Add: `mcp_server: { url: "http://localhost:8520/sse", timeout: 30, rebalance_timeout: 120 }` |
| `src/investor/__init__.py` | Re-export mcp_client functions |
| `requirements.txt` | Add `financial-mcp-server` and `mcp` (client SDK). Remove `bcrypt` (auth removed). |
| `CLAUDE.md` | Update architecture diagram, project structure, key modules, and SQLite schema sections to reflect MCP integration. Remove investor module descriptions, add MCP client description. Remove auth references. |

### Deleted Files

| File | Reason |
|---|---|
| `src/investor/engine.py` | Replaced by `score_ticker` / `scan_universe` MCP tools |
| `src/investor/market_data.py` | Replaced by `get_fundamentals` / `get_momentum` / `get_price` MCP tools |
| `src/investor/broker.py` | Replaced by `execute_buy` / `execute_sell` MCP tools |
| `src/investor/portfolio.py` | Replaced by `create_portfolio` / `analyze_portfolio` MCP tools |
| `src/investor/risk.py` | Replaced by `check_risk` MCP tool |
| `src/investor/rebalancer.py` | Replaced by `run_rebalance` MCP tool |
| `src/auth/auth.py` | No auth — single user |
| `src/auth/__init__.py` | Package removed |
| `src/agent/reviewer.py` | No Claude review layer |
| `app/components/auth_guard.py` | No auth |
| `app/pages/2_Portfolio.py` | Consolidated into `2_Trading_Bot.py` |
| `app/pages/3_Trades.py` | Consolidated into `2_Trading_Bot.py` |
| `scripts/rebalance.py` | Rebalance triggered via MCP tool call from UI |

### Kept Unchanged

| File | Reason |
|---|---|
| `app/pages/1_Ticker_Detail.py` | Sentiment-focused, independent of trading bot |
| `src/ingestion/*` | Sentiment pipeline unchanged |
| `src/labeling/*` | Labeling unchanged |
| `src/models/*` | ML pipeline unchanged |
| `src/storage/db.py` | Modified — see DB Migration section below |
| `src/agent/briefing.py` | Claude verdict on ticker search — independent |

---

## Section 5: DB Migration

### Changes to `src/storage/db.py`

**Remove from `init_db()`:**
- `CREATE TABLE IF NOT EXISTS users`
- `CREATE TABLE IF NOT EXISTS portfolios`
- `CREATE TABLE IF NOT EXISTS holdings`
- `CREATE TABLE IF NOT EXISTS trades`
- `CREATE TABLE IF NOT EXISTS portfolio_snapshots`
- `CREATE TABLE IF NOT EXISTS etf_universe`
- `_seed_etf_universe()` call
- `PRAGMA foreign_keys = ON` (only needed for investor tables)

**Delete these functions from `db.py`:**
- `create_user()`, `get_user_by_email()`, `get_user_by_id()`
- `create_portfolio()`, `get_portfolio()`, `update_portfolio()`
- `get_holdings()`, `upsert_holding()`, `delete_holding()`
- `save_trade()`, `get_trades()`, `update_trade()`
- `save_snapshot()`, `get_snapshots()`
- `get_etf_universe()`, `_seed_etf_universe()`

**Keep:**
- `init_db()` (with only `posts`, `ticker_cache`, `model_training_log` table creation)
- All sentiment/ingestion-related functions

**Existing data:** Old tables are left in place (no `DROP TABLE`). They become inert — nothing reads or writes to them. Users can delete `marketpulse.db` and re-run to get a clean schema if desired.

---

## Section 6: Data Flow & Error Handling

### Startup Sequence

```
1. User starts MCP server: financial-mcp (or it's already running)
2. User launches: streamlit run app/MarketPulse.py
3. On first page load, mcp_client.py establishes SSE connection
4. If connection fails → lite section hidden, trading terminal shows "Start the MCP server" message
5. If connection succeeds → tools available, data flows
```

### Caching Strategy

| Data | TTL | Reason |
|---|---|---|
| Market regime, VIX | 5 min | Changes slowly, expensive signal computation |
| Anomalies, volume leaders, gap movers | 2 min | More volatile during market hours |
| Fundamentals, momentum, score | 60 sec | Per-ticker, refreshed on selection |
| Portfolio summary, holdings, trades | No cache | Always fresh — user expects immediate feedback after rebalance |
| Risk assessment | 60 sec | Recomputed on demand |

All caching via `@st.cache_data(ttl=...)`.

### Error Handling — Three Tiers

1. **MCP server down** — `get_session()` raises `ConnectionError`. UI shows a clear "MCP server not running" banner with instructions. All MCP-dependent sections collapse gracefully. Existing sentiment features work independently.

2. **Individual tool failure** — MCP tool returns `{"error": "..."}`. UI shows inline error on that specific card/section (e.g., "Could not load fundamentals for XYZZ"). Other sections unaffected.

3. **Timeout** — 30 second timeout per tool call (configurable). On timeout, same behavior as tool failure. `run_rebalance` gets a longer timeout (120 sec) since it scores a full universe.

### Portfolio Persistence

- `portfolio_id` saved to `data/portfolio_id.txt` on creation
- On app load, read from file → `session_state`
- If file missing → show onboarding setup
- If portfolio_id exists but MCP server returns error (e.g., DB was wiped) → show onboarding again

---

## Section 7: Testing Strategy

### Deleted Tests

`tests/test_investor.py` — tests the old `src/investor/` modules and `src/auth/` via `db.py` fixtures. Delete entirely since the modules it tests no longer exist.

### New Tests

**`tests/test_mcp_client.py`** — Tests for `src/investor/mcp_client.py`:
- Mock the MCP server responses (patch `call_tool` to return canned JSON)
- Test each wrapper function: correct tool name called, correct parameter serialization (list → comma-separated string), correct return type
- Test error handling: `{"error": "..."}` responses returned correctly
- Test connection failure: `ConnectionError` raised when server unreachable

**`tests/test_trading_bot_page.py`** — Streamlit page tests:
- Test portfolio onboarding flow (mock MCP `create_portfolio`)
- Test that MCP server down shows graceful fallback, not crash
- Test `portfolio_id` persistence to `data/portfolio_id.txt`

**Integration test (manual / CI optional):**
- Start `financial-mcp` server
- Run `create_portfolio` → `run_rebalance` → `get_holdings` → `get_trades` end-to-end
- Verify portfolio state is consistent

### Existing Tests

All other tests in `tests/` (sentiment, ingestion, labeling, models, storage) remain unchanged. The DB test fixtures should be updated to remove investor-related table assertions if any exist outside `test_investor.py`.

---

## What's NOT Included

- No sentiment integration in scoring — MCP server scores on fundamentals + momentum + risk only
- No Claude review layer — trades auto-execute via `run_rebalance`
- No auth/users/premium gating — single user, direct access
- No real brokerage — paper trading only (MCP server's `PaperBroker`)
- No scheduled rebalancing — manual "Rebalance Now" button only (cron can be added later)
- No advisory mode — all trades auto-execute (autopilot only)
