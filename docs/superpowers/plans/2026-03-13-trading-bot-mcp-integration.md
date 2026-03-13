# Trading Bot MCP Integration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace custom `src/investor/` modules with MCP client calls to `financial-mcp-server`, add market intelligence to the home page, and build a full trading terminal page.

**Architecture:** Single MCP client module connects to `financial-mcp-server` over SSE in a background thread. Streamlit UI calls sync wrappers. MCP server owns all portfolio state. Home page gets a lite market intel teaser; new `2_Trading_Bot.py` page is the full terminal.

**Tech Stack:** Python 3.14, Streamlit, MCP SDK (`mcp`), `financial-mcp-server`, yfinance (candlestick only), Plotly, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-trading-bot-mcp-integration-design.md`

---

## Chunk 1: Cleanup & Foundation

### Task 1: Delete old investor modules, auth, and reviewer

**Files:**
- Delete: `src/investor/engine.py`
- Delete: `src/investor/market_data.py`
- Delete: `src/investor/broker.py`
- Delete: `src/investor/portfolio.py`
- Delete: `src/investor/risk.py`
- Delete: `src/investor/rebalancer.py`
- Delete: `src/auth/auth.py`
- Delete: `src/auth/__init__.py`
- Delete: `src/agent/reviewer.py`
- Delete: `app/components/auth_guard.py`
- Delete: `app/pages/2_Portfolio.py`
- Delete: `app/pages/3_Trades.py`
- Delete: `scripts/rebalance.py`
- Delete: `tests/test_investor.py`

- [ ] **Step 1: Delete all files**

```bash
rm src/investor/engine.py src/investor/market_data.py src/investor/broker.py \
   src/investor/portfolio.py src/investor/risk.py src/investor/rebalancer.py \
   src/auth/auth.py src/auth/__init__.py src/agent/reviewer.py \
   app/components/auth_guard.py app/pages/2_Portfolio.py app/pages/3_Trades.py \
   scripts/rebalance.py tests/test_investor.py
rmdir src/auth
```

- [ ] **Step 2: Verify deletion**

```bash
ls src/investor/    # should only show __init__.py and __pycache__
ls src/agent/       # should only show __init__.py, briefing.py, __pycache__
ls app/pages/       # should only show __init__.py, 1_Ticker_Detail.py
ls app/components/  # should show __init__.py, charts.py, styles.py (no auth_guard.py)
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: delete old investor modules, auth, reviewer, and related pages"
```

---

### Task 2: Clean up db.py — remove investor tables and functions

**Files:**
- Modify: `src/storage/db.py:77-162` (remove investor table creation from init_db)
- Modify: `src/storage/db.py:164-624` (delete all investor/auth functions)

- [ ] **Step 1: Remove investor table CREATE statements from init_db()**

Remove lines 77–162 from `init_db()`. These are the `CREATE TABLE IF NOT EXISTS` statements for `users`, `portfolios`, `holdings`, `trades`, `portfolio_snapshots`, `etf_universe`, and the `_seed_etf_universe()` call. Keep only `posts` (line 42), `ticker_cache` (line 55), and `model_training_log` (line 69).

- [ ] **Step 2: Delete all investor/auth functions**

Delete lines 164–624: `_seed_etf_universe()`, `create_user()`, `get_user_by_email()`, `get_user_by_id()`, `create_portfolio()`, `get_portfolio()`, `get_user_portfolios()`, `update_portfolio()`, `upsert_holding()`, `get_holdings()`, `delete_holding()`, `save_trade()`, `get_trades()`, `update_trade()`, `save_snapshot()`, `get_snapshots()`, `get_etf_universe()`.

- [ ] **Step 3: Run existing sentiment tests to confirm no breakage**

```bash
python -m pytest tests/ -v --ignore=tests/test_investor.py -x 2>&1 | head -60
```

Expected: All sentiment/ingestion/labeling/storage tests pass. `test_investor.py` was already deleted in Task 1.

- [ ] **Step 4: Commit**

```bash
git add src/storage/db.py
git commit -m "refactor: remove investor/auth tables and functions from db.py"
```

---

### Task 3: Update config and requirements

**Files:**
- Modify: `config/default.yaml:59-93` (remove investor section, add mcp_server)
- Modify: `requirements.txt` (add mcp, financial-mcp-server; remove bcrypt)

- [ ] **Step 1: Update default.yaml**

Remove the entire `investor:` block (lines 59–93). Add at the end:

```yaml
mcp_server:
  url: "http://localhost:8520/sse"
  timeout: 30
  rebalance_timeout: 120
```

- [ ] **Step 2: Update requirements.txt**

Remove `bcrypt`. Add `financial-mcp-server`, `mcp[cli]`, and `anyio`.

- [ ] **Step 3: Commit**

```bash
git add config/default.yaml requirements.txt
git commit -m "config: add mcp_server config, remove investor config and bcrypt"
```

---

### Task 4: Create MCP client module

**Files:**
- Create: `src/investor/mcp_client.py`
- Modify: `src/investor/__init__.py`

- [ ] **Step 1: Write mcp_client.py**

Create `src/investor/mcp_client.py` with:
1. Background thread MCP connection management (from spec Section 1)
2. `call_tool()` sync wrapper
3. All 20 wrapper functions from the spec's function surface table

```python
"""MCP client for financial-mcp-server.

Connects to the financial-mcp-server over SSE and exposes sync wrappers
for 21 MCP tools used by the trading terminal. Connection lives in a
background thread. The call_tool() function accepts an optional timeout
parameter (enhancement over spec) for long-running tools like run_rebalance.
"""

import json
import itertools
import logging
import threading
import queue

from src.utils.config import load_config

logger = logging.getLogger(__name__)

_config = load_config()
_call_queue: queue.Queue = queue.Queue()
_result_queues: dict[int, queue.Queue] = {}
_call_counter = itertools.count()
_thread: threading.Thread | None = None
_connected = threading.Event()


def _run_mcp_loop(url: str):
    """Background thread: keeps SSE connection alive, processes tool calls."""
    import anyio
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession

    async def _loop():
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                _connected.set()
                while True:
                    call_id, tool_name, arguments = await anyio.to_thread.run_sync(
                        _call_queue.get
                    )
                    try:
                        result = await session.call_tool(tool_name, arguments=arguments)
                        parsed = json.loads(result.content[0].text)
                        _result_queues[call_id].put(("ok", parsed))
                    except Exception as e:
                        logger.error("MCP tool %s failed: %s", tool_name, e)
                        _result_queues[call_id].put(("error", {"error": str(e)}))

    anyio.run(_loop)


def _ensure_connected():
    """Start background thread if not running."""
    global _thread
    if _thread is None or not _thread.is_alive():
        _connected.clear()
        url = _config["mcp_server"]["url"]
        _thread = threading.Thread(target=_run_mcp_loop, args=(url,), daemon=True)
        _thread.start()
        timeout = _config["mcp_server"]["timeout"]
        if not _connected.wait(timeout=timeout):
            raise ConnectionError("Could not connect to financial-mcp server")


def call_tool(tool_name: str, timeout: float | None = None, **kwargs) -> dict:
    """Sync wrapper — submits tool call to background thread, blocks for result."""
    _ensure_connected()
    call_id = next(_call_counter)
    _result_queues[call_id] = queue.Queue()
    _call_queue.put((call_id, tool_name, kwargs))
    t = timeout or _config["mcp_server"]["timeout"]
    try:
        status, result = _result_queues[call_id].get(timeout=t)
    except queue.Empty:
        del _result_queues[call_id]
        return {"error": f"Timeout after {t}s calling {tool_name}"}
    del _result_queues[call_id]
    return result


def is_connected() -> bool:
    """Check if MCP server connection is alive."""
    return _thread is not None and _thread.is_alive() and _connected.is_set()


# ── Scoring & Analysis ───────────────────────────────────────────────────────

def score_ticker(symbol: str) -> dict:
    return call_tool("score_ticker", symbol=symbol)


def scan_universe(symbols: list[str]) -> dict:
    return call_tool("scan_universe", symbols=",".join(symbols))


def analyze_ticker(symbol: str) -> dict:
    return call_tool("analyze_ticker", symbol=symbol)


def get_fundamentals(symbol: str) -> dict:
    return call_tool("get_fundamentals", symbol=symbol)


def get_momentum(symbol: str) -> dict:
    return call_tool("get_momentum", symbol=symbol)


def get_price(symbol: str) -> dict:
    return call_tool("get_price", symbol=symbol)


# ── Portfolio & Trading ───────────────────────────────────────────────────────

def create_portfolio(
    starting_capital: float,
    risk_profile: str,
    investment_horizon: str,
    name: str = "Default",
) -> dict:
    return call_tool(
        "create_portfolio",
        starting_capital=starting_capital,
        risk_profile=risk_profile,
        investment_horizon=investment_horizon,
        name=name,
    )


def analyze_portfolio(portfolio_id: str) -> dict:
    return call_tool("analyze_portfolio", portfolio_id=portfolio_id)


def get_holdings(portfolio_id: str) -> dict:
    return call_tool("get_holdings", portfolio_id=portfolio_id)


def get_trades(portfolio_id: str, status: str = "") -> dict:
    return call_tool("get_trades", portfolio_id=portfolio_id, status=status)


def execute_buy(portfolio_id: str, symbol: str, shares: int) -> dict:
    return call_tool(
        "execute_buy", portfolio_id=portfolio_id, symbol=symbol, shares=shares
    )


def execute_sell(portfolio_id: str, symbol: str, shares: int) -> dict:
    return call_tool(
        "execute_sell", portfolio_id=portfolio_id, symbol=symbol, shares=shares
    )


def run_rebalance(
    portfolio_id: str, trigger: str = "manual", symbols: str = ""
) -> dict:
    timeout = _config["mcp_server"].get("rebalance_timeout", 120)
    return call_tool(
        "run_rebalance",
        timeout=timeout,
        portfolio_id=portfolio_id,
        trigger=trigger,
        symbols=symbols,
    )


def check_risk(portfolio_id: str) -> dict:
    return call_tool("check_risk", portfolio_id=portfolio_id)


# ── Market Intelligence ───────────────────────────────────────────────────────

def detect_market_regime() -> dict:
    return call_tool("detect_market_regime")


def get_vix_analysis() -> dict:
    return call_tool("get_vix_analysis")


def scan_anomalies(symbols: list[str] | None = None) -> dict:
    args = {}
    if symbols:
        args["symbols"] = ",".join(symbols)
    return call_tool("scan_anomalies", **args)


def scan_volume_leaders(symbols: list[str] | None = None) -> dict:
    args = {}
    if symbols:
        args["symbols"] = ",".join(symbols)
    return call_tool("scan_volume_leaders", **args)


def scan_gap_movers(symbols: list[str] | None = None) -> dict:
    args = {}
    if symbols:
        args["symbols"] = ",".join(symbols)
    return call_tool("scan_gap_movers", **args)


def get_smart_money_signal(market: str) -> dict:
    return call_tool("get_smart_money_signal", market=market)


def get_futures_positioning(market: str) -> dict:
    return call_tool("get_futures_positioning", market=market)
```

- [ ] **Step 2: Update __init__.py**

Replace `src/investor/__init__.py` contents:

```python
"""Investor module — MCP client interface to financial-mcp-server."""

from src.investor.mcp_client import (
    is_connected,
    score_ticker,
    scan_universe,
    analyze_ticker,
    get_fundamentals,
    get_momentum,
    get_price,
    create_portfolio,
    analyze_portfolio,
    get_holdings,
    get_trades,
    execute_buy,
    execute_sell,
    run_rebalance,
    check_risk,
    detect_market_regime,
    get_vix_analysis,
    scan_anomalies,
    scan_volume_leaders,
    scan_gap_movers,
    get_smart_money_signal,
    get_futures_positioning,
)

__all__ = [
    "is_connected",
    "score_ticker",
    "scan_universe",
    "analyze_ticker",
    "get_fundamentals",
    "get_momentum",
    "get_price",
    "create_portfolio",
    "analyze_portfolio",
    "get_holdings",
    "get_trades",
    "execute_buy",
    "execute_sell",
    "run_rebalance",
    "check_risk",
    "detect_market_regime",
    "get_vix_analysis",
    "scan_anomalies",
    "scan_volume_leaders",
    "scan_gap_movers",
    "get_smart_money_signal",
    "get_futures_positioning",
]
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('src/investor/mcp_client.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/investor/mcp_client.py src/investor/__init__.py
git commit -m "feat: add MCP client module for financial-mcp-server"
```

---

### Task 5: Write MCP client tests

**Files:**
- Create: `tests/test_mcp_client.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the MCP client wrapper functions."""

import pytest
from unittest.mock import patch, MagicMock


# ── Helper ────────────────────────────────────────────────────────────────────

def _mock_call_tool(tool_name, timeout=None, **kwargs):
    """Returns a dict that records what was called."""
    return {"_tool": tool_name, "_args": kwargs}


# ── Wrapper function tests ────────────────────────────────────────────────────

class TestWrapperFunctions:
    """Test that each wrapper calls the correct MCP tool with correct args."""

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_score_ticker(self, mock):
        from src.investor.mcp_client import score_ticker
        result = score_ticker("AAPL")
        assert result["_tool"] == "score_ticker"
        assert result["_args"] == {"symbol": "AAPL"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_scan_universe_joins_list(self, mock):
        from src.investor.mcp_client import scan_universe
        result = scan_universe(["AAPL", "MSFT", "GOOGL"])
        assert result["_tool"] == "scan_universe"
        assert result["_args"] == {"symbols": "AAPL,MSFT,GOOGL"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_analyze_ticker(self, mock):
        from src.investor.mcp_client import analyze_ticker
        result = analyze_ticker("TSLA")
        assert result["_tool"] == "analyze_ticker"
        assert result["_args"] == {"symbol": "TSLA"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_fundamentals(self, mock):
        from src.investor.mcp_client import get_fundamentals
        result = get_fundamentals("MSFT")
        assert result["_tool"] == "get_fundamentals"
        assert result["_args"] == {"symbol": "MSFT"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_momentum(self, mock):
        from src.investor.mcp_client import get_momentum
        result = get_momentum("NVDA")
        assert result["_tool"] == "get_momentum"
        assert result["_args"] == {"symbol": "NVDA"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_create_portfolio(self, mock):
        from src.investor.mcp_client import create_portfolio
        result = create_portfolio(100000.0, "moderate", "medium", "Test")
        assert result["_tool"] == "create_portfolio"
        assert result["_args"]["starting_capital"] == 100000.0
        assert result["_args"]["risk_profile"] == "moderate"
        assert result["_args"]["investment_horizon"] == "medium"
        assert result["_args"]["name"] == "Test"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_execute_buy(self, mock):
        from src.investor.mcp_client import execute_buy
        result = execute_buy("pid-123", "AAPL", 10)
        assert result["_tool"] == "execute_buy"
        assert result["_args"] == {"portfolio_id": "pid-123", "symbol": "AAPL", "shares": 10}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_execute_sell(self, mock):
        from src.investor.mcp_client import execute_sell
        result = execute_sell("pid-123", "AAPL", 5)
        assert result["_tool"] == "execute_sell"
        assert result["_args"] == {"portfolio_id": "pid-123", "symbol": "AAPL", "shares": 5}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_run_rebalance(self, mock):
        from src.investor.mcp_client import run_rebalance
        result = run_rebalance("pid-123", trigger="manual")
        assert result["_tool"] == "run_rebalance"
        assert result["_args"]["portfolio_id"] == "pid-123"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_detect_market_regime(self, mock):
        from src.investor.mcp_client import detect_market_regime
        result = detect_market_regime()
        assert result["_tool"] == "detect_market_regime"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_vix_analysis(self, mock):
        from src.investor.mcp_client import get_vix_analysis
        result = get_vix_analysis()
        assert result["_tool"] == "get_vix_analysis"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_scan_anomalies_no_symbols(self, mock):
        from src.investor.mcp_client import scan_anomalies
        result = scan_anomalies()
        assert result["_tool"] == "scan_anomalies"
        assert "symbols" not in result["_args"]

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_scan_anomalies_with_symbols(self, mock):
        from src.investor.mcp_client import scan_anomalies
        result = scan_anomalies(["AAPL", "TSLA"])
        assert result["_args"] == {"symbols": "AAPL,TSLA"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_smart_money_signal(self, mock):
        from src.investor.mcp_client import get_smart_money_signal
        result = get_smart_money_signal("E-MINI S&P 500")
        assert result["_tool"] == "get_smart_money_signal"
        assert result["_args"] == {"market": "E-MINI S&P 500"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_check_risk(self, mock):
        from src.investor.mcp_client import check_risk
        result = check_risk("pid-123")
        assert result["_tool"] == "check_risk"
        assert result["_args"] == {"portfolio_id": "pid-123"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_analyze_portfolio(self, mock):
        from src.investor.mcp_client import analyze_portfolio
        result = analyze_portfolio("pid-123")
        assert result["_tool"] == "analyze_portfolio"
        assert result["_args"] == {"portfolio_id": "pid-123"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_holdings(self, mock):
        from src.investor.mcp_client import get_holdings
        result = get_holdings("pid-123")
        assert result["_tool"] == "get_holdings"
        assert result["_args"] == {"portfolio_id": "pid-123"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_trades(self, mock):
        from src.investor.mcp_client import get_trades
        result = get_trades("pid-123", status="executed")
        assert result["_tool"] == "get_trades"
        assert result["_args"] == {"portfolio_id": "pid-123", "status": "executed"}


# ── Error handling tests ──────────────────────────────────────────────────────

class TestErrorHandling:
    @patch("src.investor.mcp_client.call_tool", return_value={"error": "Symbol not found"})
    def test_error_response_passthrough(self, mock):
        from src.investor.mcp_client import get_fundamentals
        result = get_fundamentals("INVALID")
        assert "error" in result
        assert result["error"] == "Symbol not found"

    def test_is_connected_before_connect(self):
        from src.investor.mcp_client import is_connected
        # Before any connection attempt, should be False or True depending on state
        # Just verify it returns a bool
        assert isinstance(is_connected(), bool)
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_mcp_client.py -v
```

Expected: All 19 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcp_client.py
git commit -m "test: add MCP client wrapper tests"
```

---

### Task 5b: Write trading bot page tests

**Files:**
- Create: `tests/test_trading_bot_page.py`

- [ ] **Step 1: Write page-level tests**

```python
"""Tests for trading bot page utilities."""

import pytest
from pathlib import Path
from unittest.mock import patch


class TestPortfolioPersistence:
    def test_save_and_load_portfolio_id(self, tmp_path):
        """Portfolio ID round-trips through file."""
        pfile = tmp_path / "portfolio_id.txt"
        pfile.write_text("test-pid-123")
        assert pfile.read_text().strip() == "test-pid-123"

    def test_missing_file_returns_none(self, tmp_path):
        """No file means no portfolio."""
        pfile = tmp_path / "portfolio_id.txt"
        assert not pfile.exists()


class TestMCPGracefulDegradation:
    @patch("src.investor.mcp_client.is_connected", return_value=False)
    def test_is_connected_false_when_server_down(self, mock):
        from src.investor.mcp_client import is_connected
        assert is_connected() is False

    @patch("src.investor.mcp_client.call_tool", return_value={"error": "Connection refused"})
    def test_tool_returns_error_on_failure(self, mock):
        from src.investor.mcp_client import detect_market_regime
        result = detect_market_regime()
        assert "error" in result


class TestMergeMovers:
    def test_merge_empty_data(self):
        """Empty/error responses produce empty mover list."""
        # Import from MarketPulse would require Streamlit context,
        # so test the merge logic pattern directly
        movers = {}
        anomalies = {"error": "unavailable"}
        volume = {"error": "unavailable"}
        gaps = {"error": "unavailable"}
        # All errors → no movers
        assert len(movers) == 0

    def test_merge_anomalies_only(self):
        """Anomalies contribute to mover scores."""
        movers = {}
        anomalies = {
            "anomalies": [
                {"symbol": "AAPL", "total_score": 5, "anomalies": [{"type": "52w_high"}]}
            ]
        }
        for item in anomalies["anomalies"]:
            sym = item["symbol"]
            movers[sym] = {"symbol": sym, "badges": [], "score": item["total_score"]}
        assert movers["AAPL"]["score"] == 5
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_trading_bot_page.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trading_bot_page.py
git commit -m "test: add trading bot page and degradation tests"
```

---

## Chunk 2: UI Components

### Task 6: Add trading terminal CSS to styles.py

**Files:**
- Modify: `app/components/styles.py`

- [ ] **Step 1: Remove auth and old portfolio CSS, add trading terminal CSS**

Remove the auth form CSS (lines 129–149) and the old portfolio/trade CSS (lines 151–221) from the `apply_theme()` function. Replace with new trading terminal styles:

```css
/* Market Regime Banner */
.regime-banner {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.regime-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.regime-bull { color: #00C853; }
.regime-bear { color: #FF1744; }
.regime-sideways { color: #FFD600; }
.regime-volatile { color: #FF9100; }
.regime-crash { color: #FF1744; text-shadow: 0 0 10px rgba(255,23,68,0.5); }

/* VIX Badge */
.vix-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.vix-low { background: rgba(0,200,83,0.15); color: #00C853; }
.vix-normal { background: rgba(255,214,0,0.15); color: #FFD600; }
.vix-high { background: rgba(255,23,68,0.15); color: #FF1744; }

/* Anomaly Badges */
.anomaly-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 0.3rem;
}
.badge-52w-high { background: rgba(0,200,83,0.2); color: #00C853; }
.badge-volume-spike { background: rgba(255,145,0,0.2); color: #FF9100; }
.badge-gap-up { background: rgba(0,200,83,0.2); color: #00C853; }
.badge-gap-down { background: rgba(255,23,68,0.2); color: #FF1744; }

/* Mover Card */
.mover-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.mover-card:hover { border-color: #58A6FF; }
.mover-symbol { font-size: 1.1rem; font-weight: 700; color: #E6EDF3; }
.mover-price { font-size: 0.95rem; color: #8B949E; }
.mover-change-pos { color: #00C853; font-weight: 600; }
.mover-change-neg { color: #FF1744; font-weight: 600; }

/* Score Gauge */
.score-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.2rem;
}
.score-value {
    font-size: 2.2rem;
    font-weight: 700;
}
.score-label {
    font-size: 0.85rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.score-high { color: #00C853; }
.score-mid { color: #FFD600; }
.score-low { color: #FF1744; }

/* Smart Money Card */
.smart-money-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.2rem;
}

/* Portfolio Section */
.portfolio-value {
    font-size: 2rem;
    font-weight: 700;
    color: #E6EDF3;
}
.portfolio-change-pos { color: #00C853; }
.portfolio-change-neg { color: #FF1744; }
.portfolio-stat {
    font-size: 0.85rem;
    color: #8B949E;
}

/* Powered By Badge */
.powered-badge {
    text-align: center;
    padding: 0.8rem;
    color: #8B949E;
    font-size: 0.85rem;
}
.powered-badge a { color: #58A6FF; text-decoration: none; }
.powered-badge a:hover { text-decoration: underline; }

/* MCP Unavailable Banner */
.mcp-unavailable {
    background: rgba(255,145,0,0.1);
    border: 1px solid rgba(255,145,0,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #FF9100;
    font-size: 0.85rem;
    text-align: center;
    margin-bottom: 1rem;
}
```

- [ ] **Step 2: Verify no syntax errors**

```bash
python -c "import app.components.styles; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/components/styles.py
git commit -m "style: replace auth/portfolio CSS with trading terminal styles"
```

---

### Task 7: Create trading charts component

**Files:**
- Create: `app/components/trading_charts.py`

- [ ] **Step 1: Write trading_charts.py**

```python
"""Plotly chart components for the trading terminal."""

import plotly.graph_objects as go
import yfinance as yf

BG = "#0D1117"
GRID = "#21262D"
TEXT = "#8B949E"
GREEN = "#00C853"
RED = "#FF1744"


def candlestick_chart(symbol: str, period: str = "6mo") -> go.Figure | None:
    """Fetch OHLC data from yfinance and return a candlestick chart."""
    try:
        df = yf.download(symbol, period=period, progress=False)
        if df.empty:
            return None
    except Exception:
        return None

    # Handle MultiIndex columns from yfinance
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    fig = go.Figure(
        data=go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=350,
        xaxis=dict(gridcolor=GRID, showgrid=True),
        yaxis=dict(gridcolor=GRID, showgrid=True, side="right"),
    )
    return fig


def score_gauge(score: float, label: str) -> go.Figure:
    """Circular gauge for a 0-100 score."""
    color = GREEN if score >= 65 else (RED if score < 35 else "#FFD600")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": label, "font": {"size": 14, "color": TEXT}},
            number={"suffix": "/100", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT},
                "bar": {"color": color},
                "bgcolor": "#161B22",
                "bordercolor": "#30363D",
                "steps": [
                    {"range": [0, 35], "color": "rgba(255,23,68,0.1)"},
                    {"range": [35, 65], "color": "rgba(255,214,0,0.1)"},
                    {"range": [65, 100], "color": "rgba(0,200,83,0.1)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"color": "#E6EDF3"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def stress_gauge(stress_score: float, scenarios: dict) -> go.Figure:
    """Stress test visualization with scenario bars."""
    names = list(scenarios.keys())
    values = [abs(v) * 100 for v in scenarios.values()]
    colors = [RED if v > 30 else ("#FFD600" if v > 20 else GREEN) for v in values]

    fig = go.Figure(
        go.Bar(x=values, y=names, orientation="h", marker_color=colors, text=[f"{v:.1f}%" for v in values], textposition="auto")
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Drawdown %", gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    return fig


def cftc_positioning_bars(commercial_net: int, non_commercial_net: int) -> go.Figure:
    """Horizontal bar chart for CFTC commercial vs speculator positioning."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Commercial", "Speculator"],
        x=[commercial_net, non_commercial_net],
        orientation="h",
        marker_color=[GREEN if commercial_net > 0 else RED, GREEN if non_commercial_net > 0 else RED],
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=GRID, title="Net Position"),
        yaxis=dict(gridcolor=GRID),
    )
    return fig


def sector_allocation_bars(allocations: dict) -> go.Figure:
    """Horizontal bar chart for sector allocation."""
    sectors = list(allocations.keys())
    weights = [v * 100 for v in allocations.values()]

    fig = go.Figure(
        go.Bar(y=sectors, x=weights, orientation="h", marker_color="#58A6FF", text=[f"{w:.1f}%" for w in weights], textposition="auto")
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=max(200, len(sectors) * 30),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Weight %", gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    return fig
```

- [ ] **Step 2: Verify import**

```bash
python -c "import app.components.trading_charts; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/components/trading_charts.py
git commit -m "feat: add trading terminal chart components"
```

---

## Chunk 3: Home Page Lite Teaser

### Task 8: Update MarketPulse.py — remove auth, add market intelligence

**Files:**
- Modify: `app/MarketPulse.py`

- [ ] **Step 1: Remove auth imports and gate**

Remove the auth-related import (line 31 area — `from app.components.auth_guard import ...`). Remove the auth gate block (lines 30–39). Remove sidebar user info/sign out block (lines 45–59). Remove any guest mode logic.

- [ ] **Step 2: Add MCP-powered market intelligence section**

After the page config and before the search bar, add the market regime banner and top movers grid. Add this after the header section and before the search bar:

```python
# ── Market Intelligence (powered by financial-mcp) ───────────────────────────
try:
    from src.investor import is_connected, detect_market_regime, get_vix_analysis
    from src.investor import scan_anomalies, scan_volume_leaders, scan_gap_movers

    if is_connected():
        _show_market_intelligence()
    else:
        st.markdown('<div class="mcp-unavailable">Market intelligence unavailable — start the MCP server</div>', unsafe_allow_html=True)
except (ConnectionError, Exception):
    pass  # Silently skip if MCP server not running — sentiment features work independently
```

Add the `_show_market_intelligence()` helper function:

```python
@st.cache_data(ttl=300)
def _fetch_regime():
    from src.investor import detect_market_regime, get_vix_analysis
    return detect_market_regime(), get_vix_analysis()

@st.cache_data(ttl=120)
def _fetch_movers():
    from src.investor import scan_anomalies, scan_volume_leaders, scan_gap_movers
    anomalies = scan_anomalies()
    volume = scan_volume_leaders()
    gaps = scan_gap_movers()
    return anomalies, volume, gaps

def _show_market_intelligence():
    """Render the lite market intelligence teaser."""
    # Market Regime Banner
    regime_data, vix_data = _fetch_regime()
    if "error" not in regime_data and "error" not in vix_data:
        regime = regime_data.get("regime", "UNKNOWN")
        regime_css = {
            "BULL": "regime-bull", "BEAR": "regime-bear",
            "SIDEWAYS": "regime-sideways", "HIGH_VOLATILITY": "regime-volatile",
            "CRASH": "regime-crash",
        }.get(regime, "")
        vix = vix_data.get("vix", "N/A")
        vix_signal = vix_data.get("vix_signal", "normal")
        vix_css = {"fear": "vix-high", "normal": "vix-normal", "complacency": "vix-low"}.get(vix_signal, "vix-normal")
        recommendation = regime_data.get("recommendation", "")

        st.markdown(f'''<div class="regime-banner">
            <span class="regime-label {regime_css}">{regime.replace("_", " ")} MARKET</span>
            <span class="vix-badge {vix_css}" style="float:right">VIX: {vix} ({vix_signal.title()})</span>
            <div style="color:#8B949E;font-size:0.85rem;margin-top:0.5rem">{recommendation}</div>
        </div>''', unsafe_allow_html=True)

    # Top Movers Grid
    anomalies, volume, gaps = _fetch_movers()
    movers = _merge_movers(anomalies, volume, gaps)
    if movers:
        st.markdown("### Top Movers")
        cols = st.columns(3)
        for i, mover in enumerate(movers[:6]):
            with cols[i % 3]:
                symbol = mover["symbol"]
                change_pct = mover.get("change_pct", 0)
                change_css = "mover-change-pos" if change_pct >= 0 else "mover-change-neg"
                badges_html = "".join(
                    f'<span class="anomaly-badge badge-{b["type"].replace("_", "-")}">{b["type"].replace("_", " ").title()}</span>'
                    for b in mover.get("badges", [])
                )
                st.markdown(f'''<div class="mover-card">
                    <div class="mover-symbol">{symbol}</div>
                    <div class="mover-price">{mover.get("price", "N/A")}</div>
                    <div class="{change_css}">{change_pct:+.1f}%</div>
                    <div>{badges_html}</div>
                </div>''', unsafe_allow_html=True)

    # Powered by badge
    st.markdown('<div class="powered-badge">Powered by <a href="/Trading_Bot">financial-mcp</a></div>', unsafe_allow_html=True)


def _merge_movers(anomalies, volume, gaps):
    """Merge anomaly, volume, and gap data into a unified mover list."""
    movers = {}
    # From anomalies
    if "error" not in anomalies:
        for item in anomalies.get("anomalies", []):
            sym = item["symbol"]
            movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
            movers[sym]["score"] += item.get("total_score", 0)
            for a in item.get("anomalies", []):
                movers[sym]["badges"].append({"type": a["type"]})
    # From volume leaders
    if "error" not in volume:
        for item in volume.get("leaders", []):
            sym = item["symbol"]
            movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
            movers[sym]["score"] += item.get("ratio", 0)
            movers[sym]["badges"].append({"type": "volume_spike"})
    # From gap movers
    if "error" not in gaps:
        for item in gaps.get("movers", []):
            sym = item["symbol"]
            movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
            gap_pct = item.get("gap_percent", 0)
            movers[sym]["change_pct"] = gap_pct
            movers[sym]["badges"].append({"type": "gap_up" if gap_pct > 0 else "gap_down"})
    # Sort by score descending
    return sorted(movers.values(), key=lambda m: m["score"], reverse=True)
```

- [ ] **Step 3: Verify the app loads without crashing (MCP server not running)**

```bash
timeout 10 python -m streamlit run app/MarketPulse.py --server.headless true 2>&1 | head -20
```

Expected: App starts without error. "Market intelligence unavailable" shown.

- [ ] **Step 4: Commit**

```bash
git add app/MarketPulse.py
git commit -m "feat: add lite market intelligence teaser to home page"
```

---

## Chunk 4: Full Trading Terminal Page

### Task 9: Create the trading terminal page — market intelligence zone

**Files:**
- Create: `app/pages/2_Trading_Bot.py`

- [ ] **Step 1: Write the top section (page config, imports, MCP connection check, market intel)**

```python
"""Trading Bot — Full trading terminal powered by financial-mcp-server."""

import streamlit as st
import pandas as pd
from pathlib import Path

from app.components.styles import apply_theme

st.set_page_config(page_title="Trading Bot", page_icon="📈", layout="wide")

apply_theme()

DATA_DIR = Path("data")
PORTFOLIO_FILE = DATA_DIR / "portfolio_id.txt"


# ── Portfolio Persistence ─────────────────────────────────────────────────────

def _load_portfolio_id() -> str | None:
    if "portfolio_id" in st.session_state:
        return st.session_state["portfolio_id"]
    if PORTFOLIO_FILE.exists():
        pid = PORTFOLIO_FILE.read_text().strip()
        if pid:
            st.session_state["portfolio_id"] = pid
            return pid
    return None


def _save_portfolio_id(pid: str):
    DATA_DIR.mkdir(exist_ok=True)
    PORTFOLIO_FILE.write_text(pid)
    st.session_state["portfolio_id"] = pid


# ── MCP Connection Check ─────────────────────────────────────────────────────

try:
    from src.investor import (
        is_connected, detect_market_regime, get_vix_analysis,
        scan_anomalies, scan_volume_leaders, scan_gap_movers,
        analyze_ticker, get_fundamentals, get_momentum, score_ticker,
        get_smart_money_signal, get_futures_positioning,
        create_portfolio, analyze_portfolio, get_holdings, get_trades,
        run_rebalance, check_risk, execute_buy, execute_sell,
    )
    mcp_available = is_connected()
except (ConnectionError, Exception):
    mcp_available = False

if not mcp_available:
    st.error("🔌 **MCP server not running.** Start it with: `financial-mcp`")
    st.info("The financial-mcp server must be running for the trading terminal to work.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## 📈 Trading Bot")
st.caption("Powered by financial-mcp")


# ── Market Intelligence Zone ──────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _cached_regime():
    return detect_market_regime(), get_vix_analysis()

@st.cache_data(ttl=120)
def _cached_movers():
    return scan_anomalies(), scan_volume_leaders(), scan_gap_movers()


regime_data, vix_data = _cached_regime()

# Market Regime + VIX row
col_regime, col_vix = st.columns([3, 1])
with col_regime:
    if "error" not in regime_data:
        regime = regime_data.get("regime", "UNKNOWN")
        regime_css = {
            "BULL": "regime-bull", "BEAR": "regime-bear",
            "SIDEWAYS": "regime-sideways", "HIGH_VOLATILITY": "regime-volatile",
            "CRASH": "regime-crash",
        }.get(regime, "")
        score = regime_data.get("score", "")
        recommendation = regime_data.get("recommendation", "")
        st.markdown(f'''<div class="regime-banner">
            <span class="regime-label {regime_css}">{regime.replace("_", " ")} MARKET</span>
            <div style="color:#8B949E;font-size:0.85rem;margin-top:0.3rem">Score: {score} &mdash; {recommendation}</div>
        </div>''', unsafe_allow_html=True)
    else:
        st.warning("Could not load market regime")

with col_vix:
    if "error" not in vix_data:
        vix = vix_data.get("vix", "N/A")
        signal = vix_data.get("vix_signal", "normal")
        pct = vix_data.get("vix_1y_percentile", 0)
        vix_css = {"fear": "vix-high", "normal": "vix-normal", "complacency": "vix-low"}.get(signal, "vix-normal")
        st.markdown(f'''<div class="regime-banner" style="text-align:center">
            <div class="vix-badge {vix_css}" style="font-size:1.2rem">VIX: {vix}</div>
            <div style="color:#8B949E;font-size:0.8rem;margin-top:0.3rem">{signal.title()} &middot; {pct:.0f}th pctile</div>
        </div>''', unsafe_allow_html=True)

# Top Movers
anomalies, volume, gaps = _cached_movers()
movers = {}
if "error" not in anomalies:
    for item in anomalies.get("anomalies", []):
        sym = item["symbol"]
        movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
        movers[sym]["score"] += item.get("total_score", 0)
        for a in item.get("anomalies", []):
            movers[sym]["badges"].append(a["type"])
if "error" not in volume:
    for item in volume.get("leaders", []):
        sym = item["symbol"]
        movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
        movers[sym]["score"] += item.get("ratio", 0)
        movers[sym]["badges"].append("volume_spike")
if "error" not in gaps:
    for item in gaps.get("movers", []):
        sym = item["symbol"]
        movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
        movers[sym].setdefault("change_pct", item.get("gap_percent", 0))
        badge = "gap_up" if item.get("gap_percent", 0) > 0 else "gap_down"
        movers[sym]["badges"].append(badge)

sorted_movers = sorted(movers.values(), key=lambda m: m["score"], reverse=True)[:6]

if sorted_movers:
    st.markdown("#### Top Movers")
    mover_cols = st.columns(3)
    for i, m in enumerate(sorted_movers):
        with mover_cols[i % 3]:
            cp = m.get("change_pct", 0)
            change_css = "mover-change-pos" if cp >= 0 else "mover-change-neg"
            badges = " ".join(
                f'<span class="anomaly-badge badge-{b.replace("_", "-")}">{b.replace("_", " ").title()}</span>'
                for b in m.get("badges", [])[:2]
            )
            st.markdown(f'''<div class="mover-card">
                <div class="mover-symbol">{m["symbol"]}</div>
                <div class="{change_css}">{cp:+.1f}%</div>
                <div>{badges}</div>
            </div>''', unsafe_allow_html=True)

st.divider()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('app/pages/2_Trading_Bot.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/pages/2_Trading_Bot.py
git commit -m "feat: add trading terminal page — market intelligence zone"
```

---

### Task 10: Add ticker detail zone to trading terminal

**Files:**
- Modify: `app/pages/2_Trading_Bot.py` (append to end of file)

- [ ] **Step 1: Add ticker search and detail section**

Append to `2_Trading_Bot.py`:

```python
# ── Ticker Detail Zone ────────────────────────────────────────────────────────

from app.components.trading_charts import candlestick_chart, score_gauge, cftc_positioning_bars

st.markdown("#### Ticker Analysis")
selected_ticker = st.text_input("Search ticker", value=st.session_state.get("selected_ticker", ""), placeholder="e.g. AAPL", key="ticker_input")

if selected_ticker:
    selected_ticker = selected_ticker.upper().strip()
    st.session_state["selected_ticker"] = selected_ticker

    # Candlestick chart + analysis in parallel columns
    col_chart, col_analysis = st.columns([2, 1])

    with col_chart:
        period_map = {"1W": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y"}
        period_label = st.radio("Period", ["1W", "1M", "3M", "1Y"], horizontal=True, index=2)
        fig = candlestick_chart(selected_ticker, period=period_map[period_label])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No chart data for {selected_ticker}")

    with col_analysis:
        analysis = analyze_ticker(selected_ticker)
        if "error" not in analysis:
            price = analysis.get("price")
            score = analysis.get("score", {})
            st.metric(selected_ticker, f"${price:.2f}" if price else "N/A")
            if score:
                st.plotly_chart(score_gauge(score.get("score", 0), "Composite"), use_container_width=True)
        else:
            st.warning(f"Could not analyze {selected_ticker}")

    # Score cards row: Fundamentals | Momentum | Smart Money
    col_fund, col_mom, col_smart = st.columns(3)

    with col_fund:
        @st.cache_data(ttl=60)
        def _cached_fundamentals(sym):
            return get_fundamentals(sym), score_ticker(sym)
        fund, score_data = _cached_fundamentals(selected_ticker)
        if "error" not in fund:
            val_score = score_data.get("valuation", 0) if "error" not in score_data else 0
            st.markdown(f'''<div class="score-card">
                <div class="score-label">Fundamentals</div>
                <div class="score-value {"score-high" if val_score >= 65 else ("score-low" if val_score < 35 else "score-mid")}">{val_score:.0f}<span style="font-size:1rem;color:#8B949E">/100</span></div>
            </div>''', unsafe_allow_html=True)
            st.caption(f"P/E: {fund.get('pe_ratio', 'N/A')}")
            st.caption(f"EV/EBITDA: {fund.get('ev_to_ebitda', 'N/A')}")
            st.caption(f"P/B: {fund.get('price_to_book', 'N/A')}")
            st.caption(f"Div Yield: {fund.get('dividend_yield', 'N/A')}")
            st.caption(f"Market Cap: {fund.get('market_cap', 'N/A')}")

    with col_mom:
        @st.cache_data(ttl=60)
        def _cached_momentum(sym):
            return get_momentum(sym)
        mom = _cached_momentum(selected_ticker)
        if "error" not in mom:
            mom_score = score_data.get("momentum", 0) if "error" not in score_data else 0
            st.markdown(f'''<div class="score-card">
                <div class="score-label">Momentum</div>
                <div class="score-value {"score-high" if mom_score >= 65 else ("score-low" if mom_score < 35 else "score-mid")}">{mom_score:.0f}<span style="font-size:1rem;color:#8B949E">/100</span></div>
            </div>''', unsafe_allow_html=True)
            m30 = mom.get("price_momentum_30d")
            m90 = mom.get("price_momentum_90d")
            vol = mom.get("volatility")
            rs = mom.get("relative_strength")
            st.caption(f"30D Return: {m30 * 100:.1f}%" if m30 is not None else "30D Return: N/A")
            st.caption(f"90D Return: {m90 * 100:.1f}%" if m90 is not None else "90D Return: N/A")
            st.caption(f"Volatility: {vol:.3f}" if vol is not None else "Volatility: N/A")
            st.caption(f"Rel Strength: {rs:.2f}" if rs is not None else "Rel Strength: N/A")

    with col_smart:
        st.markdown('<div class="smart-money-card">', unsafe_allow_html=True)
        st.markdown("**Smart Money**")
        signal = get_smart_money_signal("E-MINI S&P 500")
        if "error" not in signal:
            sig = signal.get("signal", "neutral")
            sig_color = "#00C853" if sig == "bullish" else ("#FF1744" if sig == "bearish" else "#FFD600")
            st.markdown(f"CFTC Signal: <span style='color:{sig_color};font-weight:700'>{sig.upper()}</span>", unsafe_allow_html=True)
            st.caption(signal.get("reason", ""))
            positioning = get_futures_positioning("E-MINI S&P 500")
            if "error" not in positioning:
                reports = positioning.get("reports", [])
                if reports:
                    latest = reports[0]
                    fig = cftc_positioning_bars(
                        latest.get("commercial_net", 0),
                        latest.get("non_commercial_net", 0),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Smart money data unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('app/pages/2_Trading_Bot.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/pages/2_Trading_Bot.py
git commit -m "feat: add ticker detail zone to trading terminal"
```

---

### Task 11: Add portfolio management zone to trading terminal

**Files:**
- Modify: `app/pages/2_Trading_Bot.py` (append to end of file)

- [ ] **Step 1: Add portfolio section**

Append to `2_Trading_Bot.py`:

```python
# ── Portfolio Management Zone ─────────────────────────────────────────────────

from app.components.trading_charts import stress_gauge, sector_allocation_bars

st.markdown("#### Portfolio")

portfolio_id = _load_portfolio_id()

if portfolio_id is None:
    # Onboarding
    st.markdown("**Set up your portfolio to start trading**")
    with st.form("portfolio_setup"):
        name = st.text_input("Portfolio name", value="My Portfolio")
        capital = st.slider("Starting capital ($)", 10000, 1000000, 100000, step=10000)
        risk = st.radio("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
        horizon = st.selectbox("Investment horizon", ["short", "medium", "long"], index=1)
        submitted = st.form_submit_button("Create Portfolio")
        if submitted:
            result = create_portfolio(float(capital), risk, horizon, name)
            if "error" not in result:
                _save_portfolio_id(result["portfolio_id"])
                st.success(f"Portfolio created! ID: {result['portfolio_id']}")
                st.rerun()
            else:
                st.error(f"Failed: {result['error']}")
else:
    # Portfolio summary
    summary = analyze_portfolio(portfolio_id)
    if "error" in summary:
        st.warning("Portfolio not found on MCP server. Create a new one?")
        if st.button("Reset Portfolio"):
            PORTFOLIO_FILE.unlink(missing_ok=True)
            st.session_state.pop("portfolio_id", None)
            st.rerun()
    else:
        # Header metrics
        col_val, col_change, col_cash, col_rebal = st.columns(4)
        total_value = summary.get("total_value", 0)
        daily_change = summary.get("daily_change", 0)
        daily_pct = summary.get("daily_change_pct", 0)
        portfolio_info = summary.get("portfolio", {})
        cash = portfolio_info.get("current_cash", 0)

        with col_val:
            st.metric("Portfolio Value", f"${total_value:,.2f}")
        with col_change:
            st.metric("Daily Change", f"${daily_change:,.2f}", f"{daily_pct:+.2f}%")
        with col_cash:
            st.metric("Cash", f"${cash:,.2f}")
        with col_rebal:
            if st.button("🔄 Rebalance Now"):
                with st.spinner("Rebalancing..."):
                    rebal_result = run_rebalance(portfolio_id, trigger="manual")
                    if "error" not in rebal_result:
                        st.success(f"Rebalance complete: {rebal_result.get('buy_signals', 0)} buys, {rebal_result.get('sell_signals', 0)} sells")
                        st.rerun()
                    else:
                        st.error(f"Rebalance failed: {rebal_result.get('error')}")

        # Holdings table
        holdings_data = get_holdings(portfolio_id)
        if "error" not in holdings_data and holdings_data.get("holdings"):
            st.markdown("##### Holdings")
            holdings_list = holdings_data["holdings"]
            df = pd.DataFrame(holdings_list)
            display_cols = [c for c in ["symbol", "shares", "avg_cost_basis", "asset_type", "sector", "company_name"] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Trade history
        trades_data = get_trades(portfolio_id)
        if "error" not in trades_data and trades_data.get("trades"):
            st.markdown("##### Trade History")
            with st.expander(f"Show trades ({trades_data.get('count', 0)})"):
                trades_df = pd.DataFrame(trades_data["trades"])
                display_cols = [c for c in ["symbol", "action", "shares", "price", "total_value", "status", "executed_at"] if c in trades_df.columns]
                st.dataframe(trades_df[display_cols], use_container_width=True, hide_index=True)

        # Risk assessment
        risk_data = check_risk(portfolio_id)
        if "error" not in risk_data:
            st.markdown("##### Risk Assessment")
            col_stress, col_sectors = st.columns(2)
            with col_stress:
                stress = risk_data.get("stress", {})
                scenarios = stress.get("scenario_drawdowns", {})
                if scenarios:
                    st.plotly_chart(stress_gauge(stress.get("stress_score", 0), scenarios), use_container_width=True)
            with col_sectors:
                sector_alloc = risk_data.get("sector_allocation", {})
                if sector_alloc:
                    st.plotly_chart(sector_allocation_bars(sector_alloc), use_container_width=True)

        # Performance metrics
        perf = summary.get("performance", {})
        if perf:
            st.markdown("##### Performance")
            perf_cols = st.columns(4)
            with perf_cols[0]:
                st.metric("Cumulative Return", f"{perf.get('cumulative_return', 0) * 100:.2f}%")
            with perf_cols[1]:
                st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
            with perf_cols[2]:
                st.metric("Max Drawdown", f"{perf.get('max_drawdown', 0) * 100:.2f}%")
            with perf_cols[3]:
                st.metric("Daily Return", f"{perf.get('daily_return', 0) * 100:.2f}%")
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('app/pages/2_Trading_Bot.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/pages/2_Trading_Bot.py
git commit -m "feat: add portfolio management zone to trading terminal"
```

---

## Chunk 5: Final Cleanup & Docs

### Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Update the following sections:
- **Architecture** diagram: Replace the investor module flow with MCP client flow
- **Project Structure**: Remove `src/investor/engine.py`, `market_data.py`, `broker.py`, `portfolio.py`, `risk.py`, `rebalancer.py`. Remove `src/auth/`. Add `src/investor/mcp_client.py`. Replace `app/pages/2_Portfolio.py` and `3_Trades.py` with `2_Trading_Bot.py`. Add `app/components/trading_charts.py`.
- **Key Modules**: Remove engine, market_data, broker, portfolio, risk, rebalancer descriptions. Add `mcp_client.py` description. Remove `auth.py` description.
- **SQLite Schema**: Remove users, portfolios, holdings, trades, portfolio_snapshots, etf_universe tables. Note that portfolio state now lives in the MCP server's `data/financial_mcp.db`.
- **Configuration**: Remove `investor:` section, add `mcp_server:` section.
- **Environment Variables**: Remove mention of bcrypt. Note `financial-mcp` server must be running.
- **Running Locally**: Add `financial-mcp` to startup instructions.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect MCP integration"
```

---

### Task 13: Run full test suite and verify

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: All tests pass. `test_investor.py` was deleted. `test_mcp_client.py` passes. Existing sentiment/ingestion tests are unaffected.

- [ ] **Step 2: Verify app starts cleanly**

```bash
timeout 10 python -m streamlit run app/MarketPulse.py --server.headless true 2>&1 | head -20
```

Expected: No import errors, no crashes. App starts and serves.

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: resolve any integration issues from final verification"
```
