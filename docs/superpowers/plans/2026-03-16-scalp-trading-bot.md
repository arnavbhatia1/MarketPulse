# Scalp Trading Bot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an autonomous 5-minute paper-trading bot to the MarketPulse Trading Bot page that scans a 50-ticker universe, enters/exits positions based on composite score signals, and displays a live rich dashboard.

**Architecture:** A module-level `BotState` singleton in `src/investor/bot_engine.py` is updated by a background daemon thread running 5-minute cycles. The Streamlit page reads from this singleton and auto-refreshes every 10 seconds. All trade execution goes through the existing `src.investor.mcp_client` wrappers.

**Tech Stack:** Python `threading`, `dataclasses`, Streamlit, existing MCP client wrappers, pytest + `unittest.mock`

**Spec:** `docs/superpowers/specs/2026-03-16-scalp-trading-bot-design.md`

---

## Chunk 1: BotState + BotEngine Core

### Task 1: BotState singleton and pure helper functions

**Files:**
- Create: `src/investor/bot_engine.py`
- Create: `tests/test_bot_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bot_engine.py
"""Tests for autonomous scalp trading bot engine."""
import threading
import time as _time
import pytest
from unittest.mock import patch


class TestGetState:
    def test_returns_same_instance(self):
        from src.investor.bot_engine import get_state
        assert get_state() is get_state()

    def test_initial_values(self):
        from src.investor.bot_engine import get_state
        s = get_state()
        assert s.portfolio_cash == 10_000.0
        assert s.portfolio_value == 10_000.0
        assert s.open_positions == {}
        assert s.pending_sells == set()
        assert s.trade_log == []
        assert s.cycle_count == 0
        assert s.portfolio_id is None


class TestGetAllocationPct:
    def test_tier_90_to_100(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(95, False) == pytest.approx(0.12)

    def test_tier_70_to_89(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(75, False) == pytest.approx(0.08)

    def test_tier_60_to_69(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(65, False) == pytest.approx(0.05)

    def test_below_60_returns_zero(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(59, False) == 0.0

    def test_high_vix_halves_all_tiers(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(95, True) == pytest.approx(0.06)
        assert _get_allocation_pct(75, True) == pytest.approx(0.04)
        assert _get_allocation_pct(65, True) == pytest.approx(0.025)


class TestGetCompositeScore:
    def test_extracts_score(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"score": {"score": 82.5}}) == pytest.approx(82.5)

    def test_returns_zero_on_error_key(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"error": "not found"}) == 0.0

    def test_returns_zero_when_missing(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({}) == 0.0

    def test_returns_zero_when_score_is_none(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"score": {"score": None}}) == 0.0
```

- [ ] **Step 2: Run tests — confirm they fail**

```
pytest tests/test_bot_engine.py -v
```
Expected: `ModuleNotFoundError` (file doesn't exist yet)

- [ ] **Step 3: Create `src/investor/bot_engine.py` with BotState + helpers**

```python
"""Autonomous scalp trading bot engine.

Module-level BotState singleton updated by a background daemon thread
running 5-minute trading cycles. All trade execution goes through the
existing MCP client wrappers.

Design note: the spec defines a `lock` field inside BotState, but we
intentionally use a separate module-level `_lock` instead — a Lock inside
a dataclass is non-picklable and awkward to use. Never re-instantiate
_state or _engine; only one instance of each exists for the Streamlit
server's lifetime.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.investor.mcp_client import (
    create_portfolio,
    analyze_portfolio,
    execute_buy,
    execute_sell,
    detect_market_regime,
    get_vix_analysis,
    scan_universe,
    scan_anomalies,
    scan_volume_leaders,
    analyze_ticker,
)

logger = logging.getLogger(__name__)

CYCLE_INTERVAL = 300          # seconds between cycles
MAX_POSITIONS = 5
MIN_SCORE = 60
EXIT_SCORE_DROP_THRESHOLD = 0.30   # 30% drop from entry score
EXIT_ABSOLUTE_THRESHOLD = 40

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "LLY", "MA", "HD", "PG", "COST", "JNJ", "MRK", "ABBV",
    "CVX", "BAC", "WMT", "KO", "NFLX", "PEP", "DIS", "ADBE", "AMD", "INTC",
    "CRM", "PYPL", "QCOM", "TXN", "CSCO", "NEE", "RTX", "CAT", "GS", "AMGN",
    "T", "VZ", "PFE", "BMY", "MO", "SBUX", "DE", "MMM", "GE", "F",
]

# (min_score_inclusive, max_score_inclusive, pct_of_cash)
_ALLOCATION_TIERS = [
    (90, 100, 0.12),
    (70, 89,  0.08),
    (60, 69,  0.05),
]


@dataclass
class BotState:
    is_running: bool = False
    portfolio_id: Optional[str] = None
    portfolio_cash: float = 10_000.0
    portfolio_value: float = 10_000.0
    total_pnl: float = 0.0
    open_positions: dict = field(default_factory=dict)
    # { ticker: {entry_price, shares, entry_score, entry_time, current_price, current_score} }
    pending_sells: set = field(default_factory=set)
    trade_log: list = field(default_factory=list)
    # [ {time, action, ticker, price, shares, score, reason, pnl} ]
    cycle_count: int = 0
    last_cycle_time: Optional[datetime] = None
    next_cycle_time: Optional[datetime] = None


_state = BotState()
_lock = threading.Lock()


def get_state() -> BotState:
    """Return the global BotState singleton."""
    return _state


def _get_allocation_pct(score: float, high_vix: bool) -> float:
    """Map composite score to cash allocation fraction. Returns 0.0 if below MIN_SCORE."""
    for min_s, max_s, pct in _ALLOCATION_TIERS:
        if min_s <= score <= max_s:
            return pct * 0.5 if high_vix else pct
    return 0.0


def _get_composite_score(analysis: dict) -> float:
    """Extract composite score from analyze_ticker response. Returns 0.0 on any error."""
    if "error" in analysis:
        return 0.0
    raw = analysis.get("score", {}).get("score")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0
```

- [ ] **Step 4: Run tests — confirm they pass**

```
pytest tests/test_bot_engine.py::TestGetState tests/test_bot_engine.py::TestGetAllocationPct tests/test_bot_engine.py::TestGetCompositeScore -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/investor/bot_engine.py tests/test_bot_engine.py
git commit -m "feat: add BotState singleton and pure helper functions"
```

---

### Task 2: BotEngine start/stop/is_running

**Files:**
- Modify: `src/investor/bot_engine.py` (append stubs + BotEngine class)
- Modify: `tests/test_bot_engine.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_bot_engine.py
# (time as _time and threading are already imported at the top of the file)


def _reset_state():
    """Reset module-level state between tests."""
    from src.investor import bot_engine
    bot_engine._state.is_running = False
    bot_engine._state.portfolio_id = None
    bot_engine._state.cycle_count = 0
    bot_engine._state.open_positions = {}
    bot_engine._state.pending_sells = set()
    bot_engine._state.trade_log = []
    bot_engine._engine._stop_event.set()


class TestBotEngine:
    def setup_method(self):
        _reset_state()

    def teardown_method(self):
        from src.investor.bot_engine import get_engine
        get_engine().stop()
        _time.sleep(0.1)

    def test_get_engine_returns_same_instance(self):
        from src.investor.bot_engine import get_engine
        assert get_engine() is get_engine()

    def test_not_running_initially(self):
        from src.investor.bot_engine import get_engine
        assert get_engine().is_running() is False

    @patch("src.investor.bot_engine.create_portfolio",
           return_value={"portfolio_id": "test-pid"})
    @patch("src.investor.bot_engine._run_cycle")
    def test_start_sets_running(self, mock_cycle, mock_create):
        from src.investor.bot_engine import get_engine, get_state
        get_engine().start()
        _time.sleep(0.15)
        assert get_state().is_running is True

    @patch("src.investor.bot_engine.create_portfolio",
           return_value={"portfolio_id": "test-pid"})
    @patch("src.investor.bot_engine._run_cycle")
    def test_stop_clears_running(self, mock_cycle, mock_create):
        from src.investor.bot_engine import get_engine, get_state
        get_engine().start()
        _time.sleep(0.1)
        get_engine().stop()
        _time.sleep(0.2)
        # stop() directly sets is_running=False under _lock, so this passes
        # immediately — it does not wait for the _loop thread to exit.
        assert get_state().is_running is False

    @patch("src.investor.bot_engine.create_portfolio",
           return_value={"error": "server down"})
    def test_start_aborts_when_portfolio_creation_fails(self, mock_create):
        from src.investor.bot_engine import get_engine, get_state
        get_engine().start()
        _time.sleep(0.1)
        assert get_state().is_running is False

    @patch("src.investor.bot_engine.create_portfolio")
    @patch("src.investor.bot_engine._run_cycle")
    def test_start_reuses_existing_portfolio_id(self, mock_cycle, mock_create):
        from src.investor import bot_engine
        bot_engine._state.portfolio_id = "already-set"
        bot_engine.get_engine().start()
        _time.sleep(0.1)
        assert mock_create.call_count == 0
        assert bot_engine._state.portfolio_id == "already-set"
```

- [ ] **Step 2: Run tests — confirm they fail**

```
pytest tests/test_bot_engine.py::TestBotEngine -v
```
Expected: `AttributeError` (BotEngine not defined yet)

- [ ] **Step 3: Append stubs and BotEngine to `src/investor/bot_engine.py`**

Add these to the bottom of the file (below `_get_composite_score`):

```python
# ---------------------------------------------------------------------------
# Cycle step stubs — implemented incrementally in Tasks 3-5
# ---------------------------------------------------------------------------

def _run_cycle(portfolio_id: str, stop_event: threading.Event) -> None:
    """Execute one full trading cycle. Filled in incrementally."""
    pass


# ---------------------------------------------------------------------------
# BotEngine — controls the background loop
# ---------------------------------------------------------------------------

class BotEngine:
    """Start/stop the background trading thread."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the bot. Creates ScalpBot portfolio on first run."""
        with _lock:
            if _state.is_running:
                return

        if _state.portfolio_id is None:
            result = create_portfolio(
                starting_capital=10_000,
                risk_profile="aggressive",
                investment_horizon="short",
                name="ScalpBot",
            )
            if "error" not in result:
                with _lock:
                    _state.portfolio_id = result["portfolio_id"]
                logger.info("Created ScalpBot portfolio: %s", _state.portfolio_id)
            else:
                logger.error("Portfolio creation failed: %s", result["error"])
                return

        self._stop_event.clear()
        with _lock:
            _state.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Bot started")

    def stop(self) -> None:
        """Signal graceful shutdown after current cycle completes."""
        self._stop_event.set()
        with _lock:
            _state.is_running = False
        logger.info("Bot stop signaled")

    def is_running(self) -> bool:
        return (
            _state.is_running
            and self._thread is not None
            and self._thread.is_alive()
        )

    def _loop(self) -> None:
        from datetime import timedelta

        # Capture portfolio_id once under lock before the loop begins.
        # portfolio_id is only set in start() before this thread is created,
        # so it's safe to read once and reuse for all cycles.
        with _lock:
            portfolio_id = _state.portfolio_id

        while not self._stop_event.is_set():
            with _lock:
                _state.cycle_count += 1
                _state.last_cycle_time = datetime.now()
            try:
                _run_cycle(portfolio_id, self._stop_event)
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
            with _lock:
                _state.next_cycle_time = datetime.now() + timedelta(seconds=CYCLE_INTERVAL)
            # Sleep CYCLE_INTERVAL seconds, wake every second to check stop
            for _ in range(CYCLE_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        with _lock:
            _state.is_running = False
        logger.info("Bot loop exited")


_engine = BotEngine()


def get_engine() -> BotEngine:
    """Return the global BotEngine singleton."""
    return _engine
```

- [ ] **Step 4: Run tests — confirm they pass**

```
pytest tests/test_bot_engine.py::TestBotEngine -v
```
Expected: All PASS

- [ ] **Step 5: Run all bot_engine tests**

```
pytest tests/test_bot_engine.py -v
```
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/investor/bot_engine.py tests/test_bot_engine.py
git commit -m "feat: add BotEngine start/stop/loop control"
```

---

## Chunk 2: Trading Cycle Steps

### Task 3: VIX check + exit logic

**Files:**
- Modify: `src/investor/bot_engine.py`
- Modify: `tests/test_bot_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_bot_engine.py


class TestCheckVix:
    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": 35, "vix_signal": "fear"})
    def test_returns_true_when_vix_above_30(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is True

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": 18, "vix_signal": "normal"})
    def test_returns_false_when_vix_normal(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": "N/A"})
    def test_returns_false_when_vix_non_numeric(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"error": "timeout"})
    def test_returns_false_on_error(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False


class TestCheckExits:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 148.0, "score": {"score": 55.0}})
    def test_exits_when_score_drops_30_percent(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        bot_engine._state.open_positions["AAPL"] = {
            "entry_price": 150.0, "shares": 10, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 150.0, "current_score": 80.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "AAPL" not in bot_engine._state.open_positions
        assert bot_engine._state.trade_log[0]["action"] == "SELL"
        assert "dropped" in bot_engine._state.trade_log[0]["reason"]

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 148.0, "score": {"score": 35.0}})
    def test_exits_when_score_below_absolute_threshold(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        bot_engine._state.open_positions["MSFT"] = {
            "entry_price": 300.0, "shares": 5, "entry_score": 50.0,
            "entry_time": datetime.now(), "current_price": 300.0, "current_score": 50.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "MSFT" not in bot_engine._state.open_positions

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 155.0, "score": {"score": 75.0}})
    def test_no_exit_when_score_holds(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        bot_engine._state.open_positions["NVDA"] = {
            "entry_price": 800.0, "shares": 2, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 800.0, "current_score": 80.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "NVDA" in bot_engine._state.open_positions
        mock_sell.assert_not_called()

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"error": "network error"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 148.0, "score": {"score": 30.0}})
    def test_failed_sell_added_to_pending_sells(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        bot_engine._state.open_positions["TSLA"] = {
            "entry_price": 200.0, "shares": 3, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 200.0, "current_score": 80.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "TSLA" in bot_engine._state.pending_sells
        assert "TSLA" in bot_engine._state.open_positions  # not removed on fail
```

- [ ] **Step 2: Run tests — confirm they fail**

```
pytest tests/test_bot_engine.py::TestCheckVix tests/test_bot_engine.py::TestCheckExits -v
```
Expected: `ImportError` (_check_vix, _check_exits not defined)

- [ ] **Step 3: Replace the `_run_cycle` stub with cycle step functions**

Replace the existing stub in `src/investor/bot_engine.py`:

```python
# ---------------------------------------------------------------------------
# Cycle step functions
# ---------------------------------------------------------------------------

def _check_vix() -> bool:
    """Returns True if VIX is numeric and > 30."""
    try:
        data = get_vix_analysis()
        vix = data.get("vix")
        return isinstance(vix, (int, float)) and vix > 30
    except Exception as e:
        logger.warning("VIX check failed: %s", e)
        return False


def _check_exits(portfolio_id: str, stop_event: threading.Event) -> None:
    """Evaluate open positions; sell those where signal has reversed."""
    with _lock:
        positions = dict(_state.open_positions)

    for ticker, pos in positions.items():
        if stop_event.is_set():
            return
        analysis = analyze_ticker(ticker)
        current_score = _get_composite_score(analysis)
        current_price = (
            analysis.get("price") or pos["entry_price"]
            if "error" not in analysis
            else pos["entry_price"]
        )

        # Always update current metrics in state
        with _lock:
            if ticker in _state.open_positions:
                _state.open_positions[ticker]["current_price"] = current_price
                _state.open_positions[ticker]["current_score"] = current_score

        if current_score == 0:
            logger.debug("Skipping exit check for %s — no score data", ticker)
            continue

        entry_score = pos["entry_score"]
        should_exit = (
            current_score < entry_score * (1 - EXIT_SCORE_DROP_THRESHOLD)
            or current_score < EXIT_ABSOLUTE_THRESHOLD
        )
        if not should_exit:
            continue

        reason = (
            f"score dropped {entry_score:.0f}→{current_score:.0f}"
            if current_score >= EXIT_ABSOLUTE_THRESHOLD
            else f"below threshold ({current_score:.0f})"
        )
        result = execute_sell(portfolio_id, ticker, pos["shares"])
        pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)

        if "error" not in result:
            with _lock:
                _state.open_positions.pop(ticker, None)
                _state.trade_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "SELL",
                    "ticker": ticker,
                    "price": current_price,
                    "shares": pos["shares"],
                    "score": current_score,
                    "reason": reason,
                    "pnl": pnl,
                })
            logger.info("Exited %s: %s", ticker, reason)
        else:
            with _lock:
                _state.pending_sells.add(ticker)
            logger.warning("Sell failed for %s → pending: %s", ticker, result["error"])


def _retry_pending_sells(portfolio_id: str, stop_event: threading.Event) -> None:
    """Retry execute_sell for any tickers in pending_sells."""
    with _lock:
        pending = set(_state.pending_sells)
    for ticker in pending:
        if stop_event.is_set():
            return
        with _lock:
            pos = _state.open_positions.get(ticker)
        if pos is None:
            with _lock:
                _state.pending_sells.discard(ticker)
            continue
        result = execute_sell(portfolio_id, ticker, pos["shares"])
        if "error" not in result:
            current_price = pos.get("current_price", pos["entry_price"])
            pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)
            with _lock:
                _state.pending_sells.discard(ticker)
                _state.open_positions.pop(ticker, None)
                _state.trade_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "SELL",
                    "ticker": ticker,
                    "price": current_price,
                    "shares": pos["shares"],
                    "score": pos.get("current_score", 0),
                    "reason": "pending retry",
                    "pnl": pnl,
                })
            logger.info("Pending sell completed: %s", ticker)
        else:
            logger.warning("Pending sell still failing for %s: %s", ticker, result["error"])


def _scan_candidates(stop_event: threading.Event) -> list:
    """Scan universe + anomalies + volume leaders. Returns deduped candidate symbols."""
    raw: list = []

    universe_result = scan_universe(DEFAULT_UNIVERSE)
    if "error" not in universe_result:
        for item in universe_result.get("scores", []):
            sym = item.get("symbol", "")
            if sym:
                raw.append(sym)

    if stop_event.is_set():
        return []

    anomaly_result = scan_anomalies(DEFAULT_UNIVERSE)
    if "error" not in anomaly_result:
        for item in anomaly_result.get("anomalies", []):
            sym = item.get("symbol", "")
            if sym:
                raw.append(sym)

    if stop_event.is_set():
        return []

    volume_result = scan_volume_leaders(DEFAULT_UNIVERSE)
    if "error" not in volume_result:
        for item in volume_result.get("leaders", []):
            sym = item.get("symbol", "")
            if sym:
                raw.append(sym)

    with _lock:
        held = set(_state.open_positions.keys()) | _state.pending_sells

    seen: set = set()
    candidates = []
    for sym in raw:
        if sym not in held and sym not in seen:
            seen.add(sym)
            candidates.append(sym)
    return candidates


def _score_candidates(candidates: list, stop_event: threading.Event) -> list:
    """Score top-10 candidates. Returns [{ticker, score, price}] sorted score desc.

    Uses the price returned by analyze_ticker to avoid a redundant get_price call.
    """
    scored = []
    for sym in candidates[:10]:
        if stop_event.is_set():
            return scored
        analysis = analyze_ticker(sym)
        score = _get_composite_score(analysis)
        if score < MIN_SCORE:
            continue
        price = analysis.get("price") if "error" not in analysis else None
        if price and float(price) > 0:
            scored.append({"ticker": sym, "score": score, "price": float(price)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _enter_positions(
    portfolio_id: str,
    scored: list,
    high_vix: bool,
    stop_event: threading.Event,
) -> None:
    """Enter up to MAX_POSITIONS using score-based cash allocation.

    Tracks remaining_cash locally so each successive buy within the same
    cycle uses updated (decremented) cash rather than the stale portfolio_cash.
    """
    with _lock:
        remaining_cash = _state.portfolio_cash

    for candidate in scored:
        if stop_event.is_set():
            return
        with _lock:
            if len(_state.open_positions) >= MAX_POSITIONS:
                break

        alloc_pct = _get_allocation_pct(candidate["score"], high_vix)
        if alloc_pct == 0:
            continue

        shares = int(remaining_cash * alloc_pct / candidate["price"])
        if shares < 1:
            continue

        result = execute_buy(portfolio_id, candidate["ticker"], shares)
        if "error" not in result:
            with _lock:
                _state.open_positions[candidate["ticker"]] = {
                    "entry_price": candidate["price"],
                    "shares": shares,
                    "entry_score": candidate["score"],
                    "entry_time": datetime.now(),
                    "current_price": candidate["price"],
                    "current_score": candidate["score"],
                    "allocation_pct": alloc_pct,
                }
                _state.trade_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "BUY",
                    "ticker": candidate["ticker"],
                    "price": candidate["price"],
                    "shares": shares,
                    "score": candidate["score"],
                    "reason": f"score {candidate['score']:.0f} ({alloc_pct * 100:.0f}% cash)",
                    "pnl": 0.0,
                })
            remaining_cash -= shares * candidate["price"]
            logger.info(
                "Bought %s: %d shares @ $%.2f (score %.0f)",
                candidate["ticker"], shares, candidate["price"], candidate["score"],
            )
        else:
            logger.warning("Buy failed for %s: %s", candidate["ticker"], result["error"])


def _snapshot_portfolio(portfolio_id: str) -> None:
    """Reconcile BotState cash/value from MCP server."""
    result = analyze_portfolio(portfolio_id)
    if "error" not in result:
        portfolio_info = result.get("portfolio", {})
        with _lock:
            _state.portfolio_cash = portfolio_info.get("current_cash", _state.portfolio_cash)
            _state.portfolio_value = result.get("total_value", _state.portfolio_value)
            _state.total_pnl = _state.portfolio_value - 10_000.0
    else:
        logger.warning("Portfolio snapshot failed: %s", result["error"])


def _run_cycle(portfolio_id: str, stop_event: threading.Event) -> None:
    """Execute one full trading cycle: regime → VIX → retries → exits → scan → score → enter → snapshot."""
    logger.info("=== Cycle %d start ===", _state.cycle_count)

    regime = detect_market_regime()
    if "error" not in regime:
        logger.info("Market regime: %s (score %s)", regime.get("regime"), regime.get("score"))

    high_vix = _check_vix()
    _retry_pending_sells(portfolio_id, stop_event)
    if stop_event.is_set():
        return
    _check_exits(portfolio_id, stop_event)
    if stop_event.is_set():
        return
    candidates = _scan_candidates(stop_event)
    if candidates:
        scored = _score_candidates(candidates, stop_event)
        if not stop_event.is_set():
            _enter_positions(portfolio_id, scored, high_vix, stop_event)
    else:
        logger.warning("All scans returned no candidates — skipping entry phase")
    _snapshot_portfolio(portfolio_id)
```

- [ ] **Step 4: Run tests — confirm they pass**

```
pytest tests/test_bot_engine.py::TestCheckVix tests/test_bot_engine.py::TestCheckExits -v
```
Expected: All PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/test_bot_engine.py -v
```
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/investor/bot_engine.py tests/test_bot_engine.py
git commit -m "feat: implement VIX check and exit logic"
```

---

### Task 4: Scan candidates + score candidates

**Files:**
- Modify: `tests/test_bot_engine.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_bot_engine.py


class TestScanCandidates:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.scan_volume_leaders", return_value={"leaders": []})
    @patch("src.investor.bot_engine.scan_anomalies", return_value={"anomalies": []})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"scores": [{"symbol": "AAPL"}, {"symbol": "MSFT"}]})
    def test_returns_symbols_from_universe(self, mock_u, mock_a, mock_v):
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert "AAPL" in result
        assert "MSFT" in result

    @patch("src.investor.bot_engine.scan_volume_leaders",
           return_value={"leaders": [{"symbol": "AAPL"}]})
    @patch("src.investor.bot_engine.scan_anomalies",
           return_value={"anomalies": [{"symbol": "AAPL"}]})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"scores": [{"symbol": "AAPL"}, {"symbol": "MSFT"}]})
    def test_deduplicates_across_sources(self, mock_u, mock_a, mock_v):
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert result.count("AAPL") == 1

    @patch("src.investor.bot_engine.scan_volume_leaders", return_value={"leaders": []})
    @patch("src.investor.bot_engine.scan_anomalies", return_value={"anomalies": []})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"scores": [{"symbol": "AAPL"}, {"symbol": "HELD"}]})
    def test_filters_out_held_positions(self, mock_u, mock_a, mock_v):
        from src.investor import bot_engine
        bot_engine._state.open_positions["HELD"] = {}
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert "HELD" not in result
        assert "AAPL" in result

    @patch("src.investor.bot_engine.scan_volume_leaders",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.scan_anomalies",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"error": "timeout"})
    def test_returns_empty_when_all_scans_fail(self, mock_u, mock_a, mock_v):
        from src.investor.bot_engine import _scan_candidates
        assert _scan_candidates(threading.Event()) == []


class TestScoreCandidates:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 150.0, "score": {"score": 75.0}})
    def test_returns_scored_candidates_above_min(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["AAPL"], threading.Event())
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["score"] == pytest.approx(75.0)
        assert result[0]["price"] == pytest.approx(150.0)

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 45.0}})
    def test_filters_out_score_below_60(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["WEAK"], threading.Event())
        assert result == []

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"score": {"score": 85.0}})   # no "price" key
    def test_skips_when_price_unavailable(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["NOPRICE"], threading.Event())
        assert result == []

    @patch("src.investor.bot_engine.analyze_ticker", side_effect=[
        {"price": 100.0, "score": {"score": 65.0}},
        {"price": 200.0, "score": {"score": 90.0}},
    ])
    def test_sorted_descending_by_score(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["LOW", "HIGH"], threading.Event())
        assert result[0]["score"] > result[1]["score"]
```

- [ ] **Step 2: Run tests — confirm they fail**

```
pytest tests/test_bot_engine.py::TestScanCandidates tests/test_bot_engine.py::TestScoreCandidates -v
```
Expected: FAIL (functions exist but behaviour untested until now — some may pass, some fail)

- [ ] **Step 3: Run all tests to check nothing regressed**

```
pytest tests/test_bot_engine.py -v
```
Expected: All PASS (new tests should pass since the functions were implemented in Task 3)

- [ ] **Step 4: Commit**

```bash
git add tests/test_bot_engine.py
git commit -m "test: add scan and score candidate coverage"
```

---

### Task 5: Enter positions + portfolio snapshot

**Files:**
- Modify: `tests/test_bot_engine.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_bot_engine.py


class TestEnterPositions:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_enters_position_and_logs_buy(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 10_000.0
        from src.investor.bot_engine import _enter_positions
        _enter_positions("pid", [{"ticker": "AAPL", "score": 85, "price": 100.0}],
                         False, threading.Event())
        assert "AAPL" in bot_engine._state.open_positions
        assert bot_engine._state.trade_log[0]["action"] == "BUY"
        # remaining_cash starts at 10_000, score=85 → tier 70-89 → 8% → 800 → 8 shares @ $100
        mock_buy.assert_called_once_with("pid", "AAPL", 8)

    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_does_not_exceed_max_positions(self, mock_buy):
        from src.investor import bot_engine
        from src.investor.bot_engine import _enter_positions, MAX_POSITIONS
        bot_engine._state.portfolio_cash = 100_000.0
        # Fill to max
        for i in range(MAX_POSITIONS):
            bot_engine._state.open_positions[f"HELD{i}"] = {}
        candidates = [{"ticker": f"NEW{i}", "score": 85, "price": 100.0} for i in range(3)]
        _enter_positions("pid", candidates, False, threading.Event())
        mock_buy.assert_not_called()

    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_high_vix_halves_share_count(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 10_000.0
        from src.investor.bot_engine import _enter_positions
        _enter_positions("pid", [{"ticker": "AAPL", "score": 85, "price": 100.0}],
                         True, threading.Event())
        # high_vix: 0.08 * 0.5 = 0.04 → 10000 * 0.04 / 100 = 4 shares
        mock_buy.assert_called_once_with("pid", "AAPL", 4)

    @patch("src.investor.bot_engine.execute_buy", return_value={"error": "rejected"})
    def test_failed_buy_not_added_to_positions(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 10_000.0
        from src.investor.bot_engine import _enter_positions
        _enter_positions("pid", [{"ticker": "FAIL", "score": 85, "price": 100.0}],
                         False, threading.Event())
        assert "FAIL" not in bot_engine._state.open_positions


class TestSnapshotPortfolio:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.analyze_portfolio", return_value={
        "total_value": 10_500.0,
        "portfolio": {"current_cash": 8_000.0},
    })
    def test_updates_cash_and_value(self, mock_analyze):
        from src.investor import bot_engine
        from src.investor.bot_engine import _snapshot_portfolio
        _snapshot_portfolio("pid")
        assert bot_engine._state.portfolio_cash == pytest.approx(8_000.0)
        assert bot_engine._state.portfolio_value == pytest.approx(10_500.0)
        assert bot_engine._state.total_pnl == pytest.approx(500.0)

    @patch("src.investor.bot_engine.analyze_portfolio",
           return_value={"error": "not found"})
    def test_keeps_previous_values_on_failure(self, mock_analyze):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 5_000.0
        bot_engine._state.portfolio_value = 11_000.0
        from src.investor.bot_engine import _snapshot_portfolio
        _snapshot_portfolio("pid")
        assert bot_engine._state.portfolio_cash == pytest.approx(5_000.0)
        assert bot_engine._state.portfolio_value == pytest.approx(11_000.0)
```

- [ ] **Step 2: Run tests — confirm they pass**

```
pytest tests/test_bot_engine.py::TestEnterPositions tests/test_bot_engine.py::TestSnapshotPortfolio -v
```
Expected: All PASS

- [ ] **Step 3: Run full test suite**

```
pytest tests/test_bot_engine.py -v && pytest tests/ -v --ignore=tests/test_bot_engine.py -x
```
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_bot_engine.py
git commit -m "test: add entry, sizing, and portfolio snapshot coverage"
```

---

## Chunk 3: Dashboard UI + Bug Fixes

### Task 6: Bot Control section in 2_Trading_Bot.py

**Files:**
- Modify: `app/pages/2_Trading_Bot.py` (append Zone 4)
- Modify: `tests/test_trading_bot_page.py` (append smoke test)

- [ ] **Step 1: Write a smoke test for the Bot Control import**

```python
# Append to tests/test_trading_bot_page.py

class TestBotEngineImport:
    def test_bot_engine_importable(self):
        from src.investor.bot_engine import get_state, get_engine, MAX_POSITIONS
        assert MAX_POSITIONS == 5

    def test_get_state_has_expected_fields(self):
        from src.investor.bot_engine import get_state
        s = get_state()
        assert hasattr(s, "is_running")
        assert hasattr(s, "open_positions")
        assert hasattr(s, "trade_log")
        assert hasattr(s, "portfolio_value")
        assert hasattr(s, "total_pnl")
```

- [ ] **Step 2: Run tests — confirm they pass**

```
pytest tests/test_trading_bot_page.py::TestBotEngineImport -v
```
Expected: PASS

- [ ] **Step 3: Append Zone 4 Bot Control section to `app/pages/2_Trading_Bot.py`**

Add the following after the final line of the existing file (after line 372):

```python

# ==============================================================================
# ZONE 4: BOT CONTROL
# ==============================================================================

import time as _time

from src.investor.bot_engine import get_state as _get_bot_state, get_engine as _get_engine, MAX_POSITIONS

st.divider()
st.markdown("#### Bot Control")

_bot_state = _get_bot_state()
_engine = _get_engine()

# -- Start / Stop + Status row ------------------------------------------------
col_btn, col_status, col_timer = st.columns([1, 1, 2])

with col_btn:
    if _bot_state.is_running:
        if st.button("Stop Bot", type="secondary", key="bot_stop"):
            _engine.stop()
            st.rerun()
    else:
        if st.button("Start Bot", type="primary", key="bot_start"):
            _engine.start()
            st.rerun()

with col_status:
    if _bot_state.is_running:
        st.markdown(
            '<span style="color:#00C853;font-weight:700;font-size:1rem">● RUNNING</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#8B949E;font-weight:700;font-size:1rem">● STOPPED</span>',
            unsafe_allow_html=True,
        )

with col_timer:
    if _bot_state.next_cycle_time and _bot_state.is_running:
        remaining = (_bot_state.next_cycle_time - datetime.now()).total_seconds()
        if remaining > 0:
            mins, secs = divmod(int(remaining), 60)
            st.caption(
                f"Next cycle in **{mins}m {secs:02d}s** · Cycle #{_bot_state.cycle_count}"
            )
        else:
            st.caption(f"Cycle #{_bot_state.cycle_count} — running now...")

# -- Portfolio metrics --------------------------------------------------------
if _bot_state.portfolio_id:
    col_pv, col_pnl, col_npos = st.columns(3)
    with col_pv:
        st.metric("Portfolio Value", f"${_bot_state.portfolio_value:,.2f}")
    with col_pnl:
        pnl = _bot_state.total_pnl
        delta_str = f"${pnl:+,.2f}"
        st.metric("Total P&L", delta_str)
    with col_npos:
        st.metric("Open Positions", f"{len(_bot_state.open_positions)} / {MAX_POSITIONS}")

    # -- Open positions table -------------------------------------------------
    if _bot_state.open_positions:
        st.markdown("##### Open Positions")
        rows = []
        for ticker, pos in _bot_state.open_positions.items():
            entry = pos["entry_price"]
            current = pos.get("current_price", entry)
            pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
            pnl_amt = (current - entry) * pos["shares"]
            rows.append({
                "Ticker": ticker,
                "Entry": f"${entry:.2f}",
                "Current": f"${current:.2f}",
                "P&L %": f"{pnl_pct:+.2f}%",
                "P&L $": f"${pnl_amt:+.2f}",
                "Score": f"{pos.get('current_score', pos['entry_score']):.0f}",
                "Since": pos["entry_time"].strftime("%H:%M"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # -- Activity log ---------------------------------------------------------
    if _bot_state.trade_log:
        st.markdown("##### Activity Log")
        with st.expander(
            f"Recent trades ({len(_bot_state.trade_log)})", expanded=True
        ):
            for entry in _bot_state.trade_log[:50]:
                color = "#00C853" if entry["action"] == "BUY" else "#FF1744"
                pnl_str = (
                    f" &nbsp;|&nbsp; P&L: ${entry['pnl']:+.2f}"
                    if entry["action"] == "SELL"
                    else ""
                )
                st.markdown(
                    f"<small style='color:#8B949E'>[{entry['time']}]</small> "
                    f"<span style='color:{color};font-weight:700'>{entry['action']}</span> "
                    f"<b>{entry['ticker']}</b> {entry['shares']}sh "
                    f"@ ${entry['price']:.2f} (score {entry['score']:.0f})"
                    f"{pnl_str} — {entry['reason']}",
                    unsafe_allow_html=True,
                )

# -- Auto-refresh (non-blocking) ----------------------------------------------
if _bot_state.is_running:
    _now = _time.time()
    _last = st.session_state.get("bot_last_refresh", 0)
    if _now - _last >= 10:
        st.session_state["bot_last_refresh"] = _now
        st.rerun()
```

Note: `datetime` is already imported via `from src.investor.bot_engine import ...` — if not, add `from datetime import datetime` at the top of the file.

- [ ] **Step 4: Add `datetime` import to `2_Trading_Bot.py`**

The existing file does not import `datetime`. Add it after the `from pathlib import Path` line:

```python
from datetime import datetime
```

This is required for `datetime.now()` in the Bot Control countdown and open positions table.

- [ ] **Step 5: Run existing tests to confirm nothing broke**

```
pytest tests/test_trading_bot_page.py -v
```
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/pages/2_Trading_Bot.py tests/test_trading_bot_page.py
git commit -m "feat: add Bot Control dashboard section (Zone 4)"
```

---

### Task 7: Bug fixes — ticker hint + VIX/Score defensive display

**Files:**
- Modify: `app/pages/2_Trading_Bot.py`

- [ ] **Step 1: Add ticker input hint**

Find the ticker input block (around line 160-163):

```python
selected_ticker = st.text_input(
    "Search ticker", value=st.session_state.get("selected_ticker", ""),
    placeholder="e.g. AAPL", key="ticker_input",
)
```

Replace with:

```python
selected_ticker = st.text_input(
    "Search ticker",
    value=st.session_state.get("selected_ticker", ""),
    placeholder="e.g. AAPL, LCID — enter symbol, not company name",
    key="ticker_input",
    help="Enter a ticker symbol (e.g. LCID, not 'Lucid Motors')",
)
```

- [ ] **Step 2: Fix VIX percentile defensive display**

Find the VIX block (around line 103):

```python
st.markdown(f'''<div class="regime-banner" style="text-align:center">
    <div class="vix-badge {vix_css}" style="font-size:1.2rem">VIX: {vix}</div>
    <div style="color:#8B949E;font-size:0.8rem;margin-top:0.3rem">{signal.title()} &middot; {pct:.0f}th pctile</div>
</div>''', unsafe_allow_html=True)
```

Replace with:

```python
pct_display = f"{pct:.0f}th pctile" if isinstance(pct, (int, float)) else "N/A"
st.markdown(f'''<div class="regime-banner" style="text-align:center">
    <div class="vix-badge {vix_css}" style="font-size:1.2rem">VIX: {vix}</div>
    <div style="color:#8B949E;font-size:0.8rem;margin-top:0.3rem">{signal.title()} &middot; {pct_display}</div>
</div>''', unsafe_allow_html=True)
```

- [ ] **Step 3: Add defensive guard before `pct:.0f` in VIX block**

Also find where `pct` is defined (line ~99):

```python
pct = vix_data.get("vix_1y_percentile", 0)
```

Replace with:

```python
pct = vix_data.get("vix_1y_percentile", 0) or 0
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/pages/2_Trading_Bot.py
git commit -m "fix: add ticker symbol hint and defensive VIX percentile display"
```

---

## Final Verification

- [ ] **Start the app and verify**

```bash
python -m streamlit run app/MarketPulse.py
```

1. Navigate to Trading Bot page
2. Verify Zone 1 shows regime (no Score: 0 / VIX: N/A crash)
3. Type "LCID" in ticker search — should show "enter symbol not company name" hint
4. Scroll to Bot Control section — should see Start Bot button
5. Click Start Bot — status changes to RUNNING, countdown appears
6. Wait one cycle (or reduce `CYCLE_INTERVAL` to 30 temporarily for testing)
7. Verify activity log shows BUY entries
8. Click Stop Bot — status changes to STOPPED

- [ ] **Run full test suite one final time**

```
pytest tests/ -v
```
Expected: All PASS

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: autonomous scalp trading bot with live dashboard"
```
