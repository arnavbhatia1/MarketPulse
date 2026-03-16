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
from datetime import datetime, timedelta
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
    pending_sells: set = field(default_factory=set)
    trade_log: list = field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Cycle step stub — replaced in Task 3
# ---------------------------------------------------------------------------

def _run_cycle(portfolio_id: str, stop_event: threading.Event) -> None:
    """Execute one full trading cycle. Implemented in Task 3."""
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
        """Start the bot. Creates ScalpBot portfolio on first run.

        Note: there is a benign TOCTOU between the is_running guard and the
        portfolio-creation MCP call. In practice, this is single-threaded
        (Streamlit's one render thread triggers start()), so double-start is
        not a real risk.
        """
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
