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
