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

CYCLE_INTERVAL = 5            # minimum seconds between cycles (avoids hammering MCP)
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
# Cycle step functions
# ---------------------------------------------------------------------------

def _check_vix() -> bool:
    """Returns True if VIX is numeric and > 30."""
    try:
        data = get_vix_analysis()
        vix = data.get("vix")
        return isinstance(vix, (int, float)) and not isinstance(vix, bool) and vix > 30
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
            # Brief pause before next cycle to avoid hammering MCP server,
            # then immediately scan again — no fixed timer.
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
