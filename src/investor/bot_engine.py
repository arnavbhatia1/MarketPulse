"""Autonomous momentum + event-driven trading bot — probability-based.

Think like a casino, not a gambler. The edge comes from:
- Expected Value (EV): only take trades where math favors us
- Kelly Criterion: size positions based on actual win/loss statistics
- Risk of Ruin: never let position sizing threaten account survival
- Law of Large Numbers: many small trades > few big bets
- Variance awareness: drawdowns are normal, don't abandon edge during them

Strategy: momentum rides trends, event-driven catches catalysts.
Hold as long as the thesis holds. Exit on score decay OR price stops.

Module-level BotState singleton. Background daemon thread runs continuous
cycles. All trade execution goes through MCP client wrappers.
"""
import json
import logging
import math
import os
import random
import statistics
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
    scan_anomalies,
    scan_volume_leaders,
    scan_gap_movers,
    scan_universe,
    analyze_ticker,
)
from src.storage.db import load_ticker_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CYCLE_INTERVAL = 60           # seconds between cycles (MCP scores need time to update)
MAX_CONSECUTIVE_FAILURES = 10 # auto-stop bot after this many failed cycles
INITIAL_BACKOFF = 10          # seconds before first retry after failure
MAX_BACKOFF = 300             # cap exponential backoff at 5 minutes
MAX_POSITIONS = 20
MIN_SCORE = 60
EXIT_SCORE_DROP_THRESHOLD = 0.30
EXIT_ABSOLUTE_THRESHOLD = 40
# Standard paper-trading account size. At 2%/trade this is ~$2k per position,
# enough to take real share counts across all price ranges (a $10k account
# can't hold a $300–600 mega-cap inside a 2% cap, so it filled only penny names).
STARTING_CAPITAL = 100_000.0
SEED_BASE_CAPITAL = 10_000.0  # the illustrative seed ledger was authored at $10k

# Exit thresholds — price-based
TRAILING_STOP_PCT = 0.03      # sell if price drops 3% from peak since entry
HARD_STOP_PCT = 0.05          # sell if price drops 5% from entry price

# Re-entry gate — prevent churn
REENTRY_SCORE_BOOST = 15      # must score this many points above sell score to re-buy
SELL_HISTORY_EXPIRY_HOURS = 24 # clear stale sell history after this long

# Risk management — casino rules
MAX_RISK_PER_TRADE = 0.02     # never risk more than 2% of portfolio per trade
MIN_TRADES_FOR_STATS = 10     # minimum closed trades before using Kelly sizing
DEFAULT_RISK_PER_TRADE = 0.01 # conservative 1% until we have enough data
MAX_KELLY_FRACTION = 0.5      # use half-Kelly (full Kelly is too aggressive)
MAX_ACCEPTABLE_RUIN = 0.05    # 5% risk of ruin = reduce sizing
VARIANCE_LOOKBACK = 50        # last N trades for rolling stats

SCORE_CANDIDATES_LIMIT = 25   # candidates batch-scored per cycle (one scan_universe call)
MAX_UNIVERSE_SIZE = 250       # full candidate pool; sampled SCORE_CANDIDATES_LIMIT/cycle
SENTIMENT_TILT_MAX = 10       # max points RSS sentiment adds/subtracts from a score
CATALYST_BONUS_MAX = 10       # max points the event scanners add to a candidate's score
EQUITY_CURVE_MAX = 1000       # cap stored equity-curve points

# Seed tickers — only used as fallback if no dynamic sources return anything
_SEED_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM",
    "SPY", "QQQ",
]

# Where the bot persists its ledger (portfolio id + trade log) across restarts.
_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "bot_state.json",
)
_loaded = False  # guard so load_state() only reads the file once per process


# ---------------------------------------------------------------------------
# Trade Statistics — the math that matters
# ---------------------------------------------------------------------------

@dataclass
class TradeStats:
    """Rolling statistics computed from closed trades. Updated each cycle."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expected_value: float = 0.0     # EV per trade in dollars
    kelly_fraction: float = 0.0     # optimal fraction of capital to risk
    risk_of_ruin: float = 0.0       # probability of total account loss
    std_dev: float = 0.0            # standard deviation of trade P&L
    reward_risk_ratio: float = 0.0  # avg_win / avg_loss
    largest_win: float = 0.0
    largest_loss: float = 0.0
    current_streak: int = 0         # positive = win streak, negative = loss streak
    has_edge: bool = False          # True if EV > 0 with enough sample size


def _compute_trade_stats(trade_log: list) -> TradeStats:
    """Compute all trading statistics from closed trades (SELL entries in log).

    This is the foundation of the entire system. Every sizing and risk
    decision flows from these numbers.
    """
    stats = TradeStats()
    closed = [t for t in trade_log if t["action"] == "SELL"]
    if not closed:
        return stats

    pnls = [t["pnl"] for t in closed]
    stats.total_trades = len(pnls)
    stats.wins = sum(1 for p in pnls if p > 0)
    stats.losses = sum(1 for p in pnls if p <= 0)

    # Win rate
    stats.win_rate = stats.wins / stats.total_trades if stats.total_trades > 0 else 0

    # Average win and average loss (absolute value for loss)
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [abs(p) for p in pnls if p <= 0]
    stats.avg_win = statistics.mean(win_pnls) if win_pnls else 0
    stats.avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0

    # Expected Value: EV = (WinRate × AvgWin) - (LossRate × AvgLoss)
    loss_rate = 1 - stats.win_rate
    stats.expected_value = (stats.win_rate * stats.avg_win) - (loss_rate * stats.avg_loss)

    # Reward/Risk ratio
    stats.reward_risk_ratio = stats.avg_win / stats.avg_loss if stats.avg_loss > 0 else 0

    # Standard deviation of all trade P&Ls
    if len(pnls) >= 2:
        stats.std_dev = statistics.stdev(pnls)

    # Extremes
    stats.largest_win = max(pnls) if pnls else 0
    stats.largest_loss = min(pnls) if pnls else 0

    # Current streak
    streak = 0
    for t in closed:  # most recent first (list is insert(0, ...))
        if streak == 0:
            streak = 1 if t["pnl"] > 0 else -1
        elif (streak > 0 and t["pnl"] > 0) or (streak < 0 and t["pnl"] <= 0):
            streak += 1 if streak > 0 else -1
        else:
            break
    stats.current_streak = streak

    # Kelly Criterion: Kelly% = W - (1-W)/R
    # where W = win rate, R = reward/risk ratio
    if stats.reward_risk_ratio > 0 and stats.total_trades >= MIN_TRADES_FOR_STATS:
        kelly = stats.win_rate - (1 - stats.win_rate) / stats.reward_risk_ratio
        stats.kelly_fraction = max(0, kelly * MAX_KELLY_FRACTION)  # half-Kelly, floor at 0

    # Risk of Ruin: RoR = ((1-W)/W × 1/R) ^ (Account/Risk)
    # Only meaningful with enough trades
    if (stats.win_rate > 0 and stats.reward_risk_ratio > 0
            and stats.total_trades >= MIN_TRADES_FOR_STATS):
        base = ((1 - stats.win_rate) / stats.win_rate) * (1 / stats.reward_risk_ratio)
        if 0 < base < 1:
            # Use current position sizing to estimate units at risk
            risk_per_trade = stats.kelly_fraction if stats.kelly_fraction > 0 else DEFAULT_RISK_PER_TRADE
            units = 1 / risk_per_trade if risk_per_trade > 0 else 100
            stats.risk_of_ruin = base ** units
        elif base >= 1:
            stats.risk_of_ruin = 1.0  # no edge, guaranteed ruin

    # Do we have a statistically meaningful edge?
    stats.has_edge = (
        stats.expected_value > 0
        and stats.total_trades >= MIN_TRADES_FOR_STATS
    )

    return stats


# ---------------------------------------------------------------------------
# Position Sizing — Kelly-based with safety rails
# ---------------------------------------------------------------------------

def _compute_position_size(score: float, high_vix: bool, stats: TradeStats,
                           portfolio_value: float,
                           max_risk: float = MAX_RISK_PER_TRADE) -> float:
    """Calculate dollar amount to risk on this trade.

    Sizing hierarchy:
    1. If enough trade history: use half-Kelly from actual statistics
    2. If not enough history: use conservative 1% of portfolio
    3. Apply VIX discount if markets are volatile
    4. Cap at 2% of portfolio (professional standard)
    5. Scale by score conviction (higher score = closer to full size)

    Returns dollar amount to allocate (not percentage).
    """
    if portfolio_value <= 0:
        return 0

    # Base risk fraction
    if stats.has_edge and stats.kelly_fraction > 0:
        # Kelly-based: we have enough data to size intelligently
        base_risk = stats.kelly_fraction
        logger.debug("Using Kelly sizing: %.4f", base_risk)
    else:
        # Not enough data yet — be conservative, many small bets
        base_risk = DEFAULT_RISK_PER_TRADE
        logger.debug("Using default conservative sizing: %.4f", base_risk)

    # Risk of ruin check — if we're in danger zone, reduce sizing
    if stats.risk_of_ruin > MAX_ACCEPTABLE_RUIN:
        base_risk *= 0.5
        logger.warning("Risk of ruin %.2f%% > %.0f%% — halving position size",
                        stats.risk_of_ruin * 100, MAX_ACCEPTABLE_RUIN * 100)

    # VIX discount — high volatility means smaller bets
    if high_vix:
        base_risk *= 0.5

    # Score conviction scaling: score 60 = 60% of base size, score 100 = 100%
    conviction = score / 100.0
    adjusted_risk = base_risk * conviction

    # Hard cap per trade — never more, regardless of Kelly
    capped_risk = min(adjusted_risk, max_risk)

    return portfolio_value * capped_risk


# ---------------------------------------------------------------------------
# BotState
# ---------------------------------------------------------------------------

@dataclass
class BotState:
    is_running: bool = False
    portfolio_id: Optional[str] = None
    portfolio_cash: float = STARTING_CAPITAL
    portfolio_value: float = STARTING_CAPITAL
    total_pnl: float = 0.0
    open_positions: dict = field(default_factory=dict)
    pending_sells: set = field(default_factory=set)
    trade_log: list = field(default_factory=list)
    cycle_count: int = 0
    last_cycle_time: Optional[datetime] = None
    # Quant stats — updated every cycle
    stats: TradeStats = field(default_factory=TradeStats)
    # Sell history — {ticker: {"score": float, "time": datetime}} for re-entry gate
    sell_history: dict = field(default_factory=dict)
    # Equity curve — [{"time": iso_str, "value": float}] appended each snapshot
    equity_curve: list = field(default_factory=list)
    # Runtime-adjustable parameters (editable from the UI while stopped)
    max_positions: int = MAX_POSITIONS
    min_score: float = MIN_SCORE
    max_risk_per_trade: float = MAX_RISK_PER_TRADE
    starting_capital: float = STARTING_CAPITAL
    # Blend MarketPulse RSS sentiment into MCP scores (the two-tab bridge)
    sentiment_bridge: bool = True
    # Seed an illustrative trade history on a cold start so the Edge panel and
    # equity curve are populated immediately instead of empty.
    warm_start: bool = True


_state = BotState()
_lock = threading.Lock()


def get_state() -> BotState:
    """Return the global BotState singleton."""
    return _state


# ---------------------------------------------------------------------------
# Persistence — survive Streamlit restarts and warm-start the Edge panel
# ---------------------------------------------------------------------------

def _save_state() -> None:
    """Persist the bot's ledger (portfolio id + realized trade log) to disk.

    Best-effort: never raises into a trading cycle. No-ops until a portfolio
    exists so unit tests that snapshot a bare state don't write a file.
    """
    if not _state.portfolio_id:
        return
    try:
        os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
        with _lock:
            payload = {
                "portfolio_id": _state.portfolio_id,
                "total_pnl": _state.total_pnl,
                "trade_log": _state.trade_log,
                "equity_curve": _state.equity_curve,
            }
        with open(_STATE_PATH, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.debug("Could not persist bot state: %s", e)


def load_state() -> bool:
    """Load a previously saved ledger so the bot resumes warm. Call once.

    Restores portfolio_id and the closed-trade log, then recomputes stats so
    the Edge Statistics panel is populated immediately. Returns True if a
    state file was loaded. Guarded so repeated Streamlit reruns don't clobber
    the live in-memory log.
    """
    global _loaded
    if _loaded:
        return False
    _loaded = True
    if not os.path.exists(_STATE_PATH):
        return False
    try:
        with open(_STATE_PATH) as f:
            payload = json.load(f)
        with _lock:
            _state.portfolio_id = payload.get("portfolio_id") or _state.portfolio_id
            _state.trade_log = payload.get("trade_log", [])
            # total_pnl is mark-to-market (value - capital), recomputed on the
            # first snapshot; a fresh account correctly shows $0 until it trades.
            _state.total_pnl = 0.0
            _state.equity_curve = payload.get("equity_curve", [])
            _state.stats = _compute_trade_stats(_state.trade_log)
        logger.info("Loaded persisted bot state: %d trades, portfolio %s",
                    len(_state.trade_log), _state.portfolio_id)
        return True
    except Exception as e:
        logger.warning("Could not load bot state: %s", e)
        return False


# ---------------------------------------------------------------------------
# Warm start — illustrative ledger so the Edge panel is never empty on a cold
# start. These round trips are demo history; once the live bot runs, real
# trades append to and eventually dominate this log. Shared with scripts/seed_demo.py.
# ---------------------------------------------------------------------------

# (ticker, shares, entry_price, exit_price, sell_reason) — each <=2% of $10k.
_SEED_ROUND_TRIPS = [
    ("NIO", 33, 5.99, 6.35, "take profit +6.0% (score fading 88→70)"),
    ("SOFI", 10, 18.69, 17.95, "signal reversal 74→52"),
    ("GME", 9, 21.38, 22.70, "take profit +6.2% (score fading 86→69)"),
    ("KO", 2, 78.73, 81.10, "trailing stop -2.9% from peak $83.50→$81.10"),
    ("INTC", 1, 111.11, 108.50, "hard stop -2.3% (score reversal 81→44)"),
    ("CSCO", 1, 121.03, 124.80, "take profit +3.1% (score fading 79→63)"),
    ("WMT", 1, 114.16, 116.90, "take profit +2.4% (score fading 77→62)"),
    ("COIN", 1, 186.39, 179.20, "hard stop -3.9% (limit -5%)"),
    ("DIS", 1, 102.35, 105.60, "take profit +3.2% (score fading 82→66)"),
    ("PLTR", 1, 163.16, 170.40, "take profit +4.4% (score fading 90→71)"),
    ("AMD", 1, 119.10, 116.00, "signal reversal 72→51"),
    ("F", 25, 7.40, 7.18, "trailing stop -3.0% from peak $7.62→$7.18"),
    ("SNAP", 18, 10.20, 10.75, "take profit +5.4% (score fading 84→69)"),
    ("HOOD", 8, 24.80, 26.10, "take profit +5.2% (score fading 80→70)"),
]


def _seed_clock(minute: int) -> str:
    """Minutes after 09:00 ET → HH:MM, rolling hours over correctly."""
    return f"{9 + minute // 60:02d}:{minute % 60:02d}"


def build_seed_trade_log(starting_capital: float = STARTING_CAPITAL) -> list:
    """Newest-first trade log of illustrative BUY/SELL round trips.

    Share counts (authored at $10k) scale with *starting_capital* so each seeded
    position stays the same ~fraction of the account regardless of its size.
    """
    factor = max(1, round(starting_capital / SEED_BASE_CAPITAL))
    log: list = []
    minute = 31  # start at 09:31 ET
    for i, (ticker, base_shares, entry, exit_, reason) in enumerate(_SEED_ROUND_TRIPS):
        shares = base_shares * factor
        buy_time = _seed_clock(minute)
        minute += 3
        sell_time = _seed_clock(minute)
        minute += 4
        entry_score = 70 + (i * 2) % 25
        pnl = round((exit_ - entry) * shares, 2)
        risk_pct = round(shares * entry / starting_capital * 100, 1)
        log.insert(0, {
            "time": buy_time, "action": "BUY", "ticker": ticker,
            "price": entry, "shares": shares, "score": entry_score,
            "reason": f"score {entry_score} · {risk_pct}% risk", "pnl": 0.0,
        })
        log.insert(0, {
            "time": sell_time, "action": "SELL", "ticker": ticker,
            "price": exit_, "shares": shares, "score": max(40, entry_score - 12),
            "reason": reason, "pnl": pnl,
        })
    return log


def build_seed_equity_curve(trade_log: list,
                            starting_capital: float = STARTING_CAPITAL) -> list:
    """Equity curve: cumulative realized P&L over time (oldest → newest)."""
    curve = [{"time": "09:30", "value": starting_capital}]
    running = starting_capital
    for t in reversed(trade_log):
        if t["action"] == "SELL":
            running = round(running + t["pnl"], 2)
            curve.append({"time": t["time"], "value": running})
    return curve


def seed_warm_start() -> bool:
    """Populate an illustrative ledger so the Edge panel + equity curve start
    warm. No-op (returns False) if the log already has trades."""
    with _lock:
        if _state.trade_log:
            return False
        capital = _state.starting_capital
        log = build_seed_trade_log(capital)
        _state.trade_log = log
        _state.equity_curve = build_seed_equity_curve(log, capital)
        _state.total_pnl = round(sum(t["pnl"] for t in log if t["action"] == "SELL"), 2)
        _state.stats = _compute_trade_stats(log)
    logger.info("Warm-started bot with %d illustrative trades", len(log))
    return True


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


def _sell_position(portfolio_id: str, ticker: str, pos: dict, reason: str) -> bool:
    """Execute a sell and update state. Returns True if successful."""
    current_price = pos.get("current_price", pos["entry_price"])
    result = execute_sell(portfolio_id, ticker, pos["shares"])
    pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)

    if "error" not in result:
        sell_score = pos.get("current_score", 0)
        with _lock:
            _state.open_positions.pop(ticker, None)
            _state.sell_history[ticker] = {
                "score": sell_score,
                "time": datetime.now(),
            }
            _state.trade_log.insert(0, {
                "time": datetime.now().strftime("%H:%M"),
                "action": "SELL",
                "ticker": ticker,
                "price": current_price,
                "shares": pos["shares"],
                "score": sell_score,
                "reason": reason,
                "pnl": pnl,
            })
        logger.info("Sold %s: %s (P&L: $%.2f)", ticker, reason, pnl)
        return True
    else:
        with _lock:
            _state.pending_sells.add(ticker)
        logger.warning("Sell failed for %s → pending: %s", ticker, result["error"])
        return False


def _check_exits(
    portfolio_id: str,
    batch_scores: dict,
    stop_event: threading.Event,
) -> None:
    """Exit engine — score + price triggers, whichever fires first.

    Reads each held position's fresh score and price from *batch_scores* (the
    single scan_universe call for this cycle, which includes held symbols). Only
    falls back to a per-ticker ``analyze_ticker`` when a held name is missing
    from the batch (rare), so the common case costs zero extra round-trips.

    Exit triggers (checked in order):
    1. HARD STOP: price dropped 5% from entry — non-negotiable capital protection
    2. TRAILING STOP: price dropped 3% from peak — lock gains after a run-up
    3. SIGNAL REVERSAL: score dropped 30%+ or below absolute threshold
    4. PROFIT TAKING: score fading 15%+ while position is green
    5. OUTLIER LOSS: loss exceeds 2 standard deviations (cut the outlier)
    """
    batch_scores = batch_scores or {}
    with _lock:
        positions = dict(_state.open_positions)
        current_stats = _state.stats

    for ticker, pos in positions.items():
        if stop_event.is_set():
            return
        info = batch_scores.get(ticker.upper())
        if info and info.get("price"):
            current_score = info["score"]
            current_price = float(info["price"])
        else:
            analysis = analyze_ticker(ticker)
            current_score = _get_composite_score(analysis)
            current_price = (
                analysis.get("price") or pos["entry_price"]
                if "error" not in analysis
                else pos["entry_price"]
            )

        # Update position tracking
        peak_price = max(pos.get("peak_price", pos["entry_price"]), current_price)
        with _lock:
            if ticker in _state.open_positions:
                _state.open_positions[ticker]["current_price"] = current_price
                _state.open_positions[ticker]["current_score"] = current_score
                _state.open_positions[ticker]["peak_price"] = peak_price

        entry_price = pos["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        pnl_dollar = (current_price - entry_price) * pos["shares"]

        # Exit 1: Hard stop — price dropped 5% from entry
        if pnl_pct <= -HARD_STOP_PCT:
            reason = f"hard stop {pnl_pct*100:+.1f}% (limit {-HARD_STOP_PCT*100:.0f}%)"
            _sell_position(portfolio_id, ticker, pos, reason)
            continue

        # Exit 2: Trailing stop — price dropped 3% from peak
        if peak_price > entry_price:
            drop_from_peak = (current_price - peak_price) / peak_price if peak_price > 0 else 0
            if drop_from_peak <= -TRAILING_STOP_PCT:
                reason = (
                    f"trailing stop {drop_from_peak*100:+.1f}% from peak "
                    f"${peak_price:.2f}→${current_price:.2f}"
                )
                _sell_position(portfolio_id, ticker, pos, reason)
                continue

        if current_score == 0:
            continue

        entry_score = pos["entry_score"]
        score_change_pct = (current_score - entry_score) / entry_score if entry_score > 0 else 0

        # Exit 3: Signal reversal — score decayed significantly
        if (current_score < entry_score * (1 - EXIT_SCORE_DROP_THRESHOLD)
                or current_score < EXIT_ABSOLUTE_THRESHOLD):
            reason = (
                f"signal reversal {entry_score:.0f}→{current_score:.0f}"
                if current_score >= EXIT_ABSOLUTE_THRESHOLD
                else f"below threshold ({current_score:.0f})"
            )
            _sell_position(portfolio_id, ticker, pos, reason)
            continue

        # Exit 4: Profit taking — lock gains before they evaporate
        if pnl_pct > 0.005 and score_change_pct < -0.15:
            reason = f"take profit +{pnl_pct*100:.1f}% (score fading {entry_score:.0f}→{current_score:.0f})"
            _sell_position(portfolio_id, ticker, pos, reason)
            continue

        # Exit 5: Outlier loss — if unrealized loss > 2 std deviations, cut it
        if current_stats.std_dev > 0 and current_stats.total_trades >= MIN_TRADES_FOR_STATS:
            mean_pnl = current_stats.expected_value
            if pnl_dollar < mean_pnl - 2 * current_stats.std_dev:
                reason = f"outlier loss ${pnl_dollar:.2f} > 2σ (cut variance)"
                _sell_position(portfolio_id, ticker, pos, reason)


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


def _catalyst_bonus(vol_ratio: float, gap_pct: float, anomaly_score: float) -> float:
    """Map raw scanner signals to a bounded score bonus (0..CATALYST_BONUS_MAX).

    Event-driven catalysts are exactly what a scalp bot should lean into, so a
    name surfacing on a volume surge / open gap / anomaly gets a few points of
    lift on top of its composite score.
    """
    bonus = 0.0
    if vol_ratio > 2:
        bonus += min((vol_ratio - 2) * 1.5, 4.0)   # 2x→0, 4.7x→4
    bonus += min(abs(gap_pct) / 2.0, 4.0)           # 2% gap→1, 8%+→4
    bonus += min(anomaly_score / 3.0, 6.0)          # anomaly total_score → up to 6
    return round(min(bonus, CATALYST_BONUS_MAX), 1)


def _build_dynamic_universe() -> tuple[list[str], dict[str, float]]:
    """Build a broad, ordered ticker pool plus a per-symbol catalyst bonus.

    Sources (in priority order — most actionable first):
    1. MCP scanners — anomalies, volume leaders, gap movers (event-driven catalysts)
    2. RSS ticker_cache — every ticker the news is talking about right now
    3. Seed tickers — small fallback if everything else is empty

    Returns ``(symbols, catalyst_map)``. ``symbols`` is up to MAX_UNIVERSE_SIZE
    tickers; ``catalyst_map`` is ``{symbol: bonus}`` distilled from the scanner
    payloads (the rich data the scanners already fetched, previously discarded).
    """
    ordered: list[str] = []
    seen: set = set()
    # Raw signals per symbol, merged across the three scanners.
    sig: dict[str, dict] = {}

    def _add(sym: str) -> None:
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)

    def _sig(sym: str) -> dict:
        return sig.setdefault(sym, {"vol_ratio": 0.0, "gap_pct": 0.0, "anomaly_score": 0.0})

    # 1. MCP scanners (no symbols arg = server picks its own 50-ticker universe).
    #    Capture the catalyst payload, not just the symbol.
    try:
        res = scan_anomalies()
        if "error" not in res:
            for item in res.get("anomalies", []):
                s = item.get("symbol", "")
                _add(s)
                if s:
                    _sig(s.upper())["anomaly_score"] = float(item.get("total_score", 0) or 0)
    except Exception as e:
        logger.debug("scan_anomalies failed: %s", e)

    try:
        res = scan_volume_leaders()
        if "error" not in res:
            for item in res.get("leaders", []):
                s = item.get("symbol", "")
                _add(s)
                if s:
                    _sig(s.upper())["vol_ratio"] = float(item.get("ratio", 0) or 0)
    except Exception as e:
        logger.debug("scan_volume_leaders failed: %s", e)

    try:
        res = scan_gap_movers()
        if "error" not in res:
            for item in res.get("movers", []):
                s = item.get("symbol", "")
                _add(s)
                if s:
                    _sig(s.upper())["gap_pct"] = float(item.get("gap_pct", 0) or 0)
    except Exception as e:
        logger.debug("scan_gap_movers failed: %s", e)

    # 2. RSS news mentions — fill remaining slots with what the news is buzzing about
    try:
        cache = load_ticker_cache()
        by_mentions = sorted(
            cache.values(),
            key=lambda x: x.get("mention_count", 0),
            reverse=True,
        )
        for item in by_mentions:
            _add(item.get("symbol", ""))
    except Exception as e:
        logger.debug("ticker_cache unavailable: %s", e)

    # 3. Seed fallback
    if not ordered:
        ordered = list(_SEED_UNIVERSE)

    catalyst_map = {
        sym: _catalyst_bonus(s["vol_ratio"], s["gap_pct"], s["anomaly_score"])
        for sym, s in sig.items()
    }
    catalyst_map = {sym: b for sym, b in catalyst_map.items() if b > 0}
    return ordered[:MAX_UNIVERSE_SIZE], catalyst_map


def _scan_candidates(stop_event: threading.Event) -> tuple[list, dict]:
    """Build the dynamic universe and return ``(candidates, catalyst_map)``.

    ``candidates`` are shuffled, unheld symbols (so the bot rotates across the
    whole pool over successive cycles). Scoring itself is a single batched
    ``scan_universe`` call in _run_cycle — far faster than the old per-ticker
    ``analyze_ticker`` loop, and (crucially) it ranks each name against real
    peers instead of against itself, so scores actually differentiate.
    """
    universe, catalyst_map = _build_dynamic_universe()
    if stop_event.is_set():
        return [], {}

    with _lock:
        held = set(_state.open_positions.keys()) | _state.pending_sells

    seen: set = set()
    candidates = []
    for sym in universe:
        if sym in held or sym in seen:
            continue
        seen.add(sym)
        candidates.append(sym)
    # Shuffle so the bot doesn't always evaluate the same first N tickers
    random.shuffle(candidates)
    return candidates, catalyst_map


def _score_universe_batch(symbols: list[str]) -> dict[str, dict]:
    """Batch-score *symbols* in one ``scan_universe`` call.

    Returns ``{SYMBOL: {"score": float, "price": float|None}}``. This is the
    right MCP tool for ranking: the server computes momentum/valuation
    percentiles across the whole batch (real peer context), and now also
    returns the latest price so the bot can size positions and run price-based
    exits from this single call — no per-ticker round-trips.
    """
    syms = list(dict.fromkeys(s.upper() for s in symbols if s))
    if not syms:
        return {}
    result = scan_universe(syms)
    out: dict[str, dict] = {}
    if "error" in result:
        logger.warning("scan_universe failed: %s", result.get("error"))
        return out
    for row in result.get("scores", []):
        sym = row.get("symbol")
        if not sym:
            continue
        try:
            score = float(row.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        price = row.get("price")
        out[sym.upper()] = {
            "score": score,
            "price": float(price) if isinstance(price, (int, float)) else None,
        }
    return out


def _rss_sentiment_map() -> dict:
    """Map symbol -> sentiment tilt from MarketPulse RSS sentiment.

    Tilt = (bullish_ratio - bearish_ratio) * SENTIMENT_TILT_MAX, so a strongly
    bullish news ticker gets up to +10 points and a bearish one up to -10. This
    is the bridge that lets the news-sentiment tab influence what the bot trades.
    """
    out = {}
    try:
        cache = load_ticker_cache()
        for data in cache.values():
            sym = data.get("symbol")
            if not sym:
                continue
            tilt = (data.get("bullish_ratio", 0) - data.get("bearish_ratio", 0)) * SENTIMENT_TILT_MAX
            out[sym.upper()] = round(tilt, 1)
    except Exception as e:
        logger.debug("RSS sentiment map unavailable: %s", e)
    return out


def _score_candidates(
    candidates: list,
    batch_scores: dict,
    catalyst_map: dict | None,
    stop_event: threading.Event,
) -> list:
    """Rank candidates from pre-computed batch scores. Returns sorted list.

    Each candidate's composite ``scan_universe`` score is blended with:
      - MarketPulse RSS sentiment tilt (when the bridge is enabled), and
      - an event-driven catalyst bonus (volume surge / gap / anomaly).
    Re-entry gate: a recently-sold ticker must score REENTRY_SCORE_BOOST points
    above its sell score to re-enter.
    """
    with _lock:
        sell_hist = dict(_state.sell_history)
        min_score = _state.min_score
        use_sentiment = _state.sentiment_bridge

    tilt_map = _rss_sentiment_map() if use_sentiment else {}
    catalyst_map = catalyst_map or {}

    scored = []
    for sym in candidates:
        if stop_event.is_set():
            return scored
        info = batch_scores.get(sym.upper())
        if not info:
            continue
        base = info["score"]
        if base <= 0:
            continue

        tilt = tilt_map.get(sym.upper(), 0.0)
        catalyst = catalyst_map.get(sym.upper(), 0.0)
        score = max(0.0, min(100.0, base + tilt + catalyst))
        if score < min_score:
            continue

        # Re-entry gate: need higher conviction to re-buy
        if sym in sell_hist:
            required = sell_hist[sym]["score"] + REENTRY_SCORE_BOOST
            if score < required:
                logger.debug(
                    "Re-entry blocked for %s: score %.0f < required %.0f (sold at %.0f + %d)",
                    sym, score, required, sell_hist[sym]["score"], REENTRY_SCORE_BOOST,
                )
                continue

        price = info.get("price")
        if price and float(price) > 0:
            scored.append({
                "ticker": sym, "score": score, "price": float(price),
                "base_score": base, "sentiment_tilt": tilt, "catalyst": catalyst,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _enter_positions(
    portfolio_id: str,
    scored: list,
    high_vix: bool,
    stop_event: threading.Event,
) -> None:
    """Enter positions using Kelly-based sizing with rotation.

    Casino mindset: many small bets sized by mathematical edge.
    Position size comes from _compute_position_size (Kelly + safety rails),
    not fixed percentage tiers.
    """
    with _lock:
        remaining_cash = _state.portfolio_cash
        portfolio_value = _state.portfolio_value
        current_stats = _state.stats
        max_positions = _state.max_positions
        max_risk = _state.max_risk_per_trade

    for candidate in scored:
        if stop_event.is_set():
            return

        with _lock:
            at_max = len(_state.open_positions) >= max_positions
            weakest_ticker = None
            weakest_score = float("inf")
            if at_max and _state.open_positions:
                for t, p in _state.open_positions.items():
                    s = p.get("current_score", p["entry_score"])
                    if s < weakest_score:
                        weakest_score = s
                        weakest_ticker = t

        # Rotation: only swap if candidate is meaningfully better
        if at_max:
            if weakest_ticker is None:
                break
            if candidate["score"] <= weakest_score + 10:
                continue
            with _lock:
                weak_pos = _state.open_positions.get(weakest_ticker)
            if weak_pos:
                reason = f"rotated for {candidate['ticker']} ({weakest_score:.0f}→{candidate['score']:.0f})"
                sold = _sell_position(portfolio_id, weakest_ticker, weak_pos, reason)
                if not sold:
                    continue
                with _lock:
                    remaining_cash = _state.portfolio_cash

        # Kelly-based position sizing
        dollar_amount = _compute_position_size(
            candidate["score"], high_vix, current_stats, portfolio_value, max_risk
        )
        if dollar_amount <= 0:
            continue

        price = candidate["price"]
        shares = int(dollar_amount / price)
        if shares < 1:
            # Kelly allocation rounded to zero shares — common on a $10k account
            # where 1% (~$80) can't buy one share of a $150+ name. Buy a single
            # share if it still fits the per-trade cap, so the bot can actually
            # take positions in pricier tickers instead of only cheap ones.
            if price <= portfolio_value * max_risk and price <= remaining_cash:
                shares = 1
            else:
                continue
        # Don't spend more than available cash
        cost = shares * price
        if cost > remaining_cash:
            shares = int(remaining_cash / price)
            if shares < 1:
                continue

        result = execute_buy(portfolio_id, candidate["ticker"], shares)
        if "error" not in result:
            actual_cost = shares * candidate["price"]
            risk_pct = actual_cost / portfolio_value * 100 if portfolio_value > 0 else 0
            tilt = candidate.get("sentiment_tilt", 0.0)
            sentiment_note = (
                f" · news {tilt:+.0f}" if abs(tilt) >= 1 else ""
            )
            with _lock:
                _state.open_positions[candidate["ticker"]] = {
                    "entry_price": candidate["price"],
                    "shares": shares,
                    "entry_score": candidate["score"],
                    "entry_time": datetime.now(),
                    "entry_cycle": _state.cycle_count,
                    "current_price": candidate["price"],
                    "current_score": candidate["score"],
                    "peak_price": candidate["price"],
                }
                _state.trade_log.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "BUY",
                    "ticker": candidate["ticker"],
                    "price": candidate["price"],
                    "shares": shares,
                    "score": candidate["score"],
                    "reason": f"score {candidate['score']:.0f}{sentiment_note} · {risk_pct:.1f}% risk",
                    "pnl": 0.0,
                })
            remaining_cash -= actual_cost
            logger.info(
                "Bought %s: %d shares @ $%.2f (score %.0f, %.1f%% of portfolio)",
                candidate["ticker"], shares, candidate["price"],
                candidate["score"], risk_pct,
            )
        else:
            logger.warning("Buy failed for %s: %s", candidate["ticker"], result["error"])


def _snapshot_portfolio(portfolio_id: str) -> None:
    """Reconcile BotState cash/value from MCP server and recompute trade stats."""
    result = analyze_portfolio(portfolio_id)
    if "error" not in result:
        portfolio_info = result.get("portfolio", {})
        with _lock:
            _state.portfolio_cash = portfolio_info.get("current_cash", _state.portfolio_cash)
            _state.portfolio_value = result.get("total_value", _state.portfolio_value)
            # Mark-to-market P&L so it always reconciles with Portfolio Value
            # (the seeded illustrative trades drive the Edge panel, not this).
            _state.total_pnl = _state.portfolio_value - _state.starting_capital
    else:
        logger.warning("Portfolio snapshot failed: %s", result["error"])

    # Append to the equity curve (used for the dashboard chart)
    now = datetime.now()
    with _lock:
        _state.equity_curve.append({
            "time": now.isoformat(timespec="seconds"),
            "value": round(_state.portfolio_value, 2),
        })
        if len(_state.equity_curve) > EQUITY_CURVE_MAX:
            _state.equity_curve = _state.equity_curve[-EQUITY_CURVE_MAX:]

    # Expire stale sell history entries (>24h old)
    with _lock:
        expired = [
            t for t, info in _state.sell_history.items()
            if (now - info["time"]).total_seconds() > SELL_HISTORY_EXPIRY_HOURS * 3600
        ]
        for t in expired:
            del _state.sell_history[t]

    # Recompute trade statistics every cycle
    with _lock:
        _state.stats = _compute_trade_stats(_state.trade_log)
        s = _state.stats

    # Persist the ledger so the bot resumes warm after a restart
    _save_state()
    if s.total_trades > 0:
        logger.info(
            "Stats: %d trades, %.0f%% win rate, EV=$%.2f, Kelly=%.2f%%, RoR=%.4f%%, R:R=%.2f",
            s.total_trades, s.win_rate * 100, s.expected_value,
            s.kelly_fraction * 100, s.risk_of_ruin * 100, s.reward_risk_ratio,
        )


def _run_cycle(portfolio_id: str, stop_event: threading.Event) -> bool:
    """Execute one full trading cycle. Returns True if MCP server responded."""
    logger.info("=== Cycle %d start ===", _state.cycle_count)

    regime = detect_market_regime()
    mcp_alive = "error" not in regime
    if mcp_alive:
        logger.info("Market regime: %s (score %s)", regime.get("regime"), regime.get("score"))

    high_vix = _check_vix()
    _retry_pending_sells(portfolio_id, stop_event)
    if stop_event.is_set():
        return mcp_alive

    candidates, catalyst_map = _scan_candidates(stop_event)
    if stop_event.is_set():
        return mcp_alive

    # One batched scan_universe call ranks held positions and this cycle's
    # candidate slice together, against real peers — and returns prices. Both
    # the exit engine and the entry engine read from it, so a full cycle makes a
    # single scoring round-trip instead of ~45 per-ticker calls.
    with _lock:
        held = list(_state.open_positions.keys())
    to_score = candidates[:SCORE_CANDIDATES_LIMIT]
    batch_scores = _score_universe_batch(held + to_score)
    if stop_event.is_set():
        return mcp_alive

    _check_exits(portfolio_id, batch_scores, stop_event)
    if stop_event.is_set():
        return mcp_alive

    if to_score:
        scored = _score_candidates(to_score, batch_scores, catalyst_map, stop_event)
        if not stop_event.is_set():
            _enter_positions(portfolio_id, scored, high_vix, stop_event)
    else:
        logger.warning("No candidates — skipping entry phase")
    _snapshot_portfolio(portfolio_id)
    return mcp_alive


# ---------------------------------------------------------------------------
# BotEngine — controls the background loop
# ---------------------------------------------------------------------------

class BotEngine:
    """Start/stop the background trading thread."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        with _lock:
            if _state.is_running:
                return

        if _state.portfolio_id is None:
            result = create_portfolio(
                starting_capital=_state.starting_capital,
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

        # Warm start: on a cold account (no prior/seeded trades), populate an
        # illustrative ledger so the Edge panel and equity curve are alive
        # immediately while the first live cycle fills real positions.
        if _state.warm_start and not _state.trade_log:
            seed_warm_start()

        self._stop_event.clear()
        with _lock:
            _state.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Bot started")

    def stop(self) -> None:
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
        with _lock:
            portfolio_id = _state.portfolio_id

        consecutive_failures = 0

        while not self._stop_event.is_set():
            with _lock:
                _state.cycle_count += 1
                _state.last_cycle_time = datetime.now()
            try:
                healthy = _run_cycle(portfolio_id, self._stop_event)
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
                healthy = False

            if healthy:
                consecutive_failures = 0
                wait = CYCLE_INTERVAL
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "MCP server unreachable for %d consecutive cycles — auto-stopping bot",
                        consecutive_failures,
                    )
                    break
                wait = min(
                    INITIAL_BACKOFF * (2 ** (consecutive_failures - 1)),
                    MAX_BACKOFF,
                )
                logger.warning(
                    "MCP failure %d/%d — backing off %ds before retry",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES, wait,
                )

            for _ in range(wait):
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
