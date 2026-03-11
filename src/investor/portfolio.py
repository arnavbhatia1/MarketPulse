"""
Portfolio management — high-level operations that orchestrate broker + DB.

Provides portfolio creation, summary generation, performance computation,
daily snapshots, and position sizing for buy candidates.
"""

import logging
import math
import uuid
from datetime import datetime

from src.storage import db

logger = logging.getLogger(__name__)

VALID_RISK_PROFILES = {"conservative", "moderate", "aggressive"}
VALID_HORIZONS = {"short", "medium", "long"}
MIN_CAPITAL = 10_000.0
MAX_CAPITAL = 1_000_000.0


# ── Portfolio creation ───────────────────────────────────────────────

def create_user_portfolio(
    user_id: str,
    starting_capital: float,
    risk_profile: str,
    investment_horizon: str,
) -> str:
    """Create a new portfolio after validating inputs. Returns portfolio_id."""
    if not (MIN_CAPITAL <= starting_capital <= MAX_CAPITAL):
        raise ValueError(
            f"starting_capital must be between {MIN_CAPITAL:,.0f} and "
            f"{MAX_CAPITAL:,.0f}, got {starting_capital:,.2f}"
        )
    if risk_profile not in VALID_RISK_PROFILES:
        raise ValueError(
            f"risk_profile must be one of {VALID_RISK_PROFILES}, got '{risk_profile}'"
        )
    if investment_horizon not in VALID_HORIZONS:
        raise ValueError(
            f"investment_horizon must be one of {VALID_HORIZONS}, got '{investment_horizon}'"
        )

    portfolio_id = db.create_portfolio(
        user_id=user_id,
        starting_capital=starting_capital,
        risk_profile=risk_profile,
        investment_horizon=investment_horizon,
    )
    logger.info(
        "Created portfolio %s for user %s (capital=%.2f, risk=%s, horizon=%s)",
        portfolio_id, user_id, starting_capital, risk_profile, investment_horizon,
    )
    return portfolio_id


# ── Portfolio summary ────────────────────────────────────────────────

def get_portfolio_summary(portfolio_id: str) -> dict | None:
    """Return a rich summary with live prices, weights, and allocations."""
    portfolio = db.get_portfolio(portfolio_id)
    if portfolio is None:
        return None

    from src.investor.market_data import get_current_price

    holdings_raw = db.get_holdings(portfolio_id)
    holdings = []
    holdings_value = 0.0
    sector_totals: dict[str, float] = {}
    geo_totals: dict[str, float] = {}

    for h in holdings_raw:
        current_price = get_current_price(h['symbol'])
        if current_price is None:
            # Fall back to cost basis if market data unavailable
            current_price = h['avg_cost_basis']

        current_value = current_price * h['shares']
        cost_value = h['avg_cost_basis'] * h['shares']
        gain_loss = current_value - cost_value
        holdings_value += current_value

        holdings.append({
            **h,
            'current_price': current_price,
            'current_value': current_value,
            'gain_loss': gain_loss,
            'weight': 0.0,  # filled in after total_value is known
        })

        sector = h.get('sector') or 'Unknown'
        sector_totals[sector] = sector_totals.get(sector, 0.0) + current_value

        geo = h.get('geography') or 'unknown'
        geo_totals[geo] = geo_totals.get(geo, 0.0) + current_value

    total_value = portfolio['current_cash'] + holdings_value

    # Fill in weights now that total_value is known
    for h in holdings:
        h['weight'] = h['current_value'] / total_value if total_value > 0 else 0.0

    # Normalize allocations to fractions
    sector_allocation = {}
    for sector, val in sector_totals.items():
        sector_allocation[sector] = val / total_value if total_value > 0 else 0.0

    geo_allocation = {}
    for geo, val in geo_totals.items():
        geo_allocation[geo] = val / total_value if total_value > 0 else 0.0

    # Daily change from latest snapshot
    snapshots = db.get_snapshots(portfolio_id, limit=2)
    daily_change = 0.0
    daily_change_pct = 0.0
    if len(snapshots) >= 1:
        prev_value = snapshots[0]['total_value']
        daily_change = total_value - prev_value
        daily_change_pct = daily_change / prev_value if prev_value > 0 else 0.0

    return {
        'portfolio': portfolio,
        'holdings': holdings,
        'total_value': total_value,
        'holdings_value': holdings_value,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'sector_allocation': sector_allocation,
        'geo_allocation': geo_allocation,
    }


# ── Performance computation ──────────────────────────────────────────

def compute_performance(portfolio_id: str) -> dict:
    """Compute portfolio performance metrics from stored snapshots."""
    portfolio = db.get_portfolio(portfolio_id)
    if portfolio is None:
        return {}

    snapshots = db.get_snapshots(portfolio_id, limit=365)
    starting_capital = portfolio['starting_capital']

    result = {
        'cumulative_return': 0.0,
        'daily_return': 0.0,
        'benchmark_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
    }

    if not snapshots:
        return result

    # Snapshots come newest-first; reverse for chronological order
    snapshots_chrono = list(reversed(snapshots))

    latest_value = snapshots_chrono[-1]['total_value']
    result['cumulative_return'] = (latest_value / starting_capital) - 1.0

    # Daily return from last two snapshots
    if len(snapshots_chrono) >= 2:
        prev = snapshots_chrono[-2]['total_value']
        if prev > 0:
            result['daily_return'] = (latest_value / prev) - 1.0

    # Benchmark return (SPY) over the same period
    try:
        from src.investor.market_data import get_current_price
        import yfinance as yf

        first_date = snapshots_chrono[0]['snapshot_date']
        ticker = yf.Ticker("SPY")
        hist = ticker.history(start=first_date)
        if len(hist) >= 2:
            spy_start = hist['Close'].iloc[0]
            spy_end = hist['Close'].iloc[-1]
            if spy_start > 0:
                result['benchmark_return'] = (spy_end / spy_start) - 1.0
    except Exception:
        logger.debug("Could not compute benchmark return for SPY")

    # Sharpe ratio (annualized)
    if len(snapshots_chrono) >= 3:
        daily_returns = []
        for i in range(1, len(snapshots_chrono)):
            prev_val = snapshots_chrono[i - 1]['total_value']
            curr_val = snapshots_chrono[i]['total_value']
            if prev_val > 0:
                daily_returns.append((curr_val / prev_val) - 1.0)

        if daily_returns:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_dev = math.sqrt(variance)
            risk_free_daily = 0.05 / 252  # ~5% annual risk-free rate
            if std_dev > 0:
                result['sharpe_ratio'] = (
                    (mean_return - risk_free_daily) / std_dev * math.sqrt(252)
                )

    # Max drawdown (worst peak-to-trough)
    peak = 0.0
    max_dd = 0.0
    for snap in snapshots_chrono:
        val = snap['total_value']
        if val > peak:
            peak = val
        if peak > 0:
            drawdown = (peak - val) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    result['max_drawdown'] = max_dd

    return result


# ── Daily snapshot ───────────────────────────────────────────────────

def take_snapshot(portfolio_id: str) -> dict:
    """Create a daily portfolio snapshot with current prices and metrics."""
    summary = get_portfolio_summary(portfolio_id)
    if summary is None:
        raise ValueError(f"Portfolio {portfolio_id} not found")

    portfolio = summary['portfolio']
    perf = compute_performance(portfolio_id)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    snapshot = {
        'id': str(uuid.uuid4()),
        'portfolio_id': portfolio_id,
        'snapshot_date': today,
        'total_value': summary['total_value'],
        'cash_value': portfolio['current_cash'],
        'holdings_value': summary['holdings_value'],
        'daily_return': perf.get('daily_return', 0.0),
        'cumulative_return': perf.get('cumulative_return', 0.0),
        'benchmark_return': perf.get('benchmark_return', 0.0),
        'sharpe_ratio': perf.get('sharpe_ratio', 0.0),
        'max_drawdown': perf.get('max_drawdown', 0.0),
        'sector_allocation': summary['sector_allocation'],
        'geo_allocation': summary['geo_allocation'],
    }

    db.save_snapshot(snapshot)
    logger.info(
        "Snapshot for %s on %s: total=$%.2f, cum_return=%.4f",
        portfolio_id, today, summary['total_value'], perf.get('cumulative_return', 0.0),
    )
    return snapshot


# ── Position sizing ──────────────────────────────────────────────────

def generate_position_sizes(
    buy_candidates: list[dict],
    portfolio_value: float,
    available_cash: float,
    risk_profile: str,
    config: dict,
) -> list[dict]:
    """
    Size positions for scored buy candidates.

    Each candidate dict must have 'symbol' and 'score' keys.
    Returns a new list with 'shares', 'estimated_cost' added.
    Candidates with zero shares after flooring are dropped.
    """
    if not buy_candidates:
        return []

    # Read position limits from config
    position_limits = config.get('investor', {}).get('position_limits', {})
    profile_limits = position_limits.get(risk_profile, position_limits.get('moderate', {}))
    max_position_pct = profile_limits.get('max_position', 0.08)
    cash_deploy_ratio = config.get('investor', {}).get('scoring', {}).get(
        'cash_deploy_ratio', 0.80
    )

    # Budget: deploy up to cash_deploy_ratio of available cash,
    # but never more than max_position_pct of portfolio per position
    buy_budget = min(
        available_cash * cash_deploy_ratio,
        portfolio_value * max_position_pct * len(buy_candidates),
    )
    if buy_budget <= 0:
        return []

    # Score-proportional allocation
    total_score = sum(c['score'] for c in buy_candidates)
    if total_score <= 0:
        return []

    from src.investor.market_data import get_current_price

    sized = []
    for candidate in buy_candidates:
        weight = candidate['score'] / total_score
        allocation = buy_budget * weight

        # Cap each position at max_position_pct of portfolio
        max_alloc = portfolio_value * max_position_pct
        allocation = min(allocation, max_alloc)

        price = get_current_price(candidate['symbol'])
        if price is None or price <= 0:
            continue

        # Floor to whole shares
        shares = int(allocation / price)
        if shares <= 0:
            continue

        estimated_cost = shares * price
        sized.append({
            **candidate,
            'shares': shares,
            'estimated_cost': estimated_cost,
        })

    return sized
