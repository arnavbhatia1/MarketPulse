"""
Risk Management Framework for MarketPulse.

Provides position-limit checking, portfolio allocation analysis,
stress testing against historical scenarios, event-driven trigger
detection, and defensive ETF recommendations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Sector sensitivity coefficients ──────────────────────────────────
# How much each sector historically moved (relative to broad market = 1.0)
# during three reference crises.

SECTOR_SENSITIVITY = {
    "Technology": {
        "recession_2008": 1.20,
        "covid_2020": 0.85,
        "rate_hike_2022": 1.40,
    },
    "Consumer Discretionary": {
        "recession_2008": 1.30,
        "covid_2020": 1.10,
        "rate_hike_2022": 1.15,
    },
    "Financials": {
        "recession_2008": 1.50,
        "covid_2020": 1.00,
        "rate_hike_2022": 0.90,
    },
    "Energy": {
        "recession_2008": 1.10,
        "covid_2020": 1.40,
        "rate_hike_2022": 0.60,
    },
    "Healthcare": {
        "recession_2008": 0.75,
        "covid_2020": 0.70,
        "rate_hike_2022": 0.85,
    },
    "Consumer Staples": {
        "recession_2008": 0.60,
        "covid_2020": 0.65,
        "rate_hike_2022": 0.80,
    },
    "Utilities": {
        "recession_2008": 0.55,
        "covid_2020": 0.70,
        "rate_hike_2022": 0.75,
    },
    "Industrials": {
        "recession_2008": 1.15,
        "covid_2020": 1.05,
        "rate_hike_2022": 1.00,
    },
    "Real Estate": {
        "recession_2008": 1.40,
        "covid_2020": 0.90,
        "rate_hike_2022": 1.20,
    },
    "Materials": {
        "recession_2008": 1.10,
        "covid_2020": 0.95,
        "rate_hike_2022": 1.00,
    },
    "Communication Services": {
        "recession_2008": 0.90,
        "covid_2020": 0.80,
        "rate_hike_2022": 1.30,
    },
    "Bonds/Fixed Income": {
        "recession_2008": 0.20,
        "covid_2020": 0.15,
        "rate_hike_2022": 0.90,
    },
    "Broad Market ETFs": {
        "recession_2008": 1.00,
        "covid_2020": 1.00,
        "rate_hike_2022": 1.00,
    },
}

# Default sensitivity for unknown sectors — assumes broad-market behaviour
_DEFAULT_SENSITIVITY = {"recession_2008": 1.00, "covid_2020": 1.00, "rate_hike_2022": 1.00}


# ── Position-limit checks ───────────────────────────────────────────

def check_position_limits(
    symbol: str,
    proposed_value: float,
    holdings: list,
    portfolio_value: float,
    risk_profile: str,
    config: dict,
) -> dict:
    """Validate a proposed position against risk-profile limits.

    Returns::

        {
            "allowed": bool,
            "violations": [str, ...],
            "details": {
                "position_pct": float,
                "sector_pct": float,
                "cash_pct": float,
                "stocks_pct": float,
            },
        }
    """
    limits = (
        config.get("investor", {})
        .get("position_limits", {})
        .get(risk_profile, {})
    )
    violations: list[str] = []

    if portfolio_value <= 0:
        return {"allowed": False, "violations": ["Portfolio value is zero or negative"], "details": {}}

    max_position = limits.get("max_position", 0.08)
    max_sector = limits.get("max_sector", 0.30)
    min_cash = limits.get("min_cash", 0.10)
    max_stocks = limits.get("max_stocks", 0.70)

    # --- single position check ---
    position_pct = proposed_value / portfolio_value
    if position_pct > max_position:
        violations.append(
            f"Position {symbol} would be {position_pct:.1%} of portfolio "
            f"(limit: {max_position:.1%})"
        )

    # --- sector concentration ---
    # Find the sector of the symbol from current holdings or proposed
    symbol_sector = None
    for h in (holdings or []):
        if h.get("symbol") == symbol:
            symbol_sector = h.get("sector")
            break

    sector_alloc = get_sector_allocation(holdings or [], portfolio_value)
    sector_pct = sector_alloc.get(symbol_sector, 0.0) if symbol_sector else 0.0
    added_sector_pct = sector_pct + (proposed_value / portfolio_value)
    if symbol_sector and added_sector_pct > max_sector:
        violations.append(
            f"Sector '{symbol_sector}' would reach {added_sector_pct:.1%} "
            f"(limit: {max_sector:.1%})"
        )

    # --- cash reserve ---
    current_cash = _estimate_cash(holdings, portfolio_value)
    cash_after = (current_cash - proposed_value) / portfolio_value
    if cash_after < min_cash:
        violations.append(
            f"Cash would drop to {cash_after:.1%} (minimum: {min_cash:.1%})"
        )

    # --- stocks vs ETFs cap ---
    stocks_value = sum(
        h.get("shares", 0) * h.get("avg_cost_basis", 0)
        for h in (holdings or [])
        if h.get("asset_type") == "stock"
    )
    # Assume the proposed trade is a stock unless we know otherwise
    stocks_pct = (stocks_value + proposed_value) / portfolio_value
    if stocks_pct > max_stocks:
        violations.append(
            f"Individual stocks would be {stocks_pct:.1%} of portfolio "
            f"(limit: {max_stocks:.1%})"
        )

    return {
        "allowed": len(violations) == 0,
        "violations": violations,
        "details": {
            "position_pct": round(position_pct, 4),
            "sector_pct": round(added_sector_pct, 4) if symbol_sector else None,
            "cash_pct": round(cash_after, 4),
            "stocks_pct": round(stocks_pct, 4),
        },
    }


def _estimate_cash(holdings: list, portfolio_value: float) -> float:
    """Estimate current cash as portfolio_value minus total holdings value."""
    if not holdings:
        return portfolio_value
    holdings_value = sum(
        h.get("shares", 0) * h.get("avg_cost_basis", 0) for h in holdings
    )
    return max(0.0, portfolio_value - holdings_value)


# ── Allocation analysis ─────────────────────────────────────────────

def get_sector_allocation(holdings: list, portfolio_value: float) -> dict:
    """Return sector -> weight (0-1) mapping.

    Example: ``{"Technology": 0.28, "Healthcare": 0.15, ...}``
    """
    if not holdings or portfolio_value <= 0:
        return {}

    sector_totals: dict[str, float] = {}
    for h in holdings:
        sector = h.get("sector") or "Unknown"
        value = h.get("shares", 0) * h.get("avg_cost_basis", 0)
        sector_totals[sector] = sector_totals.get(sector, 0.0) + value

    return {
        sector: round(val / portfolio_value, 4)
        for sector, val in sorted(sector_totals.items(), key=lambda x: -x[1])
    }


def get_geo_allocation(holdings: list, portfolio_value: float) -> dict:
    """Return geography -> weight (0-1) mapping.

    Example: ``{"us": 0.70, "intl_developed": 0.20, "emerging": 0.10}``
    """
    if not holdings or portfolio_value <= 0:
        return {}

    geo_totals: dict[str, float] = {}
    for h in holdings:
        geo = h.get("geography") or "us"
        value = h.get("shares", 0) * h.get("avg_cost_basis", 0)
        geo_totals[geo] = geo_totals.get(geo, 0.0) + value

    return {
        geo: round(val / portfolio_value, 4)
        for geo, val in sorted(geo_totals.items(), key=lambda x: -x[1])
    }


# ── Stress testing ───────────────────────────────────────────────────

def compute_stress_score(
    holdings: list,
    portfolio_value: float,
    config: dict,
) -> dict:
    """Estimate portfolio vulnerability to historical stress scenarios.

    Returns::

        {
            "stress_score": float (0-1),
            "scenario_drawdowns": {"recession_2008": -0.28, ...},
            "vulnerable_sectors": ["Technology", "Financials"],
        }
    """
    if not holdings or portfolio_value <= 0:
        return {
            "stress_score": 0.0,
            "scenario_drawdowns": {},
            "vulnerable_sectors": [],
        }

    stress_cfg = config.get("investor", {}).get("stress_scenarios", {})
    sector_alloc = get_sector_allocation(holdings, portfolio_value)

    scenario_drawdowns: dict[str, float] = {}

    for scenario, broad_drawdown in stress_cfg.items():
        weighted_drawdown = 0.0
        for sector, weight in sector_alloc.items():
            sensitivity = SECTOR_SENSITIVITY.get(sector, _DEFAULT_SENSITIVITY)
            sector_multiplier = sensitivity.get(scenario, 1.0)
            weighted_drawdown += weight * broad_drawdown * sector_multiplier
        scenario_drawdowns[scenario] = round(weighted_drawdown, 4)

    # stress_score = worst-case drawdown magnitude scaled to 0-1
    if scenario_drawdowns:
        worst_drawdown = min(scenario_drawdowns.values())  # most negative
        stress_score = min(1.0, abs(worst_drawdown))
    else:
        stress_score = 0.0

    # vulnerable sectors: those with sensitivity > 1.0 in the worst scenario
    worst_scenario = (
        min(scenario_drawdowns, key=scenario_drawdowns.get)
        if scenario_drawdowns
        else None
    )
    vulnerable_sectors = []
    if worst_scenario:
        for sector in sector_alloc:
            sensitivity = SECTOR_SENSITIVITY.get(sector, _DEFAULT_SENSITIVITY)
            if sensitivity.get(worst_scenario, 1.0) > 1.0:
                vulnerable_sectors.append(sector)

    return {
        "stress_score": round(stress_score, 4),
        "scenario_drawdowns": scenario_drawdowns,
        "vulnerable_sectors": vulnerable_sectors,
    }


# ── Event-driven triggers ───────────────────────────────────────────

def check_event_triggers(
    holdings: list,
    portfolio_value: float,
    sentiment_cache: dict,
    prev_sentiment_cache: dict,
    config: dict,
) -> list:
    """Check the four event-driven rebalance triggers.

    Triggers checked:
      1. **Sentiment flip** — a held ticker's dominant sentiment reverses
         with high confidence.
      2. **Drawdown breach** — a holding's unrealized loss exceeds the
         drawdown threshold.
      3. **Concentration breach** — a sector exceeds its limit + buffer.
      4. **Stress spike** — portfolio stress score exceeds action threshold.

    Returns list of triggered event dicts.
    """
    investor_cfg = config.get("investor", {})
    triggers_cfg = investor_cfg.get("rebalance", {}).get("event_triggers", {})
    events: list[dict] = []

    # 1. Sentiment flip
    flip_conf = triggers_cfg.get("sentiment_flip_confidence", 0.7)
    events.extend(
        _check_sentiment_flip(holdings, sentiment_cache, prev_sentiment_cache, flip_conf)
    )

    # 2. Drawdown breach
    drawdown_thresh = triggers_cfg.get("drawdown_threshold", 0.10)
    events.extend(_check_drawdown_breach(holdings, drawdown_thresh))

    # 3. Concentration breach
    risk_profile = _infer_risk_profile(holdings, config)
    limits = investor_cfg.get("position_limits", {}).get(risk_profile, {})
    buffer = triggers_cfg.get("concentration_breach_buffer", 0.05)
    events.extend(
        _check_concentration_breach(holdings, portfolio_value, limits, buffer)
    )

    # 4. Stress spike
    stress = compute_stress_score(holdings, portfolio_value, config)
    stress_thresholds = investor_cfg.get("stress_thresholds", {}).get(risk_profile, {})
    action_level = stress_thresholds.get("action", 0.33)
    if stress["stress_score"] >= action_level:
        events.append({
            "trigger": "stress_spike",
            "severity": "high",
            "stress_score": stress["stress_score"],
            "action_threshold": action_level,
            "vulnerable_sectors": stress["vulnerable_sectors"],
            "message": (
                f"Stress score {stress['stress_score']:.2f} exceeds "
                f"action threshold {action_level:.2f}"
            ),
        })

    if events:
        logger.info("Event triggers fired: %d events", len(events))
    return events


def _check_sentiment_flip(
    holdings: list,
    sentiment_cache: dict,
    prev_sentiment_cache: dict,
    min_confidence: float,
) -> list:
    """Detect sentiment reversals for held symbols."""
    events = []
    if not holdings or not sentiment_cache or not prev_sentiment_cache:
        return events

    _opposites = {
        "bullish": "bearish",
        "bearish": "bullish",
    }

    for h in holdings:
        sym = h.get("symbol", "")
        current = sentiment_cache.get(sym, {})
        previous = prev_sentiment_cache.get(sym, {})

        cur_sent = current.get("dominant_sentiment")
        prev_sent = previous.get("dominant_sentiment")
        cur_conf = current.get("avg_confidence", 0.0)

        if (
            cur_sent
            and prev_sent
            and _opposites.get(prev_sent) == cur_sent
            and cur_conf >= min_confidence
        ):
            events.append({
                "trigger": "sentiment_flip",
                "symbol": sym,
                "severity": "high",
                "from_sentiment": prev_sent,
                "to_sentiment": cur_sent,
                "confidence": cur_conf,
                "message": (
                    f"{sym} sentiment flipped from {prev_sent} to {cur_sent} "
                    f"(confidence: {cur_conf:.2f})"
                ),
            })
    return events


def _check_drawdown_breach(holdings: list, threshold: float) -> list:
    """Flag holdings whose unrealized loss exceeds threshold."""
    events = []
    for h in (holdings or []):
        avg_cost = h.get("avg_cost_basis", 0)
        current_price = h.get("current_price")
        if avg_cost <= 0 or current_price is None:
            continue
        drawdown = (current_price - avg_cost) / avg_cost
        if drawdown < -threshold:
            events.append({
                "trigger": "drawdown_breach",
                "symbol": h.get("symbol", ""),
                "severity": "high" if drawdown < -2 * threshold else "medium",
                "drawdown": round(drawdown, 4),
                "threshold": threshold,
                "message": (
                    f"{h.get('symbol', '')} is down {abs(drawdown):.1%} "
                    f"(threshold: {threshold:.1%})"
                ),
            })
    return events


def _check_concentration_breach(
    holdings: list,
    portfolio_value: float,
    limits: dict,
    buffer: float,
) -> list:
    """Flag sectors exceeding max_sector + buffer."""
    events = []
    max_sector = limits.get("max_sector", 0.30)
    breach_level = max_sector + buffer

    sector_alloc = get_sector_allocation(holdings, portfolio_value)
    for sector, weight in sector_alloc.items():
        if weight > breach_level:
            events.append({
                "trigger": "concentration_breach",
                "sector": sector,
                "severity": "medium",
                "current_weight": round(weight, 4),
                "limit": max_sector,
                "breach_level": breach_level,
                "message": (
                    f"Sector '{sector}' at {weight:.1%} exceeds "
                    f"limit+buffer ({breach_level:.1%})"
                ),
            })
    return events


def _infer_risk_profile(holdings: list, config: dict) -> str:
    """Best-effort inference of risk profile. Defaults to 'moderate'."""
    # In a real integration the profile comes from the portfolio record.
    # Here we return a sensible default.
    return "moderate"


# ── Defensive ETF recommendations ───────────────────────────────────

def get_defensive_etfs(stress_score: float, risk_profile: str) -> list:
    """Recommend defensive ETF symbols based on stress level.

    Higher stress -> heavier tilt toward treasuries and bonds.
    Lower stress -> lighter defensive allocation.
    """
    # Tiered recommendations
    light = ["BND", "VIG"]                              # low stress
    moderate_list = ["BND", "VGIT", "VIG", "SCHD"]      # moderate stress
    heavy = ["VGSH", "VGIT", "VGLT", "VTIP", "BND", "AGG"]  # high stress

    # Thresholds vary by risk profile
    _thresholds = {
        "conservative": (0.15, 0.22),
        "moderate": (0.25, 0.32),
        "aggressive": (0.32, 0.40),
    }
    low_thresh, high_thresh = _thresholds.get(risk_profile, (0.25, 0.32))

    if stress_score >= high_thresh:
        return heavy
    elif stress_score >= low_thresh:
        return moderate_list
    else:
        return light
