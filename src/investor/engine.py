"""
Alpha Engine — proprietary scoring formula for MarketPulse.

Computes a composite score (0-100) for each ticker by blending:
  - Sentiment composite  (crowd mood from Reddit / Stocktwits / news)
  - Valuation composite  (PE, EV/EBITDA, P/B, dividend yield, market cap)
  - Momentum composite   (30d/90d price momentum, relative strength, volatility)
  - Risk penalty          (concentration, geography, correlation, drawdown)

Weights are risk-profile-dependent (config/default.yaml → investor.formula_weights).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Sentiment label → numeric mapping ────────────────────────────────
_SENTIMENT_NUMERIC = {
    "bullish": 1.0,
    "neutral": 0.0,
    "bearish": -1.0,
    "meme": 0.2,
}


# ── Helper functions ─────────────────────────────────────────────────

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* to [min_val, max_val] then scale to [0, 1].

    Returns 0.5 when the range is degenerate (min == max).
    """
    if min_val == max_val:
        return 0.5
    clamped = max(min_val, min(max_val, value))
    return (clamped - min_val) / (max_val - min_val)


def percentile_rank(value: float, all_values: list) -> float:
    """Scipy-free percentile rank (0-100).

    Returns the percentage of *all_values* that are <= *value*.
    Returns 50.0 for an empty list.
    """
    if not all_values:
        return 50.0
    count_le = sum(1 for v in all_values if v is not None and v <= value)
    total = sum(1 for v in all_values if v is not None)
    if total == 0:
        return 50.0
    return (count_le / total) * 100.0


# ── Sentiment composite ─────────────────────────────────────────────

def compute_sentiment_composite(ticker_data: dict) -> Optional[float]:
    """Compute a 0-100 sentiment composite from a ticker_cache row.

    Returns None when *ticker_data* is empty or lacks sentiment info.
    """
    if not ticker_data:
        return None

    dominant = ticker_data.get("dominant_sentiment")
    if dominant is None:
        return None

    # --- sentiment_score: map label → numeric ---
    sentiment_score = _SENTIMENT_NUMERIC.get(dominant, 0.0)

    # --- sentiment_momentum: last-3-day avg vs first-4-day avg ---
    sentiment_by_day = ticker_data.get("sentiment_by_day") or {}
    momentum = _compute_sentiment_momentum(sentiment_by_day)

    # --- confidence: raw avg_confidence ---
    confidence = float(ticker_data.get("avg_confidence", 0.0))

    # --- source_agreement: fraction of per-source sentiments matching dominant ---
    source_keys = ("reddit_sentiment", "stocktwits_sentiment", "news_sentiment")
    available_sources = [
        ticker_data.get(k) for k in source_keys if ticker_data.get(k) is not None
    ]
    if available_sources:
        matches = sum(1 for s in available_sources if s == dominant)
        agreement = matches / len(available_sources)
    else:
        agreement = 0.0

    composite = (
        0.40 * normalize(sentiment_score, -1, 1)
        + 0.25 * normalize(momentum, -1, 1)
        + 0.20 * normalize(confidence, 0, 1)
        + 0.15 * normalize(agreement, 0, 1)
    ) * 100.0

    return composite


def _compute_sentiment_momentum(sentiment_by_day: dict) -> float:
    """Compare last-3-day avg sentiment vs first-4-day avg.

    Returns a value in [-1, 1].  0.0 when there's no data.
    """
    if not sentiment_by_day:
        return 0.0

    sorted_days = sorted(sentiment_by_day.keys())
    day_scores = [_SENTIMENT_NUMERIC.get(sentiment_by_day[d], 0.0) for d in sorted_days]

    if len(day_scores) < 2:
        return 0.0

    first_part = day_scores[: max(1, len(day_scores) - 3)]
    last_part = day_scores[-min(3, len(day_scores)):]

    first_avg = sum(first_part) / len(first_part)
    last_avg = sum(last_part) / len(last_part)

    momentum = last_avg - first_avg  # range: roughly -2 to +2
    return max(-1.0, min(1.0, momentum))


# ── Valuation composite ─────────────────────────────────────────────

def compute_valuation_composite(
    fundamentals: dict,
    sector_medians: dict,
) -> Optional[float]:
    """Compute a 0-100 valuation composite from fundamentals + sector medians.

    Weight redistribution: if some signals are None, their weight is spread
    equally among the available signals.  Returns None when *all* signals
    are unavailable.
    """
    if not fundamentals:
        return None

    base_weights = {
        "pe": 0.25,
        "ebitda": 0.25,
        "pb": 0.20,
        "div": 0.15,
        "cap": 0.15,
    }

    scores: dict[str, float] = {}

    # --- PE ratio ---
    pe = fundamentals.get("pe_ratio")
    sector_pe = (sector_medians or {}).get("pe_ratio")
    if pe is not None:
        if pe > 0 and sector_pe is not None and sector_pe > 0:
            # Lower PE relative to sector is better
            scores["pe"] = normalize(sector_pe / pe, 0, 2) * 100
        elif pe < 0:
            scores["pe"] = 0.0
        # pe == 0: skip (undefined)

    # --- EV/EBITDA ---
    ev_ebitda = fundamentals.get("ev_to_ebitda")
    sector_ebitda = (sector_medians or {}).get("ev_to_ebitda")
    if ev_ebitda is not None:
        if ev_ebitda > 0 and sector_ebitda is not None and sector_ebitda > 0:
            scores["ebitda"] = normalize(sector_ebitda / ev_ebitda, 0, 2) * 100
        elif ev_ebitda < 0:
            scores["ebitda"] = 0.0

    # --- Price-to-Book ---
    pb = fundamentals.get("price_to_book")
    if pb is not None and pb > 0:
        scores["pb"] = normalize(1.0 / pb, 0, 2) * 100

    # --- Dividend yield ---
    div_yield = fundamentals.get("dividend_yield")
    if div_yield is not None:
        scores["div"] = normalize(div_yield, 0, 0.10) * 100  # 10% cap

    # --- Market cap tier ---
    cap_tier = fundamentals.get("market_cap_tier")
    _cap_scores = {"large": 70, "mid": 50, "small": 30}
    if cap_tier is not None:
        scores["cap"] = _cap_scores.get(cap_tier.lower(), 50)

    if not scores:
        return None

    # Redistribute weights of missing signals
    available_weight = sum(base_weights[k] for k in scores)
    if available_weight <= 0:
        return None

    composite = sum(
        (base_weights[k] / available_weight) * scores[k] for k in scores
    )

    return max(0.0, min(100.0, composite))


# ── Momentum composite ──────────────────────────────────────────────

def compute_momentum_composite(
    momentum: dict,
    all_momentum: list,
) -> Optional[float]:
    """Compute a 0-100 momentum composite using cross-sectional percentile ranks.

    *momentum* is one ticker's dict; *all_momentum* is every ticker's dict
    (needed for percentile computation).
    """
    if not momentum:
        return None

    component_cfg = [
        ("momentum_30d", 0.35, False),
        ("momentum_90d", 0.30, False),
        ("relative_strength", 0.20, False),
        ("volatility", 0.15, True),   # True = invert (lower vol is better)
    ]

    weighted_sum = 0.0
    weight_used = 0.0

    for key, weight, invert in component_cfg:
        val = momentum.get(key)
        if val is None:
            continue

        all_vals = [
            (-m[key] if invert else m[key])
            for m in all_momentum
            if m.get(key) is not None
        ]
        rank_val = -val if invert else val
        pct = percentile_rank(rank_val, all_vals)

        weighted_sum += weight * pct
        weight_used += weight

    if weight_used <= 0:
        return None

    composite = (weighted_sum / weight_used)
    return max(0.0, min(100.0, composite))


# ── Risk penalty ─────────────────────────────────────────────────────

def compute_risk_penalty(
    symbol: str,
    holdings: list,
    portfolio_value: float,
    risk_profile: str,
    config: dict,
    fundamentals: dict,
) -> float:
    """Compute risk penalty (0-100) for adding/holding *symbol*.

    Components:
      - sector_concentration:  current sector weight / sector limit
      - geo_deviation:         deviation from geography target
      - correlation:           same-sector holding fraction (simplified)
      - drawdown_risk:         from fundamentals max_drawdown
    """
    investor_cfg = config.get("investor", {})
    limits = investor_cfg.get("position_limits", {}).get(risk_profile, {})
    geo_targets = investor_cfg.get("geo_targets", {}).get(risk_profile, {})

    sector = (fundamentals or {}).get("sector", "Unknown")
    geography = (fundamentals or {}).get("geography", "us")

    # --- sector_concentration ---
    sector_limit = limits.get("max_sector", 0.30)
    sector_weight = _current_sector_weight(sector, holdings, portfolio_value)
    sector_penalty = min(1.0, sector_weight / sector_limit) if sector_limit > 0 else 0.0

    # --- geo_deviation ---
    geo_target = geo_targets.get(geography, 0.0)
    geo_weight = _current_geo_weight(geography, holdings, portfolio_value)
    geo_penalty = min(1.0, abs(geo_weight - geo_target)) if geo_target > 0 else 0.0

    # --- correlation (simplified) ---
    total = len(holdings) if holdings else 0
    same_sector = sum(
        1 for h in (holdings or []) if h.get("sector") == sector
    )
    correlation_penalty = (same_sector / total) if total > 0 else 0.0

    # --- drawdown_risk ---
    max_drawdown = abs((fundamentals or {}).get("max_drawdown", 0.0))
    drawdown_penalty = normalize(max_drawdown, 0, 1)

    penalty = (
        0.35 * sector_penalty
        + 0.25 * geo_penalty
        + 0.25 * correlation_penalty
        + 0.15 * drawdown_penalty
    ) * 100.0

    return max(0.0, min(100.0, penalty))


def _current_sector_weight(sector: str, holdings: list, portfolio_value: float) -> float:
    """Sum of (shares * avg_cost_basis) for holdings in *sector* / portfolio_value."""
    if not holdings or portfolio_value <= 0:
        return 0.0
    sector_value = sum(
        h.get("shares", 0) * h.get("avg_cost_basis", 0)
        for h in holdings
        if h.get("sector") == sector
    )
    return sector_value / portfolio_value


def _current_geo_weight(geography: str, holdings: list, portfolio_value: float) -> float:
    """Sum of (shares * avg_cost_basis) for holdings in *geography* / portfolio_value."""
    if not holdings or portfolio_value <= 0:
        return 0.0
    geo_value = sum(
        h.get("shares", 0) * h.get("avg_cost_basis", 0)
        for h in holdings
        if h.get("geography") == geography
    )
    return geo_value / portfolio_value


# ── Main scoring function ───────────────────────────────────────────

def score_ticker(
    symbol: str,
    sentiment_data: Optional[dict],
    fundamentals: Optional[dict],
    momentum: Optional[dict],
    all_momentum: list,
    sector_medians: dict,
    holdings: list,
    portfolio_value: float,
    risk_profile: str,
    config: dict,
) -> dict:
    """Score a single ticker on a 0-100 scale.

    Missing composites have their weight redistributed equally among the
    remaining composites.  Risk penalty always applies.
    """
    investor_cfg = config.get("investor", {})
    weights = investor_cfg.get("formula_weights", {}).get(risk_profile, {
        "sentiment": 0.25, "valuation": 0.30, "momentum": 0.25, "risk": 0.20,
    })

    # --- compute each composite ---
    sent_comp = compute_sentiment_composite(sentiment_data)
    val_comp = compute_valuation_composite(fundamentals, sector_medians)
    mom_comp = compute_momentum_composite(momentum, all_momentum)
    risk_pen = compute_risk_penalty(
        symbol, holdings, portfolio_value, risk_profile, config, fundamentals or {},
    )

    # --- build composite map (only non-None entries) ---
    composite_map = {}
    if sent_comp is not None:
        composite_map["sentiment"] = (weights.get("sentiment", 0.25), sent_comp)
    if val_comp is not None:
        composite_map["valuation"] = (weights.get("valuation", 0.30), val_comp)
    if mom_comp is not None:
        composite_map["momentum"] = (weights.get("momentum", 0.25), mom_comp)

    # --- redistribute weights among available composites ---
    if composite_map:
        total_weight = sum(w for w, _ in composite_map.values())
        if total_weight <= 0:
            total_weight = 1.0  # fallback

        raw_positive = sum(
            (w / total_weight) * score for w, score in composite_map.values()
        )
    else:
        raw_positive = 50.0  # no data — neutral default

    risk_weight = weights.get("risk", 0.20)
    raw_score = raw_positive * (1.0 - risk_weight) - risk_pen * risk_weight
    final_score = max(0.0, min(100.0, raw_score))

    return {
        "symbol": symbol,
        "score": round(final_score, 2),
        "sentiment_composite": round(sent_comp, 2) if sent_comp is not None else None,
        "valuation_composite": round(val_comp, 2) if val_comp is not None else None,
        "momentum_composite": round(mom_comp, 2) if mom_comp is not None else None,
        "risk_penalty": round(risk_pen, 2),
        "components": {
            "weights_used": {
                k: round(w, 4) for k, (w, _) in composite_map.items()
            } if composite_map else {},
            "risk_weight": round(risk_weight, 4),
            "raw_positive": round(raw_positive, 2),
        },
    }


# ── Universe scoring ─────────────────────────────────────────────────

def score_universe(
    ticker_universe: list,
    sentiment_cache: dict,
    batch_fundamentals: dict,
    batch_momentum: dict,
    holdings: list,
    portfolio_value: float,
    risk_profile: str,
    config: dict,
) -> list:
    """Score every ticker in *ticker_universe*.

    Returns a list of score dicts sorted by score descending.
    """
    all_momentum = [
        batch_momentum[sym]
        for sym in ticker_universe
        if sym in batch_momentum and batch_momentum[sym]
    ]

    results = []
    for sym in ticker_universe:
        sentiment_data = sentiment_cache.get(sym)
        fundamentals = (batch_fundamentals or {}).get(sym)
        momentum = (batch_momentum or {}).get(sym)

        # Sector medians: derive from batch_fundamentals for the ticker's sector
        sector = (fundamentals or {}).get("sector")
        sector_medians = _derive_sector_medians(sector, batch_fundamentals)

        score_result = score_ticker(
            symbol=sym,
            sentiment_data=sentiment_data,
            fundamentals=fundamentals,
            momentum=momentum,
            all_momentum=all_momentum,
            sector_medians=sector_medians,
            holdings=holdings,
            portfolio_value=portfolio_value,
            risk_profile=risk_profile,
            config=config,
        )
        results.append(score_result)

    results.sort(key=lambda r: r["score"], reverse=True)
    logger.info(
        "Scored %d tickers. Top: %s (%.1f), Bottom: %s (%.1f)",
        len(results),
        results[0]["symbol"] if results else "N/A",
        results[0]["score"] if results else 0,
        results[-1]["symbol"] if results else "N/A",
        results[-1]["score"] if results else 0,
    )
    return results


def _derive_sector_medians(sector: str, batch_fundamentals: dict) -> dict:
    """Compute median PE, EV/EBITDA for a sector from the batch."""
    if not sector or not batch_fundamentals:
        return {}

    sector_entries = [
        f for f in batch_fundamentals.values()
        if f and f.get("sector") == sector
    ]
    if not sector_entries:
        return {}

    def _median(values):
        clean = sorted(v for v in values if v is not None and v > 0)
        if not clean:
            return None
        mid = len(clean) // 2
        if len(clean) % 2 == 0:
            return (clean[mid - 1] + clean[mid]) / 2
        return clean[mid]

    return {
        "pe_ratio": _median([e.get("pe_ratio") for e in sector_entries]),
        "ev_to_ebitda": _median([e.get("ev_to_ebitda") for e in sector_entries]),
    }
