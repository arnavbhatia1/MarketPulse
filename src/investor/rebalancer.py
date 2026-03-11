"""
Rebalancer — orchestrates the full rebalance cycle.

Weekly scheduled + event-driven triggers. Ties together:
engine (scoring) -> risk (checks) -> reviewer (Claude) -> broker (execution).
"""

import logging
import uuid
from datetime import datetime

from src.storage import db
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_rebalance(portfolio_id: str, trigger: str = "weekly_rebalance") -> dict:
    """
    Run a full rebalance cycle for one portfolio.

    Args:
        portfolio_id: UUID of the portfolio
        trigger: 'weekly_rebalance' | 'event_driven' | 'manual'

    Returns:
        dict with results summary
    """
    from src.investor import engine, risk, market_data, portfolio as portfolio_mod
    from src.investor.broker import PaperBroker
    from src.agent.reviewer import review_trades

    config = load_config()
    investor_config = config.get("investor", {})

    # 1. Load portfolio state
    port = db.get_portfolio(portfolio_id)
    if not port:
        return {"error": "Portfolio not found"}

    risk_profile = port["risk_profile"]
    holdings = db.get_holdings(portfolio_id)
    summary = portfolio_mod.get_portfolio_summary(portfolio_id)
    if not summary:
        return {"error": "Could not compute portfolio summary"}

    portfolio_value = summary["total_value"]
    available_cash = port["current_cash"]

    logger.info("Rebalance start: portfolio=%s value=$%.0f cash=$%.0f trigger=%s",
                portfolio_id, portfolio_value, available_cash, trigger)

    # 2. Build ticker universe
    etf_symbols = [e["symbol"] for e in db.get_etf_universe()]
    from src.extraction.ticker_extractor import TickerExtractor
    extractor = TickerExtractor()
    stock_symbols = list(extractor.ticker_map.keys())
    universe = list(set(stock_symbols + etf_symbols))

    # 3. Fetch market data
    logger.info("Fetching market data for %d tickers...", len(universe))
    batch_fundamentals = market_data.get_batch_fundamentals(universe)
    batch_momentum = {}
    for sym in universe:
        mom = market_data.get_momentum_signals(sym)
        if mom:
            batch_momentum[sym] = mom

    sector_medians = market_data.get_sector_medians(batch_fundamentals)

    # 4. Load sentiment data
    sentiment_cache = db.load_ticker_cache()
    # Map symbols to sentiment data (ticker_cache is keyed by company name)
    symbol_to_sentiment = {}
    for company, data in sentiment_cache.items():
        sym = data.get("symbol")
        if sym:
            symbol_to_sentiment[sym] = data

    # 5. Score all tickers
    logger.info("Scoring universe...")
    scores = engine.score_universe(
        ticker_universe=universe,
        sentiment_cache=symbol_to_sentiment,
        batch_fundamentals=batch_fundamentals,
        batch_momentum=batch_momentum,
        holdings=holdings,
        portfolio_value=portfolio_value,
        risk_profile=risk_profile,
        config=investor_config,
    )

    # 6. Generate trade candidates
    scoring_config = investor_config.get("scoring", {})
    buy_threshold = scoring_config.get("buy_threshold", 65)
    sell_threshold = scoring_config.get("sell_threshold", 35)
    strong_sell_threshold = scoring_config.get("strong_sell_threshold", 20)

    held_symbols = {h["symbol"] for h in holdings}
    proposed_trades = []

    # Sell signals
    for s in scores:
        sym = s["symbol"]
        if sym in held_symbols and s["score"] < sell_threshold:
            holding = next(h for h in holdings if h["symbol"] == sym)
            price = market_data.get_current_price(sym) or 0
            if price <= 0:
                continue

            if s["score"] < strong_sell_threshold:
                sell_shares = holding["shares"]  # full exit
            else:
                # Trim to reduce overweight
                sell_shares = max(1, int(holding["shares"] * 0.3))

            proposed_trades.append({
                "trade_id": str(uuid.uuid4()),
                "portfolio_id": portfolio_id,
                "symbol": sym,
                "action": "sell",
                "shares": sell_shares,
                "price": price,
                "total_value": price * sell_shares,
                "formula_score": s["score"],
                "reason": f"Score {s['score']:.0f} below sell threshold {sell_threshold}",
                "status": "proposed",
                "trigger": trigger,
                "proposed_at": datetime.utcnow().isoformat(),
            })

    # Buy signals
    buy_candidates = [
        s for s in scores
        if s["score"] >= buy_threshold and s["symbol"] not in held_symbols
    ]

    if buy_candidates and available_cash > 1000:
        sized = portfolio_mod.generate_position_sizes(
            buy_candidates=buy_candidates,
            portfolio_value=portfolio_value,
            available_cash=available_cash,
            risk_profile=risk_profile,
            config=investor_config,
        )
        for pos in sized:
            if pos["shares"] < 1:
                continue
            price = market_data.get_current_price(pos["symbol"]) or 0
            if price <= 0:
                continue

            # Risk check before proposing
            limit_check = risk.check_position_limits(
                symbol=pos["symbol"],
                proposed_value=pos["estimated_cost"],
                holdings=holdings,
                portfolio_value=portfolio_value,
                risk_profile=risk_profile,
                config=investor_config,
            )
            if not limit_check["allowed"]:
                logger.info("Skipping %s: %s", pos["symbol"], limit_check["violations"])
                continue

            proposed_trades.append({
                "trade_id": str(uuid.uuid4()),
                "portfolio_id": portfolio_id,
                "symbol": pos["symbol"],
                "action": "buy",
                "shares": pos["shares"],
                "price": price,
                "total_value": price * pos["shares"],
                "formula_score": pos["score"],
                "reason": f"Score {pos['score']:.0f} above buy threshold {buy_threshold}",
                "status": "proposed",
                "trigger": trigger,
                "proposed_at": datetime.utcnow().isoformat(),
            })

    if not proposed_trades:
        logger.info("No trades proposed this cycle")
        # Still take a snapshot
        portfolio_mod.take_snapshot(portfolio_id)
        db.update_portfolio(portfolio_id, last_rebalanced_at=datetime.utcnow().isoformat())
        return {"trades_proposed": 0, "trades_executed": 0, "commentary": "No action needed."}

    logger.info("Proposed %d trades", len(proposed_trades))

    # 7. Claude review
    stress = risk.compute_stress_score(holdings, portfolio_value, investor_config)
    market_context = _build_market_context(sentiment_cache)
    risk_summary = {
        "stress_score": stress["stress_score"],
        "cash_pct": available_cash / portfolio_value if portfolio_value > 0 else 0,
        "post_trade_cash_pct": available_cash / portfolio_value if portfolio_value > 0 else 0,
    }
    portfolio_state = {
        "total_value": portfolio_value,
        "current_cash": available_cash,
        "risk_profile": risk_profile,
        "mode": port["mode"],
        "sector_allocation": summary.get("sector_allocation", {}),
        "geo_allocation": summary.get("geo_allocation", {}),
    }

    review = review_trades(proposed_trades, portfolio_state, market_context, risk_summary)

    # 8. Apply review decisions
    approved_trades = []
    decision_map = {d["trade_id"]: d for d in review.get("decisions", [])}

    for trade in proposed_trades:
        decision = decision_map.get(trade["trade_id"])
        if decision and decision.get("action") == "VETO":
            trade["status"] = "rejected"
            trade["claude_review"] = decision.get("reasoning", "Vetoed by reviewer")
            db.save_trade(trade)
            logger.info("VETO: %s %s — %s", trade["action"], trade["symbol"],
                       decision.get("reasoning", ""))
        else:
            trade["claude_review"] = (decision or {}).get("reasoning", "Auto-approved")
            approved_trades.append(trade)

    # 9. Execute or queue based on mode
    executed_count = 0
    if port["mode"] == "autopilot":
        broker = PaperBroker(portfolio_id)
        for trade in approved_trades:
            if trade["action"] == "buy":
                result = broker.execute_buy(trade["symbol"], trade["shares"])
            else:
                result = broker.execute_sell(trade["symbol"], trade["shares"])

            if result.success:
                executed_count += 1
                trade["status"] = "executed"
                trade["executed_at"] = result.executed_at
                trade["trade_id"] = result.trade_id
            else:
                trade["status"] = "proposed"
                logger.warning("Trade failed: %s %s — %s",
                             trade["action"], trade["symbol"], result.error)
            db.save_trade(trade)
    else:
        # Advisory mode — save as proposed for user approval
        for trade in approved_trades:
            trade["status"] = "approved"  # approved by Claude, awaiting user
            db.save_trade(trade)

    # 10. Take snapshot and update portfolio
    portfolio_mod.take_snapshot(portfolio_id)
    db.update_portfolio(portfolio_id, last_rebalanced_at=datetime.utcnow().isoformat())

    commentary = review.get("portfolio_commentary", "Rebalance complete.")
    logger.info("Rebalance complete: %d proposed, %d executed. %s",
                len(proposed_trades), executed_count, commentary)

    return {
        "trades_proposed": len(proposed_trades),
        "trades_executed": executed_count,
        "trades_rejected": len(proposed_trades) - len(approved_trades),
        "commentary": commentary,
        "review": review,
    }


def run_rebalance_all(trigger: str = "weekly_rebalance") -> list:
    """Run rebalance for all active portfolios. Used by cron script."""
    conn = db.get_connection()
    try:
        rows = conn.execute(
            "SELECT portfolio_id FROM portfolios WHERE is_active = 1"
        ).fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        pid = row["portfolio_id"]
        logger.info("Rebalancing portfolio %s", pid)
        try:
            result = run_rebalance(pid, trigger=trigger)
            results.append({"portfolio_id": pid, **result})
        except Exception as exc:
            logger.error("Rebalance failed for %s: %s", pid, exc)
            results.append({"portfolio_id": pid, "error": str(exc)})
    return results


def _build_market_context(sentiment_cache: dict) -> dict:
    """Extract overall market context from sentiment cache for Claude."""
    bullish = []
    bearish = []
    sentiments = []

    for company, data in sentiment_cache.items():
        dominant = data.get("dominant_sentiment", "neutral")
        sentiments.append(dominant)
        sym = data.get("symbol", company)
        if dominant == "bullish":
            bullish.append(sym)
        elif dominant == "bearish":
            bearish.append(sym)

    # Overall sentiment = mode of all sentiments
    if sentiments:
        from collections import Counter
        overall = Counter(sentiments).most_common(1)[0][0]
    else:
        overall = "unknown"

    return {
        "overall_sentiment": overall,
        "trending_bullish": bullish[:5],
        "trending_bearish": bearish[:5],
    }
