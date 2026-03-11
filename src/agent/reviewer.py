"""
Claude trade reviewer for MarketPulse Investor Bot.

One API call per rebalance cycle. Claude reviews proposed trades and returns
APPROVE/VETO decisions with reasoning. Falls back to formula-only mode on failure.
"""

import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)

_FALLBACK_REVIEW = {
    "decisions": [],
    "portfolio_commentary": "Automated review unavailable. Trades executed based on formula scores only.",
}


def review_trades(proposed_trades: list, portfolio_state: dict,
                  market_context: dict, risk_summary: dict) -> dict:
    """
    Have Claude review proposed trades before execution.

    Args:
        proposed_trades: list of trade dicts with trade_id, symbol, action, shares,
                        price, formula_score, reason
        portfolio_state: dict with total_value, current_cash, risk_profile,
                        sector_allocation, geo_allocation
        market_context: dict with overall_sentiment, trending_bullish, trending_bearish
        risk_summary: dict with stress_score, sector_exposure, cash_pct

    Returns:
        dict with 'decisions' list and 'portfolio_commentary' string.
        Each decision: {"trade_id": str, "action": "APPROVE"|"VETO", "reasoning": str}
    """
    if not proposed_trades:
        return {"decisions": [], "portfolio_commentary": "No trades proposed this cycle."}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; skipping trade review")
        return _approve_all(proposed_trades)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_review_prompt(proposed_trades, portfolio_state,
                                      market_context, risk_summary)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        if not message.content:
            logger.warning("Claude returned empty review content")
            return _approve_all(proposed_trades)

        raw = message.content[0].text.strip()
        return _parse_review(raw, proposed_trades)

    except Exception as exc:
        logger.warning("Claude trade review failed: %s", exc)
        return _approve_all(proposed_trades)


def _build_review_prompt(trades: list, portfolio: dict,
                         context: dict, risk: dict) -> str:
    """Build the structured review prompt."""
    trades_text = []
    for i, t in enumerate(trades, 1):
        trades_text.append(
            f"{i}. {t['action'].upper()} {t.get('shares', '?')} shares {t['symbol']} "
            f"@ ${t.get('price', '?'):.2f} (${t.get('total_value', 0):.0f})\n"
            f"   Trade ID: {t['trade_id']}\n"
            f"   Formula score: {t.get('formula_score', 'N/A')} | "
            f"Reason: {t.get('reason', 'N/A')}"
        )

    sector_text = ", ".join(
        f"{k} {v:.0%}" for k, v in sorted(
            portfolio.get('sector_allocation', {}).items(),
            key=lambda x: -x[1]
        )[:5]
    )

    return f"""You are a senior portfolio analyst reviewing proposed trades for a client's investment portfolio.

PORTFOLIO STATE:
- Total value: ${portfolio.get('total_value', 0):,.0f} | Cash: ${portfolio.get('current_cash', 0):,.0f} ({risk.get('cash_pct', 0):.1%})
- Risk profile: {portfolio.get('risk_profile', 'moderate').title()} | Mode: {portfolio.get('mode', 'autopilot')}
- Top sectors: {sector_text or 'N/A'}

PROPOSED TRADES:
{chr(10).join(trades_text)}

MARKET CONTEXT (from MarketPulse sentiment):
- Overall sentiment: {context.get('overall_sentiment', 'unknown')}
- Trending bullish: {', '.join(context.get('trending_bullish', [])) or 'None'}
- Trending bearish: {', '.join(context.get('trending_bearish', [])) or 'None'}

RISK SUMMARY:
- Recession stress score: {risk.get('stress_score', 0):.2f}
- Cash after trades: {risk.get('post_trade_cash_pct', 0):.1%}

For each trade, respond with APPROVE or VETO and 1-2 sentences of reasoning.
Respond in valid JSON format:
{{
  "decisions": [
    {{"trade_id": "<id>", "action": "APPROVE", "reasoning": "..."}},
    ...
  ],
  "portfolio_commentary": "1-2 sentence overall assessment for the client."
}}

Only output the JSON object, nothing else."""


def _parse_review(raw: str, proposed_trades: list) -> dict:
    """Parse Claude's JSON response. Falls back to approve-all on parse failure."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        if "decisions" not in result:
            logger.warning("Claude review missing 'decisions' key")
            return _approve_all(proposed_trades)
        return result
    except json.JSONDecodeError:
        logger.warning("Could not parse Claude review as JSON: %s", text[:200])
        return _approve_all(proposed_trades)


def _approve_all(trades: list) -> dict:
    """Fallback: approve all trades when review is unavailable."""
    return {
        "decisions": [
            {
                "trade_id": t["trade_id"],
                "action": "APPROVE",
                "reasoning": "Auto-approved (review unavailable)",
            }
            for t in trades
        ],
        "portfolio_commentary": _FALLBACK_REVIEW["portfolio_commentary"],
    }
