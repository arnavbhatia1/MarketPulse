"""
Claude briefing agent for MarketPulse.

One API call per ticker search. Claude writes the AI Verdict paragraph only.
All charts, stats, and post samples come from SQLite — Claude is narrative only.
"""

import json
import os

import anthropic

_FALLBACK = (
    "Sentiment data has been aggregated from Reddit, Stocktwits, and financial news. "
    "See the source breakdown below for details."
)


def generate_briefing(company: str, ticker: str, ticker_data: dict) -> str:
    """
    Generate a 2-3 sentence AI verdict for a ticker briefing card.

    Args:
        company: canonical company name e.g. "Tesla"
        ticker:  stock symbol e.g. "TSLA"
        ticker_data: dict from ticker_cache (SQLite row with all fields)

    Returns:
        2-3 sentence verdict string. Returns a safe fallback if Claude unavailable.
    """
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        prompt = _build_prompt(company, ticker, ticker_data)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return _FALLBACK


def _build_prompt(company: str, ticker: str, ticker_data: dict) -> str:
    """Build the structured prompt sent to Claude."""
    top_posts = ticker_data.get('top_posts', {})

    samples = []
    for source in ('reddit', 'stocktwits', 'news'):
        for post in top_posts.get(source, [])[:2]:
            text = str(post.get('text', ''))[:140]
            samples.append(f"[{source}] {text}")

    trend = json.dumps(ticker_data.get('sentiment_by_day', {}))

    return f"""You are a financial sentiment analyst writing for a market intelligence dashboard.

Ticker: {ticker} ({company})
Overall sentiment: {ticker_data.get('dominant_sentiment', 'unknown')}
Mentions: {ticker_data.get('mention_count', 0)} posts
Reddit: {ticker_data.get('reddit_sentiment', 'N/A')}
Stocktwits: {ticker_data.get('stocktwits_sentiment', 'N/A')}
News: {ticker_data.get('news_sentiment', 'N/A')}
7-day trend: {trend}

Sample posts:
{chr(10).join(samples) if samples else 'No sample posts available.'}

Write a 2-3 sentence verdict summarizing the current sentiment. Be specific about \
which sources are most bearish/bullish and any trend direction. Plain prose, no \
bullets, no headers. Do not start with "Based on" or "According to"."""
