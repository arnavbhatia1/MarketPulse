import hashlib
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Google News RSS — no API key required
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
# Yahoo Finance RSS per ticker — no API key required
_YAHOO_FINANCE_RSS = "https://finance.yahoo.com/rss/headline?s={symbol}"


class NewsIngester(BaseIngester):
    """
    Ingest financial news headlines via free RSS feeds — no API key required.

    Sources:
    - Google News RSS: searched by keyword (stock market, earnings, IPO, etc.)
    - Yahoo Finance RSS: per tracked ticker symbol

    post_id format: "news_{source_slug}_{md5(url)[:12]}"
    """

    def __init__(self, config):
        self.config = config
        news_cfg = config.get('ingestion', {}).get('news', {})
        self.query_terms = news_cfg.get(
            'query_terms', ['stock market', 'earnings', 'IPO', 'SEC', 'Fed']
        )
        # Ticker symbols to fetch per-ticker Yahoo Finance feeds
        stocktwits_cfg = config.get('ingestion', {}).get('stocktwits', {})
        self.symbols = stocktwits_cfg.get(
            'symbols', ['AAPL', 'TSLA', 'NVDA', 'GME', 'MSFT', 'AMZN', 'SPY']
        )

    def is_available(self) -> bool:
        """Always available — uses free RSS feeds, no API key needed."""
        return True

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch news via Google News RSS (keyword searches) and
        Yahoo Finance RSS (per-ticker feeds). Filters by date range.
        """
        rows = []
        seen_urls = set()

        # Google News: one feed per query term
        for term in self.query_terms:
            url = _GOOGLE_NEWS_RSS.format(query=term.replace(' ', '+') + '+stock')
            rows += self._parse_feed(url, f"google_news_{term[:20]}", seen_urls, start_date, end_date)

        # Yahoo Finance: one feed per ticker
        for symbol in self.symbols:
            url = _YAHOO_FINANCE_RSS.format(symbol=symbol)
            rows += self._parse_feed(url, f"yahoo_{symbol}", seen_urls, start_date, end_date)

        logger.info(f"News: {len(rows)} articles from RSS feeds")

        if not rows:
            return self._empty_dataframe()

        return self.validate_output(pd.DataFrame(rows))

    def _parse_feed(self, feed_url: str, source_slug: str,
                    seen_urls: set, start_date: datetime, end_date: datetime) -> list:
        rows = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                url = entry.get('link', '')
                if not url or url in seen_urls:
                    continue

                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = f"{title} {summary}".strip()[:500]
                if not text:
                    continue

                ts = self._parse_date(entry)
                if start_date and ts < start_date:
                    continue
                if end_date and ts > end_date:
                    continue

                seen_urls.add(url)
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                rows.append({
                    'post_id': f"news_{source_slug}_{url_hash}",
                    'text': text,
                    'source': 'news',
                    'timestamp': ts.isoformat(),
                    'author': entry.get('author', 'unknown'),
                    'score': 0,
                    'url': url,
                    'metadata': str({'news_source': source_slug, 'article_url': url}),
                })
        except Exception as e:
            logger.warning(f"News: failed to parse feed {feed_url}: {e}")
        return rows

    def _parse_date(self, entry) -> datetime:
        for field in ('published', 'updated'):
            raw = entry.get(field)
            if raw:
                try:
                    return parsedate_to_datetime(raw).replace(tzinfo=None)
                except Exception:
                    pass
        return datetime.now()
