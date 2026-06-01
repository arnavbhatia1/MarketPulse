import hashlib
import feedparser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
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
    - Yahoo Finance RSS: per tracked ticker symbol (20 tickers)
    - Additional static feeds: CNBC Markets, MarketWatch (configurable)

    post_id format: "news_{source_slug}_{md5(url)[:12]}"
    """

    _TRACKING_PARAMS = frozenset({
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term',
        'utm_content', 'ref', 'fbclid', 'gclid', 'source',
    })

    def _normalize_url(self, url):
        """Strip tracking parameters from URL for better deduplication."""
        parsed = urlparse(url)
        params = {k: v for k, v in parse_qs(parsed.query).items()
                  if k not in self._TRACKING_PARAMS}
        cleaned_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=cleaned_query, fragment=''))

    def __init__(self, config):
        self.config = config
        news_cfg = config.get('ingestion', {}).get('news', {})
        self.query_terms = news_cfg.get(
            'query_terms', ['stock market', 'earnings', 'IPO', 'SEC', 'Fed']
        )
        # Ticker symbols to fetch per-ticker Yahoo Finance feeds
        self.symbols = news_cfg.get(
            'symbols', ['AAPL', 'TSLA', 'NVDA', 'GME', 'AMC', 'SPY', 'MSFT', 'AMZN']
        )
        self.additional_feeds = news_cfg.get('additional_feeds', [])

    def is_available(self) -> bool:
        """Always available — uses free RSS feeds, no API key needed."""
        return True

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch news via Google News RSS (keyword searches) and
        Yahoo Finance RSS (per-ticker feeds). Filters by date range.
        """
        # Build the full list of feeds to fetch, then pull them concurrently —
        # RSS fetching is network-bound, so a thread pool turns a ~50s serial
        # crawl into a few seconds.
        feeds: list[tuple[str, str]] = []
        for term in self.query_terms:
            feeds.append((
                _GOOGLE_NEWS_RSS.format(query=term.replace(' ', '+') + '+stock'),
                f"google_news_{term[:20]}",
            ))
        for symbol in self.symbols:
            feeds.append((_YAHOO_FINANCE_RSS.format(symbol=symbol), f"yahoo_{symbol}"))
        for feed_info in self.additional_feeds:
            feed_url = feed_info.get('url', '')
            if feed_url:
                feeds.append((feed_url, feed_info.get('name', 'extra')))

        with ThreadPoolExecutor(max_workers=16) as pool:
            per_feed = pool.map(
                lambda f: self._parse_feed(f[0], f[1], start_date, end_date), feeds
            )

        # Global dedup across feeds by normalized URL.
        rows = []
        seen_urls = set()
        for feed_rows in per_feed:
            for row in feed_rows:
                if row['url'] in seen_urls:
                    continue
                seen_urls.add(row['url'])
                rows.append(row)

        logger.info(f"News: {len(rows)} articles from {len(feeds)} RSS feeds")

        if not rows:
            return self._empty_dataframe()

        return self.validate_output(pd.DataFrame(rows))

    def _parse_feed(self, feed_url: str, source_slug: str,
                    start_date: datetime, end_date: datetime) -> list:
        rows = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                try:
                    url = entry.get('link', '')
                    if not url:
                        continue
                    normalized = self._normalize_url(url)

                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    text = f"{title} {summary}".strip()[:500]
                    if not text:
                        continue

                    ts = self._parse_date(entry)
                    if ts is None:
                        continue
                    if start_date and ts < start_date:
                        continue
                    if end_date and ts > end_date:
                        continue

                    url_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
                    rows.append({
                        'post_id': f"news_{source_slug}_{url_hash}",
                        'text': text,
                        'source': 'news',
                        'timestamp': ts.isoformat(),
                        'author': entry.get('author', 'unknown'),
                        'score': 0,
                        'url': normalized,
                        'metadata': str({'news_source': source_slug, 'article_url': normalized}),
                    })
                except Exception as e:
                    logger.warning(f"News: skipping malformed entry in {feed_url}: {e}")
        except Exception as e:
            logger.warning(f"News: failed to parse feed {feed_url}: {e}")
        return rows

    def _parse_date(self, entry):
        """Parse date from feed entry. Returns None if unparseable."""
        for field in ('published', 'updated'):
            raw = entry.get(field)
            if raw:
                try:
                    return parsedate_to_datetime(raw).replace(tzinfo=None)
                except Exception:
                    pass
        logger.debug("News: unparseable date in entry, skipping")
        return None
