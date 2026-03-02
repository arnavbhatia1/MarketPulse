import os
import hashlib
import requests
from datetime import datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsIngester(BaseIngester):
    """
    Ingest financial news headlines and summaries from NewsAPI.org.

    API endpoint: https://newsapi.org/v2/everything
    Parameters: q=query, from=date, to=date, language=en, sortBy=publishedAt

    Implementation details:
    - Query financial keywords from config
    - Use article title + description as text (not full article body)
    - Filter to English only
    - Deduplicate by URL hash
    - post_id format: "news_{source_name}_{md5(url)[:12]}"

    Metadata captured:
    - news_source: str (CNBC, Reuters, Bloomberg, etc.)
    - article_url: str
    - image_url: str or null
    - published_at: datetime
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, config):
        self.config = config
        news_cfg = config.get('ingestion', {}).get('news', {})
        self.query_terms = news_cfg.get(
            'query_terms', ['stock market', 'earnings', 'IPO', 'SEC', 'Fed']
        )
        self.language = news_cfg.get('language', 'en')
        self.page_size = news_cfg.get('page_size', 100)

    def is_available(self) -> bool:
        """Check if News API key exists in environment."""
        return bool(os.getenv('NEWS_API_KEY'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch articles for each configured query term within the date range.

        Loops over query_terms, calls the NewsAPI /v2/everything endpoint,
        parses articles into the standard schema, and deduplicates by URL.
        Gracefully skips query terms that return API errors.
        """
        if not self.is_available():
            logger.info("News: no API credentials configured, returning empty")
            return self._empty_dataframe()

        api_key = os.getenv('NEWS_API_KEY')
        rows = []

        for term in self.query_terms:
            logger.info(f"News: fetching articles for '{term}'...")
            try:
                params = {
                    'q': term,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'language': self.language,
                    'sortBy': 'publishedAt',
                    'pageSize': self.page_size,
                    'apiKey': api_key,
                }
                resp = requests.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                articles = data.get('articles', [])
                for article in articles:
                    url = article.get('url', '')
                    if not url:
                        continue

                    # Build text from title + description
                    title = article.get('title') or ''
                    description = article.get('description') or ''
                    text = f"{title} {description}".strip()
                    if not text:
                        continue

                    # Truncate to 500 chars
                    text = text[:500]

                    # Generate post_id from source name and URL hash
                    source_name = (article.get('source') or {}).get('name', 'unknown')
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                    post_id = f"news_{source_name}_{url_hash}".replace(' ', '_')

                    # Parse published timestamp
                    published = article.get('publishedAt', '')
                    try:
                        ts = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
                    except (ValueError, TypeError):
                        ts = datetime.now()

                    rows.append({
                        'post_id': post_id,
                        'text': text,
                        'source': 'news',
                        'timestamp': ts.isoformat(),
                        'author': article.get('author') or 'unknown',
                        'score': 0,
                        'url': url,
                        'metadata': str({
                            'news_source': source_name,
                            'article_url': url,
                            'image_url': article.get('urlToImage'),
                            'published_at': published,
                        }),
                    })
            except Exception as e:
                logger.warning(f"News: error fetching '{term}': {e}")

        logger.info(f"News: {len(rows)} articles fetched across {len(self.query_terms)} query terms")

        if not rows:
            return self._empty_dataframe()

        df = pd.DataFrame(rows)
        return self.validate_output(df)
