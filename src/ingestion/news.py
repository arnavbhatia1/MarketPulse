from datetime import datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsIngester(BaseIngester):
    """
    Ingest financial news headlines and summaries.

    Primary source: NewsAPI (https://newsapi.org/)

    Why news matters:
    - News posts are almost always NEUTRAL (factual reporting)
    - They provide good training signal for the neutral class
    - News about earnings, mergers, regulation gives entity extraction data
    - Mixing news with social media posts creates a realistic class distribution

    API endpoint: https://newsapi.org/v2/everything
    Parameters: q=query, from=date, to=date, language=en, sortBy=publishedAt

    Implementation details:
    - Query financial keywords from config
    - Use article title + description as text (not full article body)
    - Filter to English only
    - Deduplicate by headline similarity (news gets syndicated)
    - post_id format: "news_{source}_{hash_of_url}"

    Metadata captured:
    - news_source: str (CNBC, Reuters, Bloomberg, etc.)
    - article_url: str
    - image_url: str or null
    - published_at: datetime
    """

    def __init__(self, config):
        self.config = config

    def is_available(self) -> bool:
        """Check if News API key exists in environment."""
        import os
        return bool(os.getenv('NEWS_API_KEY'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not self.is_available():
            logger.info("News: no API credentials configured")
            return self._empty_dataframe()
        logger.info("News: live ingestion not implemented yet")
        return self._empty_dataframe()
