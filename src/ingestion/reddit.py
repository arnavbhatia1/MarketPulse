from datetime import datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedditIngester(BaseIngester):
    """
    Ingest posts from Reddit using PRAW (Python Reddit API Wrapper).

    Targets subreddits: wallstreetbets, stocks, investing

    Implementation details:
    - Use praw.Reddit() with credentials from .env
    - Fetch posts using subreddit.new() and subreddit.hot() within date range
    - Extract: title + selftext as the text field
    - Filter posts with score < min_score to remove low-quality content
    - Handle rate limits with exponential backoff
    - post_id format: "reddit_{subreddit}_{original_id}"

    Metadata captured:
    - subreddit: str
    - num_comments: int
    - flair: str (DD, YOLO, Discussion, etc.)
    - is_self: bool
    - link_flair_text: str
    """

    def __init__(self, config):
        self.config = config

    def is_available(self) -> bool:
        """Check if Reddit credentials exist in environment."""
        import os
        return bool(os.getenv('REDDIT_CLIENT_ID') and os.getenv('REDDIT_CLIENT_SECRET'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not self.is_available():
            logger.info("Reddit: no API credentials configured")
            return self._empty_dataframe()
        logger.info("Reddit: live ingestion not implemented yet")
        return self._empty_dataframe()
