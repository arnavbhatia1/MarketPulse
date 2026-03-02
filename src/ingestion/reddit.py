import os
import praw
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
    - Fetch posts using subreddit.new() within date range
    - Extract: title + selftext as the text field
    - Filter posts with score < min_score to remove low-quality content
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
        reddit_cfg = config.get('ingestion', {}).get('reddit', {})
        self.subreddits = reddit_cfg.get(
            'subreddits', ['wallstreetbets', 'stocks', 'investing']
        )
        self.post_limit = reddit_cfg.get('post_limit_per_sub', 200)
        self.min_score = reddit_cfg.get('min_score', 5)

    def is_available(self) -> bool:
        """Check if Reddit credentials exist in environment."""
        return bool(
            os.getenv('REDDIT_CLIENT_ID') and os.getenv('REDDIT_CLIENT_SECRET')
        )

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch posts from each configured subreddit within the date range.

        Uses subreddit.new() to walk posts in reverse chronological order.
        Stops early once posts fall outside the requested date window.
        Gracefully skips subreddits that raise exceptions.
        """
        if not self.is_available():
            logger.info("Reddit: no API credentials configured, returning empty")
            return self._empty_dataframe()

        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'MarketPulse/1.0'),
        )

        rows = []
        for sub_name in self.subreddits:
            logger.info(f"Reddit: fetching r/{sub_name} (limit={self.post_limit})...")
            try:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.new(limit=self.post_limit):
                    created = datetime.utcfromtimestamp(post.created_utc)

                    # Skip posts outside the requested window
                    if created < start_date or created > end_date:
                        continue

                    # Skip low-engagement posts
                    if post.score < self.min_score:
                        continue

                    text = post.title
                    if post.selftext and post.selftext.strip():
                        text = text + " " + post.selftext

                    rows.append({
                        'post_id': f"reddit_{sub_name}_{post.id}",
                        'text': text[:500],
                        'source': 'reddit',
                        'timestamp': created.isoformat(),
                        'author': str(post.author) if post.author else 'unknown',
                        'score': int(post.score),
                        'url': f"https://reddit.com{post.permalink}",
                        'metadata': str({
                            'subreddit': sub_name,
                            'num_comments': post.num_comments,
                            'flair': post.link_flair_text or '',
                            'is_self': post.is_self,
                        }),
                    })
            except Exception as e:
                logger.warning(f"Reddit: error fetching r/{sub_name}: {e}")

        logger.info(f"Reddit: {len(rows)} posts fetched across {len(self.subreddits)} subreddits")

        if not rows:
            return self._empty_dataframe()

        df = pd.DataFrame(rows)
        return self.validate_output(df)
