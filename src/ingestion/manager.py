"""
Ingestion Manager — orchestrates data ingestion from free News RSS feeds.

News RSS is always available (no API key needed). Pulls from Google News,
Yahoo Finance, CNBC, and MarketWatch.
"""

from datetime import datetime, timedelta
import pandas as pd
from .news import NewsIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionManager:
    """
    Orchestrates data ingestion from News RSS feeds.

    The manager handles:
    - Ingesting from News RSS (always available, no API key)
    - Validating schema for all ingested data
    - Storing summary statistics for dashboard display
    """

    def __init__(self, config):
        self.config = config
        self.live_sources = [
            ('news', NewsIngester(config)),
        ]
        self._summary = None

    def ingest(self, start_date=None, end_date=None):
        """
        Main ingestion entry point.

        Args:
            start_date: datetime or None (defaults to N days ago from config)
            end_date: datetime or None (defaults to now)

        Returns:
            Unified DataFrame with data from News RSS.
        """
        date_cfg = self.config.get('ingestion', {}).get('date_range', {})
        lookback = date_cfg.get('default_lookback_days', 7)

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=lookback)

        logger.info(f"Ingestion date range: {start_date.date()} to {end_date.date()}")

        frames = []
        sources_used = []

        for source_name, source in self.live_sources:
            if source.is_available():
                logger.info(f"Ingesting from {source_name}...")
                try:
                    df = source.ingest(start_date, end_date)
                    if len(df) > 0:
                        df = source.validate_output(df)
                        frames.append(df)
                        sources_used.append(source_name)
                        logger.info(f"  {source_name}: {len(df)} posts")
                    else:
                        logger.info(f"  {source_name}: no data in date range")
                except Exception as e:
                    logger.warning(f"  {source_name} ingestion failed: {e}")

        if not frames:
            raise RuntimeError(
                "No data ingested. News RSS should always provide data — check network connectivity."
            )

        combined = pd.concat(frames, ignore_index=True)
        initial_count = len(combined)

        combined = combined.drop_duplicates(subset=['post_id'], keep='first').reset_index(drop=True)
        deduped_count = len(combined)

        if initial_count != deduped_count:
            logger.info(
                f"Deduplication: {initial_count} posts -> {deduped_count} unique posts "
                f"({initial_count - deduped_count} duplicates removed)"
            )

        posts_per_source = combined.groupby('source').size().to_dict()

        self._summary = {
            'total_posts': len(combined),
            'sources_used': sources_used,
            'sources_unavailable': [],
            'date_range': {'start': start_date, 'end': end_date},
            'posts_per_source': posts_per_source,
            'mode': 'auto',
        }

        logger.info(f"Ingestion complete: {len(combined)} total posts from {sources_used}")
        return combined

    def get_source_summary(self):
        if self._summary is None:
            return {
                'total_posts': 0, 'sources_used': [], 'sources_unavailable': [],
                'date_range': {'start': None, 'end': None},
                'posts_per_source': {}, 'mode': 'not_run',
            }
        return self._summary
