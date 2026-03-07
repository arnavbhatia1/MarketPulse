"""
Ingestion Manager — orchestrates data ingestion across all sources.

Sources: Reddit (requires API key), Stocktwits (requires API key),
         News (free RSS — always available).

At least one source always has data because NewsIngester uses free RSS feeds.
"""

from datetime import datetime, timedelta
import pandas as pd
from .reddit import RedditIngester
from .stocktwits import StocktwitsIngester
from .news import NewsIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionManager:
    """
    Orchestrates data ingestion across all sources with smart fallback logic.

    The manager handles:
    - Checking which sources are available (have API credentials)
    - Ingesting from multiple sources in parallel conceptually
    - Deduplicating results across sources
    - Validating schema for all ingested data
    - Falling back gracefully when sources unavailable
    - Storing summary statistics for dashboard display
    """

    def __init__(self, config):
        """
        Initialize the ingestion manager.

        Args:
            config: Configuration dict with structure from default.yaml
        """
        self.config = config
        self.live_sources = [
            ('reddit', RedditIngester(config)),
            ('stocktwits', StocktwitsIngester(config)),
            ('news', NewsIngester(config)),
        ]

        # Summary statistics from last ingestion
        self._summary = None

    def ingest(self, start_date=None, end_date=None):
        """
        Main ingestion entry point.

        Orchestrates the full ingestion process:
        1. Resolve date range from args or config defaults
        2. Check mode and determine which sources to use
        3. Ingest from each available source
        4. Combine DataFrames
        5. Deduplicate by post_id
        6. Validate schema
        7. Fall back to synthetic if needed (in auto mode)
        8. Store summary statistics

        Args:
            start_date: datetime or None (defaults to N days ago from config)
            end_date: datetime or None (defaults to now)

        Returns:
            Unified DataFrame with data from all available sources.
            Includes 'source' column indicating where each post came from.
        """
        # Resolve date range
        date_cfg = self.config.get('ingestion', {}).get('date_range', {})
        lookback = date_cfg.get('default_lookback_days', 7)

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=lookback)

        logger.info(f"Ingestion date range: {start_date.date()} to {end_date.date()}")

        # Determine operating mode
        mode = self.config.get('data', {}).get('mode', 'auto')
        logger.info(f"Ingestion mode: {mode}")

        frames = []
        sources_used = []
        sources_unavailable = []

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
                    sources_unavailable.append(source_name)
            else:
                sources_unavailable.append(source_name)
                logger.info(f"  {source_name}: not available (missing API credentials)")

        if not frames:
            raise RuntimeError(
                "No data ingested. News RSS should always provide data — check network connectivity."
            )

        combined = pd.concat(frames, ignore_index=True)
        initial_count = len(combined)

        # Deduplicate by post_id
        combined = combined.drop_duplicates(subset=['post_id'], keep='first').reset_index(drop=True)
        deduped_count = len(combined)

        if initial_count != deduped_count:
            logger.info(
                f"Deduplication: {initial_count} posts -> {deduped_count} unique posts "
                f"({initial_count - deduped_count} duplicates removed)"
            )

        # Compute posts per source
        posts_per_source = combined.groupby('source').size().to_dict()

        # Store summary for dashboard/reporting
        self._summary = {
            'total_posts': len(combined),
            'sources_used': sources_used,
            'sources_unavailable': sources_unavailable,
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'posts_per_source': posts_per_source,
            'mode': mode,
        }

        logger.info(
            f"Ingestion complete: {len(combined)} total posts from {sources_used}"
        )

        return combined

    def get_source_summary(self):
        """
        Return dict with ingestion statistics from the last ingest() call.

        Returns:
            Dict with keys:
            - total_posts: int
            - sources_used: list of source names
            - sources_unavailable: list of source names that were not available
            - date_range: dict with 'start' and 'end' datetime objects
            - posts_per_source: dict mapping source name to post count
            - mode: str ('live' or 'auto')

            If ingest() has not been called yet, returns a minimal dict with
            total_posts=0 and mode='not_run'.
        """
        if self._summary is None:
            return {
                'total_posts': 0,
                'sources_used': [],
                'sources_unavailable': [],
                'date_range': {'start': None, 'end': None},
                'posts_per_source': {},
                'mode': 'not_run',
            }
        return self._summary
