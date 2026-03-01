from datetime import datetime
import pandas as pd
from .base import BaseIngester
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StocktwitsIngester(BaseIngester):
    """
    Ingest messages from Stocktwits API.

    Stocktwits is valuable because:
    - Posts are already associated with ticker symbols
    - Some posts have user-submitted sentiment tags (bullish/bearish)
    - These user tags can be used as additional weak supervision signal

    API endpoint: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json

    Implementation details:
    - Fetch streams for configured symbols
    - Extract message body as text
    - Capture user-submitted sentiment tag if present (in metadata)
    - Handle pagination for date range filtering
    - post_id format: "stocktwits_{message_id}"

    Metadata captured:
    - symbols: list of associated ticker symbols
    - user_sentiment: str or null (user-tagged bullish/bearish)
    - reshares: int
    - likes: int
    """

    def __init__(self, config):
        self.config = config

    def is_available(self) -> bool:
        """Check if Stocktwits access token exists in environment."""
        import os
        return bool(os.getenv('STOCKTWITS_ACCESS_TOKEN'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not self.is_available():
            logger.info("Stocktwits: no API credentials configured")
            return self._empty_dataframe()
        logger.info("Stocktwits: live ingestion not implemented yet")
        return self._empty_dataframe()
