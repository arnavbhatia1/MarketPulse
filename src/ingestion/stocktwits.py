import os
import requests
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
    - post_id format: "stocktwits_{message_id}"

    Metadata captured:
    - symbols: list of associated ticker symbols
    - user_sentiment: str or null (user-tagged bullish/bearish)
    - reshares: int
    - likes: int
    """

    BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    def __init__(self, config):
        self.config = config
        st_cfg = config.get('ingestion', {}).get('stocktwits', {})
        self.symbols = st_cfg.get(
            'symbols', ['AAPL', 'TSLA', 'NVDA', 'GME', 'AMC', 'SPY', 'MSFT', 'AMZN']
        )
        self.limit_per_symbol = st_cfg.get('limit_per_symbol', 50)

    def is_available(self) -> bool:
        """Check if Stocktwits access token exists in environment."""
        return bool(os.getenv('STOCKTWITS_ACCESS_TOKEN'))

    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch recent messages for each configured symbol.

        Passes the access token as a query parameter when available.
        Filters messages to the requested date window.
        Captures user-submitted bullish/bearish sentiment tags for use as
        weak supervision signals downstream.
        Gracefully skips symbols that return API errors.
        """
        if not self.is_available():
            logger.info("Stocktwits: no access token configured, returning empty")
            return self._empty_dataframe()

        token = os.getenv('STOCKTWITS_ACCESS_TOKEN')
        rows = []

        for symbol in self.symbols:
            logger.info(f"Stocktwits: fetching {symbol} (limit={self.limit_per_symbol})...")
            try:
                url = self.BASE_URL.format(symbol=symbol)
                params = {'access_token': token}
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                messages = data.get('messages', [])
                for msg in messages[:self.limit_per_symbol]:
                    created = datetime.strptime(
                        msg['created_at'], "%Y-%m-%dT%H:%M:%SZ"
                    )

                    # Skip messages outside the requested window
                    if created < start_date or created > end_date:
                        continue

                    # Extract user-submitted sentiment tag if present
                    user_sentiment = None
                    entities = msg.get('entities') or {}
                    sentiment_block = entities.get('sentiment')
                    if sentiment_block:
                        user_sentiment = sentiment_block.get('basic')

                    rows.append({
                        'post_id': f"stocktwits_{msg['id']}",
                        'text': msg.get('body', '')[:500],
                        'source': 'stocktwits',
                        'timestamp': created.isoformat(),
                        'author': (msg.get('user') or {}).get('username', 'unknown'),
                        'score': (msg.get('likes') or {}).get('total', 0),
                        'url': f"https://stocktwits.com/message/{msg['id']}",
                        'metadata': str({
                            'symbols': [s['symbol'] for s in msg.get('symbols', [])],
                            'user_sentiment': user_sentiment,
                            'reshares': (msg.get('reshares') or {}).get('reshared_count', 0),
                            'likes': (msg.get('likes') or {}).get('total', 0),
                        }),
                    })
            except Exception as e:
                logger.warning(f"Stocktwits: error fetching {symbol}: {e}")

        logger.info(f"Stocktwits: {len(rows)} messages fetched across {len(self.symbols)} symbols")

        if not rows:
            return self._empty_dataframe()

        df = pd.DataFrame(rows)
        return self.validate_output(df)
