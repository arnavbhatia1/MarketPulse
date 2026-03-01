from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseIngester(ABC):
    """Abstract base class for all data sources."""

    REQUIRED_COLUMNS = [
        'post_id', 'text', 'source', 'timestamp',
        'author', 'score', 'url', 'metadata',
    ]

    @abstractmethod
    def ingest(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Ingest data for the given date range.
        Must return DataFrame matching REQUIRED_COLUMNS schema.
        Must handle API rate limits gracefully.
        Must log ingestion progress.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this source is configured and reachable.
        Returns False if API keys are missing or invalid.
        """
        pass

    def validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, drop null texts, deduplicate by post_id."""
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        before = len(df)
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip().astype(bool)]
        df = df.drop_duplicates(subset=['post_id'], keep='first')
        after = len(df)
        if before != after:
            logger.info(f"Validation: {before} -> {after} posts (dropped {before - after})")
        return df.reset_index(drop=True)

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
