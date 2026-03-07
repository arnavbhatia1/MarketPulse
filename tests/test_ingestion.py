"""
Tests for ingestion system: manager and news (mocked).
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.base import BaseIngester
from src.ingestion.manager import IngestionManager
from src.ingestion.news import NewsIngester


class TestIngestionManager:
    def _news_df(self):
        """Minimal valid news DataFrame for mocking."""
        return pd.DataFrame([{
            'post_id': 'news_test_001',
            'text': 'Fed announces rate decision today',
            'source': 'news',
            'timestamp': '2026-03-07 10:00:00',
            'author': 'Reuters',
            'score': 0,
            'url': 'https://example.com/1',
            'metadata': '{}',
        }])

    def test_auto_mode_uses_news(self, config):
        """Auto mode ingests from news RSS (always available, no keys needed)."""
        config['data']['mode'] = 'auto'
        with patch.object(NewsIngester, 'ingest', return_value=self._news_df()):
            manager = IngestionManager(config)
            df = manager.ingest()
        assert len(df) > 0
        assert 'news' in df['source'].values

    def test_get_source_summary_before_ingest(self, config):
        manager = IngestionManager(config)
        summary = manager.get_source_summary()
        assert summary['total_posts'] == 0
        assert summary['mode'] == 'not_run'

    def test_get_source_summary_after_ingest(self, config):
        config['data']['mode'] = 'auto'
        with patch.object(NewsIngester, 'ingest', return_value=self._news_df()):
            manager = IngestionManager(config)
            manager.ingest()
        summary = manager.get_source_summary()
        assert summary['total_posts'] > 0
        assert 'news' in summary['sources_used']

    def test_date_range_defaults(self, config):
        config['data']['mode'] = 'auto'
        with patch.object(NewsIngester, 'ingest', return_value=self._news_df()):
            manager = IngestionManager(config)
            manager.ingest()
        summary = manager.get_source_summary()
        assert summary['date_range']['start'] is not None
        assert summary['date_range']['end'] is not None

    def test_raises_when_all_sources_fail(self, config):
        """Raises RuntimeError if all sources return no data."""
        empty_df = pd.DataFrame(columns=BaseIngester.REQUIRED_COLUMNS)
        with patch.object(NewsIngester, 'ingest', return_value=empty_df):
            manager = IngestionManager(config)
            with pytest.raises(RuntimeError):
                manager.ingest()


class TestNewsIngester:
    def _make_entry(self, url='https://example.com/article1'):
        data = {
            'link': url,
            'title': 'Fed announces rate decision',
            'summary': 'The Federal Reserve announced its latest rate decision today.',
            'author': 'Reuters',
            'published': 'Fri, 28 Feb 2026 10:00:00 GMT',
        }
        entry = MagicMock()
        entry.get.side_effect = lambda key, default='': data.get(key, default)
        for k, v in data.items():
            setattr(entry, k, v)
        return entry

    def test_always_available(self, config):
        """RSS-based ingester needs no API key — always available."""
        ingester = NewsIngester(config)
        assert ingester.is_available() is True

    def test_mocked_feed_returns_valid_df(self, config):
        mock_feed = MagicMock()
        mock_feed.entries = [
            self._make_entry('https://example.com/a1'),
            self._make_entry('https://example.com/a2'),
        ]
        mock_feed.entries[1].title = 'AAPL earnings beat expectations'
        mock_feed.entries[1].link = 'https://example.com/a2'

        with patch('src.ingestion.news.feedparser.parse', return_value=mock_feed):
            ingester = NewsIngester(config)
            # Wide date range so mock articles (Feb 28) are not filtered out
            start = datetime(2026, 1, 1)
            end = datetime(2026, 12, 31)
            df = ingester.ingest(start, end)

        assert len(df) > 0
        for col in BaseIngester.REQUIRED_COLUMNS:
            assert col in df.columns
        assert (df['source'] == 'news').all()
        assert df['text'].str.strip().str.len().gt(0).all()
        assert df['post_id'].str.startswith('news_').all()

    def test_feed_error_returns_empty(self, config):
        with patch('src.ingestion.news.feedparser.parse', side_effect=Exception("network error")):
            ingester = NewsIngester(config)
            end = datetime.now()
            start = end - timedelta(days=7)
            df = ingester.ingest(start, end)
        assert len(df) == 0
        for col in BaseIngester.REQUIRED_COLUMNS:
            assert col in df.columns

    def test_deduplicates_by_url(self, config):
        mock_feed = MagicMock()
        mock_feed.entries = [self._make_entry()]  # same URL in every feed call

        with patch('src.ingestion.news.feedparser.parse', return_value=mock_feed):
            ingester = NewsIngester(config)
            end = datetime.now()
            start = end - timedelta(days=7)
            df = ingester.ingest(start, end)

        assert df['url'].duplicated().sum() == 0
