"""
Tests for ingestion system: synthetic, manager, and news (mocked).
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.base import BaseIngester
from src.ingestion.synthetic import SyntheticIngester
from src.ingestion.manager import IngestionManager
from src.ingestion.news import NewsIngester


class TestSyntheticIngester:
    def test_is_always_available(self, config):
        ingester = SyntheticIngester(config)
        assert ingester.is_available() is True

    def test_returns_valid_schema(self, config):
        ingester = SyntheticIngester(config)
        end = datetime.now()
        start = end - timedelta(days=7)
        df = ingester.ingest(start, end)

        for col in BaseIngester.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_returns_expected_row_count(self, config):
        ingester = SyntheticIngester(config)
        end = datetime.now()
        start = end - timedelta(days=7)
        df = ingester.ingest(start, end)
        # Synthetic data should have substantial content
        assert len(df) > 100

    def test_source_column_is_synthetic(self, config):
        ingester = SyntheticIngester(config)
        end = datetime.now()
        start = end - timedelta(days=7)
        df = ingester.ingest(start, end)
        assert (df['source'] == 'synthetic').all()

    def test_no_empty_texts(self, config):
        ingester = SyntheticIngester(config)
        end = datetime.now()
        start = end - timedelta(days=7)
        df = ingester.ingest(start, end)
        assert df['text'].str.strip().str.len().gt(0).all()


class TestIngestionManager:
    def test_synthetic_mode(self, config):
        config['data']['mode'] = 'synthetic'
        manager = IngestionManager(config)
        df = manager.ingest()
        assert len(df) > 0
        assert 'synthetic' in df['source'].values

    def test_get_source_summary_before_ingest(self, config):
        manager = IngestionManager(config)
        summary = manager.get_source_summary()
        assert summary['total_posts'] == 0
        assert summary['mode'] == 'not_run'

    def test_get_source_summary_after_ingest(self, config):
        config['data']['mode'] = 'synthetic'
        manager = IngestionManager(config)
        manager.ingest()
        summary = manager.get_source_summary()
        assert summary['total_posts'] > 0
        assert 'synthetic' in summary['sources_used']
        assert summary['mode'] == 'synthetic'
        assert summary['used_fallback'] is True

    def test_auto_mode_falls_back(self, config):
        config['data']['mode'] = 'auto'
        manager = IngestionManager(config)
        df = manager.ingest()
        # With no API keys, should fall back to synthetic
        assert len(df) > 0
        summary = manager.get_source_summary()
        assert summary['used_fallback'] is True

    def test_date_range_defaults(self, config):
        config['data']['mode'] = 'synthetic'
        manager = IngestionManager(config)
        manager.ingest()
        summary = manager.get_source_summary()
        assert summary['date_range']['start'] is not None
        assert summary['date_range']['end'] is not None


class TestNewsIngester:
    def test_not_available_without_key(self, config):
        with patch.dict(os.environ, {}, clear=True):
            ingester = NewsIngester(config)
            assert ingester.is_available() is False

    def test_available_with_key(self, config):
        with patch.dict(os.environ, {'NEWS_API_KEY': 'test_key'}):
            ingester = NewsIngester(config)
            assert ingester.is_available() is True

    def test_returns_empty_without_key(self, config):
        with patch.dict(os.environ, {}, clear=True):
            ingester = NewsIngester(config)
            end = datetime.now()
            start = end - timedelta(days=7)
            df = ingester.ingest(start, end)
            assert len(df) == 0
            for col in BaseIngester.REQUIRED_COLUMNS:
                assert col in df.columns

    def test_mocked_api_returns_valid_df(self, config):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            'status': 'ok',
            'totalResults': 2,
            'articles': [
                {
                    'source': {'name': 'CNBC'},
                    'author': 'John Doe',
                    'title': 'Fed announces rate decision',
                    'description': 'The Federal Reserve announced its latest rate decision today.',
                    'url': 'https://cnbc.com/article1',
                    'urlToImage': 'https://cnbc.com/img1.jpg',
                    'publishedAt': '2026-02-28T10:00:00Z',
                },
                {
                    'source': {'name': 'Reuters'},
                    'author': 'Jane Smith',
                    'title': 'AAPL earnings beat expectations',
                    'description': 'Apple reported quarterly earnings above analyst estimates.',
                    'url': 'https://reuters.com/article2',
                    'urlToImage': None,
                    'publishedAt': '2026-02-28T12:00:00Z',
                },
            ]
        }

        with patch.dict(os.environ, {'NEWS_API_KEY': 'test_key'}):
            with patch('src.ingestion.news.requests.get', return_value=mock_response):
                ingester = NewsIngester(config)
                end = datetime.now()
                start = end - timedelta(days=7)
                df = ingester.ingest(start, end)

                # Should have articles (may be multiplied by number of query terms
                # since each term hits the mock)
                assert len(df) > 0

                # Check schema
                for col in BaseIngester.REQUIRED_COLUMNS:
                    assert col in df.columns

                # Check source is 'news'
                assert (df['source'] == 'news').all()

                # Check text is non-empty
                assert df['text'].str.strip().str.len().gt(0).all()

                # Check post_id format starts with 'news_'
                assert df['post_id'].str.startswith('news_').all()

    def test_mocked_api_error_handled(self, config):
        with patch.dict(os.environ, {'NEWS_API_KEY': 'test_key'}):
            with patch('src.ingestion.news.requests.get', side_effect=Exception("API error")):
                ingester = NewsIngester(config)
                end = datetime.now()
                start = end - timedelta(days=7)
                df = ingester.ingest(start, end)
                # Should return empty, not raise
                assert len(df) == 0
