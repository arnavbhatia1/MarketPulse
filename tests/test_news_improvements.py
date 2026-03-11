"""Tests for NewsIngester improvements — URL normalization, date handling, per-entry error handling."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd


class TestUrlNormalization:
    def test_strips_utm_params(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article?utm_source=google&utm_medium=rss&id=123"
        normalized = ingester._normalize_url(url)
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized

    def test_strips_fbclid(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article?fbclid=abc123&page=2"
        normalized = ingester._normalize_url(url)
        assert "fbclid" not in normalized
        assert "page=2" in normalized

    def test_strips_fragment(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article?id=5#comments"
        normalized = ingester._normalize_url(url)
        assert "#comments" not in normalized
        assert "id=5" in normalized

    def test_preserves_clean_url(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article/12345"
        normalized = ingester._normalize_url(url)
        assert normalized == url


class TestDateHandling:
    def test_missing_date_returns_none(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        entry = {}
        result = ingester._parse_date(entry)
        assert result is None

    def test_malformed_date_returns_none(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        entry = {'published': 'not-a-date'}
        result = ingester._parse_date(entry)
        assert result is None

    def test_valid_date_parses(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        entry = {'published': 'Fri, 28 Feb 2026 10:00:00 GMT'}
        result = ingester._parse_date(entry)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2


class TestPerEntryErrorHandling:
    @patch('src.ingestion.news.feedparser')
    def test_bad_entry_skipped_others_survive(self, mock_fp):
        from src.ingestion.news import NewsIngester

        good_entry = MagicMock()
        good_entry.get = lambda k, d='': {
            'link': 'https://example.com/good',
            'title': 'Good article',
            'summary': 'This is good',
            'published': 'Fri, 28 Feb 2026 10:00:00 GMT',
            'author': 'tester',
        }.get(k, d)

        bad_entry = MagicMock()
        bad_entry.get = MagicMock(side_effect=Exception("corrupt entry"))

        mock_feed = MagicMock()
        mock_feed.entries = [bad_entry, good_entry]
        mock_fp.parse.return_value = mock_feed

        ingester = NewsIngester({'ingestion': {'news': {}}})
        rows = ingester._parse_feed(
            'https://fake.com/rss', 'test',
            set(), datetime(2026, 1, 1), datetime(2026, 12, 31)
        )
        assert len(rows) == 1
        assert rows[0]['text'].startswith('Good article')
