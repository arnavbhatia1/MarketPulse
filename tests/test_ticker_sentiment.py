"""Tests for extended TickerSentimentAnalyzer fields."""
import os, sys, pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.ticker_sentiment import TickerSentimentAnalyzer


@pytest.fixture
def multi_source_df():
    return pd.DataFrame([
        {'text': 'Tesla reports Q3 deliveries per SEC filing', 'source': 'news',
         'programmatic_label': 'neutral', 'label_confidence': 0.9,
         'tickers': ['Tesla'], 'timestamp': '2026-03-05 09:00:00'},
        {'text': 'Tesla stock tumbles on delivery miss', 'source': 'news',
         'programmatic_label': 'bearish', 'label_confidence': 0.8,
         'tickers': ['Tesla'], 'timestamp': '2026-03-06 11:00:00'},
        {'text': 'Tesla beats EV market expectations', 'source': 'news',
         'programmatic_label': 'bullish', 'label_confidence': 0.75,
         'tickers': ['Tesla'], 'timestamp': '2026-03-07 10:00:00'},
        {'text': 'Tesla cybertruck recall widens', 'source': 'news',
         'programmatic_label': 'bearish', 'label_confidence': 0.7,
         'tickers': ['Tesla'], 'timestamp': '2026-03-07 14:00:00'},
    ])


def test_per_source_sentiment(multi_source_df):
    analyzer = TickerSentimentAnalyzer()
    results = analyzer.analyze(multi_source_df)
    tesla = results['Tesla']
    assert 'news_sentiment' in tesla
    assert tesla['news_sentiment'] == 'bearish'


def test_sentiment_by_day(multi_source_df):
    analyzer = TickerSentimentAnalyzer()
    results = analyzer.analyze(multi_source_df)
    tesla = results['Tesla']
    assert 'sentiment_by_day' in tesla
    assert isinstance(tesla['sentiment_by_day'], dict)
    assert len(tesla['sentiment_by_day']) == 3


def test_top_posts_per_source(multi_source_df):
    analyzer = TickerSentimentAnalyzer()
    results = analyzer.analyze(multi_source_df)
    tesla = results['Tesla']
    assert 'top_posts' in tesla
    assert isinstance(tesla['top_posts'], dict)
    assert 'news' in tesla['top_posts']
    assert isinstance(tesla['top_posts']['news'], list)
    assert len(tesla['top_posts']['news']) <= 5
