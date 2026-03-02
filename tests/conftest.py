"""
Shared pytest fixtures for MarketPulse test suite.
"""

import os
import sys
import pytest
import pandas as pd
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def config():
    """Load default configuration from config/default.yaml."""
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_posts():
    """10 varied financial posts covering all sentiment classes."""
    return [
        # Bullish (3)
        "Just bought 500 shares of $AAPL. This is going to $200 easy.",
        "Loading up on NVDA calls before earnings. Bullish af.",
        "MSFT breakout confirmed, added to my position today.",
        # Bearish (2)
        "Shorting $TSLA at these levels. P/E is insane, crash incoming.",
        "Bought puts on SPY. This rally is fake, get out now.",
        # Neutral (2)
        "What do you think about GOOG earnings this Thursday?",
        "Tesla reports Q3 deliveries at 435K units according to filing.",
        # Meme (3)
        "GME to the moon 🚀🚀🚀 apes together strong 💎🙌",
        "My wife's boyfriend picks better stocks than me. Down 90% on weeklies.",
        "GUH",
    ]


@pytest.fixture
def sample_df(sample_posts):
    """DataFrame with all required columns from BaseIngester schema."""
    rows = []
    sources = ['reddit', 'reddit', 'stocktwits',
               'reddit', 'stocktwits',
               'reddit', 'news',
               'reddit', 'reddit', 'reddit']
    for i, (text, source) in enumerate(zip(sample_posts, sources)):
        rows.append({
            'post_id': f'test_{i}',
            'text': text,
            'source': source,
            'timestamp': '2026-02-28T12:00:00',
            'author': f'user_{i}',
            'score': i * 10,
            'url': f'https://example.com/{i}',
            'metadata': str({'subreddit': 'wallstreetbets'} if source == 'reddit' else {}),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def labeled_df(sample_df):
    """sample_df after running through the LabelAggregator."""
    from src.labeling.aggregator import LabelAggregator
    agg = LabelAggregator(strategy="majority")
    return agg.aggregate_batch(sample_df)


@pytest.fixture
def trained_pipeline(config):
    """SentimentPipeline trained on a small synthetic subset."""
    from src.models.pipeline import SentimentPipeline

    texts = [
        "Buying AAPL calls, very bullish",
        "Loading up on NVDA, going up",
        "Long MSFT, breakout confirmed",
        "All in on TSLA, price target $300",
        "Buy the dip on GOOG, undervalued",
        "Shorting SPY, puts loaded, crash incoming",
        "Selling everything, market going down",
        "Bearish on TSLA, overvalued bubble",
        "Get out of AMC, this is a dump",
        "Sold all my NVDA, dead cat bounce",
        "What do you think about earnings?",
        "MSFT reports quarterly results today",
        "How does options pricing work? Thoughts?",
        "Anyone know when the Fed meeting is?",
        "According to analysts, revenue grew 10% year-over-year",
        "GME to the moon apes together strong 💎🙌",
        "My wife's boyfriend picks better stocks, loss porn",
        "YOLO tendies diamond hands hodl",
        "Sir this is a casino, stonks only go up",
        "Smooth brain degen gambling behind Wendy's",
    ]
    labels = (
        ['bullish'] * 5 + ['bearish'] * 5 +
        ['neutral'] * 5 + ['meme'] * 5
    )

    pipeline = SentimentPipeline(config.get('model', {}))
    pipeline.train(texts, labels, validation_split=False)
    return pipeline


@pytest.fixture
def ticker_extractor():
    """Initialized TickerExtractor."""
    from src.extraction.ticker_extractor import TickerExtractor
    return TickerExtractor()


@pytest.fixture
def normalizer():
    """Initialized EntityNormalizer."""
    from src.extraction.normalizer import EntityNormalizer
    return EntityNormalizer()
