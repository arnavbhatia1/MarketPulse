"""Tests for SQLite storage layer."""
import os
import json
import pytest
import pandas as pd
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Override DB_PATH to a temp file for each test."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("src.storage.db.DB_PATH", db_file)
    from src.storage import db
    db.init_db()
    return db


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            'post_id': 'reddit_wsb_001',
            'text': 'Loading NVDA calls, bullish af 🚀',
            'source': 'reddit',
            'timestamp': '2026-03-07 10:00:00',
            'author': 'user1',
            'score': 42,
            'tickers': ['NVIDIA'],
            'sentiment': 'bullish',
            'confidence': 0.82,
            'url': '',
        },
        {
            'post_id': 'news_001',
            'text': 'Tesla reports Q3 deliveries according to SEC filing.',
            'source': 'news',
            'timestamp': '2026-03-06 09:00:00',
            'author': 'Reuters',
            'score': 0,
            'tickers': ['Tesla'],
            'sentiment': 'neutral',
            'confidence': 0.71,
            'url': 'https://example.com',
        },
    ])


def test_init_db_creates_tables(tmp_db):
    import sqlite3
    from src.storage.db import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert 'posts' in tables
    assert 'ticker_cache' in tables
    assert 'model_training_log' in tables


def test_save_and_load_posts(tmp_db, sample_df):
    from src.storage.db import save_posts, load_posts
    save_posts(sample_df)
    result = load_posts()
    assert len(result) == 2
    assert set(result['post_id']) == {'reddit_wsb_001', 'news_001'}


def test_load_posts_filters_by_date(tmp_db, sample_df):
    from src.storage.db import save_posts, load_posts
    save_posts(sample_df)
    result = load_posts(start_date='2026-03-07', end_date='2026-03-07')
    assert len(result) == 1
    assert result.iloc[0]['post_id'] == 'reddit_wsb_001'


def test_save_posts_upserts(tmp_db, sample_df):
    """Saving same post twice should not duplicate it."""
    from src.storage.db import save_posts, load_posts
    save_posts(sample_df)
    save_posts(sample_df)
    result = load_posts()
    assert len(result) == 2


def test_save_and_load_ticker_cache(tmp_db):
    from src.storage.db import save_ticker_cache, load_ticker_cache
    ticker_results = {
        'Tesla': {
            'symbol': 'TSLA',
            'dominant_sentiment': 'bearish',
            'mention_count': 45,
            'reddit_sentiment': 'bearish',
            'news_sentiment': 'neutral',
            'stocktwits_sentiment': 'bearish',
            'sentiment_by_day': {'2026-03-07': 'bearish'},
            'top_posts': {'reddit': [{'text': 'TSLA puts loaded', 'sentiment': 'bearish'}]},
        }
    }
    save_ticker_cache(ticker_results)
    result = load_ticker_cache()
    assert 'Tesla' in result
    assert result['Tesla']['dominant_sentiment'] == 'bearish'
    assert result['Tesla']['sentiment_by_day'] == {'2026-03-07': 'bearish'}


def test_log_training_run(tmp_db):
    from src.storage.db import log_training_run, get_training_history
    log_training_run('run_001', num_samples=400, weighted_f1=0.74)
    history = get_training_history()
    assert len(history) == 1
    assert history[0]['run_id'] == 'run_001'
    assert history[0]['weighted_f1'] == pytest.approx(0.74)
