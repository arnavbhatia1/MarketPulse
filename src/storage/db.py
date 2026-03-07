"""
SQLite storage layer for MarketPulse.

Single .db file at data/marketpulse.db — free, persistent, no cloud required.
All reads/writes go through this module. No direct sqlite3 calls elsewhere.
"""

import sqlite3
import json
import os
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "marketpulse.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call repeatedly."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id    TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            source     TEXT NOT NULL,
            timestamp  TEXT,
            author     TEXT,
            score      INTEGER DEFAULT 0,
            tickers    TEXT DEFAULT '[]',
            sentiment  TEXT,
            confidence REAL DEFAULT 0.0,
            url        TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS ticker_cache (
            ticker                TEXT PRIMARY KEY,
            symbol                TEXT,
            last_updated          TEXT,
            dominant_sentiment    TEXT,
            mention_count         INTEGER DEFAULT 0,
            reddit_sentiment      TEXT,
            news_sentiment        TEXT,
            stocktwits_sentiment  TEXT,
            sentiment_by_day      TEXT DEFAULT '{}',
            top_posts             TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS model_training_log (
            run_id       TEXT PRIMARY KEY,
            trained_at   TEXT,
            num_samples  INTEGER,
            weighted_f1  REAL,
            label_source TEXT DEFAULT 'keyword_majority'
        );
    """)
    conn.commit()
    conn.close()


def save_posts(df: pd.DataFrame):
    """Upsert posts DataFrame into SQLite. Safe to call multiple times."""
    conn = get_connection()
    for _, row in df.iterrows():
        tickers = row.get('tickers', [])
        if not isinstance(tickers, list):
            tickers = []
        conn.execute(
            """INSERT OR REPLACE INTO posts
               (post_id, text, source, timestamp, author, score,
                tickers, sentiment, confidence, url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(row['post_id']),
                str(row['text']),
                str(row.get('source', 'unknown')),
                str(row.get('timestamp', '')),
                str(row.get('author', 'unknown')),
                int(row.get('score', 0)),
                json.dumps(tickers),
                row.get('sentiment'),
                float(row.get('confidence', 0.0)),
                str(row.get('url', '')),
            )
        )
    conn.commit()
    conn.close()


def load_posts(start_date=None, end_date=None) -> pd.DataFrame:
    """Load posts, optionally filtered by date range (YYYY-MM-DD strings)."""
    conn = get_connection()
    if start_date and end_date:
        rows = conn.execute(
            "SELECT * FROM posts WHERE timestamp >= ? AND timestamp <= ?",
            (str(start_date), str(end_date) + " 23:59:59")
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM posts").fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            'post_id', 'text', 'source', 'timestamp', 'author',
            'score', 'tickers', 'sentiment', 'confidence', 'url'
        ])

    df = pd.DataFrame([dict(r) for r in rows])
    df['tickers'] = df['tickers'].apply(
        lambda x: json.loads(x) if x else []
    )
    return df


def save_ticker_cache(ticker_results: dict):
    """Upsert ticker_results dict into ticker_cache table."""
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    for company, data in ticker_results.items():
        conn.execute(
            """INSERT OR REPLACE INTO ticker_cache
               (ticker, symbol, last_updated, dominant_sentiment, mention_count,
                reddit_sentiment, news_sentiment, stocktwits_sentiment,
                sentiment_by_day, top_posts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                company,
                data.get('symbol', ''),
                now,
                data.get('dominant_sentiment', 'neutral'),
                int(data.get('mention_count', 0)),
                data.get('reddit_sentiment', 'neutral'),
                data.get('news_sentiment', 'neutral'),
                data.get('stocktwits_sentiment', 'neutral'),
                json.dumps(data.get('sentiment_by_day', {})),
                json.dumps(data.get('top_posts', {})),
            )
        )
    conn.commit()
    conn.close()


def load_ticker_cache() -> dict:
    """Return all rows from ticker_cache as dict keyed by company name."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM ticker_cache").fetchall()
    conn.close()
    result = {}
    for row in rows:
        d = dict(row)
        d['sentiment_by_day'] = json.loads(d.get('sentiment_by_day') or '{}')
        d['top_posts'] = json.loads(d.get('top_posts') or '{}')
        result[d['ticker']] = d
    return result


def log_training_run(run_id: str, num_samples: int, weighted_f1: float,
                     label_source: str = 'keyword_majority'):
    """Record a model training run."""
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO model_training_log
           (run_id, trained_at, num_samples, weighted_f1, label_source)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, datetime.utcnow().isoformat(), num_samples, weighted_f1, label_source)
    )
    conn.commit()
    conn.close()


def get_training_history() -> list:
    """Return all training runs as list of dicts, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM model_training_log ORDER BY trained_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
