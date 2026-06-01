"""
Pipeline runner for MarketPulse.

Two distinct operations:
  refresh_pipeline() — slow: ingest → label → analyze → write SQLite
  get_ticker_cache() — fast: read SQLite → return to dashboard

Pages import get_ticker_cache() for display and call refresh_pipeline()
only on startup or when user clicks Refresh.
"""

import streamlit as st
import sys
import os
from datetime import datetime

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.extraction.ticker_extractor import TickerExtractor
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
from src.storage.db import (
    init_db, save_posts, load_posts,
    save_ticker_cache, load_ticker_cache
)


def refresh_pipeline(start_date_str=None, end_date_str=None, progress_callback=None) -> dict:
    """
    Run full pipeline: ingest → label → extract → analyze → write SQLite.

    Expensive. Call on startup or manual Refresh only.
    Returns source_summary for status display.

    Args:
        progress_callback: optional callable(str) invoked at each pipeline stage
                           for UI feedback. Defaults to None (silent).
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    init_db()
    config = load_config()

    start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

    # Ingest
    _progress("Fetching from RSS feeds...")
    mgr = IngestionManager(config)
    df = mgr.ingest(start_date=start_date, end_date=end_date)
    source_summary = mgr.get_source_summary()

    # Label
    _progress("Labeling posts...")
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    # Extract tickers
    _progress("Extracting tickers...")
    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)

    # Map programmatic_label → sentiment column for storage
    df['sentiment'] = df['programmatic_label']
    if 'label_confidence' in df.columns:
        df['confidence'] = df['label_confidence'].fillna(0.0)
    else:
        df['confidence'] = 0.0

    # Analyze per-ticker
    _progress("Analyzing sentiment...")
    analyzer = TickerSentimentAnalyzer()
    ticker_results = analyzer.analyze(df)

    # Write to SQLite
    _progress("Saving results...")
    save_posts(df)
    save_ticker_cache(ticker_results)

    # Optionally train/retrain model if enough labeled data
    _maybe_train_model(df, config)

    # Coverage stats for the UI (how much of the news we could classify)
    total_posts = len(df)
    labeled_posts = int(df['programmatic_label'].notna().sum()) if 'programmatic_label' in df.columns else 0
    source_summary['total_posts'] = total_posts
    source_summary['labeled_posts'] = labeled_posts
    source_summary['label_coverage'] = (labeled_posts / total_posts) if total_posts else 0.0
    source_summary['ticker_count'] = len(ticker_results)

    return source_summary


@st.cache_data(ttl=20)
def get_ticker_cache() -> dict:
    """
    Read ticker_cache from SQLite. Cached briefly so background auto-refreshes
    surface quickly. Fast — no ingestion, no compute.
    """
    init_db()
    return load_ticker_cache()


@st.cache_resource
def load_model():
    """Load trained sentiment model from disk. None if not yet trained."""
    from src.models.pipeline import SentimentPipeline
    model_dir = os.path.join(_root, "data", "models")
    model_path = os.path.join(model_dir, "sentiment_model.pkl")
    if not os.path.exists(model_path):
        return None
    pipeline = SentimentPipeline()
    try:
        pipeline.load(model_dir)
        return pipeline
    except Exception:
        return None


def _maybe_train_model(df, config, min_samples=200):
    """
    Train model if we have enough labeled data and no model exists yet.
    Silently skips if not enough data or model already exists.
    """
    from src.models.pipeline import SentimentPipeline
    from src.storage.db import log_training_run
    import uuid

    labeled = df[df['programmatic_label'].notna()]
    if len(labeled) < min_samples:
        return

    model_path = os.path.join(_root, "data", "models", "sentiment_model.pkl")
    if os.path.exists(model_path):
        return  # Already trained — don't auto-retrain

    pipeline = SentimentPipeline(config=config)
    report = pipeline.train(
        labeled['text'].tolist(),
        labeled['programmatic_label'].tolist()
    )
    pipeline.save(os.path.join(_root, "data", "models"))

    f1 = 0.0
    if report:
        val = report.get('validation_metrics') or report.get('training_metrics')
        if val:
            f1 = val.get('weighted_f1', 0.0)
    log_training_run(
        run_id=str(uuid.uuid4()),
        num_samples=len(labeled),
        weighted_f1=f1,
    )
