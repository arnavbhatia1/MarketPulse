"""Tests for the Claude briefing agent."""
import os, sys, pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SAMPLE_TICKER_DATA = {
    'symbol': 'TSLA',
    'dominant_sentiment': 'bearish',
    'mention_count': 120,
    'reddit_sentiment': 'bearish',
    'news_sentiment': 'neutral',
    'stocktwits_sentiment': 'bearish',
    'sentiment_by_day': {
        '2026-03-05': 'neutral',
        '2026-03-06': 'bearish',
        '2026-03-07': 'bearish',
    },
    'top_posts': {
        'reddit': [
            {'text': 'TSLA puts loaded, P/E is insane', 'sentiment': 'bearish'},
            {'text': 'Shorting TSLA at these levels', 'sentiment': 'bearish'},
        ],
        'news': [
            {'text': 'Tesla reports Q3 deliveries at 435K units', 'sentiment': 'neutral'},
        ],
        'stocktwits': [],
    },
}


def test_generate_briefing_returns_string():
    """generate_briefing returns a non-empty string."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Tesla sentiment has turned bearish this week.")]

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        from src.agent.briefing import generate_briefing
        result = generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert isinstance(result, str)
    assert len(result) > 10


def test_generate_briefing_calls_claude_once():
    """Only one API call per briefing."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Some verdict.")]

    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        import importlib
        import src.agent.briefing as briefing_mod
        importlib.reload(briefing_mod)
        briefing_mod.generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert mock_client.messages.create.call_count == 1


def test_generate_briefing_no_api_key():
    """Returns fallback string when Claude is unavailable."""
    with patch('src.agent.briefing.anthropic.Anthropic') as mock_client_cls:
        mock_client_cls.side_effect = Exception("No API key")

        import importlib
        import src.agent.briefing as briefing_mod
        importlib.reload(briefing_mod)
        result = briefing_mod.generate_briefing('Tesla', 'TSLA', SAMPLE_TICKER_DATA)

    assert isinstance(result, str)
    assert len(result) > 0  # fallback, not empty
