"""Dark financial terminal theme for MarketPulse dashboard."""

COLORS = {
    'bullish': '#00C853',
    'bearish': '#FF1744',
    'neutral': '#78909C',
    'meme': '#FFD600',
    'primary': '#58A6FF',
    'secondary': '#8B949E',
    'bg_primary': '#0D1117',
    'bg_secondary': '#161B22',
    'bg_tertiary': '#21262D',
    'text_primary': '#E6EDF3',
    'text_secondary': '#8B949E',
    'border': '#30363D',
}

SENTIMENT_COLORS = {
    'bullish': COLORS['bullish'],
    'bearish': COLORS['bearish'],
    'neutral': COLORS['neutral'],
    'meme': COLORS['meme'],
}


def apply_theme():
    """Inject custom CSS for MarketPulse-specific components.

    Base dark theme is handled by .streamlit/config.toml — this only
    styles custom HTML elements (ticker cards, sentiment badges, etc.).
    """
    import streamlit as st
    st.markdown("""
    <style>
    .ticker-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        color: #E6EDF3;
    }
    .ticker-card:hover { border-color: #58A6FF; }
    .ticker-card div { color: #E6EDF3; }
    .sentiment-bullish { color: #00C853 !important; font-weight: bold; }
    .sentiment-bearish { color: #FF1744 !important; font-weight: bold; }
    .sentiment-neutral { color: #78909C !important; font-weight: bold; }
    .sentiment-meme { color: #FFD600 !important; font-weight: bold; }
    div[data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
