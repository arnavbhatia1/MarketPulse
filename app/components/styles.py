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
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    .ticker-card:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.15);
    }
    .ticker-card div { color: #E6EDF3; }
    .sentiment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .sentiment-badge-bullish {
        background: rgba(0,200,83,0.15);
        color: #00C853;
    }
    .sentiment-badge-bearish {
        background: rgba(255,23,68,0.15);
        color: #FF1744;
    }
    .sentiment-badge-neutral {
        background: rgba(120,144,156,0.15);
        color: #78909C;
    }
    .sentiment-badge-meme {
        background: rgba(255,214,0,0.15);
        color: #FFD600;
    }
    .sentiment-bullish { color: #00C853 !important; font-weight: bold; }
    .sentiment-bearish { color: #FF1744 !important; font-weight: bold; }
    .sentiment-neutral { color: #78909C !important; font-weight: bold; }
    .sentiment-meme { color: #FFD600 !important; font-weight: bold; }
    .briefing-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
    }
    .briefing-verdict {
        background: #0D1117;
        border-left: 3px solid #58A6FF;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 0 6px 6px 0;
        font-style: italic;
        color: #E6EDF3;
    }
    .source-card {
        background: #0D1117;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 12px;
        margin: 4px 0;
    }
    div[data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px;
    }
    .evidence-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9em;
    }
    .evidence-table th {
        background: #21262D;
        color: #E6EDF3;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 2px solid #30363D;
    }
    .evidence-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #30363D;
        color: #E6EDF3;
    }
    .evidence-table tr:nth-child(even) { background: #161B22; }
    .evidence-table tr:nth-child(odd) { background: #0D1117; }
    .evidence-table tr:hover { background: #21262D; }

    /* ── Auth forms ── */
    .auth-container {
        max-width: 400px;
        margin: 80px auto;
        padding: 32px;
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
    }
    .auth-title {
        text-align: center;
        font-size: 1.8em;
        font-weight: bold;
        color: #E6EDF3;
        margin-bottom: 8px;
    }
    .auth-subtitle {
        text-align: center;
        color: #8B949E;
        margin-bottom: 24px;
    }

    /* ── Portfolio components ── */
    .portfolio-header {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .portfolio-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #E6EDF3;
    }
    .portfolio-change-positive { color: #00C853; font-weight: bold; }
    .portfolio-change-negative { color: #FF1744; font-weight: bold; }
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .risk-badge-conservative {
        background: rgba(88,166,255,0.15);
        color: #58A6FF;
    }
    .risk-badge-moderate {
        background: rgba(255,214,0,0.15);
        color: #FFD600;
    }
    .risk-badge-aggressive {
        background: rgba(255,23,68,0.15);
        color: #FF1744;
    }
    .trade-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .trade-card-buy { border-left: 4px solid #00C853; }
    .trade-card-sell { border-left: 4px solid #FF1744; }
    .bot-commentary {
        background: #0D1117;
        border-left: 3px solid #FFD600;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 0 6px 6px 0;
        color: #E6EDF3;
    }
    .premium-overlay {
        text-align: center;
        padding: 60px 20px;
        background: #161B22;
        border: 2px dashed #30363D;
        border-radius: 12px;
        margin: 20px 0;
    }
    .premium-overlay h3 { color: #FFD600; }
    .onboarding-card {
        background: #161B22;
        border: 1px solid #58A6FF;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin: 40px auto;
        max-width: 600px;
    }
    .stress-gauge { margin: 8px 0; }
    </style>
    """, unsafe_allow_html=True)
