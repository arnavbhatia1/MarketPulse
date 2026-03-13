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

    /* ── Market Regime Banner ── */
    .regime-banner {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .regime-label { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.05em; }
    .regime-bull { color: #00C853; }
    .regime-bear { color: #FF1744; }
    .regime-sideways { color: #FFD600; }
    .regime-volatile { color: #FF9100; }
    .regime-crash { color: #FF1744; text-shadow: 0 0 10px rgba(255,23,68,0.5); }

    /* ── VIX Badge ── */
    .vix-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .vix-low { background: rgba(0,200,83,0.15); color: #00C853; }
    .vix-normal { background: rgba(255,214,0,0.15); color: #FFD600; }
    .vix-high { background: rgba(255,23,68,0.15); color: #FF1744; }

    /* ── Anomaly Badges ── */
    .anomaly-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin-left: 0.3rem; }
    .badge-52w-high { background: rgba(0,200,83,0.2); color: #00C853; }
    .badge-volume-spike { background: rgba(255,145,0,0.2); color: #FF9100; }
    .badge-gap-up { background: rgba(0,200,83,0.2); color: #00C853; }
    .badge-gap-down { background: rgba(255,23,68,0.2); color: #FF1744; }

    /* ── Mover Card ── */
    .mover-card { background: #161B22; border: 1px solid #30363D; border-radius: 10px; padding: 1rem; text-align: center; transition: border-color 0.2s; }
    .mover-card:hover { border-color: #58A6FF; }
    .mover-symbol { font-size: 1.1rem; font-weight: 700; color: #E6EDF3; }
    .mover-change-pos { color: #00C853; font-weight: 600; }
    .mover-change-neg { color: #FF1744; font-weight: 600; }

    /* ── Score Card ── */
    .score-card { background: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 1.2rem; }
    .score-value { font-size: 2.2rem; font-weight: 700; }
    .score-label { font-size: 0.85rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; }
    .score-high { color: #00C853; }
    .score-mid { color: #FFD600; }
    .score-low { color: #FF1744; }

    /* ── Smart Money Card ── */
    .smart-money-card { background: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 1.2rem; }

    /* ── Powered By Badge ── */
    .powered-badge { text-align: center; padding: 0.8rem; color: #8B949E; font-size: 0.85rem; }
    .powered-badge a { color: #58A6FF; text-decoration: none; }
    .powered-badge a:hover { text-decoration: underline; }

    /* ── MCP Unavailable Banner ── */
    .mcp-unavailable { background: rgba(255,145,0,0.1); border: 1px solid rgba(255,145,0,0.3); border-radius: 8px; padding: 0.8rem 1rem; color: #FF9100; font-size: 0.85rem; text-align: center; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)
