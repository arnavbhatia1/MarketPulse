"""
MarketPulse — Financial Sentiment Hub

Run: streamlit run app/MarketPulse.py
"""

import streamlit as st
import sys, os, json
from datetime import date, timedelta
from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv(os.path.join(_root, '.env'))

from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import ticker_mentions_bar

st.set_page_config(
    page_title="MarketPulse",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("MarketPulse")
st.sidebar.markdown("**Sentiment intelligence for financial markets**")
st.sidebar.markdown("---")

_today = date.today()
start_date = st.sidebar.date_input(
    "Start date", value=_today - timedelta(days=7),
    min_value=_today - timedelta(days=30), max_value=_today,
)
end_date = st.sidebar.date_input(
    "End date", value=_today,
    min_value=_today - timedelta(days=30), max_value=_today,
)
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

st.session_state["start_date"] = start_date.isoformat()
st.session_state["end_date"] = end_date.isoformat()

from app.pipeline_runner import refresh_pipeline, load_model, get_ticker_cache

if st.sidebar.button("Refresh Data", use_container_width=True):
    with st.spinner("Ingesting and analyzing market data..."):
        source_summary = refresh_pipeline(
            start_date_str=start_date.isoformat(),
            end_date_str=end_date.isoformat(),
        )
        st.cache_data.clear()
    st.rerun()

# Model status
from src.storage.db import init_db, get_training_history

init_db()
model = load_model()
if model and model.is_trained:
    history = get_training_history()
    f1 = history[0]['weighted_f1'] if history else 0.0
    st.sidebar.success(f"Model trained (F1: {f1:.2f})")
else:
    st.sidebar.info("Keyword fallback active (model not yet trained)")

st.sidebar.markdown("---")

# ── Load ticker cache ─────────────────────────────────────────────────────────
ticker_results = get_ticker_cache()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("MarketPulse")
st.markdown("Sentiment intelligence for financial markets.")
st.markdown("---")

# ── Search bar (PRIMARY) ──────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="Research a ticker",
        placeholder="TSLA, NVDA, AAPL...",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("Research", use_container_width=True)

# ── Briefing card (inline, below search) ─────────────────────────────────────
if search_clicked and query.strip():
    from src.extraction.normalizer import EntityNormalizer
    from src.agent.briefing import generate_briefing

    normalizer = EntityNormalizer()

    # Resolve query to canonical company name
    resolved = normalizer.normalize(query.strip())
    ticker_data = ticker_results.get(resolved)

    # Fallback: try symbol lookup
    if not ticker_data:
        symbol_upper = query.strip().upper()
        for company, data in ticker_results.items():
            if data.get('symbol', '').upper() == symbol_upper:
                ticker_data = data
                resolved = company
                break

    if not ticker_data:
        st.warning(f"No data found for **{query.strip()}**. Try refreshing data or check the ticker symbol.")
    else:
        symbol = ticker_data.get('symbol', resolved.upper())
        dominant = ticker_data.get('dominant_sentiment', 'neutral')
        color = SENTIMENT_COLORS.get(dominant, COLORS['secondary'])
        mention_count = ticker_data.get('mention_count', 0)
        last_updated = ticker_data.get('last_updated', 'unknown')

        # Header card
        st.markdown(f"""
        <div class="ticker-card" style="border-left: 4px solid {color}; padding: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.6em; font-weight:bold;">{symbol}</span>
                    <span style="color:#8B949E; margin-left:10px;">{resolved}</span>
                </div>
                <div class="sentiment-{dominant}" style="font-size:1.2em; font-weight:bold;">
                    {dominant.upper()}
                </div>
            </div>
            <div style="color:#8B949E; font-size:0.85em; margin-top:4px;">
                {mention_count} mentions · updated {last_updated[:16] if last_updated != 'unknown' else 'unknown'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Verdict
        with st.container():
            st.markdown("#### AI Verdict")
            with st.spinner("Generating verdict..."):
                verdict = generate_briefing(resolved, symbol, ticker_data)
            st.info(f'"{verdict}"\n\n— MarketPulse AI')

        # Sentiment trend chart
        by_day = ticker_data.get('sentiment_by_day', {})
        if by_day:
            st.markdown("#### Sentiment Trend (7 days)")
            import plotly.graph_objects as go

            days = sorted(by_day.keys())
            bar_colors = [SENTIMENT_COLORS.get(by_day[d], COLORS['secondary']) for d in days]

            fig = go.Figure(go.Bar(
                x=days,
                y=[1] * len(days),
                marker_color=bar_colors,
                text=[by_day[d] for d in days],
                textposition='inside',
            ))
            fig.update_layout(
                template='plotly_dark', height=160,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        # By Source breakdown
        st.markdown("#### By Source")
        top_posts = ticker_data.get('top_posts', {})
        src_cols = st.columns(3)
        for i, source in enumerate(('reddit', 'stocktwits', 'news')):
            src_sentiment = ticker_data.get(f'{source}_sentiment') or 'N/A'
            src_posts = top_posts.get(source, [])
            src_color = SENTIMENT_COLORS.get(src_sentiment, COLORS['secondary'])
            with src_cols[i]:
                st.markdown(f"**{source.upper()}**")
                st.markdown(
                    f"<span style='color:{src_color}; font-weight:bold;'>"
                    f"{src_sentiment.upper() if src_sentiment else 'N/A'}</span>",
                    unsafe_allow_html=True
                )
                for post in src_posts[:3]:
                    st.caption(f"> {post['text'][:100]}...")

    st.markdown("---")

# ── Market Overview grid (SECONDARY) ─────────────────────────────────────────
if not ticker_results:
    st.info(
        "No market data yet. Click **Refresh Data** in the sidebar to ingest and analyze."
    )
else:
    st.markdown("### Market Overview")

    # KPI row
    from collections import Counter
    sentiment_dist = Counter(
        v['dominant_sentiment'] for v in ticker_results.values()
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickers Tracked", len(ticker_results))
    k2.metric("Bullish", sentiment_dist.get('bullish', 0))
    k3.metric("Bearish", sentiment_dist.get('bearish', 0))
    k4.metric("Neutral", sentiment_dist.get('neutral', 0))

    st.markdown("---")

    # Ticker card grid
    cols = st.columns(3)
    for i, (company, data) in enumerate(ticker_results.items()):
        sentiment = data.get('dominant_sentiment', 'neutral')
        color = SENTIMENT_COLORS.get(sentiment, COLORS['secondary'])
        symbol = data.get('symbol', company.upper())
        mentions = data.get('mention_count', 0)
        conf = data.get('avg_confidence', 0.0)

        with cols[i % 3]:
            st.markdown(f"""
            <div class="ticker-card">
                <div style="font-size:1.2em; font-weight:bold;">{symbol}</div>
                <div style="color:#8B949E; font-size:0.85em;">{company}</div>
                <div class="sentiment-{sentiment}" style="margin:6px 0;">
                    {sentiment.upper()}
                </div>
                <div style="color:#8B949E; font-size:0.8em;">
                    {mentions} mentions · {conf:.0%} confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Most mentioned bar chart
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, use_container_width=True)
