"""
MarketPulse — Financial Sentiment Hub

Run: streamlit run app/MarketPulse.py
"""

import html as html_mod
import streamlit as st
import sys, os, json
from datetime import date, timedelta
from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv(os.path.join(_root, '.env'))

from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import ticker_mentions_bar, sentiment_trend

st.set_page_config(
    page_title="MarketPulse",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Auth gate (inline) ──────────────────────────────────────────────────────
from src.storage.db import init_db
init_db()

if not st.session_state.get("user_id") and not st.session_state.get("guest"):
    from app.components.auth_guard import show_login_form
    user = show_login_form()
    if user:
        st.rerun()
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("MarketPulse")
st.sidebar.markdown("**Sentiment intelligence for financial markets**")

# Show user info in sidebar
if st.session_state.get("user_id"):
    user_email = st.session_state.get("user_email", "User")
    is_prem = st.session_state.get("is_premium", False)
    badge = "Premium" if is_prem else "Free"
    st.sidebar.markdown(f"**{user_email}** · {badge}")
    if st.sidebar.button("Sign Out", use_container_width=True):
        for key in ["user_id", "user_email", "is_premium", "guest"]:
            st.session_state.pop(key, None)
        st.rerun()
elif st.session_state.get("guest"):
    st.sidebar.markdown("*Browsing as guest*")
    if st.sidebar.button("Sign In", use_container_width=True):
        st.session_state.pop("guest", None)
        st.rerun()

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
    with st.status("Refreshing market data...", expanded=True) as status:
        source_summary = refresh_pipeline(
            start_date_str=start_date.isoformat(),
            end_date_str=end_date.isoformat(),
            progress_callback=st.write,
        )
        posts = source_summary.get('total_posts', 0)
        sources = source_summary.get('sources_used', [])
        status.update(label=f"Done — {posts} posts from {', '.join(sources)}", state="complete")
        st.cache_data.clear()
    st.rerun()

# Model status
model = load_model()
if model and model.is_trained:
    st.sidebar.success("AI-enhanced analysis active")
else:
    st.sidebar.info("Basic analysis mode")

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
        st.warning(f"No data for **{query.strip()}**. Try a ticker symbol like TSLA, NVDA, or AAPL — then hit **Refresh Data** if needed.")
    else:
        symbol = ticker_data.get('symbol', resolved.upper())
        dominant = ticker_data.get('dominant_sentiment', 'neutral')
        mention_count = ticker_data.get('mention_count', 0)
        last_updated = ticker_data.get('last_updated', 'unknown')

        # Briefing card with badge pill
        safe_symbol = html_mod.escape(str(symbol))
        safe_resolved = html_mod.escape(str(resolved))
        safe_dominant = html_mod.escape(str(dominant))
        safe_updated = html_mod.escape(str(last_updated[:16] if last_updated != 'unknown' else 'unknown'))
        st.markdown(f"""
        <div class="briefing-card" style="border-left: 4px solid {SENTIMENT_COLORS.get(dominant, COLORS['secondary'])};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.6em; font-weight:bold;">{safe_symbol}</span>
                    <span style="color:#8B949E; margin-left:10px;">{safe_resolved}</span>
                </div>
                <span class="sentiment-badge sentiment-badge-{safe_dominant}">{safe_dominant.upper()}</span>
            </div>
            <div style="color:#8B949E; font-size:0.85em; margin-top:4px;">
                {mention_count} mentions · updated {safe_updated}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Verdict
        st.markdown("#### AI Verdict")
        with st.spinner("Generating verdict..."):
            verdict = generate_briefing(resolved, symbol, ticker_data)
        st.markdown(f'<div class="briefing-verdict">"{html_mod.escape(str(verdict))}"<br><small style="color:#8B949E;">— MarketPulse AI</small></div>', unsafe_allow_html=True)

        # Sentiment trend chart
        by_day = ticker_data.get('sentiment_by_day', {})
        if by_day:
            st.markdown("#### Sentiment Trend (7 days)")
            fig = sentiment_trend(by_day)
            st.plotly_chart(fig, use_container_width=True)

        # By Source breakdown
        st.markdown("#### By Source")
        top_posts = ticker_data.get('top_posts', {})
        src_cols = st.columns(3)
        for i, source in enumerate(('reddit', 'stocktwits', 'news')):
            src_sentiment = ticker_data.get(f'{source}_sentiment') or 'N/A'
            src_posts = top_posts.get(source, [])
            with src_cols[i]:
                badge_class = f"sentiment-badge-{src_sentiment}" if src_sentiment != 'N/A' else "sentiment-badge-neutral"
                safe_src_sent = html_mod.escape(str(src_sentiment))
                st.markdown(f"""
                <div class="source-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <strong>{html_mod.escape(source.upper())}</strong>
                        <span class="sentiment-badge {badge_class}">{safe_src_sent.upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
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

    # Ticker card grid — clickable buttons with styled cards
    cols = st.columns(3)
    for i, (company, data) in enumerate(ticker_results.items()):
        sentiment = data.get('dominant_sentiment', 'neutral')
        symbol = data.get('symbol', company.upper())
        mentions = data.get('mention_count', 0)
        conf = data.get('avg_confidence', 0.0)

        with cols[i % 3]:
            safe_sym = html_mod.escape(str(symbol))
            safe_co = html_mod.escape(str(company))
            safe_sent = html_mod.escape(str(sentiment))
            card = st.container(border=True)
            with card:
                st.markdown(f"""
                <div style="cursor:pointer;">
                    <div style="font-size:1.2em; font-weight:bold;">{safe_sym}</div>
                    <div style="color:#8B949E; font-size:0.85em;">{safe_co}</div>
                    <div style="margin:6px 0;">
                        <span class="sentiment-badge sentiment-badge-{safe_sent}">{safe_sent.upper()}</span>
                    </div>
                    <div style="color:#8B949E; font-size:0.8em;">
                        {mentions} mentions · {conf:.0%} confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View Details", key=f"ticker_btn_{i}", use_container_width=True):
                    st.session_state["selected_ticker"] = company
                    st.switch_page("pages/1_Ticker_Detail.py")

    # Most mentioned bar chart
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, use_container_width=True)
