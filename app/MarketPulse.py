"""
MarketPulse Dashboard -- Main Entry Point

Run with:
    streamlit run app/MarketPulse.py
"""

import streamlit as st
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import ticker_mentions_bar
from app.components.metrics import source_status_indicator

st.set_page_config(
    page_title="MarketPulse",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("MarketPulse")
st.sidebar.markdown("**Sentiment Intelligence for Financial Markets**")
st.sidebar.markdown("---")

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Load pipeline data
# ---------------------------------------------------------------------------
try:
    from app.pipeline_runner import run_pipeline

    with st.spinner("Analyzing market sentiment..."):
        data = run_pipeline()

    df = data['df']
    ticker_results = data['ticker_results']
    market_summary = data['market_summary']
    source_summary = data['source_summary']

except Exception as e:
    st.title("MarketPulse")
    st.warning(
        "Run the pipeline first to load data:\n\n"
        "```\npython3 scripts/run_pipeline.py\n```\n\n"
        "Then refresh this page."
    )
    st.caption(f"({e})")
    st.stop()

# Sidebar source status
source_status_indicator(
    sources_used=source_summary.get("sources_used", []),
    sources_unavailable=source_summary.get("sources_unavailable", []),
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("MarketPulse")
st.markdown(
    "Real-time sentiment intelligence for financial markets. "
    "See which tickers are **bullish**, **bearish**, or drowning in **memes** "
    "— backed by ML analysis of social media posts."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top KPI metrics
# ---------------------------------------------------------------------------
total_posts = len(df)
tickers_tracked = len(ticker_results)

labeled_count = df['programmatic_label'].notna().sum() if 'programmatic_label' in df.columns else 0
coverage_pct = labeled_count / total_posts if total_posts > 0 else 0.0

sources_used = source_summary.get("sources_used", [])
data_source_display = ", ".join(sources_used).upper() if sources_used else "NONE"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts Analyzed", f"{total_posts:,}")
c2.metric("Tickers Tracked", f"{tickers_tracked:,}")
c3.metric("Labeling Coverage", f"{coverage_pct:.1%}")
c4.metric("Data Source", data_source_display)

st.markdown("---")

# ---------------------------------------------------------------------------
# Market snapshot — sentiment counts
# ---------------------------------------------------------------------------
st.markdown("### Market Snapshot")

dist = market_summary.get('ticker_sentiment_distribution', {})
scols = st.columns(4)
for i, sentiment in enumerate(['bullish', 'bearish', 'neutral', 'meme']):
    count = dist.get(sentiment, 0)
    color = SENTIMENT_COLORS.get(sentiment, '#78909C')
    scols[i].markdown(
        f"<div style='text-align:center;'>"
        f"<div style='font-size:2em; font-weight:bold; color:{color};'>{count}</div>"
        f"<div style='color:#8B949E;'>{sentiment.upper()} tickers</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Ticker card grid
# ---------------------------------------------------------------------------
if not ticker_results:
    st.info(
        "No ticker data yet. The pipeline may still be warming up, "
        "or no ticker mentions were found in the current dataset."
    )
else:
    st.markdown("### Ticker Sentiment")
    st.caption(
        "Each card shows the dominant sentiment direction for that ticker based "
        "on programmatic labeling of all posts mentioning it."
    )

    cols = st.columns(3)
    for i, (company, ticker_data) in enumerate(ticker_results.items()):
        col = cols[i % 3]
        with col:
            sentiment = ticker_data.get("dominant_sentiment", "neutral")
            color = SENTIMENT_COLORS.get(sentiment, COLORS["secondary"])
            symbol = ticker_data.get("symbol", company.upper())
            mention_count = ticker_data.get("mention_count", 0)
            avg_conf = ticker_data.get("avg_confidence", 0.0)

            st.markdown(
                f"""
                <div class="ticker-card">
                    <div style="font-size:1.3em; font-weight:bold;">{symbol}</div>
                    <div style="color: #8B949E;">{company}</div>
                    <div class="sentiment-{sentiment}" style="font-size:1.1em; margin:8px 0;">
                        {sentiment.upper()}
                    </div>
                    <div style="color: #8B949E; font-size:0.9em;">
                        {mention_count} mentions &middot; {avg_conf:.0%} confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(f"View {symbol}", key=f"btn_{company}"):
                st.session_state["selected_ticker"] = company
                st.switch_page("pages/1_Ticker_Detail.py")

    # -----------------------------------------------------------------------
    # Ticker mentions bar chart
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    st.caption("Bar length shows total mentions. Color reflects dominant sentiment.")

    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Market-level summary
# ---------------------------------------------------------------------------
if market_summary:
    st.markdown("---")
    st.markdown("### Market Sentiment Summary")

    overall = market_summary.get("overall_sentiment", "")
    bullish_pct = market_summary.get("bullish_pct", 0.0)
    bearish_pct = market_summary.get("bearish_pct", 0.0)

    ms_col1, ms_col2, ms_col3 = st.columns(3)
    ms_col1.metric("Overall Market Bias", overall.upper() if overall else "MIXED")
    ms_col2.metric("Bullish Tickers", f"{bullish_pct:.0%}")
    ms_col3.metric("Bearish Tickers", f"{bearish_pct:.0%}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
used_fallback = source_summary.get("used_fallback", False)
if used_fallback:
    st.info(
        "No live API keys detected. Data shown is from the **synthetic dataset**. "
        "Add credentials to `.env` to enable live Reddit, Stocktwits, or News ingestion."
    )
else:
    st.caption(f"Data mode: {source_summary.get('mode', 'auto').upper()} | Sources: {data_source_display}")
