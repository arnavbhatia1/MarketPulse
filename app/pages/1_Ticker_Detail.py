"""
MarketPulse — Ticker Detail

Deep-dive sentiment view for a single ticker. Shows sentiment distribution,
key metrics, and the individual evidence posts that drove the label.
"""

import streamlit as st
import sys
import os
import pandas as pd

# Ensure project root is importable regardless of launch directory.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.pipeline_runner import run_pipeline
from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import sentiment_pie

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ticker Detail | MarketPulse",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("MarketPulse")
st.sidebar.markdown("**Ticker Detail**")
st.sidebar.markdown("---")

if st.sidebar.button("Back to Overview", use_container_width=True):
    st.switch_page("MarketPulse.py")

# ---------------------------------------------------------------------------
# Load pipeline data
# ---------------------------------------------------------------------------
with st.spinner("Loading pipeline data..."):
    try:
        data = run_pipeline()
    except Exception as e:
        st.error(
            "Pipeline failed to run. Make sure dependencies are installed "
            "and at least synthetic data is available."
        )
        st.caption(f"Technical detail: {e}")
        st.stop()

df = data["df"]
ticker_results = data["ticker_results"]

# ---------------------------------------------------------------------------
# Ticker selection
# ---------------------------------------------------------------------------
ticker_options = list(ticker_results.keys()) if ticker_results else []

# Resolve the selected ticker: prefer session_state if it is valid,
# otherwise fall back to the selectbox default.
session_ticker = st.session_state.get("selected_ticker", None)
default_index = 0
if session_ticker and session_ticker in ticker_options:
    default_index = ticker_options.index(session_ticker)

if not ticker_options:
    st.title("Ticker Detail")
    st.warning(
        "No ticker data available yet. "
        "Return to **Market Overview** and make sure the pipeline has run."
    )
    if st.button("Go to Market Overview"):
        st.switch_page("MarketPulse.py")
    st.stop()

selected_company = st.selectbox(
    "Select ticker",
    options=ticker_options,
    index=default_index,
    help="Choose a company to explore its sentiment breakdown.",
)

# Keep session state in sync with selectbox.
st.session_state["selected_ticker"] = selected_company

ticker_data = ticker_results[selected_company]

# ---------------------------------------------------------------------------
# Header: company name, symbol, dominant sentiment
# ---------------------------------------------------------------------------
symbol = ticker_data.get("symbol", selected_company.upper())
sentiment = ticker_data.get("dominant_sentiment", "neutral")
color = SENTIMENT_COLORS.get(sentiment, COLORS["secondary"])

st.markdown(
    f"""
    <div style="margin-bottom: 4px;">
        <span style="font-size:2em; font-weight:bold;">{symbol}</span>
        &nbsp;
        <span style="color:#8B949E; font-size:1.3em;">{selected_company}</span>
    </div>
    <div class="sentiment-{sentiment}" style="font-size:1.5em; margin-bottom:16px;">
        {sentiment.upper()}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
mention_count = ticker_data.get("mention_count", 0)
avg_conf = ticker_data.get("avg_confidence", 0.0)

# Compute bullish/bearish ratios from the sentiment distribution if available.
sentiment_dist = ticker_data.get("sentiment", {})
total_labeled = sum(sentiment_dist.values()) if sentiment_dist else 0

bullish_ratio = sentiment_dist.get("bullish", 0) / total_labeled if total_labeled else 0.0
bearish_ratio = sentiment_dist.get("bearish", 0) / total_labeled if total_labeled else 0.0

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("Mentions", f"{mention_count:,}")
col_k2.metric("Avg Confidence", f"{avg_conf:.1%}")
col_k3.metric("Bullish Ratio", f"{bullish_ratio:.1%}")
col_k4.metric("Bearish Ratio", f"{bearish_ratio:.1%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Sentiment distribution pie chart
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("#### Sentiment Distribution")
    if sentiment_dist:
        fig = sentiment_pie(
            sentiment_dist,
            title=f"{symbol} Sentiment Breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment distribution data available for this ticker.")

# ---------------------------------------------------------------------------
# Evidence posts table
# ---------------------------------------------------------------------------
with right_col:
    st.markdown("#### Evidence Posts")
    st.caption("Posts mentioning this ticker that drove the sentiment label.")

# Build evidence posts from the main DataFrame.
# Filter to rows that contain the selected ticker in their extracted tickers list.
evidence_posts = pd.DataFrame()

if "tickers" in df.columns:
    # 'tickers' column is a list of canonical company names per post.
    mask = df["tickers"].apply(
        lambda tickers: selected_company in tickers
        if isinstance(tickers, (list, set))
        else False
    )
    evidence_posts = df[mask].copy()

# Select and rename display columns, handling missing columns gracefully.
display_cols_map = {
    "text": "text",
    "programmatic_label": "programmatic_label",
    "label_confidence": "label_confidence",
    "source": "source",
    "timestamp": "timestamp",
}

available_map = {
    internal: internal
    for internal in display_cols_map
    if internal in evidence_posts.columns
}

if evidence_posts.empty:
    st.info(
        "No individual posts found for this ticker in the current dataset. "
        "This can happen if the pipeline is using aggregated data."
    )
else:
    display_df = evidence_posts[list(available_map.keys())].copy()

    # Truncate text to 200 characters for readability.
    if "text" in display_df.columns:
        display_df["text"] = display_df["text"].str[:200]

    # Rename for cleaner column headers.
    rename_map = {
        "programmatic_label": "sentiment",
        "label_confidence": "confidence",
    }
    display_df = display_df.rename(columns=rename_map)

    # Format confidence as a percentage string if present.
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].apply(
            lambda v: f"{v:.0%}" if pd.notna(v) else "—"
        )

    # Format timestamp if present.
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = pd.to_datetime(
            display_df["timestamp"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")

    # Show the table — sort by confidence descending so clearest posts appear first.
    sort_col = "confidence" if "confidence" in display_df.columns else None

    st.dataframe(
        display_df,
        use_container_width=True,
        height=350,
    )

    # CSV download button.
    csv_data = evidence_posts.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Evidence Posts (CSV)",
        data=csv_data,
        file_name=f"{symbol}_evidence_posts.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Expander: raw ticker result dict (useful for debugging/exploration)
# ---------------------------------------------------------------------------
with st.expander("Raw ticker analysis data"):
    st.json(
        {
            k: v
            for k, v in ticker_data.items()
            if not isinstance(v, pd.DataFrame)  # skip any DataFrames
        }
    )

# ---------------------------------------------------------------------------
# Footer navigation
# ---------------------------------------------------------------------------
st.markdown("---")
back_col, _, next_col = st.columns([1, 6, 1])

with back_col:
    if st.button("Back to Overview", key="back_footer"):
        st.switch_page("MarketPulse.py")

with next_col:
    if ticker_options:
        current_idx = ticker_options.index(selected_company)
        next_idx = (current_idx + 1) % len(ticker_options)
        next_company = ticker_options[next_idx]
        next_symbol = ticker_results[next_company].get("symbol", next_company.upper())
        if st.button(f"Next: {next_symbol}", key="next_ticker"):
            st.session_state["selected_ticker"] = next_company
            st.rerun()
