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

from app.pipeline_runner import get_ticker_cache
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

# Query param navigation
query_ticker = st.query_params.get("ticker", None)

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
        ticker_results = get_ticker_cache()
    except Exception as e:
        st.error(
            "Failed to load pipeline data. Make sure the pipeline has run at least once."
        )
        st.caption(f"Technical detail: {e}")
        st.stop()

# Build a lightweight posts DataFrame from SQLite for evidence display.
from src.storage.db import load_posts
try:
    df = load_posts()
except Exception:
    df = pd.DataFrame()

# ---------------------------------------------------------------------------
# Ticker selection
# ---------------------------------------------------------------------------
ticker_options = list(ticker_results.keys()) if ticker_results else []

# Resolve the selected ticker: prefer query_ticker first, then session_state,
# then fall back to default index 0.
default_index = 0
if query_ticker and query_ticker in ticker_options:
    default_index = ticker_options.index(query_ticker)
elif st.session_state.get("selected_ticker") and st.session_state["selected_ticker"] in ticker_options:
    default_index = ticker_options.index(st.session_state["selected_ticker"])

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

st.markdown(
    f"""
    <div style="margin-bottom: 4px;">
        <span style="font-size:2em; font-weight:bold;">{symbol}</span>
        &nbsp;
        <span style="color:#8B949E; font-size:1.3em;">{selected_company}</span>
    </div>
    <div style="margin-bottom:16px;">
        <span class="sentiment-badge sentiment-badge-{sentiment}">{sentiment.upper()}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Metrics row — 3 columns: Mentions, Bullish %, Bearish %
# ---------------------------------------------------------------------------
mention_count = ticker_data.get("mention_count", 0)

# Compute bullish/bearish ratios from the sentiment distribution if available.
sentiment_dist = ticker_data.get("sentiment", {})
total_labeled = sum(sentiment_dist.values()) if sentiment_dist else 0

bullish_ratio = sentiment_dist.get("bullish", 0) / total_labeled if total_labeled else 0.0
bearish_ratio = sentiment_dist.get("bearish", 0) / total_labeled if total_labeled else 0.0

col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Mentions", f"{mention_count:,}")
col_k2.metric("Bullish %", f"{bullish_ratio:.1%}")
col_k3.metric("Bearish %", f"{bearish_ratio:.1%}")

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

    # Build styled HTML table with evidence-table CSS class and badge pills.
    col_order = [c for c in ["text", "sentiment", "confidence", "source", "timestamp"] if c in display_df.columns]
    display_df = display_df[col_order]

    header_cells = "".join(f"<th>{col.capitalize()}</th>" for col in col_order)
    rows_html = ""
    for _, row in display_df.iterrows():
        cells = ""
        for col in col_order:
            val = row[col] if pd.notna(row[col]) else "—"
            if col == "sentiment" and val != "—":
                cells += f'<td><span class="sentiment-badge sentiment-badge-{val}">{str(val).upper()}</span></td>'
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    table_html = f"""
    <table class="evidence-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

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
