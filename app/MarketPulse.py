"""
MarketPulse -- Financial Sentiment Hub

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

from src.storage.db import init_db
init_db()

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

if st.sidebar.button("Refresh Data", width="stretch"):
    with st.status("Refreshing market data...", expanded=True) as status:
        source_summary = refresh_pipeline(
            start_date_str=start_date.isoformat(),
            end_date_str=end_date.isoformat(),
            progress_callback=st.write,
        )
        posts = source_summary.get('total_posts', 0)
        sources = source_summary.get('sources_used', [])
        status.update(label=f"Done -- {posts} posts from {', '.join(sources)}", state="complete")
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


# ── Market Intelligence Teaser ───────────────────────────────────────────────

def _merge_movers(anomalies, volume, gaps):
    """Merge anomaly, volume-leader, and gap-mover scans into a unified list.

    Each result dict uses a different key for its list of items:
      anomalies -> "anomalies" (items have "symbol", "total_score", "anomalies")
      volume    -> "leaders"   (items have "symbol", "ratio")
      gaps      -> "movers"    (items have "symbol", "gap_percent")

    Returns a list sorted by cumulative score descending, each entry:
      {"symbol": str, "score": float, "badges": [str, ...]}
    """
    merged: dict[str, dict] = {}

    for item in (anomalies or {}).get("anomalies", []):
        sym = item.get("symbol", "")
        if not sym:
            continue
        entry = merged.setdefault(sym, {"symbol": sym, "score": 0.0, "badges": []})
        entry["score"] += float(item.get("total_score", 0))
        for a in item.get("anomalies", []):
            entry["badges"].append(a)

    for item in (volume or {}).get("leaders", []):
        sym = item.get("symbol", "")
        if not sym:
            continue
        entry = merged.setdefault(sym, {"symbol": sym, "score": 0.0, "badges": []})
        ratio = float(item.get("ratio", 0))
        entry["score"] += ratio
        entry["badges"].append(f"Vol {ratio:.1f}x")

    for item in (gaps or {}).get("movers", []):
        sym = item.get("symbol", "")
        if not sym:
            continue
        entry = merged.setdefault(sym, {"symbol": sym, "score": 0.0, "badges": []})
        gap = float(item.get("gap_percent", 0))
        entry["score"] += abs(gap)
        direction = "up" if gap >= 0 else "down"
        entry["badges"].append(f"Gap {direction} {abs(gap):.1f}%")

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)


try:
    from src.investor.mcp_client import (
        is_connected,
        detect_market_regime,
        get_vix_analysis,
        scan_anomalies,
        scan_volume_leaders,
        scan_gap_movers,
    )
    _HAS_MCP = True
except Exception:
    _HAS_MCP = False

if _HAS_MCP:
    try:
        if is_connected():
            # -- Cached data fetchers ------------------------------------------
            @st.cache_data(ttl=300)
            def _fetch_regime():
                regime = detect_market_regime()
                vix = get_vix_analysis()
                return regime, vix

            @st.cache_data(ttl=120)
            def _fetch_movers():
                anomalies = scan_anomalies()
                volume = scan_volume_leaders()
                gaps = scan_gap_movers()
                return _merge_movers(anomalies, volume, gaps)

            regime_data, vix_data = _fetch_regime()
            movers = _fetch_movers()

            # -- Market Regime Banner ------------------------------------------
            regime_name = regime_data.get("regime", "unknown").lower()
            regime_css = {
                "bull": "regime-bull",
                "bear": "regime-bear",
                "sideways": "regime-sideways",
                "volatile": "regime-volatile",
                "crash": "regime-crash",
            }.get(regime_name, "regime-sideways")
            recommendation = html_mod.escape(
                str(regime_data.get("recommendation", ""))
            )

            vix_level = vix_data.get("level", "normal").lower()
            vix_value = vix_data.get("value", "")
            vix_css = {
                "low": "vix-low",
                "normal": "vix-normal",
                "high": "vix-high",
            }.get(vix_level, "vix-normal")

            st.markdown(f"""
            <div class="regime-banner">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <span class="regime-label {regime_css}">{html_mod.escape(regime_name.upper())}</span>
                    <span class="vix-badge {vix_css}">VIX {html_mod.escape(str(vix_value))}</span>
                </div>
                <div style="color:#8B949E; font-size:0.9rem; margin-top:0.5rem;">
                    {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # -- Top Movers Grid -----------------------------------------------
            if movers:
                top6 = movers[:6]
                cols = st.columns(3)
                for idx, mover in enumerate(top6):
                    sym = html_mod.escape(str(mover["symbol"]))
                    score = mover["score"]
                    badges_html = ""
                    for badge_text in mover["badges"][:3]:
                        safe_badge = html_mod.escape(str(badge_text))
                        # Pick badge CSS class based on content keywords
                        if "gap" in badge_text.lower() and "down" in badge_text.lower():
                            bcls = "badge-gap-down"
                        elif "gap" in badge_text.lower():
                            bcls = "badge-gap-up"
                        elif "vol" in badge_text.lower():
                            bcls = "badge-volume-spike"
                        elif "52w" in badge_text.lower() or "high" in badge_text.lower():
                            bcls = "badge-52w-high"
                        else:
                            bcls = "badge-volume-spike"
                        badges_html += f'<span class="anomaly-badge {bcls}">{safe_badge}</span>'

                    change_cls = "mover-change-pos" if score >= 0 else "mover-change-neg"

                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class="mover-card">
                            <div class="mover-symbol">{sym}</div>
                            <div class="{change_cls}" style="margin:0.3rem 0;">
                                Score: {score:.1f}
                            </div>
                            <div>{badges_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # -- Powered by badge ----------------------------------------------
            st.markdown("""
            <div class="powered-badge">
                Powered by <a href="/Investor_Bot">financial-mcp</a>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(
                '<div class="mcp-unavailable">Market intelligence unavailable -- MCP server not connected</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<div class="mcp-unavailable">Market intelligence unavailable</div>',
            unsafe_allow_html=True,
        )

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
    search_clicked = st.button("Research", width="stretch")

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
        st.warning(f"No data for **{query.strip()}**. Try a ticker symbol like TSLA, NVDA, or AAPL -- then hit **Refresh Data** if needed.")
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
                {mention_count} mentions -- updated {safe_updated}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Verdict
        st.markdown("#### AI Verdict")
        with st.spinner("Generating verdict..."):
            verdict = generate_briefing(resolved, symbol, ticker_data)
        st.markdown(f'<div class="briefing-verdict">"{html_mod.escape(str(verdict))}"<br><small style="color:#8B949E;">-- MarketPulse AI</small></div>', unsafe_allow_html=True)

        # Sentiment trend chart
        by_day = ticker_data.get('sentiment_by_day', {})
        if by_day:
            st.markdown("#### Sentiment Trend (7 days)")
            fig = sentiment_trend(by_day)
            st.plotly_chart(fig, width="stretch")

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

    # Ticker card grid -- clickable buttons with styled cards
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
                        {mentions} mentions -- {conf:.0%} confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Most mentioned bar chart
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, width="stretch")
