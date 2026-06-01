"""
MarketPulse -- Financial Sentiment Hub

Run: streamlit run app/MarketPulse.py
"""

import html as html_mod
import re
import streamlit as st
import sys, os
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv(os.path.join(_root, '.env'))

from app.components.styles import apply_theme, COLORS, SENTIMENT_COLORS
from app.components.charts import ticker_mentions_bar, sentiment_trend
from app.components.trading_charts import candlestick_chart

_SENTIMENT_SCORE = {'bullish': 3, 'neutral': 2, 'meme': 1, 'bearish': 0}


def _daily_scores(data: dict):
    """Return [(day, numeric_score, label)] sorted by day from sentiment_by_day."""
    by_day = data.get('sentiment_by_day', {}) or {}
    out = []
    for day in sorted(by_day.keys()):
        label = by_day[day]
        out.append((day, _SENTIMENT_SCORE.get(label, 2), label))
    return out


def _sentiment_momentum(data: dict):
    """Compare the two most recent days. Returns (arrow, color, tooltip)."""
    scores = _daily_scores(data)
    if len(scores) < 2:
        return '', COLORS['secondary'], 'new'
    prev, latest = scores[-2][1], scores[-1][1]
    if latest > prev:
        return '▲', COLORS['bullish'], f"improving ({scores[-2][2]}→{scores[-1][2]})"
    if latest < prev:
        return '▼', COLORS['bearish'], f"cooling ({scores[-2][2]}→{scores[-1][2]})"
    return '▬', COLORS['secondary'], 'steady'


def _sparkline_html(data: dict) -> str:
    """Tiny bar sparkline of the last 7 days' dominant sentiment.

    Flex container with a fixed 20px track and flex-end alignment so the bars
    share a clean baseline and even spacing.
    """
    scores = _daily_scores(data)[-7:]
    if not scores:
        return ''
    height_for = {0: 6, 1: 10, 2: 14, 3: 18}  # bearish→bullish
    bars = ''
    for _, score, label in scores:
        h = height_for.get(score, 10)
        color = SENTIMENT_COLORS.get(label, COLORS['secondary'])
        bars += (f'<span style="width:6px;height:{h}px;background:{color};'
                 f'border-radius:1px;flex:0 0 auto"></span>')
    return (f'<div style="display:flex;align-items:flex-end;gap:3px;'
            f'height:20px;margin-top:8px">{bars}</div>')

_TAG_RE = re.compile(r'<[^>]+>')

def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities from post text."""
    return html_mod.unescape(_TAG_RE.sub('', text))


def _sentiment_bar_html(data: dict) -> str:
    """Stacked bar + legend showing the bullish/neutral/meme/bearish split."""
    from app.components.styles import SENTIMENT_COLORS
    counts = data.get('sentiment', {}) or {}
    total = sum(counts.values()) or 1
    order = [('bullish', 'Bullish'), ('neutral', 'Neutral'),
             ('meme', 'Meme'), ('bearish', 'Bearish')]
    bar, legend = '', ''
    for key, label in order:
        c = counts.get(key, 0)
        if c <= 0:
            continue
        color = SENTIMENT_COLORS.get(key, '#78909C')
        bar += f'<span style="width:{c / total * 100:.1f}%;background:{color}"></span>'
        legend += f'<span><b style="color:{color}">{c}</b> {label}</span>'
    if not bar:
        return ''
    return f'<div class="tk-bar">{bar}</div><div class="tk-bar-legend">{legend}</div>'


def _render_ticker_body(company: str, symbol: str, data: dict) -> None:
    """Shared ticker detail body: header, sentiment split, AI verdict, trend,
    clickable headlines. Used by both the inline search card and the dialog."""
    from src.agent.briefing import generate_briefing

    dominant = data.get('dominant_sentiment', 'neutral')
    color = SENTIMENT_COLORS.get(dominant, COLORS['secondary'])
    mention_count = data.get('mention_count', 0)
    last_updated = data.get('last_updated', 'unknown')
    updated = str(last_updated)[:16] if last_updated and last_updated != 'unknown' else 'unknown'

    safe_symbol = html_mod.escape(str(symbol))
    safe_company = html_mod.escape(str(company))
    safe_dom = html_mod.escape(str(dominant))

    # Header with sentiment-colored top accent
    st.markdown(f'''
    <div class="tk-header" style="border-top:3px solid {color}">
        <div>
            <div class="tk-symbol">{safe_symbol}</div>
            <div class="tk-company">{safe_company}</div>
            <div class="tk-meta">{mention_count} mentions &middot; updated {html_mod.escape(updated)}</div>
        </div>
        <span class="tk-badge sentiment-badge-{safe_dom}">{safe_dom.upper()}</span>
    </div>
    ''', unsafe_allow_html=True)

    bar = _sentiment_bar_html(data)
    if bar:
        st.markdown(bar, unsafe_allow_html=True)

    # Price candlestick — same auto-scaled chart as the Trading Bot (the price
    # line looked flat because it was anchored to $0).
    if symbol:
        cfig = candlestick_chart(str(symbol), period="3mo", interval="1d")
        if cfig is not None:
            st.markdown("#### Price (3M)")
            st.plotly_chart(cfig, width="stretch")

    # AI Verdict
    st.markdown("#### AI Verdict")
    with st.spinner("Generating verdict..."):
        verdict = generate_briefing(company, symbol, data)
    st.markdown(
        f'<div class="briefing-verdict">"{html_mod.escape(str(verdict))}"'
        f'<br><small style="color:#8B949E;">&mdash; MarketPulse AI</small></div>',
        unsafe_allow_html=True,
    )

    # Sentiment trend
    by_day = data.get('sentiment_by_day', {})
    if by_day:
        st.markdown("#### Sentiment Trend (7 days)")
        st.plotly_chart(sentiment_trend(by_day), width="stretch")

    # Clickable headlines
    news_posts = (data.get('top_posts', {}) or {}).get('news', [])
    st.markdown("#### Latest Headlines")
    if not news_posts:
        st.caption("No recent headlines for this ticker.")
        return
    for post in news_posts[:6]:
        title = _strip_html(str(post.get('text', ''))).strip()
        if len(title) > 130:
            title = title[:130].rsplit(' ', 1)[0] + '…'
        url = html_mod.escape(str(post.get('url') or '#'))
        sent = str(post.get('sentiment', 'neutral'))
        dot = SENTIMENT_COLORS.get(sent, COLORS['secondary'])
        date = html_mod.escape(str(post.get('date') or ''))
        st.markdown(f'''
        <a class="tk-headline" href="{url}" target="_blank" style="border-left-color:{dot}">
            <div class="tk-headline-title">{html_mod.escape(title)}</div>
            <div class="tk-headline-meta">
                <span class="tk-dot" style="background:{dot}"></span>{html_mod.escape(sent).upper()} &middot; {date}
            </div>
        </a>
        ''', unsafe_allow_html=True)

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
from app.auto_refresh import start_auto_refresh, get_status

# Keep data fresh in the background — no manual click required.
start_auto_refresh()

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

# ── Live status + coverage (auto-refresh) ────────────────────────────────────
_refresh = get_status()
if _refresh["running"]:
    st.sidebar.markdown(
        '<span style="color:#FFD600;font-weight:600">● Refreshing…</span>',
        unsafe_allow_html=True,
    )
elif _refresh["last_run"]:
    _ago = int((datetime.now() - _refresh["last_run"]).total_seconds())
    _ago_str = f"{_ago}s ago" if _ago < 60 else f"{_ago // 60}m ago"
    st.sidebar.markdown(
        f'<span style="color:#00C853;font-weight:600">● Live</span>'
        f'<span style="color:#8B949E"> · updated {_ago_str}</span>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<span style="color:#8B949E">● Starting auto-refresh…</span>',
        unsafe_allow_html=True,
    )

if _refresh["last_error"]:
    st.sidebar.caption(f"⚠ last refresh error: {_refresh['last_error'][:60]}")

if _refresh["total_posts"]:
    _cov = _refresh["label_coverage"] * 100
    st.sidebar.caption(
        f"{_refresh['ticker_count']} tickers · {_refresh['labeled_posts']}/"
        f"{_refresh['total_posts']} posts classified ({_cov:.0f}%)"
    )

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
        _render_ticker_body(resolved, symbol, ticker_data)

    st.markdown("---")


# ── Ticker detail dialog ─────────────────────────────────────────────────────

@st.dialog("Ticker Detail", width="large")
def _show_ticker_detail(company: str, data: dict):
    """Modal popup showing full sentiment breakdown for a ticker."""
    symbol = data.get('symbol', company.upper())
    _render_ticker_body(company, symbol, data)


# ── Market Overview grid (SECONDARY) ─────────────────────────────────────────
if not ticker_results:
    st.info(
        "No market data yet. Auto-refresh is running — give it a few seconds, "
        "or click **Refresh Data** in the sidebar."
    )
else:
    st.markdown(f"### Market Overview ({len(ticker_results)} tickers)")

    # Market mood bar — overall bullish/bearish split across every tracked ticker
    mood = {'bullish': 0, 'neutral': 0, 'meme': 0, 'bearish': 0}
    for _d in ticker_results.values():
        mood[_d.get('dominant_sentiment', 'neutral')] = mood.get(_d.get('dominant_sentiment', 'neutral'), 0) + 1
    _mood_total = sum(mood.values()) or 1
    _mood_bar = ''.join(
        f'<span style="display:block;height:100%;width:{c / _mood_total * 100:.1f}%;background:{SENTIMENT_COLORS.get(k)}"></span>'
        for k, c in mood.items() if c > 0
    )
    _mood_legend = ' &nbsp; '.join(
        f'<b style="color:{SENTIMENT_COLORS.get(k)}">{c}</b> {k}'
        for k, c in mood.items() if c > 0
    )
    st.markdown(
        f'<div style="color:#8B949E;font-size:0.85rem;margin-bottom:4px">Market mood — {_mood_legend}</div>'
        f'<div style="display:flex;height:10px;border-radius:6px;overflow:hidden;background:#21262D;margin-bottom:1rem">{_mood_bar}</div>',
        unsafe_allow_html=True,
    )

    # Controls: filter + sentiment + sort
    c_filter, c_sent, c_sort = st.columns([2, 1, 1])
    with c_filter:
        _q = st.text_input("Filter", placeholder="Filter by symbol or company…",
                           label_visibility="collapsed", key="grid_filter").strip().lower()
    with c_sent:
        _sent_filter = st.selectbox(
            "Sentiment", ["All", "Bullish", "Bearish", "Neutral", "Meme"],
            label_visibility="collapsed", key="grid_sentiment",
        )
    with c_sort:
        _sort = st.selectbox(
            "Sort", ["Most mentions", "Heating up", "Most bullish", "Most bearish", "A–Z"],
            label_visibility="collapsed", key="grid_sort",
        )

    # Apply filters
    items = list(ticker_results.items())
    if _q:
        items = [(c, d) for c, d in items
                 if _q in str(c).lower() or _q in str(d.get('symbol', '')).lower()]
    if _sent_filter != "All":
        items = [(c, d) for c, d in items
                 if d.get('dominant_sentiment') == _sent_filter.lower()]

    # Apply sort
    if _sort == "Most mentions":
        items.sort(key=lambda kv: kv[1].get('mention_count', 0), reverse=True)
    elif _sort == "Heating up":
        def _mom(d):
            s = _daily_scores(d)
            return (s[-1][1] - s[-2][1]) if len(s) >= 2 else 0
        items.sort(key=lambda kv: (_mom(kv[1]), kv[1].get('mention_count', 0)), reverse=True)
    elif _sort == "Most bullish":
        items.sort(key=lambda kv: kv[1].get('bullish_ratio', 0), reverse=True)
    elif _sort == "Most bearish":
        items.sort(key=lambda kv: kv[1].get('bearish_ratio', 0), reverse=True)
    elif _sort == "A–Z":
        items.sort(key=lambda kv: str(kv[1].get('symbol', kv[0])))

    items = items[:150]
    if not items:
        st.caption("No tickers match your filter.")
    else:
        st.caption(f"Showing {len(items)} tickers")

    # Ticker card grid
    cols = st.columns(3)
    for i, (company, data) in enumerate(items):
        sentiment = data.get('dominant_sentiment', 'neutral')
        symbol = data.get('symbol', company.upper())
        mentions = data.get('mention_count', 0)
        conf = data.get('avg_confidence', 0.0)
        arrow, arrow_color, arrow_tip = _sentiment_momentum(data)
        spark = _sparkline_html(data)

        with cols[i % 3]:
            safe_sym = html_mod.escape(str(symbol))
            safe_co = html_mod.escape(str(company))
            safe_sent = html_mod.escape(str(sentiment))
            card = st.container(border=True)
            with card:
                st.markdown(f"""
                <div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:1.2em; font-weight:bold;">{safe_sym}</span>
                        <span style="color:{arrow_color};font-weight:700" title="{html_mod.escape(arrow_tip)}">{arrow}</span>
                    </div>
                    <div style="color:#8B949E; font-size:0.85em;">{safe_co}</div>
                    <div style="margin:6px 0;">
                        <span class="sentiment-badge sentiment-badge-{safe_sent}">{safe_sent.upper()}</span>
                    </div>
                    <div style="color:#8B949E; font-size:0.8em;">
                        {mentions} mentions -- {conf:.0%} confidence
                    </div>
                    {spark}
                </div>
                """, unsafe_allow_html=True)
                if st.button("View Details", key=f"view_{i}_{symbol}", width="stretch"):
                    _show_ticker_detail(company, data)

    # Most mentioned bar chart
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, width="stretch")
