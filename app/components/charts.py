"""Reusable Plotly chart components for MarketPulse dashboard."""

import plotly.graph_objects as go
import streamlit as st
from .styles import SENTIMENT_COLORS, COLORS


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_closes(symbol: str, period: str = "1mo"):
    """Daily closing prices for a symbol (cached 15 min). Returns (dates, closes)
    or None. Kept out of the hot path so a missing/invalid ticker never errors."""
    try:
        from .yfsession import yf_download
        df = yf_download(symbol, period=period, progress=False)
        if df is None or df.empty:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        if closes.empty:
            return None
        return [d.strftime("%Y-%m-%d") for d in closes.index], [float(c) for c in closes]
    except Exception:
        return None


def price_line(symbol: str, period: str = "1mo") -> go.Figure | None:
    """Compact price line for the ticker detail card. None if no data."""
    data = _fetch_closes(symbol, period)
    if not data:
        return None
    dates, closes = data
    up = closes[-1] >= closes[0]
    color = COLORS["bullish"] if up else COLORS["bearish"]
    fig = go.Figure(go.Scatter(
        x=dates, y=closes, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({'0,200,83' if up else '255,23,68'},0.08)",
        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
    ))
    pct = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=180, margin=dict(l=0, r=0, t=28, b=20), showlegend=False,
        title=dict(text=f"Price {period} &nbsp; <span style='color:{color}'>{pct:+.1f}%</span>",
                   font=dict(size=13, color=COLORS["text_secondary"]), x=0.0),
        yaxis=dict(gridcolor="#30363D", tickprefix="$", side="right"),
        xaxis=dict(gridcolor="#30363D", showticklabels=False),
    )
    return fig


def sentiment_pie(sentiment_dict, title="Sentiment Distribution"):
    """Pie chart of sentiment distribution."""
    labels = list(sentiment_dict.keys())
    values = list(sentiment_dict.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS['secondary']) for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=[l.upper() for l in labels],
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+percent',
        textfont_size=12,
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        showlegend=True,
        height=350,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def ticker_mentions_bar(ticker_results, top_n=15):
    """Horizontal bar chart of top mentioned tickers, colored by dominant sentiment."""
    items = list(ticker_results.values())[:top_n]
    items.reverse()

    names = [f"{t['symbol']} ({t['company']})" for t in items]
    counts = [t['mention_count'] for t in items]
    colors = [SENTIMENT_COLORS.get(t['dominant_sentiment'], COLORS['secondary']) for t in items]

    fig = go.Figure(data=[go.Bar(
        x=counts,
        y=names,
        orientation='h',
        marker_color=colors,
        text=counts,
        textposition='auto',
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Most Mentioned Tickers",
        height=max(300, top_n * 30),
        margin=dict(t=40, b=20, l=150, r=20),
        xaxis_title="Mentions",
    )
    return fig


def sentiment_trend(by_day, sentiment_colors=None):
    """7-day sentiment trend as a line chart with colored markers."""
    if not sentiment_colors:
        sentiment_colors = SENTIMENT_COLORS
    days = sorted(by_day.keys())
    sentiments = [by_day[d] for d in days]
    marker_colors = [sentiment_colors.get(s, COLORS['secondary']) for s in sentiments]
    sentiment_map = {'bullish': 3, 'neutral': 2, 'meme': 1, 'bearish': 0}
    y_values = [sentiment_map.get(s, 2) for s in sentiments]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=y_values,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(color=marker_colors, size=12, line=dict(width=2, color='#0D1117')),
        text=[s.upper() for s in sentiments],
        hovertemplate='%{x}<br>%{text}<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=0, r=0, t=10, b=30),
        showlegend=False,
        yaxis=dict(
            ticktext=['BEARISH', 'MEME', 'NEUTRAL', 'BULLISH'],
            tickvals=[0, 1, 2, 3],
            gridcolor='#30363D',
        ),
        xaxis=dict(gridcolor='#30363D'),
    )
    return fig
