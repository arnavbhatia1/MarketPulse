"""Reusable Plotly chart components for MarketPulse dashboard."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .styles import SENTIMENT_COLORS, COLORS


# ── Portfolio chart components ───────────────────────────────────────────────

def portfolio_performance_line(snapshots: list, title="Portfolio Performance"):
    """Cumulative return line chart with SPY benchmark overlay."""
    if not snapshots:
        return go.Figure()

    dates = [s["snapshot_date"] for s in reversed(snapshots)]
    cum_returns = [s.get("cumulative_return", 0) or 0 for s in reversed(snapshots)]
    bench_returns = [s.get("benchmark_return", 0) or 0 for s in reversed(snapshots)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cum_returns, mode="lines",
        name="Portfolio", line=dict(color=COLORS["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=bench_returns, mode="lines",
        name="SPY Benchmark", line=dict(color=COLORS["secondary"], width=1, dash="dash"),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", title=title, height=350,
        margin=dict(t=40, b=30, l=60, r=20),
        yaxis=dict(tickformat=".1%", gridcolor="#30363D"),
        xaxis=dict(gridcolor="#30363D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def allocation_donut(allocation: dict, title="Allocation"):
    """Donut chart for sector or geographic allocation."""
    if not allocation:
        return go.Figure()

    labels = list(allocation.keys())
    values = [v * 100 for v in allocation.values()]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.5,
        textinfo="label+percent", textfont_size=11,
        marker_colors=[
            COLORS["bullish"], COLORS["primary"], COLORS["bearish"],
            COLORS["meme"], COLORS["neutral"], COLORS["secondary"],
            "#7B68EE", "#FF8C00", "#20B2AA", "#DC143C", "#9370DB",
        ][:len(labels)],
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", title=title, height=300,
        margin=dict(t=40, b=20, l=20, r=20), showlegend=True,
    )
    return fig


def stress_gauge(score: float, warning_threshold: float, action_threshold: float):
    """Gauge chart for recession stress score."""
    color = COLORS["bullish"] if score < warning_threshold else (
        COLORS["meme"] if score < action_threshold else COLORS["bearish"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": "Recession Stress Score"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 50], "tickcolor": COLORS["secondary"]},
            "bar": {"color": color},
            "bgcolor": COLORS["bg_secondary"],
            "bordercolor": COLORS["border"],
            "steps": [
                {"range": [0, warning_threshold * 100], "color": "rgba(0,200,83,0.1)"},
                {"range": [warning_threshold * 100, action_threshold * 100], "color": "rgba(255,214,0,0.1)"},
                {"range": [action_threshold * 100, 50], "color": "rgba(255,23,68,0.1)"},
            ],
        },
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=250,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig


def sentiment_pie(sentiment_dict, title="Sentiment Distribution"):
    """
    Pie chart of sentiment distribution.

    Args:
        sentiment_dict: dict mapping sentiment label -> count
        title: chart title string

    Returns:
        Plotly Figure object
    """
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


def sentiment_bar(sentiment_dict, title="Sentiment Breakdown"):
    """
    Horizontal bar chart of sentiment counts.

    Args:
        sentiment_dict: dict mapping sentiment label -> count
        title: chart title string

    Returns:
        Plotly Figure object
    """
    labels = list(sentiment_dict.keys())
    values = list(sentiment_dict.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS['secondary']) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=[l.upper() for l in labels],
        orientation='h',
        marker_color=colors,
        text=values,
        textposition='auto',
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        height=250,
        margin=dict(t=40, b=20, l=80, r=20),
        xaxis_title="Count",
    )
    return fig


def ticker_mentions_bar(ticker_results, top_n=15):
    """
    Horizontal bar chart of top mentioned tickers, colored by dominant sentiment.

    Args:
        ticker_results: dict mapping symbol -> {symbol, company, mention_count,
                        dominant_sentiment, ...}
        top_n: how many tickers to display

    Returns:
        Plotly Figure object
    """
    items = list(ticker_results.values())[:top_n]
    items.reverse()  # highest count at top

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


def probability_bar(probabilities):
    """
    Horizontal bar chart of class probabilities for a single prediction.

    Args:
        probabilities: dict mapping sentiment label -> float probability

    Returns:
        Plotly Figure object
    """
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS['secondary']) for l in labels]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=[l.upper() for l in labels],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition='auto',
    )])
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Class Probabilities",
        height=200,
        margin=dict(t=40, b=20, l=80, r=20),
        xaxis=dict(range=[0, 1], tickformat='.0%'),
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
