"""Plotly chart components for the trading terminal."""

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

BG = "#0D1117"
GRID = "#21262D"
TEXT = "#8B949E"
GREEN = "#00C853"
RED = "#FF1744"


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_ohlc(symbol: str, period: str):
    """Download OHLC data from yfinance. Cached 10 min so flipping the period
    (or any page rerun) doesn't re-hit the network for data already fetched."""
    try:
        df = yf.download(symbol, period=period, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    return df


def candlestick_chart(symbol: str, period: str = "6mo") -> go.Figure | None:
    """Return a candlestick chart for *symbol* over *period* (data is cached)."""
    df = _fetch_ohlc(symbol, period)
    if df is None or df.empty:
        return None

    fig = go.Figure(
        data=go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=350,
        xaxis=dict(gridcolor=GRID, showgrid=True),
        yaxis=dict(gridcolor=GRID, showgrid=True, side="right"),
    )
    return fig


def equity_curve_chart(points: list, starting_capital: float = 10_000.0) -> go.Figure | None:
    """Line chart of portfolio value over time from bot snapshots.

    *points* is a list of {"time": iso_str, "value": float}. Returns None if
    there's nothing to plot yet.
    """
    if not points:
        return None
    times = [p.get("time") for p in points]
    values = [p.get("value", starting_capital) for p in points]
    up = values[-1] >= starting_capital
    color = GREEN if up else RED
    fill = "rgba(0,200,83,0.08)" if up else "rgba(255,23,68,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines",
        line=dict(color=color, width=2), fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
    ))
    # Starting-capital baseline
    fig.add_hline(y=starting_capital, line_dash="dot", line_color=TEXT, opacity=0.5)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        height=260, margin=dict(l=10, r=10, t=10, b=20), showlegend=False,
        xaxis=dict(gridcolor=GRID, showticklabels=False),
        yaxis=dict(gridcolor=GRID, side="right", tickprefix="$"),
    )
    return fig


def score_gauge(score: float, label: str) -> go.Figure:
    """Circular gauge for a 0-100 score."""
    color = GREEN if score >= 65 else (RED if score < 35 else "#FFD600")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": label, "font": {"size": 14, "color": TEXT}},
            number={"suffix": "/100", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT},
                "bar": {"color": color},
                "bgcolor": "#161B22",
                "bordercolor": "#30363D",
                "steps": [
                    {"range": [0, 35], "color": "rgba(255,23,68,0.1)"},
                    {"range": [35, 65], "color": "rgba(255,214,0,0.1)"},
                    {"range": [65, 100], "color": "rgba(0,200,83,0.1)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font={"color": "#E6EDF3"}, height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def stress_gauge(stress_score: float, scenarios: dict) -> go.Figure:
    """Stress test visualization with scenario bars."""
    names = list(scenarios.keys())
    values = [abs(v) * 100 for v in scenarios.values()]
    colors = [RED if v > 30 else ("#FFD600" if v > 20 else GREEN) for v in values]
    fig = go.Figure(
        go.Bar(x=values, y=names, orientation="h", marker_color=colors,
               text=[f"{v:.1f}%" for v in values], textposition="auto")
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Drawdown %", gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    return fig


def cftc_positioning_bars(commercial_net: int, non_commercial_net: int) -> go.Figure:
    """Horizontal bar chart for CFTC commercial vs speculator positioning."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Commercial", "Speculator"],
        x=[commercial_net, non_commercial_net],
        orientation="h",
        marker_color=[GREEN if commercial_net > 0 else RED,
                      GREEN if non_commercial_net > 0 else RED],
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        height=150, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=GRID, title="Net Position"),
        yaxis=dict(gridcolor=GRID),
    )
    return fig


def sector_allocation_bars(allocations: dict) -> go.Figure:
    """Horizontal bar chart for sector allocation."""
    sectors = list(allocations.keys())
    weights = [v * 100 for v in allocations.values()]
    fig = go.Figure(
        go.Bar(y=sectors, x=weights, orientation="h", marker_color="#58A6FF",
               text=[f"{w:.1f}%" for w in weights], textposition="auto")
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        height=max(200, len(sectors) * 30),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Weight %", gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    return fig
