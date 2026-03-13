"""Plotly chart components for the trading terminal."""

import plotly.graph_objects as go
import yfinance as yf

BG = "#0D1117"
GRID = "#21262D"
TEXT = "#8B949E"
GREEN = "#00C853"
RED = "#FF1744"


def candlestick_chart(symbol: str, period: str = "6mo") -> go.Figure | None:
    """Fetch OHLC data from yfinance and return a candlestick chart."""
    try:
        df = yf.download(symbol, period=period, progress=False)
        if df.empty:
            return None
    except Exception:
        return None

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

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
