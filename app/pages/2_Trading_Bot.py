"""Trading Bot — Full trading terminal powered by financial-mcp-server."""

import streamlit as st
import pandas as pd
from datetime import datetime

from src.investor.bot_engine import (
    get_state as _get_bot_state,
    get_engine as _get_engine,
    load_state as _load_bot_state,
    MAX_POSITIONS,
)

# Resume a persisted/seeded ledger once per process so the Edge panel starts warm.
_load_bot_state()

st.set_page_config(page_title="Trading Bot", page_icon="\U0001F4C8", layout="wide")

from app.components.styles import apply_theme
apply_theme()


# -- MCP Connection Check -----------------------------------------------------

try:
    from src.investor import (
        is_connected, detect_market_regime, get_vix_analysis,
        analyze_ticker, get_fundamentals, get_momentum, score_ticker,
        get_smart_money_signal, get_futures_positioning,
        get_sec_filings, get_insider_trades, get_search_trends, check_risk,
    )
    mcp_available = is_connected()
except (ConnectionError, Exception):
    mcp_available = False

if not mcp_available:
    st.error("**MCP server not running.** Start it with: `financial-mcp`")
    st.info("The financial-mcp server must be running for the trading terminal to work.")
    st.stop()


# -- Header --------------------------------------------------------------------

st.markdown("## Trading Bot")
st.caption("Powered by financial-mcp")


# -- Cached MCP wrappers -------------------------------------------------------
# These tools hit yfinance/CFTC over the network (~1s each). Caching them keeps
# the page snappy: flipping the chart period or any other rerun reuses results
# instead of re-fetching every card.

@st.cache_data(ttl=300, show_spinner=False)
def _c_analyze(sym):
    return analyze_ticker(sym)

@st.cache_data(ttl=300, show_spinner=False)
def _c_score(sym):
    return score_ticker(sym)

@st.cache_data(ttl=300, show_spinner=False)
def _c_fundamentals(sym):
    return get_fundamentals(sym)

@st.cache_data(ttl=300, show_spinner=False)
def _c_momentum(sym):
    return get_momentum(sym)

@st.cache_data(ttl=600, show_spinner=False)
def _c_smart_money(market):
    return get_smart_money_signal(market)

@st.cache_data(ttl=600, show_spinner=False)
def _c_futures(market):
    return get_futures_positioning(market)

@st.cache_data(ttl=600, show_spinner=False)
def _c_sec(sym):
    return get_sec_filings(sym, filing_type="8-K", count=5)

@st.cache_data(ttl=600, show_spinner=False)
def _c_insider(sym):
    return get_insider_trades(sym, days=90)

@st.cache_data(ttl=900, show_spinner=False)
def _c_trends(keywords):
    return get_search_trends(keywords)

@st.cache_data(ttl=30, show_spinner=False)
def _c_risk(pid):
    return check_risk(pid)


def _score_card_html(label, score):
    """Score card that degrades to N/A when the MCP score is missing/None
    (e.g. when yfinance data is unavailable) instead of crashing."""
    if score is None:
        return (f'<div class="score-card"><div class="score-label">{label}</div>'
                f'<div class="score-value score-mid">N/A</div></div>')
    css = "score-high" if score >= 65 else ("score-low" if score < 35 else "score-mid")
    return (f'<div class="score-card"><div class="score-label">{label}</div>'
            f'<div class="score-value {css}">{score:.0f}'
            f'<span style="font-size:1rem;color:#8B949E">/100</span></div></div>')


# ==============================================================================
# ZONE 1: MARKET INTELLIGENCE
# ==============================================================================

@st.cache_data(ttl=300)
def _cached_regime():
    return detect_market_regime(), get_vix_analysis()

regime_data, vix_data = _cached_regime()

# Market Regime + VIX row
col_regime, col_vix = st.columns([3, 1])
with col_regime:
    if "error" not in regime_data:
        regime = regime_data.get("regime", "UNKNOWN")
        regime_css = {
            "BULL": "regime-bull", "BEAR": "regime-bear",
            "SIDEWAYS": "regime-sideways", "HIGH_VOLATILITY": "regime-volatile",
            "CRASH": "regime-crash",
        }.get(regime, "")
        score = regime_data.get("score", "")
        recommendation = regime_data.get("recommendation", "")
        st.markdown(f'''<div class="regime-banner">
            <span class="regime-label {regime_css}">{regime.replace("_", " ")} MARKET</span>
            <div style="color:#8B949E;font-size:0.85rem;margin-top:0.3rem">Score: {score} &mdash; {recommendation}</div>
        </div>''', unsafe_allow_html=True)
    else:
        st.warning("Could not load market regime")

with col_vix:
    if "error" not in vix_data:
        vix = vix_data.get("vix", "N/A")
        signal = vix_data.get("vix_signal", "normal")
        pct = vix_data.get("vix_1y_percentile", 0) or 0
        vix_css = {"fear": "vix-high", "normal": "vix-normal", "complacency": "vix-low"}.get(signal, "vix-normal")
        pct_display = f"{pct:.0f}th pctile" if isinstance(pct, (int, float)) else "N/A"
        st.markdown(f'''<div class="regime-banner" style="text-align:center">
            <div class="vix-badge {vix_css}" style="font-size:1.2rem">VIX: {vix}</div>
            <div style="color:#8B949E;font-size:0.8rem;margin-top:0.3rem">{signal.title()} &middot; {pct_display}</div>
        </div>''', unsafe_allow_html=True)

st.divider()


# ==============================================================================
# ZONE 2: TICKER DETAIL
# ==============================================================================

from app.components.trading_charts import (
    candlestick_chart, score_gauge, cftc_positioning_bars,
    stress_gauge, sector_allocation_bars, equity_curve_chart,
)

st.markdown("#### Ticker Analysis")
selected_ticker = st.text_input(
    "Search ticker",
    value=st.session_state.get("selected_ticker", ""),
    placeholder="e.g. AAPL, LCID — enter symbol, not company name",
    key="ticker_input",
    help="Enter a ticker symbol (e.g. LCID, not 'Lucid Motors')",
)

if selected_ticker:
    selected_ticker = selected_ticker.upper().strip()
    st.session_state["selected_ticker"] = selected_ticker

    col_chart, col_analysis = st.columns([2, 1])
    with col_chart:
        # (period, interval) — granularity scales with the window so the chart
        # always has enough candles: hourly for a week, daily for months/year.
        period_cfg = {
            "1W": ("5d", "1h"),
            "1M": ("1mo", "1d"),
            "3M": ("3mo", "1d"),
            "1Y": ("1y", "1d"),
        }
        period_label = st.radio("Period", list(period_cfg), horizontal=True, index=2)
        _period, _interval = period_cfg[period_label]
        fig = candlestick_chart(selected_ticker, period=_period, interval=_interval)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning(f"No chart data for {selected_ticker}")

    with col_analysis:
        analysis = _c_analyze(selected_ticker)
        if "error" not in analysis:
            price = analysis.get("price")
            score_data_full = analysis.get("score", {})
            st.metric(selected_ticker, f"${price:.2f}" if price else "N/A")
            if score_data_full:
                st.plotly_chart(
                    score_gauge(score_data_full.get("score", 0), "Composite"),
                    width="stretch",
                )
                # Sentiment bridge — blend MarketPulse RSS sentiment into the score
                from app.pipeline_runner import get_ticker_cache as _gtc
                _cache = _gtc()
                _rss = next((d for d in _cache.values()
                             if str(d.get("symbol", "")).upper() == selected_ticker), None)
                if _rss:
                    base = score_data_full.get("score", 0)
                    tilt = round((_rss.get("bullish_ratio", 0) - _rss.get("bearish_ratio", 0)) * 10, 1)
                    adj = max(0, min(100, base + tilt))
                    sent = _rss.get("dominant_sentiment", "neutral")
                    scolor = {"bullish": "#00C853", "bearish": "#FF1744"}.get(sent, "#FFD600")
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.85rem'>"
                        f"News: <b style='color:{scolor}'>{sent.upper()}</b> "
                        f"&nbsp;·&nbsp; score {base:.0f} <b style='color:{scolor}'>{tilt:+.0f}</b> "
                        f"&rarr; <b>{adj:.0f}</b></div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("MarketPulse RSS sentiment blended into the bot's score")
        else:
            st.warning(f"Could not analyze {selected_ticker}")

    # Score cards: Fundamentals | Momentum | Smart Money
    score_result = _c_score(selected_ticker)
    col_fund, col_mom, col_smart = st.columns(3)

    with col_fund:
        fund = _c_fundamentals(selected_ticker)
        if "error" not in fund:
            val_score = score_result.get("valuation") if "error" not in score_result else None
            st.markdown(_score_card_html("Fundamentals", val_score), unsafe_allow_html=True)
            st.caption(f"P/E: {fund.get('pe_ratio', 'N/A')}")
            st.caption(f"EV/EBITDA: {fund.get('ev_to_ebitda', 'N/A')}")
            st.caption(f"P/B: {fund.get('price_to_book', 'N/A')}")
            st.caption(f"Div Yield: {fund.get('dividend_yield', 'N/A')}")
            st.caption(f"Market Cap: {fund.get('market_cap', 'N/A')}")

    with col_mom:
        mom = _c_momentum(selected_ticker)
        if "error" not in mom:
            mom_score = score_result.get("momentum") if "error" not in score_result else None
            st.markdown(_score_card_html("Momentum", mom_score), unsafe_allow_html=True)
            m30 = mom.get("price_momentum_30d")
            m90 = mom.get("price_momentum_90d")
            vol = mom.get("volatility")
            rs = mom.get("relative_strength")
            st.caption(f"30D Return: {m30 * 100:.1f}%" if m30 is not None else "30D Return: N/A")
            st.caption(f"90D Return: {m90 * 100:.1f}%" if m90 is not None else "90D Return: N/A")
            st.caption(f"Volatility: {vol:.3f}" if vol is not None else "Volatility: N/A")
            st.caption(f"Rel Strength: {rs:.2f}" if rs is not None else "Rel Strength: N/A")

    with col_smart:
        # Use a real Streamlit bordered container — a raw <div> split across two
        # st.markdown calls renders as an empty box (Streamlit closes each block).
        with st.container(border=True):
            st.markdown("**Smart Money**")
            signal_data = _c_smart_money("E-MINI S&P 500")
            if "error" not in signal_data:
                sig = signal_data.get("signal", "neutral")
                sig_color = "#00C853" if sig == "bullish" else ("#FF1744" if sig == "bearish" else "#FFD600")
                st.markdown(
                    f"CFTC Signal: <span style='color:{sig_color};font-weight:700'>{sig.upper()}</span>",
                    unsafe_allow_html=True,
                )
                reason = signal_data.get("reason", "")
                if reason:
                    st.caption(reason)
                positioning = _c_futures("E-MINI S&P 500")
                if "error" not in positioning:
                    reports = positioning.get("reports", [])
                    if reports:
                        latest = reports[0]
                        st.plotly_chart(
                            cftc_positioning_bars(
                                latest.get("commercial_net", 0),
                                latest.get("non_commercial_net", 0),
                            ),
                            width="stretch",
                        )
            else:
                st.caption("Smart money data unavailable")

    # -- Catalysts: SEC filings · insider trades · search interest -------------
    with st.expander(f"Catalysts for {selected_ticker} — SEC · Insider · Search Trends"):
        cat1, cat2, cat3 = st.columns(3)
        with cat1:
            st.markdown("**Recent SEC Filings (8-K)**")
            sec = _c_sec(selected_ticker)
            filings = sec.get("filings", []) if "error" not in sec else []
            if filings:
                for f in filings[:5]:
                    ftype = f.get("filing_type", "Filing")
                    fdate = f.get("date", "")
                    furl = f.get("primary_document_url") or ""
                    if furl:
                        st.markdown(
                            f"<a href='{furl}' target='_blank' rel='noopener' style='color:#58A6FF;text-decoration:none'>"
                            f"{ftype}</a> <span style='color:#6E7681;font-size:0.8rem'>{fdate}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"{ftype} <span style='color:#6E7681;font-size:0.8rem'>{fdate}</span>",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No recent 8-K filings.")
        with cat2:
            st.markdown("**Insider Filings (90d)**")
            ins = _c_insider(selected_ticker)
            trades = ins.get("insider_trades", []) if "error" not in ins else []
            if trades:
                st.caption(f"{len(trades)} Form 3/4/5 filing(s)")
                for t in trades[:5]:
                    form = t.get("form_type", "Form")
                    fdate = t.get("filing_date", "")
                    furl = t.get("url") or ""
                    if furl:
                        st.markdown(
                            f"<a href='{furl}' target='_blank' rel='noopener' style='color:#58A6FF;text-decoration:none'>"
                            f"Form {form}</a> <span style='color:#6E7681;font-size:0.8rem'>{fdate}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"Form {form} <span style='color:#6E7681;font-size:0.8rem'>{fdate}</span>",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No insider filings found.")
        with cat3:
            st.markdown("**Search Interest**")
            tr = _c_trends(selected_ticker)
            series = tr.get("interest_over_time") or tr.get("data") or [] if "error" not in tr else []
            if series:
                vals = [pt.get("value") if isinstance(pt, dict) else pt for pt in series]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    trend = "rising" if vals[-1] >= vals[0] else "falling"
                    tcolor = "#00C853" if trend == "rising" else "#FF1744"
                    st.markdown(
                        f"Google search interest <b style='color:{tcolor}'>{trend}</b> "
                        f"({vals[0]:.0f} → {vals[-1]:.0f})", unsafe_allow_html=True,
                    )
                else:
                    st.caption("Trend data unavailable.")
            else:
                st.caption("Trend data unavailable.")

    st.divider()


# ==============================================================================
# ZONE 3: BOT CONTROL (entire zone is a live fragment — updates every second)
# ==============================================================================

st.divider()
st.markdown("#### Bot Control")

# -- Bot settings (outside the live fragment so widgets stay stable) ----------
_bs = _get_bot_state()
with st.expander("⚙ Bot Settings", expanded=not _bs.is_running):
    if _bs.is_running:
        st.caption("Stop the bot to change starting capital. Other settings apply live.")
    sc1, sc2, sc3, sc4 = st.columns(4)
    _cap = sc1.number_input(
        "Starting capital ($)", min_value=10_000, max_value=1_000_000,
        value=int(_bs.starting_capital), step=5_000,
        disabled=_bs.is_running, key="cfg_capital",
    )
    _maxpos = sc2.slider("Max positions", 1, 30, int(_bs.max_positions), key="cfg_maxpos")
    _minsc = sc3.slider("Min score to buy", 40, 90, int(_bs.min_score), key="cfg_minscore")
    _riskcap = sc4.slider("Max risk / trade (%)", 1, 10, int(round(_bs.max_risk_per_trade * 100)), key="cfg_risk")
    _bridge = st.checkbox(
        "Blend MarketPulse news sentiment into scores (the two-tab bridge)",
        value=_bs.sentiment_bridge, key="cfg_bridge",
    )
    _warm = st.checkbox(
        "Warm-start the Edge panel with illustrative history on a cold start",
        value=_bs.warm_start, key="cfg_warm", disabled=_bs.is_running,
        help="Populates win-rate/EV/Kelly and an equity curve immediately. "
             "Turn off to begin from a truly empty $0-history account.",
    )
    # Apply to live bot state
    if not _bs.is_running:
        _bs.starting_capital = float(_cap)
        _bs.warm_start = _warm
    _bs.max_positions = int(_maxpos)
    _bs.min_score = float(_minsc)
    _bs.max_risk_per_trade = _riskcap / 100.0
    _bs.sentiment_bridge = _bridge


@st.fragment(run_every=1)
def _bot_live_panel():
    """Entire bot control panel re-renders every 1 second for live counters."""
    state = _get_bot_state()
    engine = _get_engine()

    # -- Start / Stop + Status + Live Timer --------------------------------
    col_btn, col_status, col_timer = st.columns([1, 1, 2])

    with col_btn:
        if state.is_running:
            if st.button("Stop Bot", type="secondary", key="bot_stop"):
                engine.stop()
                st.rerun()
        else:
            if st.button("Start Bot", type="primary", key="bot_start"):
                engine.start()
                st.rerun()

    with col_status:
        if state.is_running:
            st.markdown(
                '<span style="color:#00C853;font-weight:700;font-size:1rem">● RUNNING</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#8B949E;font-weight:700;font-size:1rem">● STOPPED</span>',
                unsafe_allow_html=True,
            )

    with col_timer:
        if state.is_running and state.last_cycle_time:
            elapsed = int((datetime.now() - state.last_cycle_time).total_seconds())
            st.markdown(
                f'<span style="font-size:0.85rem;color:#8B949E;">'
                f'Cycle <b style="color:#E6EDF3;">#{state.cycle_count}</b>'
                f' &nbsp;·&nbsp; Last scan <b style="color:#E6EDF3;">{elapsed}s</b> ago'
                f'</span>',
                unsafe_allow_html=True,
            )
        elif state.is_running:
            st.caption("Starting...")

    # -- Portfolio metrics + positions + log --------------------------------
    if state.portfolio_id:
        # Market hours check (US Eastern, Mon-Fri 9:30-16:00)
        from zoneinfo import ZoneInfo
        _et = datetime.now(ZoneInfo("America/New_York"))
        _market_open = (
            _et.weekday() < 5
            and _et.hour * 60 + _et.minute >= 570   # 9:30
            and _et.hour * 60 + _et.minute < 960     # 16:00
        )
        if not _market_open:
            st.caption("Market closed — prices frozen at last close. P&L updates when market opens.")

        col_pv, col_pnl, col_npos = st.columns(3)
        with col_pv:
            st.metric("Portfolio Value", f"${state.portfolio_value:,.2f}")
        with col_pnl:
            st.metric("Total P&L", f"${state.total_pnl:+,.2f}")
        with col_npos:
            st.metric("Open Positions", f"{len(state.open_positions)} / {state.max_positions}")

        # -- Equity curve -----------------------------------------------------
        eq_fig = equity_curve_chart(state.equity_curve, state.starting_capital)
        if eq_fig is not None:
            st.markdown("##### Equity Curve")
            st.plotly_chart(eq_fig, width="stretch", key="equity_curve")

        # -- Quant stats (the math that matters) ------------------------------
        s = state.stats
        if s.total_trades > 0:
            st.markdown("##### Edge Statistics")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                ev_color = "#00C853" if s.expected_value > 0 else "#FF1744"
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">EV/Trade</div>'
                    f'<div style="color:{ev_color};font-size:1.2rem;font-weight:700">${s.expected_value:+.2f}</div>'
                    f'</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">Win Rate</div>'
                    f'<div style="color:#E6EDF3;font-size:1.2rem;font-weight:700">{s.win_rate*100:.0f}%</div>'
                    f'</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">R:R Ratio</div>'
                    f'<div style="color:#E6EDF3;font-size:1.2rem;font-weight:700">{s.reward_risk_ratio:.2f}</div>'
                    f'</div>', unsafe_allow_html=True)
            with c4:
                kelly_display = f"{s.kelly_fraction*100:.2f}%" if s.kelly_fraction > 0 else "N/A"
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">½ Kelly</div>'
                    f'<div style="color:#E6EDF3;font-size:1.2rem;font-weight:700">{kelly_display}</div>'
                    f'</div>', unsafe_allow_html=True)
            with c5:
                ruin_color = "#00C853" if s.risk_of_ruin < 0.01 else ("#FFD600" if s.risk_of_ruin < 0.05 else "#FF1744")
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">Risk of Ruin</div>'
                    f'<div style="color:{ruin_color};font-size:1.2rem;font-weight:700">{s.risk_of_ruin*100:.4f}%</div>'
                    f'</div>', unsafe_allow_html=True)

            # Second row: trades, streak, stddev
            c6, c7, c8 = st.columns(3)
            with c6:
                edge_label = "HAS EDGE" if s.has_edge else (f"NEED {10 - s.total_trades}+ TRADES" if s.total_trades < 10 else "NO EDGE")
                edge_color = "#00C853" if s.has_edge else "#FFD600"
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">Edge ({s.total_trades} trades)</div>'
                    f'<div style="color:{edge_color};font-size:0.9rem;font-weight:700">{edge_label}</div>'
                    f'</div>', unsafe_allow_html=True)
            with c7:
                streak_color = "#00C853" if s.current_streak > 0 else "#FF1744"
                streak_label = f"{'W' if s.current_streak > 0 else 'L'}{abs(s.current_streak)}"
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">Streak</div>'
                    f'<div style="color:{streak_color};font-size:1.2rem;font-weight:700">{streak_label}</div>'
                    f'</div>', unsafe_allow_html=True)
            with c8:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div style="color:#8B949E;font-size:0.75rem">Std Dev (σ)</div>'
                    f'<div style="color:#E6EDF3;font-size:1.2rem;font-weight:700">${s.std_dev:.2f}</div>'
                    f'</div>', unsafe_allow_html=True)

        # -- Open positions table ---------------------------------------------
        if state.open_positions:
            st.markdown("##### Open Positions")
            rows = []
            for ticker, pos in state.open_positions.items():
                entry = pos["entry_price"]
                current = pos.get("current_price", entry)
                pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
                pnl_amt = (current - entry) * pos["shares"]
                rows.append({
                    "Ticker": ticker,
                    "Entry": f"${entry:.2f}",
                    "Current": f"${current:.2f}",
                    "P&L %": f"{pnl_pct:+.2f}%",
                    "P&L $": f"${pnl_amt:+.2f}",
                    "Score": f"{pos.get('current_score', pos['entry_score']):.0f}",
                    "Since": pos["entry_time"].strftime("%H:%M"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            # -- Risk & exposure (MCP check_risk) -----------------------------
            with st.expander("Risk & Exposure"):
                risk = _c_risk(state.portfolio_id)
                if "error" not in risk:
                    stress = risk.get("stress", {}) or {}
                    sectors = risk.get("sector_allocation", {}) or {}
                    drawdowns = stress.get("scenario_drawdowns", {}) or {}
                    rk1, rk2 = st.columns(2)
                    with rk1:
                        sscore = stress.get("stress_score", 0)
                        scolor = "#FF1744" if sscore > 0.30 else ("#FFD600" if sscore > 0.15 else "#00C853")
                        st.markdown(
                            f"<div style='color:#8B949E;font-size:0.8rem'>Stress score (worst-case drawdown)</div>"
                            f"<div style='color:{scolor};font-size:1.6rem;font-weight:700'>{sscore * 100:.1f}%</div>",
                            unsafe_allow_html=True,
                        )
                        vuln = stress.get("vulnerable_sectors", [])
                        if vuln:
                            st.caption("Vulnerable: " + ", ".join(vuln))
                        if drawdowns:
                            st.plotly_chart(
                                stress_gauge(sscore, drawdowns),
                                width="stretch", key="risk_stress",
                            )
                    with rk2:
                        if sectors:
                            st.plotly_chart(
                                sector_allocation_bars(sectors),
                                width="stretch", key="risk_sectors",
                            )
                        else:
                            st.caption("No sector exposure yet.")
                else:
                    st.caption(f"Risk data unavailable: {risk.get('error', '')}")

        # -- Activity log -----------------------------------------------------
        if state.trade_log:
            st.markdown("##### Activity Log")
            with st.expander(
                f"Recent trades ({len(state.trade_log)})", expanded=True
            ):
                for entry in state.trade_log[:50]:
                    color = "#00C853" if entry["action"] == "BUY" else "#FF1744"
                    pnl_str = (
                        f" &nbsp;|&nbsp; P&L: ${entry['pnl']:+.2f}"
                        if entry["action"] == "SELL"
                        else ""
                    )
                    line = (
                        f"<small style='color:#8B949E'>[{entry['time']}]</small> "
                        f"<span style='color:{color};font-weight:700'>{entry['action']}</span> "
                        f"<b>{entry['ticker']}</b> {entry['shares']}sh "
                        f"@ ${entry['price']:.2f} (score {entry['score']:.0f})"
                        f"{pnl_str} — {entry['reason']}"
                    )
                    # Streamlit renders $...$ as LaTeX math, which mangles any line
                    # with two dollar signs (e.g. trailing-stop reasons). Swap $ for
                    # the HTML entity so prices always render literally.
                    st.markdown(line.replace("$", "&#36;"), unsafe_allow_html=True)
    elif state.is_running:
        st.info("Bot is starting — creating portfolio...")
    else:
        st.caption(
            f"Click **Start Bot** to begin autonomous paper trading with "
            f"${state.starting_capital:,.0f}."
        )


_bot_live_panel()
