"""Trading Bot — Full trading terminal powered by financial-mcp-server."""

import time as _time

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.investor.bot_engine import (
    get_state as _get_bot_state,
    get_engine as _get_engine,
    MAX_POSITIONS,
)

st.set_page_config(page_title="Trading Bot", page_icon="\U0001F4C8", layout="wide")

from app.components.styles import apply_theme
apply_theme()

DATA_DIR = Path("data")
PORTFOLIO_FILE = DATA_DIR / "portfolio_id.txt"


# -- Portfolio Persistence -----------------------------------------------------

def _load_portfolio_id() -> str | None:
    if "portfolio_id" in st.session_state:
        return st.session_state["portfolio_id"]
    if PORTFOLIO_FILE.exists():
        pid = PORTFOLIO_FILE.read_text().strip()
        if pid:
            st.session_state["portfolio_id"] = pid
            return pid
    return None


def _save_portfolio_id(pid: str):
    DATA_DIR.mkdir(exist_ok=True)
    PORTFOLIO_FILE.write_text(pid)
    st.session_state["portfolio_id"] = pid


# -- MCP Connection Check -----------------------------------------------------

try:
    from src.investor import (
        is_connected, detect_market_regime, get_vix_analysis,
        scan_anomalies, scan_volume_leaders, scan_gap_movers,
        analyze_ticker, get_fundamentals, get_momentum, score_ticker,
        get_smart_money_signal, get_futures_positioning,
        create_portfolio, analyze_portfolio, get_holdings, get_trades,
        run_rebalance, check_risk,
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


# ==============================================================================
# ZONE 1: MARKET INTELLIGENCE
# ==============================================================================

@st.cache_data(ttl=300)
def _cached_regime():
    return detect_market_regime(), get_vix_analysis()

@st.cache_data(ttl=120)
def _cached_movers():
    return scan_anomalies(), scan_volume_leaders(), scan_gap_movers()

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

# Top Movers
anomalies, volume, gaps = _cached_movers()
movers = {}
if "error" not in anomalies:
    for item in anomalies.get("anomalies", []):
        sym = item["symbol"]
        movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
        movers[sym]["score"] += item.get("total_score", 0)
        for a in item.get("anomalies", []):
            movers[sym]["badges"].append(a["type"])
if "error" not in volume:
    for item in volume.get("leaders", []):
        sym = item["symbol"]
        movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
        movers[sym]["score"] += item.get("ratio", 0)
        movers[sym]["badges"].append("volume_spike")
if "error" not in gaps:
    for item in gaps.get("movers", []):
        sym = item["symbol"]
        movers[sym] = movers.get(sym, {"symbol": sym, "badges": [], "score": 0})
        movers[sym].setdefault("change_pct", item.get("gap_percent", 0))
        movers[sym]["badges"].append("gap_up" if item.get("gap_percent", 0) > 0 else "gap_down")

sorted_movers = sorted(movers.values(), key=lambda m: m["score"], reverse=True)[:6]
if sorted_movers:
    st.markdown("#### Top Movers")
    mover_cols = st.columns(3)
    for i, m in enumerate(sorted_movers):
        with mover_cols[i % 3]:
            cp = m.get("change_pct", 0)
            change_css = "mover-change-pos" if cp >= 0 else "mover-change-neg"
            badges = " ".join(
                f'<span class="anomaly-badge badge-{b.replace("_", "-")}">{b.replace("_", " ").title()}</span>'
                for b in m.get("badges", [])[:2]
            )
            st.markdown(f'''<div class="mover-card">
                <div class="mover-symbol">{m["symbol"]}</div>
                <div class="{change_css}">{cp:+.1f}%</div>
                <div>{badges}</div>
            </div>''', unsafe_allow_html=True)

st.divider()


# ==============================================================================
# ZONE 2: TICKER DETAIL
# ==============================================================================

from app.components.trading_charts import (
    candlestick_chart, score_gauge, cftc_positioning_bars,
    stress_gauge, sector_allocation_bars,
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
        period_map = {"1W": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y"}
        period_label = st.radio("Period", ["1W", "1M", "3M", "1Y"], horizontal=True, index=2)
        fig = candlestick_chart(selected_ticker, period=period_map[period_label])
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning(f"No chart data for {selected_ticker}")

    with col_analysis:
        analysis = analyze_ticker(selected_ticker)
        if "error" not in analysis:
            price = analysis.get("price")
            score_data_full = analysis.get("score", {})
            st.metric(selected_ticker, f"${price:.2f}" if price else "N/A")
            if score_data_full:
                st.plotly_chart(
                    score_gauge(score_data_full.get("score", 0), "Composite"),
                    width="stretch",
                )
        else:
            st.warning(f"Could not analyze {selected_ticker}")

    # Score cards: Fundamentals | Momentum | Smart Money
    score_result = score_ticker(selected_ticker)
    col_fund, col_mom, col_smart = st.columns(3)

    with col_fund:
        fund = get_fundamentals(selected_ticker)
        if "error" not in fund:
            val_score = score_result.get("valuation", 0) if "error" not in score_result else 0
            css = "score-high" if val_score >= 65 else ("score-low" if val_score < 35 else "score-mid")
            st.markdown(f'''<div class="score-card">
                <div class="score-label">Fundamentals</div>
                <div class="score-value {css}">{val_score:.0f}<span style="font-size:1rem;color:#8B949E">/100</span></div>
            </div>''', unsafe_allow_html=True)
            st.caption(f"P/E: {fund.get('pe_ratio', 'N/A')}")
            st.caption(f"EV/EBITDA: {fund.get('ev_to_ebitda', 'N/A')}")
            st.caption(f"P/B: {fund.get('price_to_book', 'N/A')}")
            st.caption(f"Div Yield: {fund.get('dividend_yield', 'N/A')}")
            st.caption(f"Market Cap: {fund.get('market_cap', 'N/A')}")

    with col_mom:
        mom = get_momentum(selected_ticker)
        if "error" not in mom:
            mom_score = score_result.get("momentum", 0) if "error" not in score_result else 0
            css = "score-high" if mom_score >= 65 else ("score-low" if mom_score < 35 else "score-mid")
            st.markdown(f'''<div class="score-card">
                <div class="score-label">Momentum</div>
                <div class="score-value {css}">{mom_score:.0f}<span style="font-size:1rem;color:#8B949E">/100</span></div>
            </div>''', unsafe_allow_html=True)
            m30 = mom.get("price_momentum_30d")
            m90 = mom.get("price_momentum_90d")
            vol = mom.get("volatility")
            rs = mom.get("relative_strength")
            st.caption(f"30D Return: {m30 * 100:.1f}%" if m30 is not None else "30D Return: N/A")
            st.caption(f"90D Return: {m90 * 100:.1f}%" if m90 is not None else "90D Return: N/A")
            st.caption(f"Volatility: {vol:.3f}" if vol is not None else "Volatility: N/A")
            st.caption(f"Rel Strength: {rs:.2f}" if rs is not None else "Rel Strength: N/A")

    with col_smart:
        st.markdown('<div class="smart-money-card">', unsafe_allow_html=True)
        st.markdown("**Smart Money**")
        signal_data = get_smart_money_signal("E-MINI S&P 500")
        if "error" not in signal_data:
            sig = signal_data.get("signal", "neutral")
            sig_color = "#00C853" if sig == "bullish" else ("#FF1744" if sig == "bearish" else "#FFD600")
            st.markdown(
                f"CFTC Signal: <span style='color:{sig_color};font-weight:700'>{sig.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.caption(signal_data.get("reason", ""))
            positioning = get_futures_positioning("E-MINI S&P 500")
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
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()


# ==============================================================================
# ZONE 3: PORTFOLIO MANAGEMENT
# ==============================================================================

st.markdown("#### Portfolio")

portfolio_id = _load_portfolio_id()

if portfolio_id is None:
    # Onboarding
    st.markdown("**Set up your portfolio to start trading**")
    with st.form("portfolio_setup"):
        name = st.text_input("Portfolio name", value="My Portfolio")
        capital = st.slider("Starting capital ($)", 10000, 1000000, 100000, step=10000)
        risk = st.radio("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
        horizon = st.selectbox("Investment horizon", ["short", "medium", "long"], index=1)
        submitted = st.form_submit_button("Create Portfolio")
        if submitted:
            result = create_portfolio(float(capital), risk, horizon, name)
            if "error" not in result:
                _save_portfolio_id(result["portfolio_id"])
                st.success(f"Portfolio created! ID: {result['portfolio_id']}")
                st.rerun()
            else:
                st.error(f"Failed: {result['error']}")
else:
    summary = analyze_portfolio(portfolio_id)
    if "error" in summary:
        st.warning("Portfolio not found on MCP server. Create a new one?")
        if st.button("Reset Portfolio"):
            PORTFOLIO_FILE.unlink(missing_ok=True)
            st.session_state.pop("portfolio_id", None)
            st.rerun()
    else:
        # Header metrics
        col_val, col_change, col_cash, col_rebal = st.columns(4)
        total_value = summary.get("total_value", 0)
        daily_change = summary.get("daily_change", 0)
        daily_pct = summary.get("daily_change_pct", 0)
        portfolio_info = summary.get("portfolio", {})
        cash = portfolio_info.get("current_cash", 0)

        with col_val:
            st.metric("Portfolio Value", f"${total_value:,.2f}")
        with col_change:
            st.metric("Daily Change", f"${daily_change:,.2f}", f"{daily_pct:+.2f}%")
        with col_cash:
            st.metric("Cash", f"${cash:,.2f}")
        with col_rebal:
            if st.button("Rebalance Now"):
                with st.spinner("Rebalancing..."):
                    rebal_result = run_rebalance(portfolio_id, trigger="manual")
                    if "error" not in rebal_result:
                        buys = rebal_result.get("buy_signals", 0)
                        sells = rebal_result.get("sell_signals", 0)
                        st.success(f"Rebalance complete: {buys} buys, {sells} sells")
                        st.rerun()
                    else:
                        st.error(f"Rebalance failed: {rebal_result.get('error')}")

        # Holdings table
        holdings_data = get_holdings(portfolio_id)
        if "error" not in holdings_data and holdings_data.get("holdings"):
            st.markdown("##### Holdings")
            df = pd.DataFrame(holdings_data["holdings"])
            display_cols = [c for c in ["symbol", "shares", "avg_cost_basis", "asset_type", "sector", "company_name"] if c in df.columns]
            st.dataframe(df[display_cols], width="stretch", hide_index=True)

        # Trade history
        trades_data = get_trades(portfolio_id)
        if "error" not in trades_data and trades_data.get("trades"):
            st.markdown("##### Trade History")
            with st.expander(f"Show trades ({trades_data.get('count', 0)})"):
                trades_df = pd.DataFrame(trades_data["trades"])
                display_cols = [c for c in ["symbol", "action", "shares", "price", "total_value", "status", "executed_at"] if c in trades_df.columns]
                st.dataframe(trades_df[display_cols], width="stretch", hide_index=True)

        # Risk assessment
        risk_data = check_risk(portfolio_id)
        if "error" not in risk_data:
            st.markdown("##### Risk Assessment")
            col_stress, col_sectors = st.columns(2)
            with col_stress:
                stress = risk_data.get("stress", {})
                scenarios = stress.get("scenario_drawdowns", {})
                if scenarios:
                    st.plotly_chart(
                        stress_gauge(stress.get("stress_score", 0), scenarios),
                        width="stretch",
                    )
            with col_sectors:
                sector_alloc = risk_data.get("sector_allocation", {})
                if sector_alloc:
                    st.plotly_chart(
                        sector_allocation_bars(sector_alloc),
                        width="stretch",
                    )

        # Performance metrics
        perf = summary.get("performance", {})
        if perf:
            st.markdown("##### Performance")
            perf_cols = st.columns(4)
            with perf_cols[0]:
                st.metric("Cumulative Return", f"{perf.get('cumulative_return', 0) * 100:.2f}%")
            with perf_cols[1]:
                st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
            with perf_cols[2]:
                st.metric("Max Drawdown", f"{perf.get('max_drawdown', 0) * 100:.2f}%")
            with perf_cols[3]:
                st.metric("Daily Return", f"{perf.get('daily_return', 0) * 100:.2f}%")


# ==============================================================================
# ZONE 4: BOT CONTROL
# ==============================================================================

st.divider()
st.markdown("#### Bot Control")

_bot_state = _get_bot_state()
_engine = _get_engine()

# -- Start / Stop + Status row ------------------------------------------------
col_btn, col_status, col_timer = st.columns([1, 1, 2])

with col_btn:
    if _bot_state.is_running:
        if st.button("Stop Bot", type="secondary", key="bot_stop"):
            _engine.stop()
            st.rerun()
    else:
        if st.button("Start Bot", type="primary", key="bot_start"):
            _engine.start()
            st.rerun()

with col_status:
    if _bot_state.is_running:
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
    if _bot_state.is_running:
        last = _bot_state.last_cycle_time
        elapsed = f"{(datetime.now() - last).seconds}s ago" if last else "starting..."
        st.caption(f"Running continuously · Cycle #{_bot_state.cycle_count} · Last: {elapsed}")

# -- Portfolio metrics --------------------------------------------------------
if _bot_state.portfolio_id:
    col_pv, col_pnl, col_npos = st.columns(3)
    with col_pv:
        st.metric("Portfolio Value", f"${_bot_state.portfolio_value:,.2f}")
    with col_pnl:
        pnl = _bot_state.total_pnl
        st.metric("Total P&L", f"${pnl:+,.2f}")
    with col_npos:
        st.metric("Open Positions", f"{len(_bot_state.open_positions)} / {MAX_POSITIONS}")

    # -- Open positions table -------------------------------------------------
    if _bot_state.open_positions:
        st.markdown("##### Open Positions")
        rows = []
        for ticker, pos in _bot_state.open_positions.items():
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

    # -- Activity log ---------------------------------------------------------
    if _bot_state.trade_log:
        st.markdown("##### Activity Log")
        with st.expander(
            f"Recent trades ({len(_bot_state.trade_log)})", expanded=True
        ):
            for entry in _bot_state.trade_log[:50]:
                color = "#00C853" if entry["action"] == "BUY" else "#FF1744"
                pnl_str = (
                    f" &nbsp;|&nbsp; P&L: ${entry['pnl']:+.2f}"
                    if entry["action"] == "SELL"
                    else ""
                )
                st.markdown(
                    f"<small style='color:#8B949E'>[{entry['time']}]</small> "
                    f"<span style='color:{color};font-weight:700'>{entry['action']}</span> "
                    f"<b>{entry['ticker']}</b> {entry['shares']}sh "
                    f"@ ${entry['price']:.2f} (score {entry['score']:.0f})"
                    f"{pnl_str} — {entry['reason']}",
                    unsafe_allow_html=True,
                )

# -- Auto-refresh (non-blocking) ----------------------------------------------
if _bot_state.is_running:
    _now = _time.time()
    _last = st.session_state.get("bot_last_refresh", 0)
    if _now - _last >= 10:
        st.session_state["bot_last_refresh"] = _now
        st.rerun()
