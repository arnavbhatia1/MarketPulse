"""
Portfolio Dashboard — Premium feature.

Shows portfolio value, holdings, allocation, risk panel, and bot status.
"""

import html as html_mod
import streamlit as st
import sys, os

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.components.styles import apply_theme, COLORS
from app.components.charts import (
    portfolio_performance_line, allocation_donut, stress_gauge,
)

st.set_page_config(page_title="Portfolio | MarketPulse", page_icon="M", layout="wide")
apply_theme()

# ── Auth gate ────────────────────────────────────────────────────────────────
from app.components.auth_guard import require_auth
require_auth(premium=True)

from src.storage import db
from src.investor import portfolio as portfolio_mod
from src.utils.config import load_config

config = load_config()
investor_config = config.get("investor", {})
user_id = st.session_state.get("user_id")

# ── Load portfolio ───────────────────────────────────────────────────────────
portfolios = db.get_user_portfolios(user_id)

if not portfolios:
    # Onboarding — first-time setup
    st.markdown('<div class="onboarding-card">', unsafe_allow_html=True)
    st.markdown("### Set Up Your Portfolio")
    st.markdown("Configure your investment profile to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        capital = st.slider("Starting Capital ($)", 10000, 1000000, 100000, step=5000,
                           format="$%d")
    with col2:
        risk_profile = st.radio("Risk Profile",
                               ["conservative", "moderate", "aggressive"],
                               index=1)
    with col3:
        horizon_labels = {"1yr": "short", "5yr": "medium", "10yr+": "long"}
        horizon_display = st.selectbox("Investment Horizon",
                              list(horizon_labels.keys()), index=1)
        horizon = horizon_labels[horizon_display]

    if st.button("Create Portfolio", type="primary", use_container_width=True):
        pid = portfolio_mod.create_user_portfolio(user_id, capital, risk_profile, horizon)
        st.success("Portfolio created! Running initial analysis...")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Use first active portfolio
port = portfolios[0]
portfolio_id = port["portfolio_id"]

# ── Portfolio summary ────────────────────────────────────────────────────────
summary = portfolio_mod.get_portfolio_summary(portfolio_id)
if not summary:
    st.error("Could not load portfolio data.")
    st.stop()

total_value = summary["total_value"]
holdings_value = summary["holdings_value"]
daily_change = summary.get("daily_change", 0)
daily_change_pct = summary.get("daily_change_pct", 0)
starting_capital = port["starting_capital"]
overall_return = (total_value / starting_capital - 1) if starting_capital > 0 else 0

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## Portfolio Dashboard")

change_class = "portfolio-change-positive" if daily_change >= 0 else "portfolio-change-negative"
change_sign = "+" if daily_change >= 0 else ""
risk_badge = port["risk_profile"]
horizon_display = {"short": "1yr", "medium": "5yr", "long": "10yr+"}.get(port["investment_horizon"], port["investment_horizon"])

st.markdown(f"""
<div class="portfolio-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="portfolio-value">${total_value:,.2f}</div>
            <span class="{change_class}">{change_sign}${daily_change:,.2f} ({change_sign}{daily_change_pct:.2%}) today</span>
        </div>
        <div style="text-align:right;">
            <span class="risk-badge risk-badge-{risk_badge}">{risk_badge}</span>
            <div style="color:#8B949E; font-size:0.85em; margin-top:8px;">
                {port['mode'].title()} · {horizon_display} horizon
            </div>
        </div>
    </div>
    <div style="margin-top:12px; color:#8B949E;">
        Cash: ${port['current_cash']:,.2f} · Overall return: <span class="{change_class}">{overall_return:+.2%}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Portfolio Value", f"${total_value:,.0f}")
with col2:
    st.metric("Cash Available", f"${port['current_cash']:,.0f}")
with col3:
    st.metric("Holdings", f"{len(summary['holdings'])}")
with col4:
    last_rebal = port.get("last_rebalanced_at", "Never")
    if last_rebal and last_rebal != "Never":
        last_rebal = last_rebal[:16]
    st.metric("Last Rebalance", last_rebal or "Never")

# ── Performance chart ────────────────────────────────────────────────────────
st.markdown("### Performance")
snapshots = db.get_snapshots(portfolio_id)
if snapshots:
    fig = portfolio_performance_line(snapshots)
    st.plotly_chart(fig, use_container_width=True, key="perf_line")
else:
    st.info("Performance data will appear after the first rebalance cycle.")

# ── Holdings table ───────────────────────────────────────────────────────────
st.markdown("### Holdings")
holdings = summary.get("holdings", [])
if holdings:
    import pandas as pd
    df = pd.DataFrame(holdings)
    display_cols = ["symbol", "company_name", "shares", "avg_cost_basis",
                    "current_price", "current_value", "gain_loss", "weight", "sector"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available_cols].style.format({
            "avg_cost_basis": "${:,.2f}",
            "current_price": "${:,.2f}",
            "current_value": "${:,.2f}",
            "gain_loss": "${:+,.2f}",
            "weight": "{:.1%}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No holdings yet. Run a rebalance to build your portfolio.")

# ── Allocation charts ────────────────────────────────────────────────────────
st.markdown("### Allocation")
col_sector, col_geo = st.columns(2)
with col_sector:
    fig = allocation_donut(summary.get("sector_allocation", {}), "Sector Allocation")
    st.plotly_chart(fig, use_container_width=True, key="sector_donut")
with col_geo:
    fig = allocation_donut(summary.get("geo_allocation", {}), "Geographic Allocation")
    st.plotly_chart(fig, use_container_width=True, key="geo_donut")

# ── Risk panel ───────────────────────────────────────────────────────────────
st.markdown("### Risk Assessment")
from src.investor.risk import compute_stress_score

stress = compute_stress_score(
    db.get_holdings(portfolio_id), total_value, investor_config
)
thresholds = investor_config.get("stress_thresholds", {}).get(
    port["risk_profile"], {"warning": 0.28, "action": 0.33}
)

col_gauge, col_details = st.columns([1, 1])
with col_gauge:
    fig = stress_gauge(stress["stress_score"], thresholds["warning"], thresholds["action"])
    st.plotly_chart(fig, use_container_width=True, key="stress_gauge")
with col_details:
    st.markdown("**Scenario Analysis**")
    for scenario, drawdown in stress.get("scenario_drawdowns", {}).items():
        label = scenario.replace("_", " ").title()
        st.markdown(f"- {label}: **{drawdown:+.1%}**")
    vuln = stress.get("vulnerable_sectors", [])
    if vuln:
        st.markdown(f"**Vulnerable sectors:** {', '.join(vuln)}")

# ── Bot controls ─────────────────────────────────────────────────────────────
st.markdown("---")
col_mode, col_rebal = st.columns([2, 1])
with col_mode:
    new_mode = st.selectbox(
        "Bot Mode",
        ["autopilot", "advisory"],
        index=0 if port["mode"] == "autopilot" else 1,
    )
    if new_mode != port["mode"]:
        db.update_portfolio(portfolio_id, mode=new_mode)
        st.success(f"Mode changed to {new_mode}")
        st.rerun()

with col_rebal:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Rebalance Now", type="primary", use_container_width=True):
        with st.spinner("Running rebalance cycle..."):
            from src.investor.rebalancer import run_rebalance
            result = run_rebalance(portfolio_id, trigger="manual")
        commentary = result.get("commentary", "Done.")
        proposed = result.get("trades_proposed", 0)
        executed = result.get("trades_executed", 0)
        st.success(f"Rebalance complete: {proposed} proposed, {executed} executed.")
        st.markdown(f'<div class="bot-commentary">{html_mod.escape(str(commentary))}</div>',
                   unsafe_allow_html=True)
        st.rerun()
