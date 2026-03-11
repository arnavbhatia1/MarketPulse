"""
Trade Activity — Premium feature.

Shows pending trades (advisory mode), trade history, and bot commentary.
"""

import html as html_mod
import streamlit as st
import sys, os
from datetime import datetime

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.components.styles import apply_theme, COLORS

st.set_page_config(page_title="Trades | MarketPulse", page_icon="M", layout="wide")
apply_theme()

# ── Auth gate ────────────────────────────────────────────────────────────────
from app.components.auth_guard import require_auth
require_auth(premium=True)

from src.storage import db

user_id = st.session_state.get("user_id")
portfolios = db.get_user_portfolios(user_id)

if not portfolios:
    st.info("Create a portfolio first on the Portfolio page.")
    st.stop()

portfolio_id = portfolios[0]["portfolio_id"]
port = portfolios[0]

st.markdown("## Trade Activity")

# ── Pending trades (advisory mode) ──────────────────────────────────────────
pending = db.get_trades(portfolio_id, status="approved")
if pending:
    st.markdown("### Pending Approval")
    st.markdown(f"*{len(pending)} trades awaiting your review*")

    for trade in pending:
        action = trade["action"]
        card_class = "trade-card-buy" if action == "buy" else "trade-card-sell"
        action_color = COLORS["bullish"] if action == "buy" else COLORS["bearish"]
        safe_symbol = html_mod.escape(str(trade["symbol"]))
        safe_reason = html_mod.escape(str(trade.get("reason", "")))
        safe_review = html_mod.escape(str(trade.get("claude_review", "")))

        st.markdown(f"""
        <div class="trade-card {card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:{action_color}; font-weight:bold; font-size:1.1em;">
                        {action.upper()}
                    </span>
                    <span style="font-size:1.1em; font-weight:bold; margin-left:8px;">{safe_symbol}</span>
                    <span style="color:#8B949E; margin-left:8px;">
                        {trade['shares']:.0f} shares @ ${trade['price']:,.2f}
                        (${trade['total_value']:,.0f})
                    </span>
                </div>
                <span style="color:#8B949E;">Score: {trade.get('formula_score', 'N/A')}</span>
            </div>
            <div style="color:#8B949E; font-size:0.9em; margin-top:8px;">
                <strong>Formula:</strong> {safe_reason}
            </div>
            <div style="color:#E6EDF3; font-size:0.9em; margin-top:4px;">
                <strong>AI Review:</strong> {safe_review}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button("Approve", key=f"approve_{trade['trade_id']}", type="primary"):
                from src.investor.broker import PaperBroker
                broker = PaperBroker(portfolio_id)
                if action == "buy":
                    result = broker.execute_buy(trade["symbol"], trade["shares"])
                else:
                    result = broker.execute_sell(trade["symbol"], trade["shares"])
                if result.success:
                    db.update_trade(trade["trade_id"], status="executed",
                                  executed_at=result.executed_at)
                    st.success(f"Executed: {action} {trade['symbol']}")
                else:
                    st.error(f"Failed: {result.error}")
                st.rerun()
        with col_reject:
            if st.button("Reject", key=f"reject_{trade['trade_id']}"):
                db.update_trade(trade["trade_id"], status="rejected")
                st.info(f"Rejected: {action} {trade['symbol']}")
                st.rerun()

    st.markdown("---")

# ── Latest bot commentary ────────────────────────────────────────────────────
all_trades = db.get_trades(portfolio_id)
latest_reviewed = [t for t in all_trades if t.get("claude_review") and
                   t["claude_review"] != "Auto-approved"]
if latest_reviewed:
    latest = latest_reviewed[0]
    st.markdown("### Bot Commentary")
    safe_commentary = html_mod.escape(str(latest.get("claude_review", "")))
    st.markdown(
        f'<div class="bot-commentary">{safe_commentary}</div>',
        unsafe_allow_html=True,
    )

# ── Trade history ────────────────────────────────────────────────────────────
st.markdown("### Trade History")

# Filter
status_filter = st.selectbox(
    "Filter by status",
    ["All", "executed", "proposed", "approved", "rejected", "overridden"],
    index=0,
)

if status_filter == "All":
    trades = all_trades
else:
    trades = [t for t in all_trades if t["status"] == status_filter]

if trades:
    import pandas as pd
    df = pd.DataFrame(trades)
    display_cols = ["proposed_at", "action", "symbol", "shares", "price",
                    "total_value", "formula_score", "status", "trigger"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].style.format({
            "price": "${:,.2f}",
            "total_value": "${:,.0f}",
            "formula_score": "{:.0f}",
            "shares": "{:.0f}",
        }, na_rep="--"),
        use_container_width=True, hide_index=True,
    )

    # Download CSV
    csv = df[available].to_csv(index=False)
    st.download_button("Download CSV", csv, "trades.csv", "text/csv")
else:
    st.info("No trades yet. Run a rebalance from the Portfolio page to generate trades.")
