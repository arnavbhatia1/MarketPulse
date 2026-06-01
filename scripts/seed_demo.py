"""Warm-start the trading bot for a demo recording.

Writes ``data/bot_state.json`` so the Trading Bot page opens with a populated
Edge Statistics panel (win rate, EV, Kelly, risk-of-ruin) and a full Activity
Log instead of an empty bot. The bot loads this ledger on start and keeps
trading the same portfolio live.

The closed trades below are *illustrative* demo history — a realistic round of
paper trades so the math panel has something to display on camera. Once the
live bot runs, real trades append to and eventually dominate this log.

Usage:
    python scripts/seed_demo.py           # create a real MCP portfolio + seed history
    python scripts/seed_demo.py --reset    # delete the seed file (start cold)

The financial-mcp server must be reachable (SSE on :8520) for the real
portfolio to be created; if it isn't, the seed still writes (the live bot will
create a fresh portfolio on start).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.investor.bot_engine import _STATE_PATH, _compute_trade_stats, STARTING_CAPITAL

# (ticker, shares, entry_price, exit_price, sell_reason)
# Sized so each position is <=2% of $10k — matches the bot's real 2% risk cap.
_ROUND_TRIPS = [
    ("NIO", 33, 5.99, 6.35, "take profit +6.0% (score fading 88→70)"),
    ("SOFI", 10, 18.69, 17.95, "signal reversal 74→52"),
    ("GME", 9, 21.38, 22.70, "take profit +6.2% (score fading 86→69)"),
    ("KO", 2, 78.73, 81.10, "trailing stop -2.9% from peak $83.50→$81.10"),
    ("INTC", 1, 111.11, 108.50, "hard stop -2.3% (score reversal 81→44)"),
    ("CSCO", 1, 121.03, 124.80, "take profit +3.1% (score fading 79→63)"),
    ("WMT", 1, 114.16, 116.90, "take profit +2.4% (score fading 77→62)"),
    ("COIN", 1, 186.39, 179.20, "hard stop -3.9% (limit -5%)"),
    ("DIS", 1, 102.35, 105.60, "take profit +3.2% (score fading 82→66)"),
    ("PLTR", 1, 163.16, 170.40, "take profit +4.4% (score fading 90→71)"),
    ("AMD", 1, 119.10, 116.00, "signal reversal 72→51"),
    ("F", 25, 7.40, 7.18, "trailing stop -3.0% from peak $7.62→$7.18"),
    ("SNAP", 18, 10.20, 10.75, "take profit +5.4% (score fading 84→69)"),
    ("HOOD", 8, 24.80, 26.10, "take profit +5.2% (score fading 80→70)"),
]


def _clock(minute: int) -> str:
    """Minutes after 09:00 ET → HH:MM, rolling hours over correctly."""
    return f"{9 + minute // 60:02d}:{minute % 60:02d}"


def _build_trade_log() -> list:
    """Build a newest-first trade log of BUY/SELL pairs from the round trips."""
    log = []
    minute = 31  # start at 09:31 ET
    for i, (ticker, shares, entry, exit_, reason) in enumerate(_ROUND_TRIPS):
        buy_time = _clock(minute)
        minute += 3
        sell_time = _clock(minute)
        minute += 4
        entry_score = 70 + (i * 2) % 25
        pnl = round((exit_ - entry) * shares, 2)
        risk_pct = round(shares * entry / STARTING_CAPITAL * 100, 1)
        # Newest-first: insert SELL then BUY so BUY ends up just below its SELL
        log.insert(0, {
            "time": buy_time, "action": "BUY", "ticker": ticker,
            "price": entry, "shares": shares, "score": entry_score,
            "reason": f"score {entry_score} · {risk_pct}% risk", "pnl": 0.0,
        })
        log.insert(0, {
            "time": sell_time, "action": "SELL", "ticker": ticker,
            "price": exit_, "shares": shares, "score": max(40, entry_score - 12),
            "reason": reason, "pnl": pnl,
        })
    return log


def _create_portfolio_id() -> str | None:
    """Create a real MCP portfolio so the live bot trades on it. None on failure."""
    try:
        from src.investor.mcp_client import create_portfolio
        result = create_portfolio(
            starting_capital=STARTING_CAPITAL,
            risk_profile="aggressive",
            investment_horizon="short",
            name="ScalpBot",
        )
        if "error" not in result and result.get("portfolio_id"):
            return result["portfolio_id"]
        print(f"  ! Could not create MCP portfolio ({result.get('error', 'unknown')}).")
        print("    Live bot will create a fresh portfolio on start.")
    except Exception as e:
        print(f"  ! MCP unreachable ({e}). Live bot will create a portfolio on start.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start the trading bot for a demo.")
    parser.add_argument("--reset", action="store_true", help="Delete the seed file.")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(_STATE_PATH):
            os.remove(_STATE_PATH)
            print(f"Removed {_STATE_PATH} — bot will start cold.")
        else:
            print("No seed file to remove — bot already starts cold.")
        return

    print("Seeding demo bot state...")
    portfolio_id = _create_portfolio_id()
    trade_log = _build_trade_log()
    total_pnl = round(sum(t["pnl"] for t in trade_log if t["action"] == "SELL"), 2)

    # Equity curve: cumulative realized P&L over time (oldest → newest).
    equity_curve = [{"time": "09:30", "value": STARTING_CAPITAL}]
    running = STARTING_CAPITAL
    for t in reversed(trade_log):
        if t["action"] == "SELL":
            running = round(running + t["pnl"], 2)
            equity_curve.append({"time": t["time"], "value": running})

    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump({
            "portfolio_id": portfolio_id,
            "total_pnl": total_pnl,
            "trade_log": trade_log,
            "equity_curve": equity_curve,
        }, f, indent=2)

    stats = _compute_trade_stats(trade_log)
    print(f"\nWrote {_STATE_PATH}")
    print(f"  Portfolio: {portfolio_id or '(fresh on start)'}")
    print(f"  Closed trades : {stats.total_trades}  ({stats.wins}W / {stats.losses}L)")
    print(f"  Win rate      : {stats.win_rate * 100:.0f}%")
    print(f"  Expected value: ${stats.expected_value:+.2f} / trade")
    print(f"  R:R ratio     : {stats.reward_risk_ratio:.2f}")
    print(f"  Half-Kelly    : {stats.kelly_fraction * 100:.1f}%")
    print(f"  Risk of ruin  : {stats.risk_of_ruin * 100:.3f}%")
    print(f"  Realized P&L  : ${total_pnl:+.2f}")
    print("\nStart the dashboard (start.bat) and open Trading Bot — the Edge panel is warm.")
    print("Run `python scripts/seed_demo.py --reset` to clear it.")


if __name__ == "__main__":
    main()
