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

from src.investor.bot_engine import (
    _STATE_PATH,
    _compute_trade_stats,
    STARTING_CAPITAL,
    build_seed_trade_log,
    build_seed_equity_curve,
)


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
    trade_log = build_seed_trade_log()
    total_pnl = round(sum(t["pnl"] for t in trade_log if t["action"] == "SELL"), 2)
    equity_curve = build_seed_equity_curve(trade_log)

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
