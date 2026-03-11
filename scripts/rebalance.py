#!/usr/bin/env python3
"""
CLI script to run rebalance cycle for all active portfolios.

Usage:
    python scripts/rebalance.py              # rebalance all active portfolios
    python scripts/rebalance.py --portfolio <id>  # rebalance one portfolio

Designed to be called by cron:
    0 9 * * 1 cd /app && python scripts/rebalance.py
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import init_db
from src.investor.rebalancer import run_rebalance, run_rebalance_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run portfolio rebalance cycle")
    parser.add_argument("--portfolio", type=str, help="Specific portfolio ID to rebalance")
    args = parser.parse_args()

    init_db()

    if args.portfolio:
        logger.info("Rebalancing portfolio: %s", args.portfolio)
        result = run_rebalance(args.portfolio, trigger="manual")
        logger.info("Result: %s", result)
    else:
        logger.info("Rebalancing all active portfolios...")
        results = run_rebalance_all(trigger="weekly_rebalance")
        for r in results:
            logger.info("Portfolio %s: proposed=%s executed=%s",
                       r.get("portfolio_id", "?"),
                       r.get("trades_proposed", "?"),
                       r.get("trades_executed", "?"))
        logger.info("Done. %d portfolios processed.", len(results))


if __name__ == "__main__":
    main()
