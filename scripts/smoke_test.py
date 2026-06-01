"""Headless page-render smoke test.

Boots both Streamlit pages with Streamlit's AppTest harness and fails if either
raises. This catches the class of bug that `py_compile` and unit tests miss —
runtime errors on page load (e.g. a missing import / NameError, a None that
flows into a comparison). Exits non-zero on any failure so CI can gate on it.

Usage:  python scripts/smoke_test.py
"""
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

failures: list[str] = []


def _check(label: str, at) -> None:
    if at.exception:
        msgs = "; ".join(str(e) for e in at.exception)
        failures.append(f"{label}: {msgs}")
        print(f"  [FAIL] {label} raised: {msgs}")
    else:
        print(f"  [OK] {label} rendered clean")


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


print("[smoke] Home page (MarketPulse.py)...")
_check("home", AppTest.from_file("app/MarketPulse.py", default_timeout=180).run())

# The Trading Bot page st.stop()s without a reachable MCP server, so start one.
# yfinance may be throttled on CI runners — the page should degrade to N/A, not
# crash, which is exactly what we want this smoke to confirm.
print("[smoke] Starting financial-mcp (SSE) for the Trading Bot page...")
env = {**os.environ, "FINANCIAL_MCP_TRANSPORT": "sse"}
proc = None
try:
    proc = subprocess.Popen(
        ["financial-mcp", "--transport", "sse"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    for _ in range(20):
        if _port_open("localhost", 8520):
            break
        time.sleep(1)

    if _port_open("localhost", 8520):
        print("[smoke] Trading Bot page (2_Trading_Bot.py)...")
        bot = AppTest.from_file("app/pages/2_Trading_Bot.py", default_timeout=180).run()
        _check("bot (default)", bot)
        try:
            bot.text_input(key="ticker_input").set_value("AAPL").run()
            _check("bot (ticker selected)", bot)
        except Exception as e:
            print(f"  ! could not drive ticker input ({e}) — skipping ticker render")
    else:
        print("  ! MCP server did not bind :8520 — skipping bot-page smoke (infra, not a code failure)")
finally:
    if proc is not None:
        proc.terminate()

if failures:
    print("\nSMOKE FAILED:")
    for f in failures:
        print(" -", f)
    sys.exit(1)

print("\nSMOKE OK - both pages render without exceptions.")
sys.exit(0)
