"""MCP client for financial-mcp-server.

Connects to the financial-mcp-server over SSE and exposes sync wrappers
for 21 MCP tools used by the trading terminal. Connection lives in a
background thread. The call_tool() function accepts an optional timeout
parameter for long-running tools like run_rebalance.
"""

import json
import itertools
import logging
import os
import shutil
import subprocess
import threading
import queue
import time

from src.utils.config import load_config

logger = logging.getLogger(__name__)

_config = load_config()
_call_queue: queue.Queue = queue.Queue()
_result_queues: dict[int, queue.Queue] = {}
_call_counter = itertools.count()
_thread: threading.Thread | None = None
_connected = threading.Event()
_server_process: subprocess.Popen | None = None
_reconnect_attempts: int = 0
_last_reconnect_time: float = 0.0


def _server_already_listening() -> bool:
    """True if something is already accepting connections on the SSE port.

    Lets us skip launching a second server (e.g. when start.bat already started
    one), which would otherwise fail to bind 8520 and exit with an error.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(_config["mcp_server"]["url"])
    host, port = parsed.hostname or "localhost", parsed.port or 8520
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _start_mcp_server():
    """Launch financial-mcp server as a background subprocess if not already running."""
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        return  # already running
    if _server_already_listening():
        logger.info("financial-mcp already listening on %s — reusing it",
                    _config["mcp_server"]["url"])
        return

    cmd = shutil.which("financial-mcp")
    if cmd is None:
        # Check Python user Scripts directory (pip install --user)
        import sysconfig
        scripts_dir = sysconfig.get_path("scripts", "nt_user")
        if scripts_dir:
            candidate = os.path.join(scripts_dir, "financial-mcp.exe")
            if os.path.isfile(candidate):
                cmd = candidate
    if cmd is None:
        # Last resort: run as a Python module
        cmd = None  # will use sys.executable below
        logger.info("financial-mcp not on PATH, falling back to python -m financial_mcp.server")

    # financial-mcp defaults to stdio transport; MarketPulse talks to it over
    # SSE on port 8520, so the server MUST be launched in SSE mode.
    if cmd:
        launch_cmd = [cmd, "--transport", "sse"]
    else:
        import sys
        launch_cmd = [sys.executable, "-m", "financial_mcp.server", "--transport", "sse"]

    # Belt-and-suspenders: also force SSE via env in case an older build of the
    # server ignores the flag.
    env = {**os.environ, "FINANCIAL_MCP_TRANSPORT": "sse"}

    logger.info("Auto-starting financial-mcp server: %s", launch_cmd)
    _server_process = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    # Wait for the server to bind its port
    for _ in range(10):
        time.sleep(1)
        if _server_process.poll() is not None:
            logger.error("financial-mcp server exited with code %s", _server_process.returncode)
            _server_process = None
            return
        # Try a quick HTTP check to see if the server is up
        try:
            import urllib.request
            urllib.request.urlopen(_config["mcp_server"]["url"].replace("/sse", ""), timeout=1)
            break
        except Exception:
            continue
    logger.info("financial-mcp server started (pid=%s)", _server_process.pid)


def _run_mcp_loop(url: str):
    """Background thread: keeps SSE connection alive, processes tool calls."""
    import anyio
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession

    async def _loop():
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                _connected.set()
                while True:
                    call_id, tool_name, arguments = await anyio.to_thread.run_sync(
                        _call_queue.get
                    )
                    try:
                        result = await session.call_tool(tool_name, arguments=arguments)
                        parsed = json.loads(result.content[0].text)
                        _result_queues[call_id].put(("ok", parsed))
                    except Exception as e:
                        logger.error("MCP tool %s failed: %s", tool_name, e)
                        _result_queues[call_id].put(("error", {"error": str(e)}))

    anyio.run(_loop)


def _ensure_connected():
    """Start MCP server if needed, then connect via background thread.

    Uses exponential backoff between reconnection attempts to avoid
    hammering a dead server.
    """
    global _thread, _reconnect_attempts, _last_reconnect_time
    if _thread is not None and _thread.is_alive():
        return

    # Exponential backoff: 2s, 4s, 8s, ... capped at 60s
    now = time.time()
    if _reconnect_attempts > 0:
        backoff = min(2 ** _reconnect_attempts, 60)
        elapsed = now - _last_reconnect_time
        if elapsed < backoff:
            raise ConnectionError(
                f"MCP reconnect backoff: {backoff - elapsed:.0f}s remaining "
                f"(attempt {_reconnect_attempts})"
            )
    _last_reconnect_time = now
    _reconnect_attempts += 1

    _start_mcp_server()
    _connected.clear()
    url = _config["mcp_server"]["url"]
    _thread = threading.Thread(target=_run_mcp_loop, args=(url,), daemon=True)
    _thread.start()
    timeout = _config["mcp_server"]["timeout"]
    if not _connected.wait(timeout=timeout):
        raise ConnectionError("Could not connect to financial-mcp server")
    # Success — reset backoff
    _reconnect_attempts = 0


def call_tool(tool_name: str, timeout: float | None = None, **kwargs) -> dict:
    """Sync wrapper -- submits tool call to background thread, blocks for result."""
    try:
        _ensure_connected()
    except ConnectionError as e:
        logger.error("MCP connection failed: %s", e)
        return {"error": str(e)}
    call_id = next(_call_counter)
    _result_queues[call_id] = queue.Queue()
    _call_queue.put((call_id, tool_name, kwargs))
    t = timeout or _config["mcp_server"]["timeout"]
    try:
        status, result = _result_queues[call_id].get(timeout=t)
    except queue.Empty:
        del _result_queues[call_id]
        return {"error": f"Timeout after {t}s calling {tool_name}"}
    del _result_queues[call_id]
    return result


def is_connected() -> bool:
    """Check if MCP server connection is alive. Attempts to connect if not yet connected."""
    if _thread is not None and _thread.is_alive() and _connected.is_set():
        return True
    try:
        _ensure_connected()
        return True
    except ConnectionError:
        return False


# -- Scoring & Analysis --------------------------------------------------------

def score_ticker(symbol: str, sentiment: str = "") -> dict:
    """Score a ticker. `sentiment` is an optional JSON string (e.g.
    '{"score": 75}') that the engine blends into the composite score."""
    return call_tool("score_ticker", symbol=symbol, sentiment=sentiment)


def scan_universe(symbols: list[str]) -> dict:
    return call_tool("scan_universe", symbols=",".join(symbols))


def analyze_ticker(symbol: str) -> dict:
    return call_tool("analyze_ticker", symbol=symbol)


def get_fundamentals(symbol: str) -> dict:
    return call_tool("get_fundamentals", symbol=symbol)


def get_momentum(symbol: str) -> dict:
    return call_tool("get_momentum", symbol=symbol)


def get_price(symbol: str) -> dict:
    return call_tool("get_price", symbol=symbol)


# -- Portfolio & Trading -------------------------------------------------------

def create_portfolio(
    starting_capital: float,
    risk_profile: str,
    investment_horizon: str,
    name: str = "Default",
) -> dict:
    return call_tool(
        "create_portfolio",
        starting_capital=starting_capital,
        risk_profile=risk_profile,
        investment_horizon=investment_horizon,
        name=name,
    )


def analyze_portfolio(portfolio_id: str) -> dict:
    return call_tool("analyze_portfolio", portfolio_id=portfolio_id)


def get_holdings(portfolio_id: str) -> dict:
    return call_tool("get_holdings", portfolio_id=portfolio_id)


def get_trades(portfolio_id: str, status: str = "") -> dict:
    return call_tool("get_trades", portfolio_id=portfolio_id, status=status)


def execute_buy(portfolio_id: str, symbol: str, shares: int) -> dict:
    return call_tool(
        "execute_buy", portfolio_id=portfolio_id, symbol=symbol, shares=shares
    )


def execute_sell(portfolio_id: str, symbol: str, shares: int) -> dict:
    return call_tool(
        "execute_sell", portfolio_id=portfolio_id, symbol=symbol, shares=shares
    )


def run_rebalance(
    portfolio_id: str, trigger: str = "manual", symbols: str = ""
) -> dict:
    timeout = _config["mcp_server"].get("rebalance_timeout", 120)
    return call_tool(
        "run_rebalance",
        timeout=timeout,
        portfolio_id=portfolio_id,
        trigger=trigger,
        symbols=symbols,
    )


def check_risk(portfolio_id: str) -> dict:
    return call_tool("check_risk", portfolio_id=portfolio_id)


# -- Market Intelligence -------------------------------------------------------

def detect_market_regime() -> dict:
    return call_tool("detect_market_regime")


def get_vix_analysis() -> dict:
    return call_tool("get_vix_analysis")


def scan_anomalies(symbols: list[str] | None = None) -> dict:
    args = {}
    if symbols:
        args["symbols"] = ",".join(symbols)
    return call_tool("scan_anomalies", **args)


def scan_volume_leaders(symbols: list[str] | None = None) -> dict:
    args = {}
    if symbols:
        args["symbols"] = ",".join(symbols)
    return call_tool("scan_volume_leaders", **args)


def scan_gap_movers(symbols: list[str] | None = None) -> dict:
    args = {}
    if symbols:
        args["symbols"] = ",".join(symbols)
    return call_tool("scan_gap_movers", **args)


def get_smart_money_signal(market: str) -> dict:
    return call_tool("get_smart_money_signal", market=market)


# -- Catalysts (SEC / search trends) ------------------------------------------

def get_sec_filings(symbol: str, filing_type: str = "8-K", count: int = 5) -> dict:
    return call_tool("get_sec_filings", symbol=symbol, filing_type=filing_type, count=count)


def get_insider_trades(symbol: str, days: int = 90) -> dict:
    return call_tool("get_insider_trades", symbol=symbol, days=days)


def get_search_trends(keywords: str, timeframe: str = "today 3-m") -> dict:
    return call_tool("get_search_trends", keywords=keywords, timeframe=timeframe)


def get_futures_positioning(market: str) -> dict:
    return call_tool("get_futures_positioning", market=market)
