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


def _start_mcp_server():
    """Launch financial-mcp server as a background subprocess if not already running."""
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        return  # already running

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

    if cmd:
        launch_cmd = [cmd]
    else:
        import sys
        launch_cmd = [sys.executable, "-m", "financial_mcp.server"]

    logger.info("Auto-starting financial-mcp server: %s", launch_cmd)
    _server_process = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    """Start MCP server if needed, then connect via background thread."""
    global _thread
    if _thread is None or not _thread.is_alive():
        _start_mcp_server()
        _connected.clear()
        url = _config["mcp_server"]["url"]
        _thread = threading.Thread(target=_run_mcp_loop, args=(url,), daemon=True)
        _thread.start()
        timeout = _config["mcp_server"]["timeout"]
        if not _connected.wait(timeout=timeout):
            raise ConnectionError("Could not connect to financial-mcp server")


def call_tool(tool_name: str, timeout: float | None = None, **kwargs) -> dict:
    """Sync wrapper -- submits tool call to background thread, blocks for result."""
    _ensure_connected()
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

def score_ticker(symbol: str) -> dict:
    return call_tool("score_ticker", symbol=symbol)


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


def get_futures_positioning(market: str) -> dict:
    return call_tool("get_futures_positioning", market=market)
