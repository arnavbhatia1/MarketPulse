"""Background auto-refresh for MarketPulse.

Runs the RSS → sentiment pipeline on a timer in a daemon thread so the dashboard
always shows fresh data without anyone clicking "Refresh Data". Mirrors the
trading bot's background-thread pattern: a module-level singleton guarded by a
lock, safe to call from Streamlit's repeated script reruns.
"""
import logging
import threading
import time
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 180  # refresh every 3 minutes
LOOKBACK_DAYS = 7

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()
_status = {
    "enabled": False,
    "running": False,          # a refresh is in progress right now
    "last_run": None,          # datetime of last successful refresh
    "last_error": None,
    "total_posts": 0,
    "labeled_posts": 0,
    "label_coverage": 0.0,
    "ticker_count": 0,
    "sources": [],
    "interval_sec": DEFAULT_INTERVAL_SEC,
}


def get_status() -> dict:
    """Return a copy of the current auto-refresh status."""
    with _lock:
        return dict(_status)


def _run_once() -> None:
    """Run one pipeline refresh over the rolling lookback window."""
    # Imported lazily so importing this module never triggers the heavy pipeline.
    from app.pipeline_runner import refresh_pipeline, get_ticker_cache

    with _lock:
        _status["running"] = True
    try:
        today = date.today()
        summary = refresh_pipeline(
            start_date_str=(today - timedelta(days=LOOKBACK_DAYS)).isoformat(),
            end_date_str=today.isoformat(),
        )
        with _lock:
            _status["last_run"] = datetime.now()
            _status["last_error"] = None
            _status["total_posts"] = summary.get("total_posts", 0)
            _status["labeled_posts"] = summary.get("labeled_posts", 0)
            _status["label_coverage"] = summary.get("label_coverage", 0.0)
            _status["ticker_count"] = summary.get("ticker_count", 0)
            _status["sources"] = summary.get("sources_used", [])
        # Drop the cached read so the UI sees fresh data on its next rerun.
        try:
            get_ticker_cache.clear()
        except Exception:
            pass
        logger.info("Auto-refresh complete: %s posts, %s tickers",
                    _status["total_posts"], _status["ticker_count"])
    except Exception as e:
        logger.exception("Auto-refresh failed")
        with _lock:
            _status["last_error"] = str(e)
    finally:
        with _lock:
            _status["running"] = False


def _loop(interval_sec: int) -> None:
    # Refresh immediately on first launch if we've never run, then on the timer.
    while not _stop.is_set():
        with _lock:
            never_run = _status["last_run"] is None
        if never_run:
            _run_once()
        # Sleep in 1s slices so stop() is responsive.
        for _ in range(interval_sec):
            if _stop.is_set():
                return
            time.sleep(1)
        if not _stop.is_set():
            _run_once()


def start_auto_refresh(interval_sec: int = DEFAULT_INTERVAL_SEC) -> None:
    """Start the background refresh loop once per process (idempotent)."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _status["enabled"] = True
        _status["interval_sec"] = interval_sec
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(interval_sec,), daemon=True)
    _thread.start()
    logger.info("Auto-refresh started (every %ds)", interval_sec)


def trigger_now() -> None:
    """Force an immediate refresh in the background (e.g. a manual button)."""
    threading.Thread(target=_run_once, daemon=True).start()


def stop_auto_refresh() -> None:
    """Stop the background refresh loop."""
    _stop.set()
    with _lock:
        _status["enabled"] = False
