"""Browser-impersonating yfinance helper for the dashboard's OWN yfinance calls.

The candlestick and price-line charts call yfinance directly from the Streamlit
process (separate from the MCP server). On cloud/datacenter IPs Yahoo throttles
the default bot-like fingerprint, which shows up as "No chart data". Routing
through a curl_cffi Chrome session mirrors the MCP server fix so charts render
on cloud hosts too.
"""
import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def _make_session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception as e:  # pragma: no cover - optional dep
        logger.warning("curl_cffi unavailable (%s); charts may be throttled on cloud", e)
        return None


_SESSION = _make_session()


def yf_download(*args, **kwargs):
    """yf.download routed through the impersonating session when available."""
    if _SESSION is not None and "session" not in kwargs:
        try:
            return yf.download(*args, session=_SESSION, **kwargs)
        except TypeError:
            pass
    return yf.download(*args, **kwargs)
