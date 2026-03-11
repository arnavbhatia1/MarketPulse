"""
Wrapper around yfinance for fetching financial data.

Used by the Alpha Engine to score tickers. Every function handles
None/NaN/empty data gracefully and never raises — failures are logged
as warnings and fallback values (None or empty dicts) are returned.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str) -> dict:
    """Fetch fundamental data for a single ticker.

    Returns a dict with keys: pe_ratio, ev_to_ebitda, price_to_book,
    dividend_yield, market_cap, market_cap_category, sector, industry.
    Any unavailable field is set to None.
    """
    result = {
        "pe_ratio": None,
        "ev_to_ebitda": None,
        "price_to_book": None,
        "dividend_yield": None,
        "market_cap": None,
        "market_cap_category": None,
        "sector": None,
        "industry": None,
    }

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception as exc:
        logger.warning("Failed to fetch fundamentals for %s: %s", symbol, exc)
        return result

    def _safe_float(key: str):
        val = info.get(key)
        if val is None:
            return None
        try:
            fval = float(val)
            return None if np.isnan(fval) else fval
        except (TypeError, ValueError):
            return None

    def _safe_int(key: str):
        val = info.get(key)
        if val is None:
            return None
        try:
            ival = int(val)
            return ival
        except (TypeError, ValueError):
            return None

    result["pe_ratio"] = _safe_float("trailingPE")
    result["ev_to_ebitda"] = _safe_float("enterpriseToEbitda")
    result["price_to_book"] = _safe_float("priceToBook")
    result["dividend_yield"] = _safe_float("dividendYield")
    result["market_cap"] = _safe_int("marketCap")
    result["sector"] = info.get("sector") or None
    result["industry"] = info.get("industry") or None

    mc = result["market_cap"]
    if mc is not None:
        if mc > 10_000_000_000:
            result["market_cap_category"] = "large"
        elif mc >= 2_000_000_000:
            result["market_cap_category"] = "mid"
        else:
            result["market_cap_category"] = "small"

    return result


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def get_price_history(symbol: str, period: str = "6mo") -> pd.DataFrame | None:
    """Return a DataFrame with Date index, Close, and Volume columns.

    Returns None if the download fails or yields no rows.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
    except Exception as exc:
        logger.warning("Failed to fetch price history for %s: %s", symbol, exc)
        return None

    if df is None or df.empty:
        logger.warning("No price history returned for %s (period=%s)", symbol, period)
        return None

    # Normalise columns — yfinance may return varying cases
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close")
    volume_col = cols.get("volume")

    if close_col is None:
        logger.warning("Price history for %s missing 'Close' column", symbol)
        return None

    out = pd.DataFrame(index=df.index)
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    if volume_col is not None:
        out["Volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        out["Volume"] = np.nan

    out.dropna(subset=["Close"], inplace=True)
    if out.empty:
        return None

    return out


# ---------------------------------------------------------------------------
# Current price
# ---------------------------------------------------------------------------

def get_current_price(symbol: str) -> float | None:
    """Get the latest closing price using a short lookback window."""
    df = get_price_history(symbol, period="5d")
    if df is None or df.empty:
        return None

    last_close = df["Close"].iloc[-1]
    try:
        val = float(last_close)
        return None if np.isnan(val) else val
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Momentum / technical signals
# ---------------------------------------------------------------------------

def get_momentum_signals(symbol: str, benchmark: str = "SPY") -> dict:
    """Compute momentum and risk signals from 6-month price history.

    Returns a dict with: price_momentum_30d, price_momentum_90d,
    volatility, relative_strength, max_drawdown.  Any unavailable
    metric is None.
    """
    result = {
        "price_momentum_30d": None,
        "price_momentum_90d": None,
        "volatility": None,
        "relative_strength": None,
        "max_drawdown": None,
    }

    df = get_price_history(symbol, period="6mo")
    if df is None or len(df) < 2:
        logger.warning("Insufficient price data for momentum signals: %s", symbol)
        return result

    close = df["Close"]
    daily_returns = close.pct_change().dropna()

    # -- Momentum ----------------------------------------------------------
    if len(close) >= 30:
        result["price_momentum_30d"] = _safe_return(close, 30)
    if len(close) >= 90:
        result["price_momentum_90d"] = _safe_return(close, 90)

    # -- Volatility (annualised) -------------------------------------------
    if len(daily_returns) >= 2:
        vol = daily_returns.std() * np.sqrt(252)
        result["volatility"] = None if np.isnan(vol) else float(vol)

    # -- Max drawdown ------------------------------------------------------
    cummax = close.cummax()
    drawdowns = (close - cummax) / cummax
    if not drawdowns.empty:
        mdd = float(drawdowns.min())
        result["max_drawdown"] = None if np.isnan(mdd) else mdd

    # -- Relative strength vs benchmark ------------------------------------
    ticker_90d = result["price_momentum_90d"]
    if ticker_90d is not None:
        bench_df = get_price_history(benchmark, period="6mo")
        if bench_df is not None and len(bench_df) >= 90:
            bench_90d = _safe_return(bench_df["Close"], 90)
            if bench_90d is not None:
                result["relative_strength"] = ticker_90d - bench_90d

    return result


def _safe_return(close: pd.Series, lookback: int) -> float | None:
    """Percentage return over the last *lookback* trading days."""
    if len(close) < lookback:
        return None
    recent = float(close.iloc[-1])
    past = float(close.iloc[-lookback])
    if past == 0 or np.isnan(past) or np.isnan(recent):
        return None
    return (recent - past) / past


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def get_batch_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch fundamentals for a list of symbols.

    Returns a dict keyed by symbol.  Symbols that fail are still
    included with all-None values so callers can rely on the key
    being present.
    """
    results: dict[str, dict] = {}
    for sym in symbols:
        logger.info("Fetching fundamentals for %s", sym)
        results[sym] = get_fundamentals(sym)
    return results


def get_sector_medians(fundamentals: dict[str, dict]) -> dict[str, dict]:
    """Compute median PE and EV/EBITDA per sector from batch fundamentals.

    Returns a dict like::

        {"Technology": {"median_pe": 25.0, "median_ev_ebitda": 18.0}, ...}

    Sectors with no valid data points are omitted.
    """
    sector_values: dict[str, dict[str, list[float]]] = {}

    for _sym, data in fundamentals.items():
        sector = data.get("sector")
        if not sector:
            continue

        if sector not in sector_values:
            sector_values[sector] = {"pe": [], "ev_ebitda": []}

        pe = data.get("pe_ratio")
        if pe is not None:
            sector_values[sector]["pe"].append(pe)

        ev = data.get("ev_to_ebitda")
        if ev is not None:
            sector_values[sector]["ev_ebitda"].append(ev)

    medians: dict[str, dict] = {}
    for sector, vals in sector_values.items():
        pe_list = vals["pe"]
        ev_list = vals["ev_ebitda"]

        median_pe = float(np.median(pe_list)) if pe_list else None
        median_ev = float(np.median(ev_list)) if ev_list else None

        if median_pe is not None or median_ev is not None:
            medians[sector] = {
                "median_pe": median_pe,
                "median_ev_ebitda": median_ev,
            }

    return medians
