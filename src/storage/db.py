"""
SQLite storage layer for MarketPulse.

Single .db file at data/marketpulse.db — free, persistent, no cloud required.
All reads/writes go through this module. No direct sqlite3 calls elsewhere.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "marketpulse.db"
)


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_json(x, fallback):
    if not x:
        return fallback
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return fallback


def init_db():
    """Create tables if they don't exist. Safe to call repeatedly."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                post_id    TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                source     TEXT NOT NULL,
                timestamp  TEXT,
                author     TEXT,
                score      INTEGER DEFAULT 0,
                tickers    TEXT DEFAULT '[]',
                sentiment  TEXT,
                confidence REAL DEFAULT 0.0,
                url        TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS ticker_cache (
                ticker                TEXT PRIMARY KEY,
                symbol                TEXT,
                last_updated          TEXT,
                dominant_sentiment    TEXT,
                mention_count         INTEGER DEFAULT 0,
                avg_confidence        REAL DEFAULT 0.0,
                reddit_sentiment      TEXT,
                news_sentiment        TEXT,
                stocktwits_sentiment  TEXT,
                sentiment_by_day      TEXT DEFAULT '{}',
                top_posts             TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS model_training_log (
                run_id       TEXT PRIMARY KEY,
                trained_at   TEXT,
                num_samples  INTEGER,
                weighted_f1  REAL,
                label_source TEXT DEFAULT 'keyword_majority'
            );

            CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_premium    INTEGER DEFAULT 0,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id       TEXT PRIMARY KEY,
                user_id            TEXT NOT NULL REFERENCES users(user_id),
                name               TEXT DEFAULT 'My Portfolio',
                starting_capital   REAL NOT NULL,
                current_cash       REAL NOT NULL,
                risk_profile       TEXT NOT NULL,
                investment_horizon TEXT NOT NULL,
                mode               TEXT DEFAULT 'autopilot',
                is_active          INTEGER DEFAULT 1,
                created_at         TEXT NOT NULL,
                last_rebalanced_at TEXT
            );

            CREATE TABLE IF NOT EXISTS holdings (
                id            TEXT PRIMARY KEY,
                portfolio_id  TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                symbol        TEXT NOT NULL,
                company_name  TEXT,
                shares        REAL NOT NULL,
                avg_cost_basis REAL NOT NULL,
                asset_type    TEXT NOT NULL,
                sector        TEXT,
                geography     TEXT,
                acquired_at   TEXT NOT NULL,
                UNIQUE(portfolio_id, symbol)
            );

            CREATE TABLE IF NOT EXISTS trades (
                trade_id      TEXT PRIMARY KEY,
                portfolio_id  TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                symbol        TEXT NOT NULL,
                action        TEXT NOT NULL,
                shares        REAL NOT NULL,
                price         REAL NOT NULL,
                total_value   REAL NOT NULL,
                formula_score REAL,
                reason        TEXT,
                claude_review TEXT,
                status        TEXT NOT NULL,
                trigger       TEXT,
                proposed_at   TEXT NOT NULL,
                executed_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id                TEXT PRIMARY KEY,
                portfolio_id      TEXT NOT NULL REFERENCES portfolios(portfolio_id),
                snapshot_date     TEXT NOT NULL,
                total_value       REAL NOT NULL,
                cash_value        REAL NOT NULL,
                holdings_value    REAL NOT NULL,
                daily_return      REAL,
                cumulative_return REAL,
                benchmark_return  REAL,
                sharpe_ratio      REAL,
                max_drawdown      REAL,
                sector_allocation TEXT,
                geo_allocation    TEXT,
                stress_score      REAL,
                UNIQUE(portfolio_id, snapshot_date)
            );

            CREATE TABLE IF NOT EXISTS etf_universe (
                symbol       TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                category     TEXT NOT NULL,
                sector       TEXT,
                geography    TEXT,
                expense_ratio REAL NOT NULL,
                description  TEXT
            );
        """)
        conn.commit()
        _seed_etf_universe(conn)
    finally:
        conn.close()


def _seed_etf_universe(conn):
    """Seed ETF universe table if empty. Idempotent."""
    count = conn.execute("SELECT COUNT(*) FROM etf_universe").fetchone()[0]
    if count > 0:
        return

    etfs = [
        # Broad US Market
        ("VOO", "Vanguard S&P 500", "broad_market", None, "us", 0.0003, "Core US large cap"),
        ("VTI", "Vanguard Total Stock Market", "broad_market", None, "us", 0.0003, "Full US market"),
        ("SPY", "SPDR S&P 500", "broad_market", None, "us", 0.0009, "Benchmark / liquidity"),
        ("IVV", "iShares Core S&P 500", "broad_market", None, "us", 0.0003, "S&P 500 alternative"),
        # Size & Style
        ("VV", "Vanguard Large Cap", "broad_market", None, "us", 0.0004, "Large cap core"),
        ("VO", "Vanguard Mid Cap", "broad_market", None, "us", 0.0004, "Mid cap exposure"),
        ("VB", "Vanguard Small Cap", "broad_market", None, "us", 0.0005, "Small cap"),
        ("VXF", "Vanguard Extended Market", "broad_market", None, "us", 0.0006, "Mid + small ex-S&P 500"),
        ("MGK", "Vanguard Mega Cap Growth", "broad_market", None, "us", 0.0007, "Growth tilt"),
        ("MGV", "Vanguard Mega Cap Value", "broad_market", None, "us", 0.0007, "Value tilt"),
        ("VBR", "Vanguard Small Cap Value", "broad_market", None, "us", 0.0007, "Small value factor"),
        ("VOE", "Vanguard Mid Cap Value", "broad_market", None, "us", 0.0007, "Mid value factor"),
        # Dividend / Income
        ("VIG", "Vanguard Dividend Appreciation", "dividend", None, "us", 0.0006, "Dividend growth"),
        ("VYM", "Vanguard High Dividend Yield", "dividend", None, "us", 0.0006, "High yield income"),
        ("SCHD", "Schwab US Dividend", "dividend", None, "us", 0.0006, "Quality dividend"),
        # International
        ("VEA", "Vanguard Developed Markets", "international", None, "intl_developed", 0.0005, "Intl developed"),
        ("VWO", "Vanguard Emerging Markets", "international", None, "emerging", 0.0008, "Emerging markets"),
        ("VXUS", "Vanguard Total International", "international", None, "intl_developed", 0.0007, "All ex-US"),
        ("VT", "Vanguard Total World", "international", None, "global", 0.0007, "Global single-fund"),
        ("SCHF", "Schwab International Equity", "international", None, "intl_developed", 0.0006, "Developed alt"),
        ("IEMG", "iShares Core Emerging", "international", None, "emerging", 0.0009, "Emerging alt"),
        # Sector (SPDR Select)
        ("XLK", "Technology Select Sector SPDR", "sector", "Technology", "us", 0.0009, "Technology sector"),
        ("XLF", "Financial Select Sector SPDR", "sector", "Financials", "us", 0.0009, "Financials sector"),
        ("XLV", "Health Care Select Sector SPDR", "sector", "Healthcare", "us", 0.0009, "Healthcare sector"),
        ("XLE", "Energy Select Sector SPDR", "sector", "Energy", "us", 0.0009, "Energy sector"),
        ("XLI", "Industrial Select Sector SPDR", "sector", "Industrials", "us", 0.0009, "Industrials sector"),
        ("XLP", "Consumer Staples Select Sector SPDR", "sector", "Consumer Staples", "us", 0.0009, "Consumer Staples sector"),
        ("XLU", "Utilities Select Sector SPDR", "sector", "Utilities", "us", 0.0009, "Utilities sector"),
        ("XLC", "Communication Services Select Sector SPDR", "sector", "Communication Services", "us", 0.0009, "Communication Services sector"),
        ("XLRE", "Real Estate Select Sector SPDR", "sector", "Real Estate", "us", 0.0009, "Real Estate sector"),
        ("XLB", "Materials Select Sector SPDR", "sector", "Materials", "us", 0.0009, "Materials sector"),
        ("XLY", "Consumer Discretionary Select Sector SPDR", "sector", "Consumer Discretionary", "us", 0.0009, "Consumer Discretionary sector"),
        # Fixed Income
        ("BND", "Vanguard Total Bond", "bond", "Bonds/Fixed Income", "us", 0.0003, "Core bond hedge"),
        ("BNDX", "Vanguard International Bond", "bond", "Bonds/Fixed Income", "intl_developed", 0.0007, "Intl fixed income"),
        ("VGSH", "Vanguard Short-Term Treasury", "bond", "Bonds/Fixed Income", "us", 0.0004, "Cash-like safety"),
        ("VGIT", "Vanguard Intermediate Treasury", "bond", "Bonds/Fixed Income", "us", 0.0004, "Duration hedge"),
        ("VGLT", "Vanguard Long-Term Treasury", "bond", "Bonds/Fixed Income", "us", 0.0004, "Rate-sensitive hedge"),
        ("VTIP", "Vanguard TIPS", "bond", "Bonds/Fixed Income", "us", 0.0004, "Inflation protection"),
        ("AGG", "iShares Core Aggregate Bond", "bond", "Bonds/Fixed Income", "us", 0.0003, "Bond alternative"),
    ]
    conn.executemany(
        "INSERT INTO etf_universe (symbol, name, category, sector, geography, expense_ratio, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        etfs,
    )
    conn.commit()


def save_posts(df: pd.DataFrame):
    """Upsert posts DataFrame into SQLite. Safe to call multiple times."""
    if df is None or df.empty:
        return
    conn = get_connection()
    try:
        for _, row in df.iterrows():
            tickers = row.get('tickers', [])
            if not isinstance(tickers, list):
                try:
                    tickers = list(tickers)
                except TypeError:
                    tickers = []
            conn.execute(
                """INSERT OR REPLACE INTO posts
                   (post_id, text, source, timestamp, author, score,
                    tickers, sentiment, confidence, url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(row['post_id']),
                    str(row['text']),
                    str(row.get('source', 'unknown')),
                    str(row.get('timestamp', '')),
                    str(row.get('author', 'unknown')),
                    int(row.get('score', 0)),
                    json.dumps(tickers),
                    row.get('sentiment'),
                    float(row.get('confidence', 0.0)),
                    str(row.get('url', '')),
                )
            )
        conn.commit()
    finally:
        conn.close()


def load_posts(start_date=None, end_date=None) -> pd.DataFrame:
    """Load posts, optionally filtered by date range (YYYY-MM-DD strings)."""
    conn = get_connection()
    try:
        if start_date and end_date:
            rows = conn.execute(
                "SELECT * FROM posts WHERE timestamp >= ? AND timestamp <= ?",
                (str(start_date), str(end_date) + " 23:59:59")
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM posts").fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            'post_id', 'text', 'source', 'timestamp', 'author',
            'score', 'tickers', 'sentiment', 'confidence', 'url'
        ])

    df = pd.DataFrame([dict(r) for r in rows])
    df['tickers'] = df['tickers'].apply(
        lambda x: _safe_json(x, [])
    )
    return df


def save_ticker_cache(ticker_results: dict):
    """Upsert ticker_results dict into ticker_cache table."""
    conn = get_connection()
    try:
        now = datetime.utcnow().isoformat()
        for company, data in ticker_results.items():
            conn.execute(
                """INSERT OR REPLACE INTO ticker_cache
                   (ticker, symbol, last_updated, dominant_sentiment, mention_count,
                    avg_confidence, reddit_sentiment, news_sentiment,
                    stocktwits_sentiment, sentiment_by_day, top_posts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    company,
                    data.get('symbol', ''),
                    now,
                    data.get('dominant_sentiment', 'neutral'),
                    int(data.get('mention_count', 0)),
                    float(data.get('avg_confidence', 0.0)),
                    data.get('reddit_sentiment', 'neutral'),
                    data.get('news_sentiment', 'neutral'),
                    data.get('stocktwits_sentiment', 'neutral'),
                    json.dumps(data.get('sentiment_by_day', {})),
                    json.dumps(data.get('top_posts', {})),
                )
            )
        conn.commit()
    finally:
        conn.close()


def load_ticker_cache() -> dict:
    """Return all rows from ticker_cache as dict keyed by company name."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM ticker_cache").fetchall()
    finally:
        conn.close()
    result = {}
    for row in rows:
        d = dict(row)
        d['sentiment_by_day'] = _safe_json(d.get('sentiment_by_day'), {})
        d['top_posts'] = _safe_json(d.get('top_posts'), {})
        d['company'] = d['ticker']  # canonical company name == ticker key
        result[d['ticker']] = d
    return result


def log_training_run(run_id: str, num_samples: int, weighted_f1: float,
                     label_source: str = 'keyword_majority'):
    """Record a model training run."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO model_training_log
               (run_id, trained_at, num_samples, weighted_f1, label_source)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, datetime.utcnow().isoformat(), num_samples, weighted_f1, label_source)
        )
        conn.commit()
    finally:
        conn.close()


def get_training_history() -> list:
    """Return all training runs as list of dicts, newest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM model_training_log ORDER BY trained_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── User management ──────────────────────────────────────────────────

def create_user(email: str, password_hash: str) -> str:
    """Create a new user. Returns user_id. Raises on duplicate email."""
    user_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO users (user_id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, email, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return user_id


def get_user_by_email(email: str) -> dict | None:
    """Return user dict or None."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


# ── Portfolio management ─────────────────────────────────────────────

def create_portfolio(user_id: str, starting_capital: float, risk_profile: str,
                     investment_horizon: str, name: str = "My Portfolio") -> str:
    """Create a new portfolio. Returns portfolio_id."""
    portfolio_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """INSERT INTO portfolios
               (portfolio_id, user_id, name, starting_capital, current_cash,
                risk_profile, investment_horizon, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (portfolio_id, user_id, name, starting_capital, starting_capital,
             risk_profile, investment_horizon, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return portfolio_id


def get_portfolio(portfolio_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_user_portfolios(user_id: str) -> list:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM portfolios WHERE user_id = ? AND is_active = 1 ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def update_portfolio(portfolio_id: str, **kwargs):
    """Update portfolio fields. Pass column=value pairs."""
    if not kwargs:
        return
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE portfolios SET {set_clause} WHERE portfolio_id = ?",
            (*kwargs.values(), portfolio_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Holdings ─────────────────────────────────────────────────────────

def upsert_holding(portfolio_id: str, symbol: str, shares: float, avg_cost_basis: float,
                   asset_type: str, company_name: str = None, sector: str = None,
                   geography: str = None):
    """Insert or update a holding."""
    holding_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """INSERT INTO holdings
               (id, portfolio_id, symbol, company_name, shares, avg_cost_basis,
                asset_type, sector, geography, acquired_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(portfolio_id, symbol) DO UPDATE SET
                 shares = excluded.shares,
                 avg_cost_basis = excluded.avg_cost_basis,
                 sector = excluded.sector,
                 geography = excluded.geography""",
            (holding_id, portfolio_id, symbol, company_name, shares, avg_cost_basis,
             asset_type, sector, geography, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_holdings(portfolio_id: str) -> list:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM holdings WHERE portfolio_id = ? ORDER BY symbol",
            (portfolio_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def delete_holding(portfolio_id: str, symbol: str):
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM holdings WHERE portfolio_id = ? AND symbol = ?",
            (portfolio_id, symbol),
        )
        conn.commit()
    finally:
        conn.close()


# ── Trades ───────────────────────────────────────────────────────────

def save_trade(trade: dict):
    """Save a trade record."""
    conn = get_connection()
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """INSERT OR REPLACE INTO trades
               (trade_id, portfolio_id, symbol, action, shares, price, total_value,
                formula_score, reason, claude_review, status, trigger, proposed_at, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade['trade_id'], trade['portfolio_id'], trade['symbol'],
                trade['action'], trade['shares'], trade['price'], trade['total_value'],
                trade.get('formula_score'), trade.get('reason'), trade.get('claude_review'),
                trade['status'], trade.get('trigger'), trade['proposed_at'],
                trade.get('executed_at'),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_trades(portfolio_id: str, status: str = None) -> list:
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM trades WHERE portfolio_id = ? AND status = ? ORDER BY proposed_at DESC",
                (portfolio_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE portfolio_id = ? ORDER BY proposed_at DESC",
                (portfolio_id,),
            ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def update_trade(trade_id: str, **kwargs):
    if not kwargs:
        return
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
            (*kwargs.values(), trade_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Snapshots ────────────────────────────────────────────────────────

def save_snapshot(snapshot: dict):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO portfolio_snapshots
               (id, portfolio_id, snapshot_date, total_value, cash_value, holdings_value,
                daily_return, cumulative_return, benchmark_return, sharpe_ratio, max_drawdown,
                sector_allocation, geo_allocation, stress_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.get('id', str(uuid.uuid4())), snapshot['portfolio_id'],
                snapshot['snapshot_date'], snapshot['total_value'], snapshot['cash_value'],
                snapshot['holdings_value'], snapshot.get('daily_return'),
                snapshot.get('cumulative_return'), snapshot.get('benchmark_return'),
                snapshot.get('sharpe_ratio'), snapshot.get('max_drawdown'),
                json.dumps(snapshot.get('sector_allocation', {})),
                json.dumps(snapshot.get('geo_allocation', {})),
                snapshot.get('stress_score'),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_snapshots(portfolio_id: str, limit: int = 365) -> list:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE portfolio_id = ? ORDER BY snapshot_date DESC LIMIT ?",
            (portfolio_id, limit),
        ).fetchall()
    finally:
        conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['sector_allocation'] = _safe_json(d.get('sector_allocation'), {})
        d['geo_allocation'] = _safe_json(d.get('geo_allocation'), {})
        results.append(d)
    return results


# ── ETF Universe ─────────────────────────────────────────────────────

def get_etf_universe() -> list:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM etf_universe ORDER BY symbol").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
