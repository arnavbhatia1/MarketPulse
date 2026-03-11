"""Tests for the investor bot modules."""

import pytest
import os


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Isolated test database."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("src.storage.db.DB_PATH", db_file)
    from src.storage import db
    db.init_db()
    return db


@pytest.fixture
def test_user(tmp_db):
    """Create a test user."""
    user_id = tmp_db.create_user("test@example.com", "hashed_pw")
    return tmp_db.get_user_by_id(user_id)


@pytest.fixture
def test_portfolio(tmp_db, test_user):
    """Create a test portfolio."""
    pid = tmp_db.create_portfolio(
        test_user["user_id"], 100000.0, "moderate", "5yr"
    )
    return tmp_db.get_portfolio(pid)


@pytest.fixture
def sample_holdings(tmp_db, test_portfolio):
    """Create sample holdings."""
    pid = test_portfolio["portfolio_id"]
    tmp_db.upsert_holding(pid, "AAPL", 50, 150.0, "stock", "Apple", "Technology", "us")
    tmp_db.upsert_holding(pid, "VOO", 30, 400.0, "etf", "Vanguard S&P 500", None, "us")
    tmp_db.upsert_holding(pid, "VEA", 100, 45.0, "etf", "Vanguard Developed", None, "intl_developed")
    return tmp_db.get_holdings(pid)


# ── DB Schema Tests ──────────────────────────────────────────────────────────

class TestDBSchema:
    def test_etf_universe_seeded(self, tmp_db):
        etfs = tmp_db.get_etf_universe()
        assert len(etfs) == 39

    def test_etf_universe_idempotent(self, tmp_db):
        """Calling init_db again should not duplicate ETFs."""
        tmp_db.init_db()
        etfs = tmp_db.get_etf_universe()
        assert len(etfs) == 39

    def test_create_user(self, tmp_db):
        uid = tmp_db.create_user("new@test.com", "hashval")
        user = tmp_db.get_user_by_id(uid)
        assert user["email"] == "new@test.com"
        assert user["is_premium"] == 0

    def test_duplicate_email_raises(self, tmp_db):
        tmp_db.create_user("dup@test.com", "hash1")
        with pytest.raises(Exception):
            tmp_db.create_user("dup@test.com", "hash2")

    def test_get_user_by_email(self, tmp_db):
        tmp_db.create_user("find@test.com", "hash")
        user = tmp_db.get_user_by_email("find@test.com")
        assert user is not None
        assert user["email"] == "find@test.com"

    def test_get_user_not_found(self, tmp_db):
        assert tmp_db.get_user_by_email("nope@test.com") is None

    def test_create_portfolio(self, tmp_db, test_user):
        pid = tmp_db.create_portfolio(test_user["user_id"], 50000, "aggressive", "10yr+")
        port = tmp_db.get_portfolio(pid)
        assert port["starting_capital"] == 50000
        assert port["current_cash"] == 50000
        assert port["risk_profile"] == "aggressive"

    def test_get_user_portfolios(self, tmp_db, test_user):
        tmp_db.create_portfolio(test_user["user_id"], 10000, "conservative", "1yr")
        portfolios = tmp_db.get_user_portfolios(test_user["user_id"])
        assert len(portfolios) >= 1

    def test_update_portfolio(self, tmp_db, test_portfolio):
        pid = test_portfolio["portfolio_id"]
        tmp_db.update_portfolio(pid, current_cash=50000.0, mode="advisory")
        updated = tmp_db.get_portfolio(pid)
        assert updated["current_cash"] == 50000.0
        assert updated["mode"] == "advisory"

    def test_holdings_crud(self, tmp_db, test_portfolio):
        pid = test_portfolio["portfolio_id"]
        tmp_db.upsert_holding(pid, "TSLA", 10, 200.0, "stock")
        holdings = tmp_db.get_holdings(pid)
        assert len(holdings) == 1
        assert holdings[0]["symbol"] == "TSLA"
        assert holdings[0]["shares"] == 10

        # Update shares
        tmp_db.upsert_holding(pid, "TSLA", 20, 210.0, "stock")
        holdings = tmp_db.get_holdings(pid)
        assert len(holdings) == 1
        assert holdings[0]["shares"] == 20

        # Delete
        tmp_db.delete_holding(pid, "TSLA")
        holdings = tmp_db.get_holdings(pid)
        assert len(holdings) == 0

    def test_trades_crud(self, tmp_db, test_portfolio):
        pid = test_portfolio["portfolio_id"]
        trade = {
            "trade_id": "test-trade-1",
            "portfolio_id": pid,
            "symbol": "AAPL",
            "action": "buy",
            "shares": 10,
            "price": 150.0,
            "total_value": 1500.0,
            "status": "executed",
            "trigger": "manual",
            "proposed_at": "2026-03-11T10:00:00",
            "executed_at": "2026-03-11T10:00:01",
        }
        tmp_db.save_trade(trade)
        trades = tmp_db.get_trades(pid)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "AAPL"

        tmp_db.update_trade("test-trade-1", status="rejected")
        trades = tmp_db.get_trades(pid, status="rejected")
        assert len(trades) == 1

    def test_snapshots_crud(self, tmp_db, test_portfolio):
        pid = test_portfolio["portfolio_id"]
        tmp_db.save_snapshot({
            "portfolio_id": pid,
            "snapshot_date": "2026-03-11",
            "total_value": 100000,
            "cash_value": 50000,
            "holdings_value": 50000,
            "daily_return": 0.01,
            "cumulative_return": 0.05,
            "sector_allocation": {"Technology": 0.3},
            "geo_allocation": {"us": 0.8},
        })
        snaps = tmp_db.get_snapshots(pid)
        assert len(snaps) == 1
        assert snaps[0]["sector_allocation"] == {"Technology": 0.3}


# ── Auth Tests ───────────────────────────────────────────────────────────────

class TestAuth:
    def test_hash_and_verify(self):
        from src.auth.auth import hash_password, verify_password
        h = hash_password("mypassword")
        assert verify_password("mypassword", h)
        assert not verify_password("wrongpassword", h)

    def test_register_and_login(self, tmp_db):
        from src.auth.auth import register_user, login_user
        user = register_user("auth@test.com", "securepass")
        assert user["email"] == "auth@test.com"

        logged_in = login_user("auth@test.com", "securepass")
        assert logged_in is not None
        assert logged_in["email"] == "auth@test.com"

    def test_login_wrong_password(self, tmp_db):
        from src.auth.auth import register_user, login_user
        register_user("wrong@test.com", "correct")
        result = login_user("wrong@test.com", "incorrect")
        assert result is None

    def test_login_nonexistent_user(self, tmp_db):
        from src.auth.auth import login_user
        result = login_user("ghost@test.com", "pass")
        assert result is None

    def test_register_duplicate_raises(self, tmp_db):
        from src.auth.auth import register_user
        register_user("dup@test.com", "pass1")
        with pytest.raises(ValueError, match="already exists"):
            register_user("dup@test.com", "pass2")

    def test_register_invalid_email(self, tmp_db):
        from src.auth.auth import register_user
        with pytest.raises(ValueError, match="Invalid email"):
            register_user("notanemail", "pass")


# ── Engine Tests ─────────────────────────────────────────────────────────────

class TestEngine:
    def test_normalize(self):
        from src.investor.engine import normalize
        assert normalize(0.5, 0, 1) == 0.5
        assert normalize(0, 0, 1) == 0.0
        assert normalize(1, 0, 1) == 1.0
        assert normalize(2, 0, 1) == 1.0  # clamp
        assert normalize(-1, 0, 1) == 0.0  # clamp

    def test_percentile_rank(self):
        from src.investor.engine import percentile_rank
        values = [10, 20, 30, 40, 50]
        assert percentile_rank(50, values) == 100.0
        assert percentile_rank(10, values) == 20.0
        assert percentile_rank(30, values) == 60.0

    def test_sentiment_composite(self):
        from src.investor.engine import compute_sentiment_composite
        ticker_data = {
            "dominant_sentiment": "bullish",
            "avg_confidence": 0.8,
            "reddit_sentiment": "bullish",
            "stocktwits_sentiment": "bullish",
            "news_sentiment": "neutral",
            "sentiment_by_day": {
                "2026-03-05": "neutral",
                "2026-03-06": "neutral",
                "2026-03-07": "bullish",
                "2026-03-08": "bullish",
                "2026-03-09": "bullish",
                "2026-03-10": "bullish",
                "2026-03-11": "bullish",
            },
        }
        score = compute_sentiment_composite(ticker_data)
        assert score is not None
        assert 0 <= score <= 100
        assert score > 50  # bullish should score high

    def test_sentiment_composite_none(self):
        from src.investor.engine import compute_sentiment_composite
        assert compute_sentiment_composite(None) is None
        assert compute_sentiment_composite({}) is None

    def test_score_ticker_all_data(self):
        from src.investor.engine import score_ticker
        result = score_ticker(
            symbol="AAPL",
            sentiment_data={
                "dominant_sentiment": "bullish",
                "avg_confidence": 0.7,
                "reddit_sentiment": "bullish",
                "stocktwits_sentiment": "neutral",
                "news_sentiment": "bullish",
                "sentiment_by_day": {},
            },
            fundamentals={
                "pe_ratio": 25.0,
                "ev_to_ebitda": 20.0,
                "price_to_book": 8.0,
                "dividend_yield": 0.006,
                "market_cap_category": "large",
                "sector": "Technology",
            },
            momentum={
                "price_momentum_30d": 0.05,
                "price_momentum_90d": 0.12,
                "volatility": 0.25,
                "relative_strength": 0.03,
            },
            all_momentum=[{
                "price_momentum_30d": 0.05,
                "price_momentum_90d": 0.12,
                "volatility": 0.25,
                "relative_strength": 0.03,
            }],
            sector_medians={"Technology": {"median_pe": 30.0, "median_ev_ebitda": 22.0}},
            holdings=[],
            portfolio_value=100000,
            risk_profile="moderate",
            config={"position_limits": {"moderate": {"max_sector": 0.30}}},
        )
        assert "score" in result
        assert 0 <= result["score"] <= 100

    def test_score_ticker_missing_data(self):
        from src.investor.engine import score_ticker
        result = score_ticker(
            symbol="VEA",
            sentiment_data=None,
            fundamentals=None,
            momentum={"price_momentum_30d": 0.02, "price_momentum_90d": 0.05,
                      "volatility": 0.15, "relative_strength": -0.01},
            all_momentum=[{"price_momentum_30d": 0.02, "price_momentum_90d": 0.05,
                          "volatility": 0.15, "relative_strength": -0.01}],
            sector_medians={},
            holdings=[],
            portfolio_value=100000,
            risk_profile="moderate",
            config={"position_limits": {"moderate": {"max_sector": 0.30}}},
        )
        assert "score" in result
        assert 0 <= result["score"] <= 100


# ── Risk Tests ───────────────────────────────────────────────────────────────

class TestRisk:
    def test_sector_sensitivity_complete(self):
        from src.investor.risk import SECTOR_SENSITIVITY
        assert "Technology" in SECTOR_SENSITIVITY
        assert "Utilities" in SECTOR_SENSITIVITY
        assert "Bonds/Fixed Income" in SECTOR_SENSITIVITY

    def test_check_position_limits_allowed(self):
        from src.investor.risk import check_position_limits
        config = {"position_limits": {"moderate": {
            "max_position": 0.08, "max_sector": 0.30,
            "min_cash": 0.10, "max_stocks": 0.70,
        }}}
        result = check_position_limits(
            symbol="AAPL", proposed_value=5000,
            holdings=[], portfolio_value=100000,
            risk_profile="moderate", config=config,
        )
        assert result["allowed"] is True

    def test_check_position_limits_blocked(self):
        from src.investor.risk import check_position_limits
        config = {"position_limits": {"moderate": {
            "max_position": 0.08, "max_sector": 0.30,
            "min_cash": 0.10, "max_stocks": 0.70,
        }}}
        result = check_position_limits(
            symbol="AAPL", proposed_value=50000,  # 50% of portfolio
            holdings=[], portfolio_value=100000,
            risk_profile="moderate", config=config,
        )
        assert result["allowed"] is False
        assert len(result["violations"]) > 0

    def test_compute_stress_score(self):
        from src.investor.risk import compute_stress_score
        holdings = [
            {"symbol": "AAPL", "sector": "Technology", "shares": 10,
             "avg_cost_basis": 150, "asset_type": "stock", "geography": "us"},
            {"symbol": "BND", "sector": "Bonds/Fixed Income", "shares": 50,
             "avg_cost_basis": 80, "asset_type": "etf", "geography": "us"},
        ]
        config = {"stress_scenarios": {
            "recession_2008": -0.38, "covid_2020": -0.34, "rate_hike_2022": -0.25,
        }}
        result = compute_stress_score(holdings, 100000, config)
        assert "stress_score" in result
        assert 0 <= result["stress_score"] <= 1

    def test_get_sector_allocation(self):
        from src.investor.risk import get_sector_allocation
        holdings = [
            {"symbol": "AAPL", "sector": "Technology", "shares": 10,
             "avg_cost_basis": 100, "asset_type": "stock"},
            {"symbol": "JPM", "sector": "Financials", "shares": 20,
             "avg_cost_basis": 50, "asset_type": "stock"},
        ]
        alloc = get_sector_allocation(holdings, 2000)
        assert "Technology" in alloc
        assert "Financials" in alloc
        assert abs(sum(alloc.values()) - 1.0) < 0.01


# ── Broker Tests ─────────────────────────────────────────────────────────────

class TestBroker:
    def test_trade_result_dataclass(self):
        from src.investor.broker import TradeResult
        tr = TradeResult(
            trade_id="t1", symbol="AAPL", action="buy",
            shares=10, price=150.0, total_value=1500.0,
            executed_at="2026-03-11", success=True,
        )
        assert tr.success is True
        assert tr.total_value == 1500.0

    def test_broker_is_abstract(self):
        from src.investor.broker import Broker
        with pytest.raises(TypeError):
            Broker()


# ── Reviewer Tests ───────────────────────────────────────────────────────────

class TestReviewer:
    def test_empty_trades_returns_empty(self):
        from src.agent.reviewer import review_trades
        result = review_trades([], {}, {}, {})
        assert result["decisions"] == []

    def test_fallback_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from src.agent.reviewer import review_trades
        trades = [{"trade_id": "t1", "symbol": "AAPL", "action": "buy",
                   "shares": 10, "price": 150, "total_value": 1500}]
        result = review_trades(trades, {}, {}, {})
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["action"] == "APPROVE"

    def test_parse_review_valid_json(self):
        from src.agent.reviewer import _parse_review
        raw = '{"decisions": [{"trade_id": "t1", "action": "APPROVE", "reasoning": "good"}], "portfolio_commentary": "ok"}'
        result = _parse_review(raw, [{"trade_id": "t1"}])
        assert result["decisions"][0]["action"] == "APPROVE"

    def test_parse_review_malformed(self):
        from src.agent.reviewer import _parse_review
        trades = [{"trade_id": "t1"}]
        result = _parse_review("not json at all", trades)
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["action"] == "APPROVE"  # fallback
