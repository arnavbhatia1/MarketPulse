# tests/test_bot_engine.py
"""Tests for autonomous scalp trading bot engine."""
import threading       # used by Tasks 2-5
import time as _time   # used by Tasks 2-5
import pytest
from unittest.mock import patch  # used by Tasks 2-5


class TestGetState:
    def test_returns_same_instance(self):
        from src.investor.bot_engine import get_state
        assert get_state() is get_state()

    def test_initial_values(self):
        from src.investor.bot_engine import get_state
        s = get_state()
        assert s.portfolio_cash == 10_000.0
        assert s.portfolio_value == 10_000.0
        assert s.open_positions == {}
        assert s.pending_sells == set()
        assert s.trade_log == []
        assert s.cycle_count == 0
        assert s.portfolio_id is None


class TestGetAllocationPct:
    def test_tier_90_to_100(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(95, False) == pytest.approx(0.12)

    def test_tier_70_to_89(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(75, False) == pytest.approx(0.08)

    def test_tier_60_to_69(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(65, False) == pytest.approx(0.05)

    def test_below_60_returns_zero(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(59, False) == 0.0

    def test_high_vix_halves_all_tiers(self):
        from src.investor.bot_engine import _get_allocation_pct
        assert _get_allocation_pct(95, True) == pytest.approx(0.06)
        assert _get_allocation_pct(75, True) == pytest.approx(0.04)
        assert _get_allocation_pct(65, True) == pytest.approx(0.025)


class TestGetCompositeScore:
    def test_extracts_score(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"score": {"score": 82.5}}) == pytest.approx(82.5)

    def test_returns_zero_on_error_key(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"error": "not found"}) == 0.0

    def test_returns_zero_when_missing(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({}) == 0.0

    def test_returns_zero_when_score_is_none(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"score": {"score": None}}) == 0.0

    def test_returns_zero_when_score_is_zero(self):
        from src.investor.bot_engine import _get_composite_score
        assert _get_composite_score({"score": {"score": 0}}) == 0.0


def _reset_state():
    """Reset module-level state between tests."""
    from src.investor import bot_engine
    bot_engine._state.is_running = False
    bot_engine._state.portfolio_id = None
    bot_engine._state.cycle_count = 0
    bot_engine._state.open_positions = {}
    bot_engine._state.pending_sells = set()
    bot_engine._state.trade_log = []
    bot_engine._engine._stop_event.set()
    bot_engine._engine._thread = None


class TestBotEngine:
    def setup_method(self):
        _reset_state()

    def teardown_method(self):
        from src.investor import bot_engine
        bot_engine.get_engine().stop()
        # Wait for loop thread to exit (up to 2s) rather than a fixed sleep
        t = bot_engine._engine._thread
        if t is not None:
            t.join(timeout=2.0)
        bot_engine._engine._thread = None

    def test_get_engine_returns_same_instance(self):
        from src.investor.bot_engine import get_engine
        assert get_engine() is get_engine()

    def test_not_running_initially(self):
        from src.investor.bot_engine import get_engine
        assert get_engine().is_running() is False

    @patch("src.investor.bot_engine.create_portfolio",
           return_value={"portfolio_id": "test-pid"})
    @patch("src.investor.bot_engine._run_cycle")
    def test_start_sets_running(self, mock_cycle, mock_create):
        from src.investor.bot_engine import get_engine, get_state
        get_engine().start()
        _time.sleep(0.15)
        assert get_state().is_running is True

    @patch("src.investor.bot_engine.create_portfolio",
           return_value={"portfolio_id": "test-pid"})
    @patch("src.investor.bot_engine._run_cycle")
    def test_stop_clears_running(self, mock_cycle, mock_create):
        from src.investor.bot_engine import get_engine, get_state
        get_engine().start()
        _time.sleep(0.1)
        get_engine().stop()
        _time.sleep(0.2)
        # stop() directly sets is_running=False under _lock, so this passes
        # immediately — it does not wait for the _loop thread to exit.
        assert get_state().is_running is False

    @patch("src.investor.bot_engine.create_portfolio",
           return_value={"error": "server down"})
    def test_start_aborts_when_portfolio_creation_fails(self, mock_create):
        from src.investor.bot_engine import get_engine, get_state
        get_engine().start()
        _time.sleep(0.1)
        assert get_state().is_running is False

    @patch("src.investor.bot_engine.create_portfolio")
    @patch("src.investor.bot_engine._run_cycle")
    def test_start_reuses_existing_portfolio_id(self, mock_cycle, mock_create):
        from src.investor import bot_engine
        bot_engine._state.portfolio_id = "already-set"
        bot_engine.get_engine().start()
        _time.sleep(0.1)
        assert mock_create.call_count == 0
        assert bot_engine._state.portfolio_id == "already-set"
