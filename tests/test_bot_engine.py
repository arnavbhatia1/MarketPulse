# tests/test_bot_engine.py
"""Tests for autonomous scalp trading bot engine."""
import threading
import time as _time
import pytest
from unittest.mock import patch


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
