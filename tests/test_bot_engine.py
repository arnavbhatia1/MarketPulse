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


class TestCheckVix:
    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": 35, "vix_signal": "fear"})
    def test_returns_true_when_vix_above_30(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is True

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": 18, "vix_signal": "normal"})
    def test_returns_false_when_vix_normal(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": "N/A"})
    def test_returns_false_when_vix_non_numeric(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"error": "timeout"})
    def test_returns_false_on_error(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False

    @patch("src.investor.bot_engine.get_vix_analysis",
           return_value={"vix": 30})
    def test_returns_false_when_vix_exactly_30(self, mock):
        from src.investor.bot_engine import _check_vix
        assert _check_vix() is False


class TestCheckExits:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 148.0, "score": {"score": 55.0}})
    def test_exits_when_score_drops_30_percent(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["AAPL"] = {
            "entry_price": 150.0, "shares": 10, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 150.0, "current_score": 80.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "AAPL" not in bot_engine._state.open_positions
        assert bot_engine._state.trade_log[0]["action"] == "SELL"
        assert "signal reversal" in bot_engine._state.trade_log[0]["reason"]

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 148.0, "score": {"score": 35.0}})
    def test_exits_when_score_below_absolute_threshold(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["MSFT"] = {
            "entry_price": 300.0, "shares": 5, "entry_score": 50.0,
            "entry_time": datetime.now(), "current_price": 300.0, "current_score": 50.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "MSFT" not in bot_engine._state.open_positions

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 155.0, "score": {"score": 75.0}})
    def test_no_exit_when_score_holds(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["NVDA"] = {
            "entry_price": 800.0, "shares": 2, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 800.0, "current_score": 80.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "NVDA" in bot_engine._state.open_positions
        mock_sell.assert_not_called()

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"error": "network error"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 148.0, "score": {"score": 30.0}})
    def test_failed_sell_added_to_pending_sells(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["TSLA"] = {
            "entry_price": 200.0, "shares": 3, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 200.0, "current_score": 80.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "TSLA" in bot_engine._state.pending_sells
        assert "TSLA" in bot_engine._state.open_positions  # not removed on fail

    @patch("src.investor.bot_engine.execute_sell")
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"error": "timeout"})
    def test_skips_exit_when_score_is_zero(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["AMD"] = {
            "entry_price": 120.0, "shares": 5, "entry_score": 75.0,
            "entry_time": datetime.now(), "current_price": 120.0, "current_score": 75.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "AMD" in bot_engine._state.open_positions  # position kept
        mock_sell.assert_not_called()  # no sell attempted


class TestScanCandidates:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.scan_volume_leaders", return_value={"leaders": []})
    @patch("src.investor.bot_engine.scan_anomalies", return_value={"anomalies": []})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"scores": [{"symbol": "AAPL"}, {"symbol": "MSFT"}]})
    def test_returns_symbols_from_universe(self, mock_u, mock_a, mock_v):
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert "AAPL" in result
        assert "MSFT" in result

    @patch("src.investor.bot_engine.scan_volume_leaders",
           return_value={"leaders": [{"symbol": "AAPL"}]})
    @patch("src.investor.bot_engine.scan_anomalies",
           return_value={"anomalies": [{"symbol": "AAPL"}]})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"scores": [{"symbol": "AAPL"}, {"symbol": "MSFT"}]})
    def test_deduplicates_across_sources(self, mock_u, mock_a, mock_v):
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert result.count("AAPL") == 1

    @patch("src.investor.bot_engine.scan_volume_leaders", return_value={"leaders": []})
    @patch("src.investor.bot_engine.scan_anomalies", return_value={"anomalies": []})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"scores": [{"symbol": "AAPL"}, {"symbol": "HELD"}]})
    def test_filters_out_held_positions(self, mock_u, mock_a, mock_v):
        from src.investor import bot_engine
        bot_engine._state.open_positions["HELD"] = {}
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert "HELD" not in result
        assert "AAPL" in result

    @patch("src.investor.bot_engine.scan_volume_leaders",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.scan_anomalies",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.scan_universe",
           return_value={"error": "timeout"})
    def test_returns_empty_when_all_scans_fail(self, mock_u, mock_a, mock_v):
        from src.investor.bot_engine import _scan_candidates
        assert _scan_candidates(threading.Event()) == []


class TestScoreCandidates:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 150.0, "score": {"score": 75.0}})
    def test_returns_scored_candidates_above_min(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["AAPL"], threading.Event())
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["score"] == pytest.approx(75.0)
        assert result[0]["price"] == pytest.approx(150.0)

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 45.0}})
    def test_filters_out_score_below_60(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["WEAK"], threading.Event())
        assert result == []

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"score": {"score": 85.0}})   # no "price" key
    def test_skips_when_price_unavailable(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["NOPRICE"], threading.Event())
        assert result == []

    @patch("src.investor.bot_engine.analyze_ticker", side_effect=[
        {"price": 100.0, "score": {"score": 65.0}},
        {"price": 200.0, "score": {"score": 90.0}},
    ])
    def test_sorted_descending_by_score(self, mock_analyze):
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["LOW", "HIGH"], threading.Event())
        assert result[0]["score"] > result[1]["score"]


class TestEnterPositions:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_enters_position_and_logs_buy(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 10_000.0
        from src.investor.bot_engine import _enter_positions
        _enter_positions("pid", [{"ticker": "AAPL", "score": 85, "price": 100.0}],
                         False, threading.Event())
        assert "AAPL" in bot_engine._state.open_positions
        assert bot_engine._state.trade_log[0]["action"] == "BUY"
        # remaining_cash starts at 10_000, score=85 → tier 70-89 → 8% → 800 → 8 shares @ $100
        mock_buy.assert_called_once_with("pid", "AAPL", 8)

    @patch("src.investor.bot_engine._sell_position", return_value=False)
    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_does_not_exceed_max_positions(self, mock_buy, mock_sell):
        from src.investor import bot_engine
        from src.investor.bot_engine import _enter_positions, MAX_POSITIONS
        bot_engine._state.portfolio_cash = 100_000.0
        # Fill to max with positions scoring 80 — candidates at 85 won't trigger
        # rotation (need 10+ point gap, but 85 - 80 = 5)
        for i in range(MAX_POSITIONS):
            bot_engine._state.open_positions[f"HELD{i}"] = {
                "entry_score": 80, "current_score": 80, "entry_price": 100.0,
                "shares": 10, "current_price": 100.0,
            }
        candidates = [{"ticker": f"NEW{i}", "score": 85, "price": 100.0} for i in range(3)]
        _enter_positions("pid", candidates, False, threading.Event())
        mock_buy.assert_not_called()

    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_high_vix_halves_share_count(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 10_000.0
        from src.investor.bot_engine import _enter_positions
        _enter_positions("pid", [{"ticker": "AAPL", "score": 85, "price": 100.0}],
                         True, threading.Event())
        # high_vix: 0.08 * 0.5 = 0.04 → 10000 * 0.04 / 100 = 4 shares
        mock_buy.assert_called_once_with("pid", "AAPL", 4)

    @patch("src.investor.bot_engine.execute_buy", return_value={"error": "rejected"})
    def test_failed_buy_not_added_to_positions(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 10_000.0
        from src.investor.bot_engine import _enter_positions
        _enter_positions("pid", [{"ticker": "FAIL", "score": 85, "price": 100.0}],
                         False, threading.Event())
        assert "FAIL" not in bot_engine._state.open_positions


class TestSnapshotPortfolio:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.analyze_portfolio", return_value={
        "total_value": 10_500.0,
        "portfolio": {"current_cash": 8_000.0},
    })
    def test_updates_cash_and_value(self, mock_analyze):
        from src.investor import bot_engine
        from src.investor.bot_engine import _snapshot_portfolio
        _snapshot_portfolio("pid")
        assert bot_engine._state.portfolio_cash == pytest.approx(8_000.0)
        assert bot_engine._state.portfolio_value == pytest.approx(10_500.0)
        assert bot_engine._state.total_pnl == pytest.approx(500.0)

    @patch("src.investor.bot_engine.analyze_portfolio",
           return_value={"error": "not found"})
    def test_keeps_previous_values_on_failure(self, mock_analyze):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 5_000.0
        bot_engine._state.portfolio_value = 11_000.0
        from src.investor.bot_engine import _snapshot_portfolio
        _snapshot_portfolio("pid")
        assert bot_engine._state.portfolio_cash == pytest.approx(5_000.0)
        assert bot_engine._state.portfolio_value == pytest.approx(11_000.0)
