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


class TestComputeTradeStats:
    def test_empty_log_returns_defaults(self):
        from src.investor.bot_engine import _compute_trade_stats
        stats = _compute_trade_stats([])
        assert stats.total_trades == 0
        assert stats.expected_value == 0

    def test_positive_ev_from_winning_trades(self):
        from src.investor.bot_engine import _compute_trade_stats
        log = [
            {"action": "SELL", "pnl": 100}, {"action": "SELL", "pnl": 50},
            {"action": "SELL", "pnl": -30}, {"action": "SELL", "pnl": 80},
            {"action": "SELL", "pnl": -20}, {"action": "SELL", "pnl": 60},
            {"action": "SELL", "pnl": 90},  {"action": "SELL", "pnl": -40},
            {"action": "SELL", "pnl": 70},  {"action": "SELL", "pnl": 50},
            {"action": "BUY", "pnl": 0},  # BUYs should be ignored
        ]
        stats = _compute_trade_stats(log)
        assert stats.total_trades == 10
        assert stats.wins == 7
        assert stats.losses == 3
        assert stats.win_rate == pytest.approx(0.7)
        assert stats.expected_value > 0
        assert stats.has_edge is True

    def test_negative_ev_detected(self):
        from src.investor.bot_engine import _compute_trade_stats
        log = [
            {"action": "SELL", "pnl": -100}, {"action": "SELL", "pnl": -80},
            {"action": "SELL", "pnl": 20},   {"action": "SELL", "pnl": -90},
            {"action": "SELL", "pnl": -70},  {"action": "SELL", "pnl": 10},
            {"action": "SELL", "pnl": -60},  {"action": "SELL", "pnl": -50},
            {"action": "SELL", "pnl": 15},   {"action": "SELL", "pnl": -40},
        ]
        stats = _compute_trade_stats(log)
        assert stats.expected_value < 0
        assert stats.has_edge is False

    def test_kelly_fraction_positive_for_edge(self):
        from src.investor.bot_engine import _compute_trade_stats
        # 70% win rate, 2:1 reward/risk → strong Kelly
        log = []
        for _ in range(7):
            log.append({"action": "SELL", "pnl": 200})
        for _ in range(3):
            log.append({"action": "SELL", "pnl": -100})
        stats = _compute_trade_stats(log)
        assert stats.kelly_fraction > 0
        assert stats.risk_of_ruin < 0.01


class TestComputePositionSize:
    def test_conservative_without_enough_trades(self):
        from src.investor.bot_engine import _compute_position_size, TradeStats
        stats = TradeStats()  # no trades
        size = _compute_position_size(80, False, stats, 10_000)
        # Should use default 1% × conviction(0.80) = 0.8% of 10k = $80
        assert 50 < size < 200  # conservative range

    def test_high_vix_reduces_size(self):
        from src.investor.bot_engine import _compute_position_size, TradeStats
        stats = TradeStats()
        normal = _compute_position_size(80, False, stats, 10_000)
        vix_high = _compute_position_size(80, True, stats, 10_000)
        assert vix_high < normal

    def test_capped_at_2_percent(self):
        from src.investor.bot_engine import _compute_position_size, TradeStats
        # Even with aggressive Kelly, should never exceed 2%
        stats = TradeStats(kelly_fraction=0.10, has_edge=True, total_trades=50)
        size = _compute_position_size(100, False, stats, 10_000)
        assert size <= 200  # 2% of 10k


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
    bot_engine._state.sell_history = {}
    bot_engine._state.equity_curve = []
    # Reset runtime params to defaults; disable the sentiment bridge so scoring
    # tests are deterministic (no RSS-cache tilt).
    bot_engine._state.max_positions = bot_engine.MAX_POSITIONS
    bot_engine._state.min_score = bot_engine.MIN_SCORE
    bot_engine._state.max_risk_per_trade = bot_engine.MAX_RISK_PER_TRADE
    bot_engine._state.starting_capital = bot_engine.STARTING_CAPITAL
    bot_engine._state.sentiment_bridge = False
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
            "peak_price": 150.0,
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
            "peak_price": 300.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "MSFT" not in bot_engine._state.open_positions

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 795.0, "score": {"score": 75.0}})
    def test_no_exit_when_score_holds(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["NVDA"] = {
            "entry_price": 800.0, "shares": 2, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 800.0, "current_score": 80.0,
            "peak_price": 800.0,
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
            "peak_price": 200.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "TSLA" in bot_engine._state.pending_sells
        assert "TSLA" in bot_engine._state.open_positions  # not removed on fail

    @patch("src.investor.bot_engine.execute_sell")
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"error": "timeout"})
    def test_skips_score_exits_when_score_is_zero(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["AMD"] = {
            "entry_price": 120.0, "shares": 5, "entry_score": 75.0,
            "entry_time": datetime.now(), "current_price": 120.0, "current_score": 75.0,
            "peak_price": 120.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "AMD" in bot_engine._state.open_positions  # position kept
        mock_sell.assert_not_called()  # no sell attempted

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 94.0, "score": {"score": 75.0}})
    def test_hard_stop_exits_at_5_percent_loss(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["META"] = {
            "entry_price": 100.0, "shares": 5, "entry_score": 75.0,
            "entry_time": datetime.now(), "current_price": 100.0, "current_score": 75.0,
            "peak_price": 100.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "META" not in bot_engine._state.open_positions
        assert "hard stop" in bot_engine._state.trade_log[0]["reason"]

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 107.0, "score": {"score": 80.0}})
    def test_trailing_stop_exits_on_drop_from_peak(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        # Entered at 100, peaked at 112, now dropped to 107 (4.5% from peak > 3%)
        bot_engine._state.open_positions["GOOG"] = {
            "entry_price": 100.0, "shares": 5, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 112.0, "current_score": 80.0,
            "peak_price": 112.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "GOOG" not in bot_engine._state.open_positions
        assert "trailing stop" in bot_engine._state.trade_log[0]["reason"]

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 109.0, "score": {"score": 80.0}})
    def test_trailing_stop_does_not_fire_within_threshold(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        # Entered at 100, peaked at 112, now at 109 (2.7% from peak < 3%)
        bot_engine._state.open_positions["GOOG"] = {
            "entry_price": 100.0, "shares": 5, "entry_score": 80.0,
            "entry_time": datetime.now(), "current_price": 112.0, "current_score": 80.0,
            "peak_price": 112.0,
        }
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "GOOG" in bot_engine._state.open_positions
        mock_sell.assert_not_called()

    @patch("src.investor.bot_engine.execute_sell",
           return_value={"status": "executed"})
    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 70.0}})
    def test_sell_records_sell_history(self, mock_analyze, mock_sell):
        from src.investor import bot_engine
        from datetime import datetime
        bot_engine._state.open_positions["XOM"] = {
            "entry_price": 100.0, "shares": 5, "entry_score": 70.0,
            "entry_time": datetime.now(), "current_price": 100.0, "current_score": 70.0,
            "peak_price": 100.0,
        }
        # Score 70 < 40 threshold triggers exit
        mock_analyze.return_value = {"price": 100.0, "score": {"score": 35.0}}
        from src.investor.bot_engine import _check_exits
        _check_exits("pid", threading.Event())
        assert "XOM" in bot_engine._state.sell_history
        assert bot_engine._state.sell_history["XOM"]["score"] == 35.0


class TestBuildDynamicUniverse:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.scan_gap_movers", return_value={"movers": []})
    @patch("src.investor.bot_engine.scan_volume_leaders", return_value={"leaders": []})
    @patch("src.investor.bot_engine.scan_anomalies", return_value={"anomalies": []})
    @patch("src.investor.bot_engine.load_ticker_cache", return_value={
        "AAPL": {"symbol": "AAPL", "mention_count": 50},
        "TSLA": {"symbol": "TSLA", "mention_count": 30},
    })
    def test_pulls_from_ticker_cache(self, mock_cache, mock_a, mock_v, mock_g):
        from src.investor.bot_engine import _build_dynamic_universe
        result = _build_dynamic_universe()
        assert "AAPL" in result
        assert "TSLA" in result

    @patch("src.investor.bot_engine.scan_gap_movers",
           return_value={"movers": [{"symbol": "GME"}]})
    @patch("src.investor.bot_engine.scan_volume_leaders",
           return_value={"leaders": [{"symbol": "NVDA"}]})
    @patch("src.investor.bot_engine.scan_anomalies",
           return_value={"anomalies": [{"symbol": "AMC"}]})
    @patch("src.investor.bot_engine.load_ticker_cache", return_value={})
    def test_includes_mcp_scanner_results(self, mock_cache, mock_a, mock_v, mock_g):
        from src.investor.bot_engine import _build_dynamic_universe
        result = _build_dynamic_universe()
        assert "AMC" in result
        assert "NVDA" in result
        assert "GME" in result

    @patch("src.investor.bot_engine.scan_gap_movers",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.scan_volume_leaders",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.scan_anomalies",
           return_value={"error": "timeout"})
    @patch("src.investor.bot_engine.load_ticker_cache", return_value={})
    def test_falls_back_to_seed_when_all_fail(self, mock_cache, mock_a, mock_v, mock_g):
        from src.investor.bot_engine import _build_dynamic_universe, _SEED_UNIVERSE
        result = _build_dynamic_universe()
        for sym in _SEED_UNIVERSE:
            assert sym in result


class TestScanCandidates:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine._build_dynamic_universe",
           return_value=["AAPL", "MSFT", "NVDA"])
    def test_returns_symbols_from_universe(self, mock_build):
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert "AAPL" in result
        assert "MSFT" in result

    @patch("src.investor.bot_engine._build_dynamic_universe",
           return_value=["AAPL", "AAPL", "MSFT"])
    def test_deduplicates(self, mock_build):
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert result.count("AAPL") == 1

    @patch("src.investor.bot_engine._build_dynamic_universe",
           return_value=["AAPL", "HELD"])
    def test_filters_out_held_positions(self, mock_build):
        from src.investor import bot_engine
        bot_engine._state.open_positions["HELD"] = {}
        from src.investor.bot_engine import _scan_candidates
        result = _scan_candidates(threading.Event())
        assert "HELD" not in result
        assert "AAPL" in result

    @patch("src.investor.bot_engine._build_dynamic_universe", return_value=[])
    def test_returns_empty_when_no_universe(self, mock_build):
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

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 72.0}})
    def test_reentry_blocked_when_score_below_sell_plus_boost(self, mock_analyze):
        from src.investor import bot_engine
        from datetime import datetime
        # Sold BMY at score 70 — needs 70+15=85 to re-enter
        bot_engine._state.sell_history["BMY"] = {"score": 70.0, "time": datetime.now()}
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["BMY"], threading.Event())
        assert result == []

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 88.0}})
    def test_reentry_allowed_when_score_above_sell_plus_boost(self, mock_analyze):
        from src.investor import bot_engine
        from datetime import datetime
        # Sold BMY at score 70 — needs 85, scoring 88 → allowed
        bot_engine._state.sell_history["BMY"] = {"score": 70.0, "time": datetime.now()}
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["BMY"], threading.Event())
        assert len(result) == 1
        assert result[0]["ticker"] == "BMY"


class TestEnterPositions:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.execute_buy", return_value={"status": "executed"})
    def test_enters_position_and_logs_buy(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 100_000.0
        bot_engine._state.portfolio_value = 100_000.0
        from src.investor.bot_engine import _enter_positions
        # 100k portfolio, 1% default risk × 0.85 conviction = $850 → 8 shares at $100
        _enter_positions("pid", [{"ticker": "AAPL", "score": 85, "price": 100.0}],
                         False, threading.Event())
        assert "AAPL" in bot_engine._state.open_positions
        assert bot_engine._state.trade_log[0]["action"] == "BUY"
        assert mock_buy.called

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
    def test_high_vix_reduces_position_size(self, mock_buy):
        from src.investor import bot_engine
        bot_engine._state.portfolio_cash = 100_000.0
        bot_engine._state.portfolio_value = 100_000.0
        from src.investor.bot_engine import _enter_positions
        # Normal sizing
        _enter_positions("pid", [{"ticker": "NORM", "score": 85, "price": 10.0}],
                         False, threading.Event())
        normal_shares = mock_buy.call_args[0][2] if mock_buy.called else 0
        mock_buy.reset_mock()
        bot_engine._state.open_positions.clear()
        # High VIX sizing — should be smaller
        _enter_positions("pid", [{"ticker": "VIX", "score": 85, "price": 10.0}],
                         True, threading.Event())
        vix_shares = mock_buy.call_args[0][2] if mock_buy.called else 0
        assert vix_shares < normal_shares

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
        # Mark-to-market: value (10_500) - starting_capital (10_000) = 500
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

    @patch("src.investor.bot_engine.analyze_portfolio", return_value={
        "total_value": 10_250.0, "portfolio": {"current_cash": 9_000.0},
    })
    def test_snapshot_appends_equity_point(self, mock_analyze):
        from src.investor import bot_engine
        from src.investor.bot_engine import _snapshot_portfolio
        # portfolio_id stays None so _save_state no-ops (no file written in tests)
        _snapshot_portfolio("pid")
        assert len(bot_engine._state.equity_curve) == 1
        assert bot_engine._state.equity_curve[0]["value"] == pytest.approx(10_250.0)


class TestSentimentBridge:
    def setup_method(self):
        _reset_state()

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 58.0}})
    @patch("src.investor.bot_engine._rss_sentiment_map", return_value={"AAA": 5.0})
    def test_bullish_sentiment_lifts_into_buy_range(self, mock_map, mock_analyze):
        from src.investor import bot_engine
        bot_engine._state.sentiment_bridge = True
        from src.investor.bot_engine import _score_candidates
        # base 58 (< 60) + 5 tilt = 63 → now qualifies
        result = _score_candidates(["AAA"], threading.Event())
        assert len(result) == 1
        assert result[0]["sentiment_tilt"] == 5.0
        assert result[0]["score"] == pytest.approx(63.0)

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 62.0}})
    @patch("src.investor.bot_engine._rss_sentiment_map", return_value={"BBB": -8.0})
    def test_bearish_sentiment_drops_out(self, mock_map, mock_analyze):
        from src.investor import bot_engine
        bot_engine._state.sentiment_bridge = True
        from src.investor.bot_engine import _score_candidates
        # base 62 - 8 = 54 (< 60) → excluded
        assert _score_candidates(["BBB"], threading.Event()) == []

    @patch("src.investor.bot_engine.analyze_ticker",
           return_value={"price": 100.0, "score": {"score": 55.0}})
    def test_min_score_param_respected(self, mock_analyze):
        from src.investor import bot_engine
        bot_engine._state.min_score = 50  # lowered from default 60
        from src.investor.bot_engine import _score_candidates
        result = _score_candidates(["ZZZ"], threading.Event())
        assert len(result) == 1  # 55 >= 50
