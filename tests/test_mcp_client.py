"""Tests for the MCP client wrapper functions."""

import pytest
from unittest.mock import patch, MagicMock


def _mock_call_tool(tool_name, timeout=None, **kwargs):
    """Returns a dict that records what was called."""
    return {"_tool": tool_name, "_args": kwargs}


class TestWrapperFunctions:
    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_score_ticker(self, mock):
        from src.investor.mcp_client import score_ticker
        result = score_ticker("AAPL")
        assert result["_tool"] == "score_ticker"
        assert result["_args"] == {"symbol": "AAPL"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_scan_universe_joins_list(self, mock):
        from src.investor.mcp_client import scan_universe
        result = scan_universe(["AAPL", "MSFT", "GOOGL"])
        assert result["_tool"] == "scan_universe"
        assert result["_args"] == {"symbols": "AAPL,MSFT,GOOGL"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_analyze_ticker(self, mock):
        from src.investor.mcp_client import analyze_ticker
        result = analyze_ticker("TSLA")
        assert result["_tool"] == "analyze_ticker"
        assert result["_args"] == {"symbol": "TSLA"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_fundamentals(self, mock):
        from src.investor.mcp_client import get_fundamentals
        result = get_fundamentals("MSFT")
        assert result["_tool"] == "get_fundamentals"
        assert result["_args"] == {"symbol": "MSFT"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_momentum(self, mock):
        from src.investor.mcp_client import get_momentum
        result = get_momentum("NVDA")
        assert result["_tool"] == "get_momentum"
        assert result["_args"] == {"symbol": "NVDA"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_create_portfolio(self, mock):
        from src.investor.mcp_client import create_portfolio
        result = create_portfolio(100000.0, "moderate", "medium", "Test")
        assert result["_tool"] == "create_portfolio"
        assert result["_args"]["starting_capital"] == 100000.0
        assert result["_args"]["risk_profile"] == "moderate"
        assert result["_args"]["investment_horizon"] == "medium"
        assert result["_args"]["name"] == "Test"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_execute_buy(self, mock):
        from src.investor.mcp_client import execute_buy
        result = execute_buy("pid-123", "AAPL", 10)
        assert result["_tool"] == "execute_buy"
        assert result["_args"] == {"portfolio_id": "pid-123", "symbol": "AAPL", "shares": 10}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_execute_sell(self, mock):
        from src.investor.mcp_client import execute_sell
        result = execute_sell("pid-123", "AAPL", 5)
        assert result["_tool"] == "execute_sell"
        assert result["_args"] == {"portfolio_id": "pid-123", "symbol": "AAPL", "shares": 5}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_run_rebalance(self, mock):
        from src.investor.mcp_client import run_rebalance
        result = run_rebalance("pid-123", trigger="manual")
        assert result["_tool"] == "run_rebalance"
        assert result["_args"]["portfolio_id"] == "pid-123"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_detect_market_regime(self, mock):
        from src.investor.mcp_client import detect_market_regime
        result = detect_market_regime()
        assert result["_tool"] == "detect_market_regime"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_vix_analysis(self, mock):
        from src.investor.mcp_client import get_vix_analysis
        result = get_vix_analysis()
        assert result["_tool"] == "get_vix_analysis"

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_scan_anomalies_no_symbols(self, mock):
        from src.investor.mcp_client import scan_anomalies
        result = scan_anomalies()
        assert result["_tool"] == "scan_anomalies"
        assert "symbols" not in result["_args"]

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_scan_anomalies_with_symbols(self, mock):
        from src.investor.mcp_client import scan_anomalies
        result = scan_anomalies(["AAPL", "TSLA"])
        assert result["_args"] == {"symbols": "AAPL,TSLA"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_smart_money_signal(self, mock):
        from src.investor.mcp_client import get_smart_money_signal
        result = get_smart_money_signal("E-MINI S&P 500")
        assert result["_tool"] == "get_smart_money_signal"
        assert result["_args"] == {"market": "E-MINI S&P 500"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_check_risk(self, mock):
        from src.investor.mcp_client import check_risk
        result = check_risk("pid-123")
        assert result["_tool"] == "check_risk"
        assert result["_args"] == {"portfolio_id": "pid-123"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_analyze_portfolio(self, mock):
        from src.investor.mcp_client import analyze_portfolio
        result = analyze_portfolio("pid-123")
        assert result["_tool"] == "analyze_portfolio"
        assert result["_args"] == {"portfolio_id": "pid-123"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_holdings(self, mock):
        from src.investor.mcp_client import get_holdings
        result = get_holdings("pid-123")
        assert result["_tool"] == "get_holdings"
        assert result["_args"] == {"portfolio_id": "pid-123"}

    @patch("src.investor.mcp_client.call_tool", side_effect=_mock_call_tool)
    def test_get_trades(self, mock):
        from src.investor.mcp_client import get_trades
        result = get_trades("pid-123", status="executed")
        assert result["_tool"] == "get_trades"
        assert result["_args"] == {"portfolio_id": "pid-123", "status": "executed"}


class TestErrorHandling:
    @patch("src.investor.mcp_client.call_tool", return_value={"error": "Symbol not found"})
    def test_error_response_passthrough(self, mock):
        from src.investor.mcp_client import get_fundamentals
        result = get_fundamentals("INVALID")
        assert "error" in result
        assert result["error"] == "Symbol not found"

    @patch("src.investor.mcp_client._ensure_connected", side_effect=ConnectionError("no server"))
    def test_is_connected_false_without_server(self, mock):
        from src.investor.mcp_client import is_connected
        result = is_connected()
        assert result is False
